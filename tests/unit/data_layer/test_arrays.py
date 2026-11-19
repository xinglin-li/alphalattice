"""The array door's decisions, without a log: what names an array, how its body reads back against
its profile, and what start does with a staged body."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pandas  # noqa: F401  # Arrow's NumPy conversion loads it: at collection, not in a test
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer import arrays

TEST_SOURCE = "src/alphalattice/kernel/data_layer/arrays.py"
NAN = np.frombuffer(struct.pack("<Q", 0x7FF8000000000123), dtype="<f8")[0]  # a NaN with a payload
KINDS: dict[str, dict[str, Any]] = {
    "float64 NaN payload, -0": {"value": np.array([1.0, -0.0, NAN, np.inf])},
    "float32": {"value": np.array([-0.0, 2.5], dtype="<f4")},
    "int64": {"value": np.array([-(2**62), 7], dtype="<i8")},
    "int32": {"value": np.array([3, -1], dtype="<i4")},
    "bool": {"value": np.array([True, False, True])},
    "three columns of one 2-D shape": {
        "scores": np.array([[0.5, -0.0], [NAN, 1.0]]),
        "returns": np.array([[0.1, 0.2], [0.3, 0.4]]),
        "eligible": np.array([[True, False], [True, True]]),
    },
}


def _kept(root: Path, array: arrays.Described) -> tuple[Path, arrays.Read]:
    """The array's body under `root`'s blobs/, and a log that holds its profile."""
    path = root / "blobs" / array.artifact_id
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(array.body())
    rows = [(array.artifact_id, "present", json.dumps(array.profile))]
    return path, lambda _sql, _args: rows


def _packed(columns: dict[str, Any]) -> bytes:
    return b"".join(np.ascontiguousarray(values).tobytes() for values in columns.values())


@pytest.mark.proves("data_layer.array_door_keeps_every_lane_kind_bit_exact")
@pytest.mark.parametrize("kind", list(KINDS), ids=list(KINDS))
def test_every_lane_kind_reads_back_bit_exact(tmp_path: Path, kind: str) -> None:
    """Columns keep their dtype and bits, NaN payloads and signed zeros included, under the digest
    a document cites, read back flat as columns and whole as packed bytes."""
    columns = KINDS[kind]
    array = arrays.describe(columns)
    assert array.content_digest == hashlib.sha256(_packed(columns)).hexdigest()
    _path, read = _kept(tmp_path, array)
    back = arrays.read_columns(tmp_path, read, array.content_digest)
    assert {name: (v.dtype.str, v.tobytes()) for name, v in back.items()} == {
        name: (v.dtype.str, v.tobytes()) for name, v in columns.items()
    }
    assert arrays.read_packed(tmp_path, read, array.content_digest) == _packed(columns)
    with pytest.raises(DataError) as refused:
        arrays.read_packed(tmp_path, read, array.content_digest, packed_only=True)
    assert refused.value.failure.code == "data_layer.array_profile_mismatch"


@pytest.mark.proves("data_layer.array_door_keeps_every_lane_kind_bit_exact")
def test_a_legacy_packed_lane_reads_back_as_its_bytes_alone(tmp_path: Path) -> None:
    """A lane known only by its packed bytes reads back as those bytes, never as columns."""
    array = arrays.describe_packed(b"\x00\x01legacy lane")
    _path, read = _kept(tmp_path, array)
    assert arrays.read_packed(tmp_path, read, array.content_digest) == b"\x00\x01legacy lane"
    assert (
        arrays.read_packed(tmp_path, read, array.content_digest, packed_only=True)
        == b"\x00\x01legacy lane"
    )
    with pytest.raises(DataError) as refused:
        arrays.read_columns(tmp_path, read, array.content_digest)
    assert refused.value.failure.code == "data_layer.array_profile_mismatch"


@pytest.mark.proves("data_layer.array_door_refuses_a_wrong_dtype_or_shape_by_name")
def test_equal_bytes_of_another_dtype_or_shape_are_another_array() -> None:
    """One digest, three arrays: the artifact_id binds the dtype and shape with the bytes."""
    lane = np.array([1.0, 2.0, 3.0, 4.0])
    forms = (lane, lane.view("<i8"), lane.reshape(2, 2))
    named = [arrays.describe({"value": values}) for values in forms]
    assert len({array.content_digest for array in named}) == 1
    assert len({array.artifact_id for array in named}) == 3


def _rewrite(path: Path, values: np.ndarray, shape: list[int]) -> None:
    table = pa.table({"value": values}).replace_schema_metadata({"shape": json.dumps(shape)})
    pq.write_table(table, path, use_dictionary=False)


@pytest.mark.proves("data_layer.array_door_refuses_a_wrong_dtype_or_shape_by_name")
@pytest.mark.parametrize(
    ("change", "code"),
    [
        pytest.param(
            lambda path, lane: _rewrite(path, lane.view("<i8"), [4]),
            "data_layer.array_profile_mismatch",
            id="the same bytes as another dtype",
        ),
        pytest.param(
            lambda path, lane: _rewrite(path, lane, [2, 2]),
            "data_layer.array_profile_mismatch",
            id="another shape",
        ),
        pytest.param(
            lambda path, lane: _rewrite(path, lane + 1.0, [4]),
            "data_layer.array_tampered",
            id="other values",
        ),
        pytest.param(
            lambda path, _lane: path.write_bytes(b"not parquet"),
            "data_layer.array_tampered",
            id="a body that does not decode",
        ),
        pytest.param(lambda path, _lane: path.unlink(), "data_layer.array_missing", id="no body"),
    ],
)
def test_a_body_that_is_not_its_profile_is_refused_by_name(
    tmp_path: Path, change: Any, code: str
) -> None:
    """A read checks the body's columns, dtype and shape against its profile and proves its bytes;
    whatever differs is refused by its own name."""
    lane = np.array([1.0, 2.0, 3.0, 4.0])
    array = arrays.describe({"value": lane})
    path, read = _kept(tmp_path, array)
    change(path, lane)
    with pytest.raises(DataError) as refused:
        arrays.read_columns(tmp_path, read, array.content_digest)
    assert refused.value.failure.code == code


@pytest.mark.proves("data_layer.array_door_refuses_a_wrong_dtype_or_shape_by_name")
def test_a_profile_naming_another_arrays_body_under_the_asked_digest_is_refused(
    tmp_path: Path,
) -> None:
    """The proof binds the digest a read asks for: a profile that maps it to another, whole array's
    body and id is refused, never answered with that array's bytes."""
    asked = arrays.describe({"value": np.array([1.0, 2.0])})
    other = arrays.describe({"value": np.array([3.0, 4.0])})
    _path, read = _kept(tmp_path, other)
    with pytest.raises(DataError) as refused:
        arrays.read_packed(tmp_path, read, asked.content_digest)
    assert refused.value.failure.code == "data_layer.array_tampered"


@pytest.mark.proves("data_layer.array_door_refuses_a_wrong_dtype_or_shape_by_name")
@pytest.mark.parametrize("suffix", [".parquet", ".bin"], ids=["a table", "legacy packed bytes"])
def test_a_lane_file_read_in_place_proves_its_name(tmp_path: Path, suffix: str) -> None:
    """A carried lane file reads back under its name and is refused once its bytes change."""
    lane = np.array([1.0, -0.0])
    array = arrays.describe({"value": lane})
    path = tmp_path / f"{array.content_digest}{suffix}"
    if suffix == ".bin":
        path.write_bytes(lane.tobytes())
    else:
        _rewrite(path, lane, [2])
    found = arrays.read_file(path)
    assert (found if isinstance(found, bytes) else found["value"].tobytes()) == lane.tobytes()
    if suffix == ".bin":
        path.write_bytes((lane + 1.0).tobytes())
    else:
        _rewrite(path, lane + 1.0, [2])
    with pytest.raises(DataError) as refused:
        arrays.read_file(path)
    assert refused.value.failure.code == "data_layer.array_tampered"


@pytest.mark.proves("data_layer.array_start_places_only_a_committed_staged_body")
@pytest.mark.parametrize(
    ("committed", "placed"),
    [
        pytest.param(False, False, id="its profile never committed: collected"),
        pytest.param(True, True, id="its profile committed: placed"),
    ],
)
def test_start_places_a_staged_body_only_when_its_profile_committed(
    tmp_path: Path, committed: bool, placed: bool
) -> None:
    """A body a crash left staged goes into place when the log holds its profile, and is handed to
    the collector when it does not."""
    array = arrays.describe({"value": np.array([1.0, -0.0])})
    staged = tmp_path / "blobs" / f".{array.artifact_id}.4242{arrays.STAGED}"
    staged.parent.mkdir()
    staged.write_bytes(array.body())
    rows = [("present",)] if committed else []
    collected: list[str] = []

    def collect(handle: str) -> bool:
        collected.append(handle)
        (tmp_path / handle).unlink()
        return True

    report = arrays.settle(tmp_path, lambda _sql, _args: rows, collect)
    handle = f"blobs/{array.artifact_id}" if placed else f"blobs/{staged.name}"
    assert report == {
        "placed": [handle] if placed else [],
        "collected": [] if placed else [handle],
        "waiting": [],
    }
    left = [path.name for path in staged.parent.iterdir()]
    assert left == ([array.artifact_id] if placed else [])
