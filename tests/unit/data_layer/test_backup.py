"""The backup's manifest and immutable-file decisions, without databases."""

import json

import pytest
from blake3 import blake3

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer import backup

TEST_SOURCE = "src/alphalattice/kernel/data_layer/backup.py"
HEAD = {"seq": 7, "head": "blake3:" + "a" * 64, "chain_version": 1}
FILE = {"path": "data/bar.parquet", "size": 4, "digest": "blake3:v1:" + blake3(b"PAR1").hexdigest()}
MANIFEST = {
    "version": 2,
    "databases": ["index.sqlite", "log.sqlite", "catalog.sqlite"],
    "head": HEAD,
    "files": [FILE],
}


@pytest.mark.proves("data_layer.backup_manifest_is_portable")
def test_a_manifest_keeps_its_saved_head_and_relative_file_set(tmp_path):
    """The restore reader accepts portable database/file handles and keeps the saved head."""
    array = {"path": "blobs/" + "b" * 64, "size": 4, "kind": "array", "digest": "a" * 64}
    expected = MANIFEST | {"files": [FILE, array]}
    (tmp_path / "manifest.json").write_text(json.dumps(expected), encoding="utf-8")
    try:
        found = backup.read_manifest(tmp_path)
    except DataError as error:
        found = {"refused": error.failure.code}
    assert found.get("head") == HEAD, found
    assert found.get("files") == [FILE, array]


@pytest.mark.proves("data_layer.backup_manifest_is_portable")
@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"version": 1}, id="unsealed old format"),
        pytest.param({"version": True}, id="a boolean is not a version"),
        pytest.param({"databases": ["log.sqlite"]}, id="missing database"),
        pytest.param({"head": {"seq": 7, "head": "bad", "chain_version": 1}}, id="unbound head"),
        pytest.param({"files": [FILE | {"path": "../outside"}]}, id="parent traversal"),
        pytest.param({"files": [FILE | {"path": "data/invalid\0"}]}, id="null byte"),
        pytest.param({"files": [FILE | {"path": "C:/outside"}]}, id="absolute drive"),
        pytest.param({"files": [FILE | {"path": "machine/host.lock"}]}, id="process lock"),
        pytest.param({"files": [FILE | {"size": -1}]}, id="negative size"),
        pytest.param({"files": [FILE | {"kind": "unknown"}]}, id="unknown body kind"),
        pytest.param(
            {"files": [FILE | {"kind": "array", "digest": "0" * 64}]}, id="array outside blobs"
        ),
        pytest.param(
            {"files": [FILE | {"kind": "array", "path": "blobs/a", "digest": "bad"}]},
            id="malformed array identity",
        ),
        pytest.param({"files": [FILE, FILE]}, id="duplicate handle"),
        pytest.param(
            {"files": [FILE | {"digest": "blake3:v2:" + "a" * 64}]}, id="unknown seal version"
        ),
    ],
)
def test_an_unsafe_or_incomplete_manifest_is_refused_before_restore(tmp_path, change):
    """A malformed manifest cannot choose a database or write beyond the admitted file set."""
    (tmp_path / "manifest.json").write_text(json.dumps(MANIFEST | change), encoding="utf-8")
    with pytest.raises(DataError, match="manifest") as refused:
        backup.read_manifest(tmp_path)
    assert refused.value.failure.code == "data_layer.backup_manifest_invalid"


@pytest.mark.proves("data_layer.backup_copies_only_new_files")
def test_the_file_pool_reuses_a_sealed_name_and_copies_only_its_new_file(tmp_path):
    """Reuse an old file, copy a new one, and refuse a resized sealed name."""
    source, pool = tmp_path / "source", tmp_path / "pool"
    (source / "data").mkdir(parents=True)
    (source / "blobs").mkdir()
    (source / "data/bar.parquet").write_bytes(b"PAR1")
    assert backup.copy_files(source, pool, [FILE]) == ("data/bar.parquet",)
    held_at = (pool / "data/bar.parquet").stat().st_mtime_ns
    (source / "blobs/result").write_bytes(b"kept result")
    blob = {
        "path": "blobs/result",
        "size": 11,
        "digest": "blake3:v1:" + blake3(b"kept result").hexdigest(),
    }
    assert backup.copy_files(source, pool, [FILE, blob]) == ("blobs/result",)
    assert (pool / "data/bar.parquet").stat().st_mtime_ns == held_at
    assert (pool / "blobs/result").read_bytes() == b"kept result"
    (pool / "data/bar.parquet").write_bytes(b"BAD1")
    with pytest.raises(DataError) as same_size:
        backup.copy_files(source, pool, [FILE])
    assert same_size.value.failure.code == "data_layer.backup_file_changed"
    (pool / "data/bar.parquet").write_bytes(b"PAR1")
    (source / "data/bar.parquet").write_bytes(b"changed sealed name")
    with pytest.raises(DataError, match="sealed") as refused:
        backup.copy_files(source, pool, [FILE | {"size": len(b"changed sealed name")}])
    assert refused.value.failure.code == "data_layer.backup_file_changed"
