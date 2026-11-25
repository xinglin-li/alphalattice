"""SQLite online snapshots and the real DuckLake/Host readers at restore."""

import json
import shutil
import sqlite3
import struct
import threading
import uuid
from contextlib import closing
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from alphalattice.capabilities.data.errors import DataError
from alphalattice.foundation.data_layer import arrays, backup, create_data_root
from alphalattice.foundation.data_layer.chain import canonical, digest
from alphalattice.foundation.data_layer.log import HostLog, uuid7, verify_whole
from alphalattice.foundation.data_layer.project import recipe_key
from alphalattice.foundation.data_layer.store import SETTLE_FILES, StorePort, as_columns, stage

TEST_SOURCE = "src/alphalattice/foundation/data_layer/backup.py"


def _admit(root, port, close=1.0, count=1500):
    """Commit synthetic normalized bars, enough for an actual Parquet file, their source writing
    them now, so a later call's value supersedes an earlier one's."""
    written = datetime.now(UTC).isoformat()
    rows = [
        {
            "listing_id": str(uuid.UUID(int=7)),
            "session_date": f"{2000 + n // 336}-{n % 336 // 28 + 1:02d}-{n % 28 + 1:02d}",
            "close": close,
            "source_observed_at": written,
        }
        for n in range(count)
    ]
    key = {"family": "provider", "name": "synthetic", "version": 1}
    if not port.log.read("SELECT 1 FROM definition_name WHERE family = 'provider'"):
        text = canonical(key | {"text": "Synthetic bars."}).decode()
        port.log.record(
            "definition.registered", ("definition", "provider/synthetic"), key, text=text
        )
    provider = port.log.require_lineage("provider", "synthetic")
    with stage(root, "raw_bar", provider, pa.table(as_columns(rows)), floor_bytes=0) as staged:
        return port.commit(staged)


def _sqlite_rows(path):
    """All retained rows, including FTS storage, read through SQLite."""
    with closing(sqlite3.connect(path)) as con:
        tables = con.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        return {
            name: sorted(
                con.execute('SELECT * FROM "' + name.replace('"', '""') + '"').fetchall(), key=repr
            )
            for (name,) in tables
        }


def _catalog_rows(port):
    """Every retained world and results table through the existing catalog reader."""
    tables = port.lake.execute(
        "SELECT table_schema, table_name FROM information_schema.tables"
        " WHERE table_catalog = 'lake' AND table_schema IN ('world', 'results')"
    ).fetchall()
    return {
        (schema, table): sorted(
            port.lake.execute(f'SELECT * FROM lake."{schema}"."{table}"').fetchall(), key=repr
        )
        for schema, table in tables
    }


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    """An admitted root and its immutable backup, shared read-only by the restore cases."""
    base = tmp_path_factory.mktemp("machine-backup")
    root, repository = base / "source", base / "backups"
    create_data_root(root)
    with (
        HostLog.open(root) as log,
        closing(StorePort.open(root, log)) as port,
        port.lake.cursor() as reader,
    ):
        reader.execute("BEGIN").complete()
        count_docs = "SELECT count(*) FROM lake.world.evidence_doc"
        assert reader.execute(count_docs).fetchall() == [(0,)]
        _admit(root, port)
        arrays.keep(root, log, {"value": np.array([1.0, 2.0])}, workspace=None)
        arrays.keep_packed(root, log, b"retained packed array", workspace=None)
        (root / "blobs/result").write_bytes(b"retained result")
        (root / "blobs/document").write_bytes(b"admitted synthetic Evidence document")
        log.record(
            "artifact.produced",
            ("artifact", "result"),
            {
                "type": "report",
                "type_version": 1,
                "meaning": "synthetic result",
                "identity_kind": "content",
                "storage": "blob",
                "content_digest": digest(b"retained result"),
            },
        )
        port.lake.execute(
            "INSERT INTO lake.world.evidence_doc (doc_id, version_id, provider_id, source_ref,"
            " doc_digest, byte_count, media_type, title, document_type, entity_id,"
            " available_at, availability_basis, right_of_use, known_asof, retracted,"
            " admission_id) VALUES (uuidv7(), uuidv7(), 'synthetic', 'synthetic filing',"
            " 'document', 1, 'text/html', 'synthetic filing', '10-K', 'AAA', now(),"
            " 'OFFICIAL_ACCEPTANCE', 'SEC_PUBLIC_OFFICIAL_ACCESS', now(), false,"
            " 'document-admission')"
        ).complete()
        log.flush()
        (root / "staging/transport.json").write_bytes(b"unadmitted transport")
        first = backup.backup_data_root(root, repository)
        (root / "blobs/later").write_bytes(b"second retained result")
        method = {"family": "method", "name": "synthetic", "version": 1}
        log.record(
            "definition.registered",
            ("definition", "method/synthetic"),
            method,
            text=canonical(method | {"text": "A synthetic method."}).decode(),
        )
        later = recipe_key(log.require_lineage("method", "synthetic"), 1, {})
        defined = {"method": "synthetic", "method_version": 1, "params": {}, "recipe_key": later}
        log.record("recipe.defined", ("recipe", uuid7()), defined)
        log.record(  # a recipe's blob: its row records no digest, so the backup proves its own
            "artifact.produced",
            ("artifact", "later"),
            {
                "type": "report",
                "type_version": 1,
                "meaning": "second synthetic result",
                "identity_kind": "recipe",
                "recipe_key": later,
                "storage": "blob",
            },
        )
        log.flush()
        assert (root / "machine/catalog.sqlite-wal").stat().st_size > 32
        generation = backup.backup_data_root(root, repository)
        assert reader.execute(count_docs).fetchall() == [(0,)]
        reader.execute("ROLLBACK").complete()
        catalog = _catalog_rows(port)
    expected = {name: _sqlite_rows(root / "machine" / name) for name in ("log.sqlite",)}
    return generation, repository, expected, catalog, first


@pytest.mark.proves("data_layer.machine_backup_restores_all_retained_results")
def test_a_machine_backup_restores_every_retained_result_all_equal(sample, tmp_path):
    """Integration: real SQLite/vec and DuckLake readback cannot be proved by an in-memory unit."""
    generation, repository, expected, catalog, first = sample
    restored = backup.restore_data_root(generation, tmp_path / "unused/../restored")
    assert {p.name for p in (generation / "machine").glob("catalog.sqlite*")} == {"catalog.sqlite"}
    assert {p.name for p in (restored / "machine").glob("catalog.sqlite*")} == {"catalog.sqlite"}
    restored_files = backup.catalog_table_files(restored / "machine/catalog.sqlite")
    assert {"world.raw_bar"} <= {table for tables in restored_files.values() for table in tables}
    assert all((restored / handle).is_file() for handle in restored_files)
    assert {name: _sqlite_rows(restored / "machine" / name) for name in expected} == expected
    with HostLog.open(restored) as log:
        port = StorePort.open(restored, log)
        assert _catalog_rows(port) == catalog
        lane = {"value": np.array([1.0, 2.0])}
        assert (
            arrays.read_packed(restored, log.read, arrays.digest(lane)) == lane["value"].tobytes()
        )
        assert (
            arrays.read_packed(
                restored, log.read, arrays.describe_packed(b"retained packed array").content_digest
            )
            == b"retained packed array"
        )
        port.close()
    assert (restored / "blobs/result").read_bytes() == b"retained result"
    assert (restored / "blobs/document").read_bytes() == b"admitted synthetic Evidence document"
    assert (restored / "blobs/later").read_bytes() == b"second retained result"
    manifest = json.loads((generation / "manifest.json").read_text(encoding="utf-8"))
    assert backup.verify_data_root(generation)["head"] == manifest["head"]
    array_entries = [entry for entry in manifest["files"] if entry.get("kind") == "array"]
    assert len(array_entries) == 2
    assert {entry["digest"] for entry in array_entries} == {
        arrays.digest({"value": np.array([1.0, 2.0])}),
        arrays.describe_packed(b"retained packed array").content_digest,
    }
    assert manifest["copied_files"] == ["blobs/later"]
    assert next(
        entry["digest"] for entry in manifest["files"] if entry["path"] == "blobs/result"
    ) == digest(b"retained result")
    assert next(
        entry["digest"] for entry in manifest["files"] if entry["path"] == "blobs/later"
    ).startswith("blake3:v1:")
    first_head = json.loads((first / "manifest.json").read_text(encoding="utf-8"))["head"]
    assert manifest["head"]["seq"] > first_head["seq"]
    paths = [p.relative_to(repository).as_posix() for p in repository.rglob("*") if p.is_file()]
    assert all(
        "host.lock" not in p and "staging/" not in p and "transport.json" not in p for p in paths
    )
    with pytest.raises(DataError, match="exists") as refused:
        backup.restore_data_root(generation, restored)
    assert refused.value.failure.code == "data_layer.restore_target_exists"


def _left(root):
    """The dropped table's files still in `root`."""
    return sorted((root / "data/main/t").glob("*.parquet"))


def _cleaned(root):
    """The Host's maintenance past the settle time: the dropped table's files left in `root`."""
    with HostLog.open(root) as log:
        port = StorePort.open(root, log)
        port.maintain(datetime.now(UTC) + 2 * SETTLE_FILES)
        port.close()
    return _left(root)


def _as_restored_earlier(original, root):
    """Give the restored `root` what an earlier restore left where DuckLake records a scheduled
    deletion by its absolute path (Windows; macOS records it relative): the original folder's
    path for each, since that restore moved only the data path."""
    with closing(sqlite3.connect(original / "machine/catalog.sqlite")) as old:
        ((data,),) = old.execute(
            "SELECT value FROM ducklake_metadata WHERE key = 'data_path' AND scope IS NULL"
        )
    with closing(sqlite3.connect(root / "machine/catalog.sqlite")) as new:
        new.execute(
            "UPDATE ducklake_files_scheduled_for_deletion SET path = ? || path,"
            " path_is_relative = false WHERE path_is_relative",
            (data,),
        )
        new.commit()


# What the Host does with each case's root as it stands. DuckLake refuses a data path other than
# the one its catalog records, so a plain copy or move opens only through backup and restore; a
# deletion an earlier restore left pointing at the original's folder is dropped, its file kept.
RELOCATIONS = {"in place": "cleans", "copied, original present": "refused", "moved": "refused"}
RELOCATIONS |= {"restored, original present": "cleans", "restored, original gone": "cleans"}
RELOCATIONS |= {"restored by an earlier release": "drops"}
CASES = [*RELOCATIONS, "a backup that fails midway"]


@pytest.mark.proves("data_layer.machine_backup_restores_all_retained_results")
@pytest.mark.parametrize("case", CASES, ids=CASES)
def test_a_relocated_root_backs_up_and_cleans_up_only_its_own_folder(tmp_path, case):
    """Integration: DuckLake records a file it schedules for deletion by its absolute path; only
    a real catalog, its copy, move and restore, and the Host's own maintenance show that a root in
    any folder backs up, restores ALL EQUAL and deletes nothing outside itself."""
    original, first = tmp_path / "original", tmp_path / "first"
    create_data_root(original)
    with HostLog.open(original) as log:
        port = StorePort.open(original, log)
        _admit(original, port)
        for sql in ("CREATE TABLE lake.main.t AS FROM range(5000)", "DROP TABLE lake.main.t"):
            port.lake.execute(sql).fetchall()
        port.lake.execute("CALL ducklake_expire_snapshots('lake', older_than => now())").fetchall()
        log.flush()
        rows = _catalog_rows(port)
        port.close()
    assert {p.name for p in (original / "machine").glob("catalog.sqlite*")} == {"catalog.sqlite"}
    if case == "a backup that fails midway":  # a file outside the root: refused by name
        with closing(sqlite3.connect(original / "machine/catalog.sqlite")) as con:
            con.execute("UPDATE ducklake_data_file SET path = '/x.parquet', path_is_relative = 0")
            con.commit()
        with pytest.raises(DataError) as refused:
            backup.backup_data_root(original, tmp_path / "backups")
        assert refused.value.failure.code == "data_layer.backup_path_aliased"
        assert not list((tmp_path / "backups/generations").iterdir())
        return
    root = original
    if case.startswith("restored"):
        root = backup.restore_data_root(backup.backup_data_root(original, first), tmp_path / "r")
        if case.endswith("gone"):
            shutil.rmtree(original)
        elif case.endswith("release"):
            _as_restored_earlier(original, root)
    elif case == "moved":
        root = original.rename(tmp_path / "moved")
    elif case.startswith("copied"):
        root = Path(shutil.copytree(original, tmp_path / "copy"))
    others = {f: sorted(f.rglob("*")) for f in (original, first) if f != root and f.exists()}
    generation = backup.backup_data_root(root, tmp_path / "backups")
    assert original not in others or _left(original)  # each folder holds its file before cleanup
    if RELOCATIONS[case] == "refused":
        with HostLog.open(root) as log, pytest.raises(duckdb.Error, match="data path"):
            StorePort.open(root, log)
    else:
        assert _left(root)
        assert bool(_cleaned(root)) == (RELOCATIONS[case] == "drops")
    again = backup.restore_data_root(generation, tmp_path / "again")
    assert {p.name for p in (again / "machine").glob("catalog.sqlite*")} == {"catalog.sqlite"}
    with HostLog.open(again) as log:
        port = StorePort.open(again, log)
        assert _catalog_rows(port) == rows
        port.close()
    assert bool(_left(again)) != (RELOCATIONS[case] == "drops")  # a dropped one is not carried
    assert not _cleaned(again)
    assert {f: sorted(f.rglob("*")) for f in others} == others


@pytest.mark.proves("data_layer.machine_restore_refuses_a_changed_chain")
@pytest.mark.parametrize(
    "damage", ["chain", "file-bytes", "file-set", "array-bytes", "array-missing", "array-kind"]
)
def test_a_changed_backup_chain_is_refused_before_the_fresh_root_exists(sample, tmp_path, damage):
    """Integration: the real copied owners and bytes must refuse tampering before publication."""
    generation, repository, _, _, _ = sample
    damaged = tmp_path / "damaged"
    shutil.copytree(repository, damaged)
    changed = damaged / "generations" / generation.name
    if damage in {"array-bytes", "array-missing", "array-kind"}:
        manifest_path = changed / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        lane = arrays.describe({"value": np.array([1.0, 2.0])})
        entry = next(
            entry for entry in manifest["files"] if entry["path"] == "blobs/" + lane.artifact_id
        )
        body = damaged / "files" / entry["path"]
        if damage == "array-bytes":
            body.write_bytes(arrays.describe({"value": np.array([999.0, 2.0])}).body())
            entry["size"] = (
                body.stat().st_size
            )  # the untrusted manifest cannot bless altered columns
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        elif damage == "array-missing":
            body.unlink()
        else:
            entry.pop("kind")  # an array cannot be downgraded to an ordinary file seal
            entry["digest"] = digest(body.read_bytes())
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(DataError) as refused:
            backup.verify_data_root(changed)
        if damage == "array-bytes":
            assert refused.value.failure.code == "data_layer.array_tampered"
        target = tmp_path / "refused-array"
        with pytest.raises(DataError):
            backup.restore_data_root(changed, target)
        assert not target.exists()
        return
    if damage != "chain":
        manifest_path = changed / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        parquet = next(
            entry["path"] for entry in manifest["files"] if entry["path"].endswith(".parquet")
        )
        handles = (
            ("blobs/result", parquet)
            if damage == "file-bytes"
            else ("blobs/result", parquet, "blobs/document")
        )
        for number, handle in enumerate(handles):
            body = damaged / "files" / handle
            original = body.read_bytes()
            if damage == "file-bytes":
                altered = bytes([original[0] ^ 1]) + original[1:]
                if handle.endswith(".parquet"):
                    with closing(pq.ParquetFile(body, page_checksum_verification=True)) as reader:
                        columns = reader.metadata.row_group(0)
                        column = next(
                            columns.column(i)
                            for i in range(columns.num_columns)
                            if columns.column(i).path_in_schema == "close"
                        )
                        offset = original.find(
                            struct.pack("<d", 1.0),
                            column.dictionary_page_offset,
                            column.data_page_offset,
                        )
                    assert offset >= 0
                    altered = original[:offset] + struct.pack("<d", 2.0) + original[offset + 8 :]
                assert len(altered) == len(original)
                body.write_bytes(altered)
                if handle == "blobs/result":
                    resealed = json.loads(json.dumps(manifest))
                    next(entry for entry in resealed["files"] if entry["path"] == handle)[
                        "digest"
                    ] = digest(altered)
                    manifest_path.write_text(json.dumps(resealed), encoding="utf-8")
            else:
                revised = {
                    **manifest,
                    "files": [entry for entry in manifest["files"] if entry["path"] != handle],
                }
                manifest_path.write_text(json.dumps(revised), encoding="utf-8")
            target = tmp_path / f"refused-{number}"
            with pytest.raises(DataError):
                backup.restore_data_root(changed, target)
            assert not target.exists()
            if damage == "file-bytes":
                with pytest.raises(DataError):
                    backup.backup_data_root(repository.parent / "source", damaged)
            body.write_bytes(original)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        if damage == "file-set":
            with closing(sqlite3.connect(changed / "machine/catalog.sqlite")) as catalog:
                catalog.execute("UPDATE ducklake_data_file SET table_id=999999")
                catalog.commit()
            with pytest.raises(DataError) as unresolved:
                backup.restore_data_root(changed, tmp_path / "unresolved-owner")
            assert unresolved.value.failure.code == "data_layer.backup_catalog_invalid"
            assert not (tmp_path / "unresolved-owner").exists()
        return
    with closing(sqlite3.connect(changed / "machine/log.sqlite")) as log:
        log.execute("UPDATE chain_head SET head = ?", ("blake3:" + "0" * 64,))
        log.commit()
    target = tmp_path / "refused"
    with pytest.raises(DataError, match="chain") as refused:
        backup.restore_data_root(changed, target)
    assert refused.value.failure.code == "data_layer.backup_chain_broken"
    assert not target.exists()
    (changed / "machine/log.sqlite").write_bytes(b"not a SQLite database")
    with pytest.raises(DataError, match="cannot be verified") as unreadable:
        backup.restore_data_root(changed, target)
    assert unreadable.value.failure.code == "data_layer.backup_unreadable"
    assert not target.exists()


@pytest.mark.proves("data_layer.machine_backup_keeps_a_repairable_cut")
def test_a_write_burst_finishes_between_log_and_catalog_backup_and_restore_repairs(
    tmp_path, monkeypatch
):
    """Integration: the real writer commits between SQLite copies without a backup-length lock."""
    root = tmp_path / "source"
    create_data_root(root)
    copied, finished, errors = threading.Event(), threading.Event(), []

    class ObservedBackup(sqlite3.Connection):
        """Observe the public SQLite copy boundary after its real page-copy operation."""

        def backup(self, target, **kwargs):
            super().backup(target, **kwargs)
            path = self.execute("PRAGMA database_list").fetchone()[2]
            if path.endswith("log.sqlite"):
                copied.set()
                assert finished.wait(5), "backup held the writer across its copies"

    with HostLog.open(root) as log:
        port = StorePort.open(root, log)
        _admit(root, port)

        def writer():
            """Commit only while the backup is between its log and catalog copies."""
            try:
                assert copied.wait(5), "backup did not copy its log"
                _admit(root, port, close=2.0, count=3)
                for _ in range(128):
                    log.record(
                        "tool.call", attrs={"tool": "synthetic"}, phase="completed", started_us=0
                    )
                log.flush()
            except BaseException as error:
                errors.append(error)
            finally:
                finished.set()

        monkeypatch.setattr(sqlite3, "connect", partial(sqlite3.connect, factory=ObservedBackup))
        thread = threading.Thread(target=writer)
        thread.start()
        try:
            generation = backup.backup_data_root(root, tmp_path / "backups")
        finally:
            copied.set()
            thread.join(timeout=5)
        assert not thread.is_alive() and errors == []
        expected = _catalog_rows(port)
        assert log.read("SELECT count(*) FROM event WHERE type = 'tool.call'")[0][0] == 128
        port.close()
    monkeypatch.undo()
    saved = json.loads((generation / "manifest.json").read_text(encoding="utf-8"))["head"]
    restored = backup.restore_data_root(generation, tmp_path / "restored")
    with HostLog.open(restored) as log:
        port = StorePort.open(restored, log)
        assert _catalog_rows(port) == expected
        assert len(log.read("SELECT 1 FROM admission")) == 2
        assert log.read("SELECT seq FROM chain_head")[0][0] > saved["seq"]
        port.close()
    with closing(sqlite3.connect(restored / "machine/log.sqlite")) as con:
        con.row_factory = sqlite3.Row
        assert verify_whole(con)
