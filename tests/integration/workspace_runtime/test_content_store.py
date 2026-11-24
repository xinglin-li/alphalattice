"""The content store keeps each file in its DA9 type and every digest it names.

Numerical lanes are Parquet, named by the SHA-256 of their C-order bytes, so a lane sealed as
packed `.bin` bytes keeps its name; a store written before the change is still read.

A sealed source is proved where it enters, where it leaves and by the scrub, never on an ordinary
read (LAWS.md EV2): a read takes its bytes at their name, and full proof refuses a change whatever
the file's stamps.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from queue import Queue
from threading import Barrier, BoundedSemaphore, Event, Lock, Semaphore, Thread, current_thread
from time import time_ns
from typing import ClassVar, Literal
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import BaseModel, ConfigDict

from alphalattice.control.workspace_runtime.content_store import (
    ContentAddressedStore,
    ContentAddressedStoreError,
    columns_digest,
    full_content_proof,
    verified_model_read_scope,
    verified_npz_arrays,
    verified_request_proof,
    verified_request_value,
    verified_source_value,
    verify_source_checks,
)
from alphalattice.control.workspace_runtime.database import WorkspaceDatabase, table_columns
from alphalattice.kernel.config import Change, override
from alphalattice.kernel.data_layer import backup
from alphalattice.kernel.identity import file_facts as digests
from alphalattice.kernel.identity import file_stamp
from alphalattice.kernel.shared_kernel.spans import collect


def _store(tmp_path: Path) -> ContentAddressedStore:
    return ContentAddressedStore(tmp_path / "store", uri_prefix="artifact://qa")


class _FrozenReadRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    content_hash: str
    alternate_hash: str
    text: Literal["sealed"]
    validation_calls: ClassVar[int] = 0

    @classmethod
    def model_validate_json(cls, json_data, **kwargs):
        cls.validation_calls += 1
        return super().model_validate_json(json_data, **kwargs)


class _OtherFrozenReadRecord(_FrozenReadRecord):
    validation_calls: ClassVar[int] = 0


@dataclass(frozen=True)
class _FrozenReadRequest:
    operations: tuple[str, ...]


class _ReadPlan(_FrozenReadRecord):
    request: _FrozenReadRequest
    validation_calls: ClassVar[int] = 0


def test_verified_plan_reuses_frozen_dataclass_request_without_bypassing_validation(
    tmp_path, kept_here
):
    store = _store(tmp_path)
    plan = _ReadPlan(
        content_hash="a" * 64,
        alternate_hash="b" * 64,
        text="sealed",
        request=_FrozenReadRequest(("READ",)),
    )
    store.publish_model(category="plans", value=plan, identity_field="content_hash")
    _ReadPlan.validation_calls = 0
    with verified_model_read_scope():
        first = store.load_model(
            category="plans",
            content_hash=plan.content_hash,
            model=_ReadPlan,
            identity_field="content_hash",
        )
        second = store.load_model(
            category="plans",
            content_hash=plan.content_hash,
            model=_ReadPlan,
            identity_field="content_hash",
        )
    assert first == second == plan and (first is second) is kept_here
    assert _ReadPlan.validation_calls == (1 if kept_here else 2)


def _published_read_record(tmp_path: Path, *, root: str = "store"):
    store = ContentAddressedStore(tmp_path / root, uri_prefix="artifact://qa")
    value = _FrozenReadRecord(
        content_hash="a" * 64,
        alternate_hash="b" * 64,
        text="sealed",
    )
    store.publish_model(category="records", value=value, identity_field="content_hash")
    path = store.root / "records" / f"{value.content_hash}.json"
    return store, value.content_hash, path


def _load_read_record(
    store: ContentAddressedStore, identity: str, *, identity_field="content_hash"
):
    return store.load_model(
        category="records",
        content_hash=identity,
        model=_FrozenReadRecord,
        identity_field=identity_field,
    )


@pytest.mark.proves("workspace_runtime.sealed_value_built_once_per_process")
def test_a_verified_model_is_shared_inside_a_request_and_kept_in_the_process(
    tmp_path: Path, kept_here: bool
):
    """requirement: a validated sealed model is shared inside a request and kept in the process;
    only full proof validates it again."""
    _FrozenReadRecord.validation_calls = 0
    store, identity, _path = _published_read_record(tmp_path)
    second_owner_store = ContentAddressedStore(store.root, uri_prefix="artifact://other")

    with verified_model_read_scope():
        first = _load_read_record(store, identity)
        with verified_model_read_scope():
            second = _load_read_record(second_owner_store, identity)
        third = _load_read_record(store, identity)
        assert third == second == first
        assert (third is first, second is first) == (kept_here, kept_here)
    # where the file system names no file for its lifetime, nothing is kept: every read validates
    assert _FrozenReadRecord.validation_calls == (1 if kept_here else 3)

    with verified_model_read_scope():
        assert _load_read_record(store, identity) == first
    assert _FrozenReadRecord.validation_calls == (1 if kept_here else 4)

    # A sealed model's bytes never change under its name (EV2): the process keeps it past the
    # request, and only full proof validates it again.
    again = _load_read_record(store, identity)
    assert again == first and (again is first) is kept_here
    assert _FrozenReadRecord.validation_calls == (1 if kept_here else 5)
    with full_content_proof(), verified_model_read_scope():
        assert _load_read_record(store, identity) == first
    assert _FrozenReadRecord.validation_calls == (2 if kept_here else 6)


@pytest.mark.proves("workspace_runtime.sealed_value_built_once_per_process")
def test_concurrent_model_read_scopes_share_one_validation(tmp_path: Path, kept_here: bool):
    """requirement: concurrent requests reading one sealed model share one validation and each
    answers the same value."""

    _FrozenReadRecord.validation_calls = 0
    store, identity, _path = _published_read_record(tmp_path)
    ready = Barrier(2)

    def request_read():
        with verified_model_read_scope():
            ready.wait(timeout=5)
            first = _load_read_record(store, identity)
            second = _load_read_record(store, identity)
            return first is second

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(lambda _index: request_read(), range(2)))

    assert outcomes == (kept_here, kept_here)
    assert _FrozenReadRecord.validation_calls == (1 if kept_here else 4)


def test_verified_model_cache_keys_include_store_namespace_model_and_identity_field(
    tmp_path: Path,
):
    _FrozenReadRecord.validation_calls = 0
    _OtherFrozenReadRecord.validation_calls = 0
    store, identity, _path = _published_read_record(tmp_path)
    other_store, _other_identity, _other_path = _published_read_record(tmp_path, root="other")

    with verified_model_read_scope():
        _load_read_record(store, identity)
        # Same content identity in a different resolved store namespace must be verified.
        _load_read_record(other_store, identity)
        other_model_value = store.load_model(
            category="records",
            content_hash=identity,
            model=_OtherFrozenReadRecord,
            identity_field="content_hash",
        )
        assert isinstance(other_model_value, _OtherFrozenReadRecord)
        with pytest.raises(ContentAddressedStoreError, match="artifact_tampered"):
            _load_read_record(store, identity, identity_field="alternate_hash")
    assert _FrozenReadRecord.validation_calls == 3
    assert _OtherFrozenReadRecord.validation_calls == 1


@pytest.mark.proves("workspace_runtime.full_proof_refuses_a_changed_sealed_file")
def test_full_proof_refuses_changed_model_bytes_with_restored_size_and_mtime(
    tmp_path: Path,
):
    """tamper: a sealed model rewritten after a read, size and time restored, is refused by name
    under full proof (EV2)."""
    _FrozenReadRecord.validation_calls = 0
    store, identity, path = _published_read_record(tmp_path)
    original = path.read_bytes()
    stat = path.stat()
    changed = original.replace(b'"sealed"', b'"broken"')
    assert len(changed) == len(original)

    with verified_model_read_scope():
        _load_read_record(store, identity)
        path.write_bytes(changed)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        after = path.stat()
        assert (after.st_size, after.st_mtime_ns) == (stat.st_size, stat.st_mtime_ns)
        with (
            full_content_proof(),
            pytest.raises(ContentAddressedStoreError, match="artifact_tampered"),
        ):
            _load_read_record(store, identity)
    assert _FrozenReadRecord.validation_calls == 2


@pytest.mark.proves("workspace_runtime.full_proof_refuses_a_changed_sealed_file")
def test_full_proof_refuses_a_missing_model_file_behind_a_kept_value(tmp_path: Path):
    """tamper: a sealed model file gone behind a kept value is refused as missing by name under full
    proof."""
    _FrozenReadRecord.validation_calls = 0
    store, identity, path = _published_read_record(tmp_path)

    with verified_model_read_scope():
        _load_read_record(store, identity)
    assert _FrozenReadRecord.validation_calls == 1

    path.unlink()
    with (
        full_content_proof(),
        verified_model_read_scope(),
        pytest.raises(ContentAddressedStoreError) as missing,
    ):
        _load_read_record(store, identity)
    assert str(missing.value) == f"content_store.artifact_missing:{identity}"
    assert _FrozenReadRecord.validation_calls == 1


@pytest.mark.proves("workspace_runtime.a_kept_value_follows_its_file")
def test_a_kept_model_is_never_answered_for_a_file_gone_or_replaced_at_its_path(
    tmp_path, kept_here
):
    """A model kept for a file is read again once that file is gone or a new file stands at its
    path, as a reused test folder or a restored workspace leaves it; outside full proof too."""
    _FrozenReadRecord.validation_calls = 0
    store, identity, path = _published_read_record(tmp_path)
    other = path.with_name("other.json")
    other.write_bytes(path.read_bytes())  # made while the first file stands: never its file id
    with verified_model_read_scope():
        _load_read_record(store, identity)
    path.unlink()
    with verified_model_read_scope(), pytest.raises(ContentAddressedStoreError) as missing:
        _load_read_record(store, identity)
    assert str(missing.value) == f"content_store.artifact_missing:{identity}"

    os.replace(other, path)  # the same bytes in another file at the same path
    with verified_model_read_scope():
        _load_read_record(store, identity)
        _load_read_record(store, identity)
    assert _FrozenReadRecord.validation_calls == (2 if kept_here else 3)  # once per file, if kept
    kept = path.read_bytes()
    path.unlink()
    path.write_bytes(kept)  # made again at once: Linux gives it the freed inode number
    with verified_model_read_scope():
        _load_read_record(store, identity)
    assert _FrozenReadRecord.validation_calls == (3 if kept_here else 4)  # made again: read


def test_failed_model_verification_is_never_cached(tmp_path: Path):
    _FrozenReadRecord.validation_calls = 0
    store, identity, path = _published_read_record(tmp_path)
    path.write_bytes(path.read_bytes().replace(b'"sealed"', b'"broken"'))

    for _ in range(2):
        with (
            verified_model_read_scope(),
            pytest.raises(ContentAddressedStoreError, match="artifact_tampered"),
        ):
            _load_read_record(store, identity)
    assert _FrozenReadRecord.validation_calls == 2


def test_a_lane_is_parquet_named_by_its_packed_bytes_and_reads_back_bit_for_bit(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    signalling_nan = np.frombuffer(np.uint64(0x7FF8DEAD00000001).tobytes(), dtype="<f8")
    values = np.concatenate([[1.5, -0.0, np.nan, -np.nan, np.inf], signalling_nan])
    table = {"scores": np.arange(6.0).reshape(2, 3), "eligible": np.eye(2, 3, dtype=bool)}

    lane = store.publish_columns(category="lanes/x", columns={"value": values})
    joint = store.publish_columns(category="tables", columns=table)

    assert lane == hashlib.sha256(values.tobytes()).hexdigest()
    assert (tmp_path / "store/lanes/x" / f"{lane}.parquet").is_file()
    assert store.load_packed_bytes(category="lanes/x", content_hash=lane) == values.tobytes()
    assert joint == columns_digest(table)
    read = store.load_columns(category="tables", content_hash=joint)
    assert read["eligible"].dtype == np.bool_
    assert read["scores"].reshape(2, 3).tobytes() == table["scores"].tobytes()
    assert store.publish_columns(category="lanes/x", columns={"value": values}) == lane


def test_a_store_written_before_v210_is_read_from_its_bin_files(tmp_path: Path) -> None:
    store = _store(tmp_path)
    packed = np.arange(4.0).tobytes()
    page = b"<p>sealed legacy content</p>"
    for category, payload in (("lanes/old", packed), ("html", page)):
        folder = tmp_path / "store" / category
        folder.mkdir(parents=True)
        (folder / f"{hashlib.sha256(payload).hexdigest()}.bin").write_bytes(payload)

    lane = store.load_packed_bytes(
        category="lanes/old", content_hash=hashlib.sha256(packed).hexdigest()
    )
    read_page = store.load_document(
        category="html", content_hash=hashlib.sha256(page).hexdigest(), extension="html"
    )

    assert lane == packed
    assert read_page == page


def test_a_changed_or_missing_lane_is_refused_by_name(tmp_path: Path, monkeypatch) -> None:
    from alphalattice.control.workspace_runtime import content_store as owner

    store = _store(tmp_path)
    lane = store.publish_columns(category="lanes/x", columns={"value": np.array([1.0, 2.0, 3.0])})
    path = tmp_path / "store/lanes/x" / f"{lane}.parquet"
    written = path.read_bytes()
    stat = path.stat()

    table = pq.read_table(path).set_column(0, "value", pa.array([7.0, 8.0, 9.0]))
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, use_dictionary=False)
    replacement = sink.getvalue().to_pybytes()
    assert len(replacement) == len(written)
    original_read = pq.read_table

    calls = []
    original_digest = owner.columns_digest

    def new_producer(columns):
        calls.append(True)
        return original_digest(columns)

    # A lane's bytes are proved under full proof: an export, a stage's check, the scrub (EV2).
    with full_content_proof():
        with monkeypatch.context() as changing:
            changing.setattr(owner, "columns_digest", new_producer)
            store.load_columns(category="lanes/x", content_hash=lane)
            store.load_columns(category="lanes/x", content_hash=lane)
            assert calls == [True, True]  # an undeclared captured producer never reuses
        first = (value + "A" for value in ()).gi_code
        second = (value + "B" for value in ()).gi_code
        assert owner._column_fact_kind((first,), ()) != owner._column_fact_kind((second,), ())
        doc_changed = new_producer.__code__.replace(
            co_consts=("changed doc", *new_producer.__code__.co_consts[1:])
        )
        assert owner._column_fact_kind((new_producer.__code__,), ()) == owner._column_fact_kind(
            (doc_changed,), ()
        )
        original_kind = owner._column_kind(store)

        original_environment = owner.recorded_environment

        def changed_environment(packages):
            result = original_environment(packages)
            return {**result, "packages": {**result["packages"], "numpy": "changed"}}

        with monkeypatch.context() as changing:
            changing.setattr(owner, "recorded_environment", changed_environment)
            assert owner._column_kind(store) != original_kind

        def move_during_read(target):
            result = original_read(target)
            path.write_bytes(replacement)
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            return result

        with monkeypatch.context() as changing:
            changing.setattr(pq, "read_table", move_during_read)
            with pytest.raises(
                ContentAddressedStoreError, match=r"content_store\.artifact_tampered"
            ):
                store.load_columns(category="lanes/x", content_hash=lane)
        path.write_bytes(replacement)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with pytest.raises(ContentAddressedStoreError, match=r"content_store\.artifact_tampered"):
            store.load_columns(category="lanes/x", content_hash=lane)

        for changed in (written[:-1], bytes([written[0] ^ 1]) + written[1:]):
            path.write_bytes(changed)
            with pytest.raises(
                ContentAddressedStoreError, match=r"content_store\.artifact_tampered"
            ):
                store.load_packed_bytes(category="lanes/x", content_hash=lane)
    with pytest.raises(ContentAddressedStoreError, match=r"content_store\.artifact_missing"):
        store.load_packed_bytes(category="lanes/x", content_hash="0" * 64)
    with pytest.raises(ContentAddressedStoreError, match=r"content_store\.columns_invalid"):
        store.publish_columns(category="lanes/x", columns={"a": np.zeros(2), "b": np.zeros(3)})


def test_a_shared_file_is_replaced_through_a_reader_holding_it_open(tmp_path: Path) -> None:
    """regression (overlapping trial reads): on Windows a replace fails while
    another handle reads the file, and a Host's concurrent requests hold one as they read a
    trial record or the verification ledger; the replace waits out the read and lands."""

    from alphalattice.control.workspace_runtime.content_store import replace_shared_file

    destination = tmp_path / "record.json"
    destination.write_text("old", encoding="utf-8")
    staged = tmp_path / "record.json.partial"
    staged.write_text("new", encoding="utf-8")
    reader = destination.open("r", encoding="utf-8")
    closing = threading.Timer(0.05, reader.close)
    closing.start()
    try:
        replace_shared_file(staged, destination)
    finally:
        closing.join()
        reader.close()
    assert destination.read_text(encoding="utf-8") == "new" and not staged.exists()


@pytest.mark.parametrize(
    "reader",
    ["model", "columns", "packed", "document"],
    ids=["model", "columns", "packed", "document"],
)
def test_every_content_reader_distinguishes_missing_corrupt_and_malformed(
    tmp_path: Path, reader: str
) -> None:
    """BEHAVIOUR: absence names the request; corruption and invalid hashes stay distinct."""
    from pydantic import BaseModel

    from alphalattice.interface.local_application.cli_contract import refusal_words

    class Record(BaseModel):
        content_hash: str
        text: str

    store = _store(tmp_path)
    identity = "a" * 64
    category = "records"
    if reader == "model":

        def load(value: str) -> object:
            return store.load_model(
                category=category, content_hash=value, model=Record, identity_field="content_hash"
            )

        store.publish_model(
            category=category,
            value=Record(content_hash=identity, text="kept"),
            identity_field="content_hash",
        )
        target = store.root / category / f"{identity}.json"
    elif reader == "document":

        def load(value: str) -> object:
            return store.load_document(category=category, content_hash=value, extension="json")

        identity = store.publish_document(
            category=category, payload=b'{"kept":true}', extension="json"
        )
        target = store.root / category / f"{identity}.json"
    else:

        def load(value: str) -> object:
            return (store.load_columns if reader == "columns" else store.load_packed_bytes)(
                category=category, content_hash=value
            )

        identity = store.publish_columns(category=category, columns={"value": np.arange(3.0)})
        target = store.root / category / f"{identity}.parquet"
    assert load(identity) is not None
    with pytest.raises(ContentAddressedStoreError, match=r"^content_store.identity_invalid$"):
        load("not-a-hash")
    target.write_bytes(b"corrupt")
    # Corrupt bytes are refused by name where they are proved: under full proof (EV2).
    with (
        full_content_proof(),
        pytest.raises(ContentAddressedStoreError, match=r"^content_store.artifact_tampered$"),
    ):
        load(identity)
    if reader == "model":
        target.write_text(Record(content_hash="b" * 64, text="wrong identity").model_dump_json())
        with (
            full_content_proof(),
            pytest.raises(ContentAddressedStoreError, match=r"^content_store.artifact_tampered$"),
        ):
            load(identity)
    target.unlink()
    with pytest.raises(ContentAddressedStoreError) as absent:
        load(identity)
    assert str(absent.value) == f"content_store.artifact_missing:{identity}"
    words = refusal_words(str(absent.value))
    assert identity in words["detail"] and "workspace show" in words["detail"]
    assert "restore a backup" in words["detail"] and words["next_action"]


_RUN: list[str] = [""]


@pytest.fixture(autouse=True)
def _own_values() -> None:
    """Each test names its own kept values: a passed test's temporary folder is reused, and a kept
    value answers by name and content (LAWS.md EV2)."""
    _RUN[0] = uuid4().hex


def _reader(path: Path, payload: bytes = b"sealed"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    calls = []

    def verify():
        calls.append(path)
        current = path.read_bytes()
        if hashlib.sha256(current).hexdigest() != expected:
            raise ValueError("sealed bytes changed")
        return current

    def read():
        with verified_model_read_scope():
            return verified_source_value(
                (_RUN[0], "test-sealed-read", expected),
                (path,),
                verify,
                nbytes=lambda value: len(value),
            )

    return read, calls


def _proved(read):
    with full_content_proof():
        return read()


@pytest.mark.proves("workspace_runtime.sealed_value_built_once_per_process")
@pytest.mark.parametrize(
    "case",
    [
        "full-proof",
        "same-task",
        "replaced-file",
        "replaced-file-request-proof",
        "replaced-file-request-value",
        "open-writer-transaction",
    ],
)
def test_a_sealed_source_builds_once_per_process_and_full_proof_builds_again(
    tmp_path, kept_here, case
):
    """Verify source versions and committed schemas, including full proof and private writes."""
    if case == "full-proof":
        read, calls = _reader(tmp_path / "sealed.bin")
        first = read()  # Each read opens and closes its own Task's read scope.
        assert (first is read()) is kept_here  # where nothing is kept, every read builds
        assert first == b"sealed"
        assert len(calls) == (1 if kept_here else 2)
        assert _proved(read) == b"sealed"
        assert len(calls) == (2 if kept_here else 3)
        read()
        assert len(calls) == (2 if kept_here else 4)
        return

    if case in {"same-task", "replaced-file"}:
        path = tmp_path / "sealed.bin"
        read, calls = _reader(path)
        if case == "same-task":
            with verified_model_read_scope():
                assert read() == read() == b"sealed"
        else:
            assert read() == b"sealed"
            replacement = path.with_suffix(".new")
            replacement.write_bytes(b"changed")
            replacement.replace(path)
            with pytest.raises(ValueError, match="sealed bytes changed"):
                read()
        assert len(calls) == (2 if case == "replaced-file" or not kept_here else 1)
        return

    if case in {"replaced-file-request-proof", "replaced-file-request-value"}:
        path = tmp_path / "nested.bin"
        path.write_bytes(b"sealed")
        identity = (_RUN[0], case, path)

        def verify():
            value = path.read_bytes()
            if value != b"sealed":
                raise ValueError("sealed bytes changed")
            return value

        def build():
            if case == "replaced-file-request-proof":
                verified_request_proof(identity, verify)
                return b"sealed"
            return verified_request_value(identity, verify, nbytes=6)

        with verified_model_read_scope():
            assert verified_source_value(identity, (path,), build, nbytes=6) == b"sealed"
            replacement = path.with_suffix(".new")
            replacement.write_bytes(b"broken")
            replacement.replace(path)
            with pytest.raises(ValueError, match="sealed bytes changed"):
                verified_source_value(identity, (path,), build, nbytes=6)
        return

    database = WorkspaceDatabase(tmp_path / "catalog")
    with database.hold():
        with database.transaction() as writer:
            writer.execute("CREATE TABLE content_check (prior INTEGER)")
        with database.connect(read_only=True) as reader:
            before = (("prior", "INTEGER"),)
            after = (*before, ("next_value", "VARCHAR"))
            assert table_columns(reader, "content_check") == before
            with database.transaction() as writer:
                writer.execute("ALTER TABLE content_check ADD COLUMN next_value VARCHAR")
                assert table_columns(writer, "content_check") == after
                assert table_columns(reader, "content_check") == before
            assert table_columns(reader, "content_check") == after


@pytest.mark.proves("workspace_runtime.full_proof_refuses_a_changed_sealed_file")
@pytest.mark.parametrize(
    "change",
    ("same-size-restored-time", "atomic-replacement", "recreate"),
    ids=["same-size-restored-time", "atomic-replacement", "recreate"],
)
def test_a_changed_sealed_file_reads_at_its_name_and_full_proof_refuses_it(
    tmp_path, change, kept_here
):
    """tamper: a sealed file changed under its name is refused by name by full proof; an ordinary
    read keeps its proved bytes only while the same file stands at the name."""
    path = tmp_path / "source" / "sealed.bin"
    read, calls = _reader(path)
    read()
    before = path.stat()
    if change == "atomic-replacement":
        alternate = tmp_path / "replacement.bin"
        alternate.write_bytes(b"broken")
        os.replace(alternate, path)
    elif change == "recreate":
        path.unlink()
        path.write_bytes(b"broken")
    else:
        path.write_bytes(b"broken")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    if change == "same-size-restored-time" and kept_here:  # the same file, kept
        assert read() == b"sealed"
        assert len(calls) == 1
    else:
        with pytest.raises(ValueError, match="sealed bytes changed"):
            read()
    with pytest.raises(ValueError, match="sealed bytes changed"):
        _proved(read)


@pytest.mark.proves("workspace_runtime.full_proof_refuses_a_changed_sealed_file")
def test_a_store_read_takes_sealed_bytes_at_their_name_and_full_proof_refuses_a_change(tmp_path):
    """tamper: a sealed document rewritten in place reads at its name and is refused by name
    under full proof."""
    store = ContentAddressedStore(tmp_path, uri_prefix="artifact://qa")
    digest = store.publish_document(category="pages", payload=b"<p>one</p>", extension="html")
    (tmp_path / "pages" / f"{digest}.html").write_bytes(b"<p>two</p>")
    read = {"category": "pages", "content_hash": digest, "extension": "html"}
    assert store.load_document(**read) == b"<p>two</p>"
    with full_content_proof(), pytest.raises(ContentAddressedStoreError, match="tampered"):
        store.load_document(**read)


@pytest.mark.proves("workspace_runtime.sealed_read_leaves_nothing_behind")
def test_a_sealed_read_holds_no_file_open_and_starts_no_thread(tmp_path):
    """requirement: reading sealed bytes leaves nothing behind: no watching thread, and the file
    renames and goes at once."""
    path = tmp_path / "sealed.bin"
    read, _calls = _reader(path)
    before = {thread.ident for thread in threading.enumerate()}
    read()
    _proved(read)
    assert {thread.ident for thread in threading.enumerate()} <= before
    moved = path.with_name("moved.bin")
    os.replace(path, moved)
    moved.unlink()


@contextmanager
def _concurrent_source_reads(read, entered, release, monkeypatch, kept):
    """Four readers of one source at once. Where the store keeps values, three wait on the first
    one's producer; where it keeps nothing, each enters its own builder."""
    waiting = Queue()
    result = Future.result

    def wait(future, *args, **kwargs):
        if current_thread().name.startswith("source-waiter-"):
            waiting.put(None)
        return result(future, *args, **kwargs)

    monkeypatch.setattr(Future, "result", wait)
    outcomes = []

    def request():
        try:
            outcomes.append((read(), None))
        except BaseException as error:
            outcomes.append((None, error))

    workers = [Thread(target=request, name="source-producer")]
    try:
        workers[0].start()
        assert entered.acquire(timeout=5)
        for index in range(3):
            worker = Thread(target=request, name=f"source-waiter-{index}")
            workers.append(worker)
            worker.start()
        for _ in range(3):
            if kept:
                waiting.get(timeout=5)
            else:
                assert entered.acquire(timeout=5)
        yield outcomes
    finally:
        release.set()
        for worker in workers:
            worker.join(timeout=5)
            assert not worker.is_alive()


@pytest.mark.proves("workspace_runtime.sealed_reads_share_one_producer")
def test_concurrent_reads_of_one_source_share_one_producer_without_blocking_other_keys(
    tmp_path, monkeypatch, kept_here
):
    """requirement: concurrent readers of one sealed source share one producer and never block
    another source."""
    path = tmp_path / "sealed.bin"
    path.write_bytes(b"sealed")
    entered, release = Semaphore(0), Event()
    calls = []
    other, other_calls = _reader(tmp_path / "independent.bin")

    def verify():
        calls.append(True)
        payload = path.read_bytes()
        assert payload == b"sealed"
        entered.release()
        assert release.wait(timeout=5)
        return payload

    def read():
        with verified_model_read_scope():
            return verified_source_value(
                (
                    _RUN[0],
                    "single-flight",
                ),
                (path,),
                verify,
                nbytes=6,
            )

    builders = 1 if kept_here else 4  # where nothing is kept, each reader builds its own
    with _concurrent_source_reads(read, entered, release, monkeypatch, kept_here) as outcomes:
        independent = Thread(target=other, name="independent-source")
        independent.start()
        independent.join(timeout=5)
        assert not independent.is_alive()
        assert len(other_calls) == 1
        assert len(calls) == builders
    assert len(outcomes) == 4
    assert all(value == b"sealed" and error is None for value, error in outcomes)
    assert len(calls) == builders
    assert read() == b"sealed"
    assert len(calls) == (1 if kept_here else 5)


@pytest.mark.proves("workspace_runtime.sealed_reads_share_one_producer")
@pytest.mark.parametrize("fault", ("builder", "invalid-size"), ids=["builder", "invalid-size"])
def test_concurrent_producer_failure_releases_waiters_and_later_reads_verify_again(
    tmp_path, monkeypatch, fault, kept_here
):
    """requirement: a failed producer releases its waiters with its refusal, and a later read builds
    again."""
    path = tmp_path / "sealed.bin"
    path.write_bytes(b"sealed")
    entered, release = Semaphore(0), Event()
    calls = []
    failing = True

    def verify():
        calls.append(True)
        entered.release()
        assert release.wait(timeout=5)
        if failing and fault == "builder":
            raise ValueError("sealed verification failed")
        return path.read_bytes()

    def read():
        with verified_model_read_scope():
            return verified_source_value(
                (
                    _RUN[0],
                    "single-flight-failure",
                ),
                (path,),
                verify,
                nbytes=lambda value: -1 if failing and fault == "invalid-size" else len(value),
            )

    builders = 1 if kept_here else 4  # where nothing is kept, each reader builds its own
    with _concurrent_source_reads(read, entered, release, monkeypatch, kept_here) as outcomes:
        assert len(calls) == builders
    assert len(calls) == builders
    assert len(outcomes) == 4
    if kept_here or fault == "builder":
        assert all(value is None and isinstance(error, ValueError) for value, error in outcomes)
    else:  # a size is measured only to keep a value; where nothing is kept, each reader has it
        assert all(value == b"sealed" and error is None for value, error in outcomes)
    failing = False
    assert read() == b"sealed"
    assert len(calls) == builders + 1


@pytest.mark.proves("workspace_runtime.sealed_reads_share_one_producer")
def test_a_nested_read_of_the_same_source_key_does_not_wait_for_itself(tmp_path):
    """requirement: a builder that reads its own source again builds it in full instead of waiting
    on itself."""
    path = tmp_path / "sealed.bin"
    path.write_bytes(b"sealed")
    calls = []

    def verify():
        calls.append(True)
        if len(calls) == 1:
            return verified_source_value(
                (
                    _RUN[0],
                    "nested-source",
                ),
                (path,),
                verify,
                nbytes=6,
            )
        return path.read_bytes()

    with verified_model_read_scope():
        assert (
            verified_source_value(
                (
                    _RUN[0],
                    "nested-source",
                ),
                (path,),
                verify,
                nbytes=6,
            )
            == b"sealed"
        )
    assert len(calls) == 2


@pytest.mark.proves("workspace_runtime.parallel_source_checks_are_bounded_and_isolated")
def test_parallel_checks_have_fresh_contexts_and_inherit_full_proof(tmp_path):
    """requirement: parallel checks start in fresh contexts, answer from the process memory and
    inherit their caller's full proof."""
    marker = ContextVar("outside-check-context", default="unset")
    marker.set("caller")
    values, proofs = [], []
    identity = ("isolated-check-scopes", _RUN[0])

    def value():
        values.append(True)
        return b"sealed"

    def check():
        assert marker.get() == "unset"
        assert verified_request_value(identity, value, nbytes=6) == b"sealed"
        verified_request_proof(identity, lambda: proofs.append(True))

    with verified_model_read_scope():
        verified_request_value(identity, value, nbytes=6)
        verified_request_proof(identity, lambda: proofs.append(True))
        verify_source_checks((check, check))
        assert len(values) == len(proofs) == 1
        with full_content_proof():
            verify_source_checks((check, check))
    # Full proof builds again from the bytes, once for the whole proof its checks share.
    assert len(values) == len(proofs) == 2
    assert marker.get() == "caller"


@pytest.mark.proves("workspace_runtime.parallel_source_checks_are_bounded_and_isolated")
def test_parallel_checks_bound_running_and_submitted_work_across_requests(monkeypatch):
    """requirement: at most two source checks run or wait submitted across every request at once."""
    release, entered = Event(), Queue()
    attempts = Queue()
    lock = Lock()
    acquire, submit = BoundedSemaphore.acquire, ThreadPoolExecutor.submit
    outstanding = peak = 0
    errors = []

    def acquire_slot(semaphore, *args, **kwargs):
        if current_thread().name.startswith("source-check-caller-"):
            attempts.put(None)
        return acquire(semaphore, *args, **kwargs)

    def submitted(pool, *args, **kwargs):
        nonlocal outstanding, peak
        with lock:
            outstanding += 1
            peak = max(peak, outstanding)
        future = submit(pool, *args, **kwargs)

        def completed(_future):
            nonlocal outstanding
            with lock:
                outstanding -= 1

        future.add_done_callback(completed)
        return future

    monkeypatch.setattr(BoundedSemaphore, "acquire", acquire_slot)
    monkeypatch.setattr(ThreadPoolExecutor, "submit", submitted)

    def check():
        entered.put(None)
        assert release.wait(timeout=5)

    def request():
        try:
            verify_source_checks((check, check))
        except BaseException as error:
            errors.append(error)

    callers = [Thread(target=request, name=f"source-check-caller-{i}") for i in range(2)]
    try:
        for caller in callers:
            caller.start()
        for _ in range(2):
            entered.get(timeout=5)
        for _ in range(3):
            attempts.get(timeout=5)
        with lock:
            assert outstanding == peak == 2
    finally:
        release.set()
        for caller in callers:
            caller.join(timeout=5)
            assert not caller.is_alive()
    assert not errors
    assert outstanding == 0
    assert peak == 2


@pytest.mark.proves("workspace_runtime.parallel_source_checks_are_bounded_and_isolated")
def test_parallel_checks_join_failures_in_input_order_and_drain_started_checks():
    """requirement: parallel checks report the first failure in input order after every started
    check ends."""
    first_failed, second_started, release = Event(), Event(), Event()
    errors = []

    def first():
        assert second_started.wait(timeout=5)
        first_failed.set()
        raise ValueError("first input")

    def second():
        second_started.set()
        assert release.wait(timeout=5)
        raise ValueError("second input")

    def request():
        try:
            verify_source_checks((first, second))
        except BaseException as error:
            errors.append(error)

    caller = Thread(target=request)
    try:
        caller.start()
        assert first_failed.wait(timeout=5)
        assert caller.is_alive()
        assert not errors
    finally:
        release.set()
        caller.join(timeout=5)
    assert not caller.is_alive()
    assert len(errors) == 1 and str(errors[0]) == "first input"
    # Both submission permits were released despite both callbacks raising.
    verify_source_checks((lambda: None, lambda: None))


@pytest.mark.proves("workspace_runtime.parallel_source_checks_are_bounded_and_isolated")
def test_nested_parallel_checks_finish_without_waiting_for_their_own_pool():
    """requirement: checks nested in a running check run serially instead of waiting on their own
    pool."""
    started, release = Queue(), Event()
    finished = []

    def outer():
        started.put(None)
        assert release.wait(timeout=5)
        verify_source_checks((lambda: finished.append(True), lambda: finished.append(True)))

    caller = Thread(target=lambda: verify_source_checks((outer, outer)))
    try:
        caller.start()
        for _ in range(2):
            started.get(timeout=5)
    finally:
        release.set()
        caller.join(timeout=5)
    assert not caller.is_alive()
    assert len(finished) == 4


@pytest.mark.proves("workspace_runtime.parallel_source_checks_are_bounded_and_isolated")
def test_parallel_checks_inherit_declared_source_ownership_without_nested_retention(
    tmp_path, kept_here
):
    """requirement: a check inside an owner's build inherits its ownership and keeps nothing of its
    own."""
    path = tmp_path / "sealed.bin"
    read, calls = _reader(path)

    def check():
        assert read() == read() == b"sealed"

    def proof():
        verify_source_checks((check,))

    for _ in range(2):
        with verified_model_read_scope():
            verified_source_value(
                (
                    _RUN[0],
                    "parallel-complete-proof",
                ),
                (path,),
                proof,
                nbytes=0,
            )
    owners = 1 if kept_here else 2  # where nothing is kept, the owner builds at every read
    assert len(calls) == 2 * owners
    assert read() == b"sealed"
    assert len(calls) == 2 * owners + 1


@pytest.mark.proves("workspace_runtime.source_owner_keeps_one_copy")
def test_complete_source_owner_does_not_retain_nested_packed_bytes(tmp_path, kept_here):
    """requirement: a complete owner keeps its value and none of the packed bytes its nested reader
    built."""
    path = tmp_path / "sealed.bin"
    read, calls = _reader(path)
    expected = hashlib.sha256(b"sealed").hexdigest()

    def proof():
        assert read() == b"sealed"

    for _ in range(2):
        with verified_model_read_scope():
            verified_source_value((_RUN[0], "complete-proof", expected), (path,), proof, nbytes=0)
    owners = 1 if kept_here else 2  # where nothing is kept, the owner builds at every read
    assert len(calls) == owners
    # The nested reader built in full but kept no duplicate byte cache.
    read()
    assert len(calls) == owners + 1


@pytest.mark.proves("workspace_runtime.source_owner_keeps_one_copy")
def test_complete_source_owner_retains_decoded_values_without_a_second_process_copy(
    tmp_path, monkeypatch, kept_here
):
    """requirement: a complete owner keeps its decoded arrays read-only and once, without a second
    process copy."""
    stream = BytesIO()
    np.savez(stream, values=np.arange(8, dtype=np.float64))
    payload = stream.getvalue()
    path = tmp_path / "arrays.bin"
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    load = np.load
    decodes = 0

    def counted_load(*args, **kwargs):
        nonlocal decodes
        decodes += 1
        return load(*args, **kwargs)

    monkeypatch.setattr(np, "load", counted_load)

    def build():
        current = path.read_bytes()
        assert hashlib.sha256(current).hexdigest() == expected
        arrays = verified_npz_arrays(current, identity=(str(path), expected))
        return tuple(arrays.items())

    def read(key):
        with verified_model_read_scope():
            return verified_source_value(
                (_RUN[0], "complete-decoded-result", key, expected),
                (path,),
                build,
                nbytes=lambda values: sum(array.nbytes for _, array in values),
            )

    first = read("one")
    second = read("two")
    assert decodes == 2  # intermediate NPZ cache was not independently retained
    again = (read("one"), read("two"))
    assert (again[0] is first, again[1] is second) == (kept_here, kept_here)
    assert decodes == (2 if kept_here else 4)  # where nothing is kept, every read decodes
    assert np.array_equal(first[0][1], np.arange(8))
    with pytest.raises(ValueError, match="WRITEABLE"):
        first[0][1].setflags(write=True)


@pytest.mark.proves("workspace_runtime.source_owner_keeps_one_copy")
def test_every_source_in_a_proof_is_proved_and_mutable_results_are_not_retained(
    tmp_path, kept_here
):
    """requirement: full proof checks every source an owner declares, and a mutable result is never
    kept."""
    paths = tuple(tmp_path / f"source-{index}.bin" for index in range(2))
    for path in paths:
        path.write_bytes(b"sealed")
    calls = []

    def verify():
        calls.append(True)
        if any(path.read_bytes() != b"sealed" for path in paths):
            raise ValueError("sealed bytes changed")
        return b"sealed"

    for _ in range(2):
        with verified_model_read_scope():
            verified_source_value(
                (
                    _RUN[0],
                    "two-source-proof",
                ),
                paths,
                verify,
                nbytes=6,
            )
    assert len(calls) == (1 if kept_here else 2)  # where nothing is kept, every read builds
    paths[1].write_bytes(b"broken")
    with full_content_proof(), pytest.raises(ValueError, match="sealed bytes changed"):
        verified_source_value(
            (
                _RUN[0],
                "two-source-proof",
            ),
            paths,
            verify,
            nbytes=6,
        )
    mutable_calls = []
    for _ in range(2):
        with verified_model_read_scope():
            result = verified_source_value(
                (
                    _RUN[0],
                    "mutable-source",
                ),
                (paths[0],),
                lambda: mutable_calls.append(True) or {},
                nbytes=1,
            )
            result["caller"] = "changed"
    assert len(mutable_calls) == 2


@contextmanager
def _kept_facts(tmp_path: Path, name: str = "workspace"):
    """A workspace whose file digests and facts this process keeps, and one file in it."""
    from alphalattice.control.workspace_runtime import verified_facts as facts

    workspace = tmp_path / name
    target = workspace / "store" / "part.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes(b"one")
    facts.keep_file_facts(workspace)
    try:
        yield facts, workspace, target
    finally:
        facts.release_file_facts(workspace)


def _aged(target: Path) -> None:
    os.utime(target, (target.stat().st_atime - 10, target.stat().st_mtime - 10))


class _Clock:
    """The trust record's clock, moved past a file's racy tick without waiting."""

    def __init__(self, facts, monkeypatch) -> None:
        self.now = time_ns()
        monkeypatch.setattr(facts, "wall_time", lambda: self.now / 1e9)
        monkeypatch.setattr(facts, "wall_time_ns", lambda: self.now)

    def settle(self, target: Path) -> None:
        self.now = max(self.now, max(file_stamp(target)[-2:]) + 10**10)


def _counted_hashes(monkeypatch) -> list[bool]:
    calls: list[bool] = []
    digest = digests.integrity_sha256_stream

    def counted(*args, **kwargs):
        calls.append(True)
        return digest(*args, **kwargs)

    monkeypatch.setattr(digests, "integrity_sha256_stream", counted)
    return calls


_ONE = hashlib.sha256(b"one").hexdigest()
_TWO = hashlib.sha256(b"two").hexdigest()


def _write_then_read(facts, target, clock, tmp_path, monkeypatch) -> str:
    facts.file_sha256(target, written=True)
    clock.settle(target)
    return facts.file_sha256(target)


def _read_twice(facts, target, clock, tmp_path, monkeypatch) -> str:
    clock.settle(target)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(facts.file_sha256, [target] * 4)) == [_ONE] * 4
    return facts.file_sha256(target)


def _racy_rewrite(facts, target, clock, tmp_path, monkeypatch) -> str:
    # A same-size rewrite inside one clock tick leaves the whole stamp unchanged.
    stamp = file_stamp(target)
    facts.record_sealed_file(target, _ONE)
    monkeypatch.setattr(digests, "handle_stamp", lambda _handle: stamp)
    target.write_bytes(b"two")
    return facts.file_sha256(target)


def _file_scrub(facts, workspace):
    from datetime import UTC, datetime, timedelta

    return facts.FileTrustScrub(
        workspace, clock=lambda: datetime(2026, 11, 22, tzinfo=UTC), period=timedelta(days=7)
    )


def _scrubbed_rewrite(facts, target, clock, tmp_path, monkeypatch) -> str:
    workspace = target.parent.parent
    facts.record_sealed_file(target, _ONE)
    target.write_bytes(b"two")
    clock.settle(target)
    assert _file_scrub(facts, workspace).run(lambda: True)
    receipt = json.loads((workspace / facts.FILE_TRUST_SCRUB).read_bytes())
    assert (receipt["checked"], receipt["mismatches"]) == (1, 1)
    assert not facts.sealed_file_holds(target, _ONE)
    return facts.file_sha256(target)


def _scrub_interrupted_append(facts, target, clock, tmp_path, monkeypatch) -> str:
    workspace = target.parent.parent
    facts.record_sealed_file(target, _ONE)
    clock.settle(target)
    captured = Event()
    release = Event()
    other = target.with_name("other.parquet")
    other.write_bytes(b"one")
    clock.settle(other)
    stamp = file_stamp(target)
    real_stamp = digests.handle_stamp
    monkeypatch.setattr(
        digests,
        "handle_stamp",
        lambda handle: stamp if Path(handle.name) == target else real_stamp(handle),
    )
    target.write_bytes(b"two")
    assert facts.file_sha256(target) == _ONE
    open_path = Path.open
    journal = workspace / facts.FILE_TRUST
    appends = 0

    def interrupt(path, mode="r", **kwargs):
        nonlocal appends
        if path == journal and mode == "a":
            appends += 1
            if appends == 2:  # the mismatch invalidation after the scan's opening record
                captured.set()
                assert release.wait(5)
                raise OSError("trust journal unavailable")
        return open_path(path, mode, **kwargs)

    monkeypatch.setattr(Path, "open", interrupt)
    scrub = _file_scrub(facts, workspace)
    with ThreadPoolExecutor(max_workers=2) as pool:
        interrupted = pool.submit(scrub.run, lambda: True)
        try:
            assert captured.wait(5)
            assert pool.submit(facts.file_sha256, other).result(timeout=2) == _ONE
        finally:
            release.set()
        with pytest.raises(OSError, match="trust journal unavailable"):
            interrupted.result(timeout=5)
    assert not facts.sealed_file_holds(target, _ONE)
    assert not (workspace / facts.FILE_TRUST_SCRUB).exists()
    facts.release_file_facts(workspace)
    facts.keep_file_facts(workspace)
    assert not facts.sealed_file_holds(target, _ONE)
    return facts.file_sha256(target)


def _mtime_changed(facts, target, clock, tmp_path, monkeypatch) -> str:
    facts.record_sealed_file(target, _ONE)
    stat = target.stat()
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    clock.settle(target)
    return facts.file_sha256(target)


def _renamed_over(facts, target, clock, tmp_path, monkeypatch) -> str:
    facts.record_sealed_file(target, _ONE)
    stat = target.stat()
    staged = target.with_name("staged.parquet")
    staged.write_bytes(b"two")
    os.utime(staged, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    os.replace(staged, target)
    clock.settle(target)
    return facts.file_sha256(target)


def _switch_off(facts, target, clock, tmp_path, monkeypatch) -> str:
    with override(Change("file-trust-off", {"ALPHALATTICE_FILE_TRUST": "0"})):
        facts.file_sha256(target, written=True)
        clock.settle(target)
        facts.file_sha256(target)
        return facts.file_sha256(target)


def _copied_root(facts, target, clock, tmp_path, monkeypatch) -> str:
    facts.record_sealed_file(target, _ONE)
    workspace = target.parents[1]
    facts.release_file_facts(workspace)
    shutil.copytree(workspace, tmp_path / "copy")  # its trust journal with it
    with _kept_facts(tmp_path, "copy") as (_facts, _copy, copied):
        clock.settle(copied)
        return facts.file_sha256(copied)


def _foreign_write(facts, target, clock, tmp_path, monkeypatch) -> str:
    facts.record_sealed_file(target, _TWO)  # the writer's digest; the path holds other bytes
    clock.settle(target)
    return facts.file_sha256(target)


def _edited_while_hashed(facts, target, clock, tmp_path, monkeypatch) -> str:
    counted = digests.integrity_sha256_stream

    def then_edited(*args, **kwargs):
        held = counted(*args, **kwargs)
        with target.open("r+b") as other:  # the same file, edited in place after its hash
            other.write(b"two")
        stat = target.stat()
        os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        return held

    monkeypatch.setattr(digests, "integrity_sha256_stream", then_edited)
    facts.file_sha256(target, written=True)
    monkeypatch.setattr(digests, "integrity_sha256_stream", counted)
    clock.settle(target)
    return facts.file_sha256(target)


def _backup(facts, target, clock, tmp_path, monkeypatch) -> str:
    facts.record_sealed_file(target, _ONE)
    clock.settle(target)
    stamp = file_stamp(target)
    monkeypatch.setattr(digests, "handle_stamp", lambda _handle: stamp)
    target.write_bytes(b"two")  # a byte change that kept its stamp, as a restored timestamp
    with facts.full_content_proof():
        assert not facts.sealed_file_holds(target, _ONE)
        assert facts.file_sha256(target) == _TWO
    return backup._file_digest(target, "sha256").removeprefix("sha256:")


def _scrub_yields_between_files(facts, target, clock, tmp_path, monkeypatch) -> str:
    from datetime import UTC, datetime, timedelta

    workspace = target.parent.parent
    other = target.with_name("other.parquet")
    other.write_bytes(b"one")
    for path in (target, other):
        facts.record_sealed_file(path, _ONE)
        clock.settle(path)

    today = datetime(2026, 11, 22, tzinfo=UTC)
    period = timedelta(days=7)
    scrub = facts.FileTrustScrub(workspace, clock=lambda: today, period=period)
    hash_bytes = digests.integrity_sha256_stream
    seen = []
    idle = False

    def task_arrives(handle):
        nonlocal idle
        seen.append(handle.name)
        idle = False
        return hash_bytes(handle)

    monkeypatch.setattr(digests, "integrity_sha256_stream", task_arrives)
    assert not scrub.run(lambda: idle)
    with override(Change("file-trust-off", {"ALPHALATTICE_FILE_TRUST": "0"})):
        assert scrub.run(lambda: True)
    assert seen == []
    idle = True
    assert not scrub.run(lambda: idle)
    assert len(seen) == 1
    idle = True
    assert not scrub.run(lambda: idle)  # a request during the last file also defers publication
    assert not (workspace / facts.FILE_TRUST_SCRUB).exists()
    assert not (workspace / facts.FILE_TRUST_SCRUB.with_suffix(".partial")).exists()
    assert set(seen) == {str(target), str(other)}
    idle = True
    assert scrub.run(lambda: idle)
    facts.release_file_facts(workspace)
    facts.keep_file_facts(workspace)
    assert facts.file_sha256(target) == _ONE

    restarted = facts.FileTrustScrub(workspace, clock=lambda: today, period=period)
    assert restarted.run(lambda: True)
    assert len(seen) == 2
    today += period
    idle = True
    assert not restarted.run(lambda: idle)
    assert len(seen) == 3
    restarted = facts.FileTrustScrub(workspace, clock=lambda: today, period=period)
    assert restarted.run(lambda: True)
    assert len(seen) == 5
    assert set(seen[-2:]) == {str(target), str(other)}
    return facts.file_sha256(target)


def _next_scrub(facts, workspace):
    """The workspace's scrub a period after `_file_scrub`'s pass: due again."""
    from datetime import UTC, datetime, timedelta

    return facts.FileTrustScrub(
        workspace, clock=lambda: datetime(2026, 11, 29, tzinfo=UTC), period=timedelta(days=7)
    )


def _removed_file_compacted(facts, target, clock, tmp_path, monkeypatch) -> str:
    workspace = target.parent.parent
    gone = target.with_name("gone.parquet")
    gone.write_bytes(b"one")
    for path in (target, gone):
        facts.record_sealed_file(path, _ONE)
        clock.settle(path)
    gone.unlink()
    facts.release_file_facts(workspace)
    facts.keep_file_facts(workspace)  # the next process: its load leaves the journal as it was
    journal = workspace / facts.FILE_TRUST
    assert "gone.parquet" in journal.read_text(encoding="utf-8")
    published = []
    publish = lambda action: (published.append(action.__name__), action(), True)[2]
    assert facts._TRUST_JOURNAL.compact(workspace, publish)
    assert published == ["replace"]  # read, probed and staged outside admission
    assert "gone.parquet" not in journal.read_text(encoding="utf-8")
    return facts.file_sha256(target)


def _request_during_the_fsync(facts, target, clock, tmp_path, monkeypatch) -> str:
    workspace = target.parent.parent
    facts.record_sealed_file(target, _ONE)
    clock.settle(target)
    journal = workspace / facts.FILE_TRUST
    journal.write_text(journal.read_text(encoding="utf-8") * 1100, encoding="utf-8")
    facts.release_file_facts(workspace)
    facts.keep_file_facts(workspace)
    other = target.with_name("other.parquet")
    other.write_bytes(b"one")
    before = journal.read_bytes()
    facts.file_sha256(other, written=True)  # a foreground fact only appends, however long
    grown = journal.read_bytes()
    assert grown.startswith(before) and len(grown) > len(before)
    requests, sync, replace, replaced = [], os.fsync, os.replace, []
    staged = journal.with_name(f"{journal.name}.{os.getpid()}.partial")

    def request_admitted(fd):  # a read-only request starts during the staged journal's sync
        if staged.exists() and not requests:
            requests.append("running")
        sync(fd)

    def counted(src, dst):
        replaced.append(Path(dst)) if Path(dst) == journal else None
        return replace(src, dst)

    def idle(action) -> bool:  # the Host's boundary: no visible write while a request runs
        if "running" in requests:
            return False
        action()
        return True

    monkeypatch.setattr(facts.os, "fsync", request_admitted)
    monkeypatch.setattr(facts.os, "replace", counted)
    scrub = _file_scrub(facts, workspace)
    assert not scrub.run(lambda: "running" not in requests, publish=idle)
    assert requests == ["running"] and replaced == []  # no replace while it is in flight
    assert journal.read_bytes().startswith(grown)
    requests[0] = "ended"
    assert scrub.run(lambda: True, publish=idle)
    assert len(journal.read_bytes()) < len(grown)
    return facts.file_sha256(target)


def _seal_compacts_once(facts, target, clock, tmp_path, monkeypatch) -> str:
    workspace = target.parent.parent
    gone = target.with_name("gone.parquet")
    gone.write_bytes(b"one")
    for path in (target, gone):
        facts.record_sealed_file(path, _ONE)
        clock.settle(path)
    gone.unlink()  # trusted in this process; its scrub finds it gone
    journal = workspace / facts.FILE_TRUST
    replaced, replace = [], os.replace
    monkeypatch.setattr(
        facts.os, "replace", lambda src, dst: (replaced.append(Path(dst)), replace(src, dst))[1]
    )
    scrub = _file_scrub(facts, workspace)
    published = lambda action: (action(), True)[1]
    assert scrub.run(lambda: True, publish=published)
    assert scrub.run(lambda: True, publish=published)  # sealed and compacted: nothing due
    assert replaced.count(journal) == 1 and "gone.parquet" not in journal.read_text(
        encoding="utf-8"
    )
    return facts.file_sha256(target)


def _growth_arms_nothing(facts, target, clock, tmp_path, monkeypatch) -> str:
    from datetime import UTC, datetime, timedelta
    from types import SimpleNamespace

    from alphalattice.control.product_host.maintenance.file_trust_scrub import IdleFileTrustScrub
    from alphalattice.kernel.execution import Timers

    workspace = target.parent.parent
    facts.record_sealed_file(target, _ONE)
    clock.settle(target)
    now = [datetime(2026, 11, 22, tzinfo=UTC)]
    published = lambda action: (action(), True)[1]
    host = SimpleNamespace(
        clock=lambda: now[0], foreground_idle=lambda: True, publish_when_idle=published
    )
    timers = Timers()
    try:
        job = IdleFileTrustScrub(workspace, host, timers)
        assert job.scrub.run(lambda: True, publish=published)  # a scheduled cycle completes
        job.wake.set_next_due(job.scrub.next_due)  # armed for the next pass, a period away
        armed = job.wake.seconds_until_due(now[0])
        journal = workspace / facts.FILE_TRUST
        base = target.stat().st_mtime_ns - 10**10
        for tick in range(1100):  # a rewritten file's foreground facts outgrow the bound
            os.utime(target, ns=(base, base + tick * 1000))
            clock.settle(target)
            facts.record_sealed_file(target, _ONE)
        job.idle()
        assert job.wake.seconds_until_due(now[0]) == armed  # the crossing arms nothing
        assert len(journal.read_text(encoding="utf-8").splitlines()) > 1100
        now[0] += timedelta(days=7)
        assert job.scrub.run(lambda: True, publish=published)  # the next scheduled cycle
        assert len(journal.read_text(encoding="utf-8").splitlines()) < 10
    finally:
        timers.close()
    return facts.file_sha256(target)


def _discard_survives_compaction(facts, target, clock, tmp_path, monkeypatch) -> str:
    workspace = target.parent.parent
    other, gone = target.with_name("other.parquet"), target.with_name("gone.parquet")
    for path in (target, other, gone):
        path.write_bytes(b"one")
        facts.record_sealed_file(path, _ONE)
        clock.settle(path)
    stamp, real_stamp = file_stamp(other), digests.handle_stamp
    monkeypatch.setattr(
        digests, "handle_stamp", lambda h: stamp if Path(h.name) == other else real_stamp(h)
    )
    other.write_bytes(b"two")  # its stamp kept, its bytes changed: the scrub's mismatch
    hashed = digests.integrity_sha256_stream

    def removed_during_the_scan(*args, **kwargs):
        gone.unlink(missing_ok=True)  # the pass then skips it, so a compaction is due at its seal
        return hashed(*args, **kwargs)

    monkeypatch.setattr(digests, "integrity_sha256_stream", removed_during_the_scan)
    assert _file_scrub(facts, workspace).run(lambda: True)
    monkeypatch.setattr(digests, "integrity_sha256_stream", hashed)
    journal = workspace / facts.FILE_TRUST
    assert facts._SCRUB_DISCARDED in journal.read_text(encoding="utf-8")  # within its bound
    fresh = target.with_name("fresh.parquet")
    fresh.write_bytes(b"one")
    base = fresh.stat().st_mtime_ns - 10**10
    for tick in range(1100):  # the same Host's later facts outgrow the bound
        os.utime(fresh, ns=(base, base + tick * 1000))
        clock.settle(fresh)
        facts.record_sealed_file(fresh, _ONE)
    assert _next_scrub(facts, workspace).run(lambda: True)  # its next scheduled cycle
    text = journal.read_text(encoding="utf-8")
    assert "gone.parquet" not in text and facts._SCRUB_DISCARDED not in text  # compacted
    facts.release_file_facts(workspace)
    facts.keep_file_facts(workspace)
    assert facts.file_fact(target, "sha256") is None  # the discard holds across the reopen
    return facts.file_sha256(target)


def _failed_replace_stays_due(facts, target, clock, tmp_path, monkeypatch) -> str:
    workspace = target.parent.parent
    gone = target.with_name("gone.parquet")
    gone.write_bytes(b"one")
    for path in (target, gone):
        facts.record_sealed_file(path, _ONE)
        clock.settle(path)
    gone.unlink()
    journal = workspace / facts.FILE_TRUST
    replace, failed = os.replace, []

    def shared_once(src, dst):
        if Path(dst) == journal and not failed:
            failed.append(dst)
            raise PermissionError("the journal is open elsewhere")
        return replace(src, dst)

    monkeypatch.setattr(facts.os, "replace", shared_once)
    scrub = _file_scrub(facts, workspace)
    with pytest.raises(PermissionError, match="open elsewhere"):
        scrub.run(lambda: True)
    assert "gone.parquet" in journal.read_text(encoding="utf-8")
    assert [name for name in os.listdir(journal.parent) if name.endswith(".partial")] == []
    assert scrub.run(lambda: True)  # still due: the retry compacts
    assert "gone.parquet" not in journal.read_text(encoding="utf-8")
    return facts.file_sha256(target)


def _load_probes_no_file(facts, target, clock, tmp_path, monkeypatch) -> str:
    workspace = target.parent.parent
    facts.record_sealed_file(target, _ONE)
    clock.settle(target)
    journal = workspace / facts.FILE_TRUST
    line = json.loads(journal.read_text(encoding="utf-8").splitlines()[-1])
    (workspace.parent / "outside.parquet").write_bytes(b"one")
    with journal.open("a", encoding="utf-8") as stream:  # an entry the confined resolver refuses
        stream.write(json.dumps(["../outside.parquet", *line[1:]]) + "\n")
    facts.release_file_facts(workspace)
    probes, is_file = [], Path.is_file

    def denied(path):
        probes.append(path)
        raise PermissionError("its parent denies traversal")

    monkeypatch.setattr(Path, "is_file", denied)
    facts.keep_file_facts(workspace)  # the lease's load opens the workspace with no probe
    monkeypatch.setattr(Path, "is_file", is_file)
    assert probes == []
    assert _file_scrub(facts, workspace).run(lambda: True)
    assert "outside.parquet" not in journal.read_text(encoding="utf-8")
    return facts.file_sha256(target)


# Each input, the trust record's hashes it costs (a writer's hash of the file it renamed into
# place included, backup's own hashlib digest not) and the digest its last read gives.
_TRUST_TABLE = {
    "write then read": (_write_then_read, 1, _ONE),
    "read twice": (_read_twice, 1, _ONE),
    "bytes changed, same size, mtime inside the racy tick": (_racy_rewrite, 2, _TWO),
    "mtime changed": (_mtime_changed, 2, _ONE),
    "replaced by rename": (_renamed_over, 2, _TWO),
    "switch off": (_switch_off, 3, _ONE),
    "copied to another root": (_copied_root, 2, _ONE),
    "a writer's digest of bytes the path does not hold": (_foreign_write, 2, _ONE),
    "edited in place while its writer hashes it": (_edited_while_hashed, 2, _TWO),
    "backup": (_backup, 3, _TWO),
    "scrub detects a settled changed stamp": (_scrubbed_rewrite, 3, _TWO),
    "scrub interruption leaves no stale journal on restart": (_scrub_interrupted_append, 4, _TWO),
    "scrub yields between files and keeps its period": (_scrub_yields_between_files, 7, _ONE),
    "entries for removed files are compacted at idle": (_removed_file_compacted, 2, _ONE),
    "a request during the staged journal's sync defers its replace": (
        _request_during_the_fsync,
        4,
        _ONE,
    ),
    "a scrub sealed with immediate publication compacts once": (_seal_compacts_once, 3, _ONE),
    "a foreground crossing arms nothing; the next scrub compacts": (
        _growth_arms_nothing,
        1103,
        _ONE,
    ),
    "a discard sealed with a compaction due holds on reopen": (
        _discard_survives_compaction,
        1109,
        _ONE,
    ),
    "a failed replace keeps the compaction due": (_failed_replace_stays_due, 3, _ONE),
    "the lease's load probes no file": (_load_probes_no_file, 2, _ONE),
}


@pytest.mark.proves("workspace_runtime.changed_file_ends_its_fact")
@pytest.mark.parametrize("case", list(_TRUST_TABLE), ids=list(_TRUST_TABLE))
def test_a_files_digest_is_trusted_until_its_stamp_changes(
    tmp_path: Path, monkeypatch, case: str
) -> None:
    """Each input's hash count and digest: a stamp's change, the racy tick, a copy and the
    switch hash again; the full proof and backup hash every file."""
    run, hashes, digest = _TRUST_TABLE[case]
    with _kept_facts(tmp_path) as (facts, _workspace, target):
        clock = _Clock(facts, monkeypatch)
        calls = _counted_hashes(monkeypatch)
        with collect() as timing:
            assert run(facts, target, clock, tmp_path, monkeypatch) == digest
    assert len(calls) == hashes
    off = any(row["detail"] == "file_trust_off" for row in timing.rows())
    assert off == (case == "switch off")


@pytest.mark.proves("workspace_runtime.changed_file_ends_its_fact")
def test_a_files_digest_outlives_its_process_for_the_lease_holders_workspace(
    tmp_path: Path, monkeypatch
) -> None:
    """requirement: a writer's digest is kept for the next process of the workspace, which
    reads the file with no hash; older journal lines cost a hash, and a long journal is
    compacted to its latest entries."""
    with _kept_facts(tmp_path) as (facts, workspace, target):
        _aged(target)
        clock = _Clock(facts, monkeypatch)
        clock.settle(target)
        facts.record_sealed_file(target, _ONE)
        facts.release_file_facts(workspace)
        facts.keep_file_facts(workspace)  # the next process: only the writer's journal
        calls = _counted_hashes(monkeypatch)
        assert facts.sealed_file_holds(target, _ONE)
        assert facts.file_fact(target, "sha256") == _ONE
        assert calls == []
        journal = workspace / facts.FILE_TRUST
        line = journal.read_bytes().splitlines(keepends=True)[-1]
        malformed = [*json.loads(line)[:-1], ""]  # the held stamp, with no SHA-256
        journal.write_text(
            json.dumps([target.relative_to(workspace).as_posix(), "sealed-sha256", _ONE])
            + "\n"
            + json.dumps(malformed)
            + "\n",
            encoding="utf-8",
        )
        facts.release_file_facts(workspace)
        facts.keep_file_facts(workspace)
        assert facts.file_fact(target, "sha256") is None
        assert facts.file_sha256(target) == _ONE
        assert len(calls) == 1
        with journal.open("ab") as stream:
            stream.write(line * 1100)
        before = journal.stat().st_size
        facts.release_file_facts(workspace)
        facts.keep_file_facts(workspace)
        assert journal.stat().st_size == before  # the load reads it; its idle job compacts it
        assert _file_scrub(facts, workspace).run(lambda: True)
        assert journal.stat().st_size < before
        assert facts.file_fact(target, "sha256") == _ONE


@pytest.mark.proves("workspace_runtime.changed_file_ends_its_fact")
def test_a_value_derived_from_a_files_bytes_is_kept_by_their_digest(
    tmp_path: Path, monkeypatch
) -> None:
    """A young reader's value is not kept and its writer's is; a copy shares it by digest; a
    rewrite, or a forced index collision, finds none; an unreadable stamp still hashes."""
    with _kept_facts(tmp_path) as (facts, workspace, target):
        clock = _Clock(facts, monkeypatch)
        facts.record_file_fact(target, "columns", "a" * 64)
        assert facts.file_fact(target, "columns") is None
        facts.record_file_fact(target, "columns", "a" * 64, written=True)
        assert facts.file_fact(target, "columns") == "a" * 64
        facts.release_file_facts(workspace)
        facts.keep_file_facts(workspace)
        assert facts.file_fact(target, "columns") == "a" * 64
        copy = target.with_name("copy.parquet")
        copy.write_bytes(target.read_bytes())
        assert facts.file_fact(copy, "columns") == "a" * 64
        original = target.stat()
        target.write_bytes(b"two")
        os.utime(target, ns=(original.st_atime_ns, original.st_mtime_ns))
        clock.settle(target)
        assert facts.file_fact(target, "columns") is None
        monkeypatch.setattr(facts, "change_xxh3_128_hex", lambda _data: "0" * 32)
        facts.record_file_fact(target, "columns", "b" * 64, written=True)
        target.write_bytes(b"one")
        clock.settle(target)
        assert facts.file_fact(target, "columns") is None

        def unavailable(_path):
            raise OSError("stamp unavailable")

        with monkeypatch.context() as blocked, collect() as timing:
            blocked.setattr(digests.os, "fstat", unavailable)
            assert facts.file_sha256(target) == _ONE
        assert any(row["detail"] == "file_stamp_unavailable" for row in timing.rows())
