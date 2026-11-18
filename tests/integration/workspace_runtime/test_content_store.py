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
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from queue import Queue
from threading import Barrier, BoundedSemaphore, Event, Lock, Semaphore, Thread, current_thread
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
def test_a_sealed_source_builds_once_per_process_and_full_proof_builds_again(tmp_path, kept_here):
    """requirement: a value built from sealed sources answers every later read in the process;
    full proof (an export, a stage's check, the scrub) builds it from the bytes again."""
    read, calls = _reader(tmp_path / "sealed.bin")
    first = read()
    assert (first is read()) is kept_here  # where nothing is kept, every read builds
    assert first == b"sealed"
    assert len(calls) == (1 if kept_here else 2)
    assert _proved(read) == b"sealed"
    assert len(calls) == (2 if kept_here else 3)
    read()
    assert len(calls) == (2 if kept_here else 4)


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
def _kept_facts(tmp_path: Path):
    """A workspace whose file facts this process keeps, and one file in it, written now."""
    from alphalattice.control.workspace_runtime import verified_facts as facts

    workspace = tmp_path / "workspace"
    target = workspace / "store" / "part.parquet"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"one")
    facts.keep_file_facts(workspace)
    try:
        yield facts, workspace, target
    finally:
        facts.release_file_facts(workspace)


def _aged(target: Path) -> None:
    os.utime(target, (target.stat().st_atime - 10, target.stat().st_mtime - 10))


def _hashed(timing) -> int:
    """How many sealed files the verifier hashed inside `timing`."""
    return sum(row["count"] for row in timing.rows() if row["detail"] == "sealed_file")


@pytest.mark.proves("workspace_runtime.changed_file_ends_its_fact")
def test_a_fresh_files_digest_is_kept_from_its_writer_never_from_a_reader(
    tmp_path: Path,
) -> None:
    """requirement: a reader's digest of a file modified within two seconds is not kept, since
    a write in the same clock tick could leave its stamp; the writer's own digest is, and a
    sealed file's is trusted only once its stat has settled."""
    with _kept_facts(tmp_path) as (facts, _workspace, target):
        facts.record_file_fact(target, "columns", "a" * 64)
        assert facts.file_fact(target, "columns") is None
        facts.record_file_fact(target, "columns", "a" * 64, written=True)
        assert facts.file_fact(target, "columns") == "a" * 64
        sealed = hashlib.sha256(b"one").hexdigest()
        facts.record_sealed_file(target, sealed)
        stat = target.stat()
        target.write_bytes(b"two")  # same size, in the same tick as far as its stat says
        os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        assert not facts.sealed_file_holds(target, sealed)


@pytest.mark.proves("workspace_runtime.changed_file_ends_its_fact")
def test_a_file_fact_outlives_its_process_for_the_lease_holders_workspace(
    tmp_path: Path,
) -> None:
    """requirement: a digest verified once is kept for the next process of the workspace, and
    a sealed file's writer's digest lets that process read it with no hash."""
    with _kept_facts(tmp_path) as (facts, workspace, target):
        _aged(target)
        sealed = hashlib.sha256(b"one").hexdigest()
        facts.record_sealed_file(target, sealed)
        facts.release_file_facts(workspace)
        facts.keep_file_facts(workspace)  # the next process: only the writer's journal
        with collect() as timing:
            assert facts.sealed_file_holds(target, sealed)
        assert _hashed(timing) == 0
        # A legacy metadata-only fact is readable but never used as a byte proof.
        stat = target.stat()
        journal = workspace / facts.FILE_FACTS
        journal.parent.mkdir(parents=True, exist_ok=True)
        journal.write_text(
            json.dumps(
                [
                    target.relative_to(workspace).as_posix(),
                    "columns",
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_ino,
                    "a" * 64,
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        facts.release_file_facts(workspace)
        facts.keep_file_facts(workspace)
        assert facts.file_fact(target, "columns") is None
        facts.record_file_fact(target, "columns", "a" * 64)
        facts.release_file_facts(workspace)
        facts.keep_file_facts(workspace)
        assert facts.file_fact(target, "columns") == "a" * 64
        facts.configure_file_facts(capacity=1)
        try:
            for _ in range(2):
                facts.file_sha256(target)
                facts.record_file_fact(target, "columns", "a" * 64)
                assert facts.file_fact(target, "sha256") is None
                assert facts.file_sha256(target) == hashlib.sha256(b"one").hexdigest()
                facts.release_file_facts(workspace)
            journal = (workspace / facts.FILE_FACTS).read_bytes()
            facts.keep_file_facts(workspace)
            assert (workspace / facts.FILE_FACTS).read_bytes() == journal
            assert facts.file_fact(target, "sha256") == hashlib.sha256(b"one").hexdigest()
            # Compaction keeps the latest content proof while bounding historical rows.
            path = workspace / facts.FILE_FACTS
            with path.open("ab") as stream:
                stream.write(journal.splitlines(keepends=True)[-1] * 1100)
            before = path.stat().st_size
            facts.release_file_facts(workspace)
            facts.keep_file_facts(workspace)
            assert path.stat().st_size < before
            assert facts.file_fact(target, "sha256") == hashlib.sha256(b"one").hexdigest()
        finally:
            facts.configure_file_facts()


@pytest.mark.proves("workspace_runtime.changed_file_ends_its_fact")
def test_a_change_to_the_file_ends_its_fact(tmp_path: Path, monkeypatch) -> None:
    """tamper: a file rewritten in place, same size, loses its fact and is digested again; a
    sealed file's read hashes it again once its stat changed, and only full proof refuses a
    rewrite whose stat was put back."""
    with _kept_facts(tmp_path) as (facts, _workspace, target):
        _aged(target)
        digest = hashlib.file_digest
        calls = []

        def counted(*args, **kwargs):
            calls.append(True)
            return digest(*args, **kwargs)

        with monkeypatch.context() as counting:
            counting.setattr(hashlib, "file_digest", counted)
            with ThreadPoolExecutor(max_workers=4) as pool:
                assert (
                    list(pool.map(facts.file_sha256, [target] * 4))
                    == [hashlib.sha256(b"one").hexdigest()] * 4
                )
            assert facts.file_fact(target, "sha256") == hashlib.sha256(b"one").hexdigest()
            assert facts.file_sha256(target) == hashlib.sha256(b"one").hexdigest()
            assert calls == [True]

        def unavailable(_path):
            raise OSError("stamp unavailable")

        with monkeypatch.context() as blocked, collect() as timing:
            blocked.setattr(facts.os, "fstat", unavailable)
            assert facts.file_sha256(target) == hashlib.sha256(b"one").hexdigest()
        assert any(row["detail"] == "file_stamp_unavailable" for row in timing.rows())
        facts.record_file_fact(target, "columns", "a" * 64)
        copy = target.with_name("copy.parquet")
        copy.write_bytes(target.read_bytes())
        assert facts.file_fact(copy, "columns") == "a" * 64
        stat = target.stat()
        os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        assert facts.file_fact(target, "columns") == "a" * 64
        original = target.stat()
        assert facts.file_sha256(target) == hashlib.sha256(b"one").hexdigest()
        target.write_bytes(b"two")
        os.utime(target, ns=(original.st_atime_ns, original.st_mtime_ns))
        assert facts.file_fact(target, "columns") is None
        assert facts.file_sha256(target) == hashlib.sha256(b"two").hexdigest()
        sealed = hashlib.sha256(b"two").hexdigest()
        _aged(target)
        facts.record_sealed_file(target, sealed)
        aged = target.stat()
        target.write_bytes(b"TWO")
        with collect() as timing:
            assert not facts.sealed_file_holds(target, sealed)  # its stat changed: hashed
        assert _hashed(timing) == 1
        os.utime(target, ns=(aged.st_atime_ns, aged.st_mtime_ns))
        assert facts.sealed_file_holds(target, sealed)  # stat put back: trusted, as git does
        with facts.full_content_proof():
            assert not facts.sealed_file_holds(target, sealed)
        target.write_bytes(b"two")
        os.utime(target, ns=(original.st_atime_ns, original.st_mtime_ns))
        # Even a forced index collision cannot match a different full content key.
        monkeypatch.setattr(facts.xxhash, "xxh3_128_hexdigest", lambda _data: "0" * 32)
        facts.record_file_fact(target, "columns", "b" * 64, written=True)
        target.write_bytes(b"one")
        os.utime(target, ns=(original.st_atime_ns, original.st_mtime_ns))
        assert facts.file_fact(target, "columns") is None
