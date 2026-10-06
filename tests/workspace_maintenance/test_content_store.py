"""The content store keeps each file in its DA9 type and every digest it names (V210).

Numerical lanes are Parquet, named by the SHA-256 of their C-order bytes, so a lane sealed as
packed `.bin` bytes keeps its name; a store written before the change is still read.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict

from alphalattice.control.workspace_runtime.content_store import (
    ContentAddressedStore,
    ContentAddressedStoreError,
    columns_digest,
    verified_model_read_scope,
)


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


def test_verified_plan_reuses_frozen_dataclass_request_without_bypassing_validation(tmp_path):
    store = _store(tmp_path)
    plan = _ReadPlan(
        content_hash="a" * 64,
        alternate_hash="b" * 64,
        text="sealed",
        request=_FrozenReadRequest(("READ",)),
    )
    store.publish_model(category="plans", value=plan, identity_field="content_hash")
    _ReadPlan.validation_calls = 0
    with verified_model_read_scope(reuse_verified=True):
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
    assert first is second and first == plan
    assert _ReadPlan.validation_calls == 1


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


def test_a_verified_model_is_shared_inside_a_request_and_reused_only_when_enabled(
    tmp_path: Path,
):
    _FrozenReadRecord.validation_calls = 0
    store, identity, _path = _published_read_record(tmp_path)
    second_owner_store = ContentAddressedStore(store.root, uri_prefix="artifact://other")

    with verified_model_read_scope(reuse_verified=True):
        first = _load_read_record(store, identity)
        with verified_model_read_scope():
            second = _load_read_record(second_owner_store, identity)
        assert _load_read_record(store, identity) is first
        assert second is first
    assert _FrozenReadRecord.validation_calls == 1

    with verified_model_read_scope(reuse_verified=True):
        assert _load_read_record(store, identity) == first
    assert _FrozenReadRecord.validation_calls == 1

    # The request map has been reset, and a scope without explicit reuse validates again.
    with verified_model_read_scope():
        assert _load_read_record(store, identity) == first
    assert _FrozenReadRecord.validation_calls == 2
    assert _load_read_record(store, identity) == first
    assert _FrozenReadRecord.validation_calls == 3


def test_concurrent_model_read_scopes_keep_request_maps_isolated(tmp_path: Path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

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

    assert outcomes == (True, True)
    assert _FrozenReadRecord.validation_calls == 2


def test_verified_model_cache_keys_include_store_namespace_model_and_identity_field(
    tmp_path: Path,
):
    _FrozenReadRecord.validation_calls = 0
    _OtherFrozenReadRecord.validation_calls = 0
    store, identity, _path = _published_read_record(tmp_path)
    other_store, _other_identity, _other_path = _published_read_record(tmp_path, root="other")

    with verified_model_read_scope(reuse_verified=True):
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


def test_changed_model_bytes_with_restored_size_and_mtime_are_reverified(
    tmp_path: Path,
):
    _FrozenReadRecord.validation_calls = 0
    store, identity, path = _published_read_record(tmp_path)
    original = path.read_bytes()
    stat = path.stat()
    changed = original.replace(b'"sealed"', b'"broken"')
    assert len(changed) == len(original)

    with verified_model_read_scope(reuse_verified=True):
        _load_read_record(store, identity)
        path.write_bytes(changed)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        after = path.stat()
        assert (after.st_size, after.st_mtime_ns) == (stat.st_size, stat.st_mtime_ns)
        with pytest.raises(ContentAddressedStoreError, match="artifact_tampered"):
            _load_read_record(store, identity)
    assert _FrozenReadRecord.validation_calls == 2


def test_missing_model_file_invalidates_a_warm_verified_entry(tmp_path: Path):
    _FrozenReadRecord.validation_calls = 0
    store, identity, path = _published_read_record(tmp_path)

    with verified_model_read_scope(reuse_verified=True):
        _load_read_record(store, identity)
    assert _FrozenReadRecord.validation_calls == 1

    path.unlink()
    with (
        verified_model_read_scope(reuse_verified=True),
        pytest.raises(ContentAddressedStoreError) as missing,
    ):
        _load_read_record(store, identity)
    assert str(missing.value) == f"content_store.artifact_missing:{identity}"
    assert _FrozenReadRecord.validation_calls == 1


def test_failed_model_verification_is_never_cached(tmp_path: Path):
    _FrozenReadRecord.validation_calls = 0
    store, identity, path = _published_read_record(tmp_path)
    path.write_bytes(path.read_bytes().replace(b'"sealed"', b'"broken"'))

    for _ in range(2):
        with (
            verified_model_read_scope(reuse_verified=True),
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
    page = b"<p>sealed before V210</p>"
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


def test_a_changed_or_missing_lane_is_refused_by_name(tmp_path: Path) -> None:
    store = _store(tmp_path)
    lane = store.publish_columns(category="lanes/x", columns={"value": np.arange(3.0)})
    path = tmp_path / "store/lanes/x" / f"{lane}.parquet"
    written = path.read_bytes()

    for changed in (written[:-1], bytes([written[0] ^ 1]) + written[1:]):
        path.write_bytes(changed)
        with pytest.raises(ContentAddressedStoreError, match=r"content_store\.artifact_tampered"):
            store.load_packed_bytes(category="lanes/x", content_hash=lane)
    with pytest.raises(ContentAddressedStoreError, match=r"content_store\.artifact_missing"):
        store.load_packed_bytes(category="lanes/x", content_hash="0" * 64)
    with pytest.raises(ContentAddressedStoreError, match=r"content_store\.columns_invalid"):
        store.publish_columns(category="lanes/x", columns={"a": np.zeros(2), "b": np.zeros(3)})


def test_a_shared_file_is_replaced_through_a_reader_holding_it_open(tmp_path: Path) -> None:
    """regression (V477, AX17's overlapping trial reads): on Windows a replace fails while
    another handle reads the file, and a Host's concurrent requests hold one as they read a
    trial record or the verification ledger; the replace waits out the read and lands."""

    import threading

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


@pytest.mark.parametrize("reader", ["model", "columns", "packed", "document"])
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
    with pytest.raises(ContentAddressedStoreError, match=r"^content_store.artifact_tampered$"):
        load(identity)
    if reader == "model":
        target.write_text(Record(content_hash="b" * 64, text="wrong identity").model_dump_json())
        with pytest.raises(ContentAddressedStoreError, match=r"^content_store.artifact_tampered$"):
            load(identity)
    target.unlink()
    with pytest.raises(ContentAddressedStoreError) as absent:
        load(identity)
    assert str(absent.value) == f"content_store.artifact_missing:{identity}"
    words = refusal_words(str(absent.value))
    assert identity in words["detail"] and "workspace show" in words["detail"]
    assert "restore a backup" in words["detail"] and words["next_action"]
