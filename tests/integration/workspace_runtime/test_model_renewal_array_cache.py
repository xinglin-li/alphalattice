"""Request-local NPZ reuse keeps full artifact-byte verification on every access."""

from __future__ import annotations

import os
from collections import Counter
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from alphalattice.control.workspace_runtime import content_store
from alphalattice.investment.alpha_research.experiments.development_artifacts import (
    AlphaDevelopmentArtifactReadbackError,
    AlphaDevelopmentArtifactStore,
)
from alphalattice.investment.alpha_research.scores import model_renewal


def _store_with_arrays(root, payload: bytes) -> tuple[AlphaDevelopmentArtifactStore, str]:
    store = AlphaDevelopmentArtifactStore(root)
    identity = store._publish_packed_bytes(category="current/lifecycle-arrays", payload=payload)
    return store, identity


def _payload() -> bytes:
    stream = BytesIO()
    np.savez(
        stream,
        features=np.arange(24, dtype=np.float64).reshape(6, 4),
        targets=np.linspace(-0.1, 0.2, 6, dtype=np.float64),
    )
    return stream.getvalue()


@pytest.mark.proves("alpha_research.lifecycle_arrays_kept_per_process")
def test_arrays_are_shared_inside_nested_lifecycle_read_scope_and_kept_after(
    tmp_path, monkeypatch, kept_here
):
    """requirement: lifecycle arrays are shared read-only inside nested admission scopes and stay
    kept in the process after them."""
    payload = _payload()
    store, identity = _store_with_arrays(tmp_path / "one", payload)
    load = np.load
    load_calls = 0

    def counted_load(*args, **kwargs):
        nonlocal load_calls
        load_calls += 1
        return load(*args, **kwargs)

    monkeypatch.setattr(model_renewal.np, "load", counted_load)
    with model_renewal.verified_lifecycle_admissions():
        first = model_renewal._arrays(store, identity)
        with model_renewal.verified_lifecycle_admissions():
            second = model_renewal._arrays(store, identity)
        assert first is not second  # callers cannot replace entries in the private memo
        assert first["features"] is second["features"]
        assert first["targets"] is second["targets"]
        assert load_calls == 1
        for values in first.values():
            assert not values.flags.writeable
            with pytest.raises(ValueError, match="WRITEABLE"):
                values.setflags(write=True)
        first["targets"] = np.zeros(1)
        third = model_renewal._arrays(store, identity)
        assert third["targets"] is second["targets"]

    model_renewal._arrays(store, identity)
    # The process keeps sealed arrays past the scope (EV2); where nothing is kept, decode again.
    assert load_calls == (1 if kept_here else 2)


@pytest.mark.proves("alpha_research.lifecycle_arrays_full_proof_refuses_tamper")
def test_full_proof_rehashes_bytes_even_when_mtime_is_restored(tmp_path):
    """tamper: sealed arrays changed after a read, their time restored, are refused by name under
    full proof; the restored bytes read again."""
    payload = _payload()
    store, identity = _store_with_arrays(tmp_path / "one", payload)
    path = store._path("current/lifecycle-arrays", identity, "bin")
    original_stat = path.stat()
    with model_renewal.verified_lifecycle_admissions():
        model_renewal._arrays(store, identity)
        damaged = bytearray(payload)
        damaged[len(damaged) // 2] ^= 0x01
        path.write_bytes(damaged)
        os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        with (
            content_store.full_content_proof(),
            pytest.raises(AlphaDevelopmentArtifactReadbackError, match="content_invalid"),
        ):
            model_renewal._arrays(store, identity)

        path.write_bytes(payload)
        os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        with content_store.full_content_proof():
            restored = model_renewal._arrays(store, identity)
        assert np.array_equal(restored["features"], np.arange(24).reshape(6, 4))


def test_identical_content_at_another_store_root_has_a_distinct_cache_key(tmp_path, monkeypatch):
    payload = _payload()
    first_store, first_hash = _store_with_arrays(tmp_path / "one", payload)
    second_store, second_hash = _store_with_arrays(tmp_path / "two", payload)
    assert first_hash == second_hash == sha256(payload).hexdigest()
    load = np.load
    calls = 0
    reads = Counter()
    tracked = {
        (first_store.root / "current/lifecycle-arrays" / f"{first_hash}.bin").resolve(),
        (second_store.root / "current/lifecycle-arrays" / f"{second_hash}.bin").resolve(),
    }
    open_file = Path.open

    def counted_load(*args, **kwargs):
        nonlocal calls
        calls += 1
        return load(*args, **kwargs)

    def counted_open(path, mode="r", *args, **kwargs):
        resolved = Path(path).resolve()
        if resolved in tracked and "r" in mode:
            reads[resolved] += 1
        return open_file(path, mode, *args, **kwargs)

    monkeypatch.setattr(model_renewal.np, "load", counted_load)
    monkeypatch.setattr(Path, "open", counted_open)
    with content_store.verified_array_read_scope():
        first = model_renewal._arrays(first_store, first_hash)
    with content_store.verified_array_read_scope():
        second = model_renewal._arrays(second_store, second_hash)
    assert calls == 2
    assert all(reads[path] == 1 for path in tracked)
    assert first["features"] is not second["features"]


@pytest.mark.proves("alpha_research.lifecycle_arrays_kept_per_process")
def test_verified_arrays_are_read_and_decoded_once_across_scopes(tmp_path, monkeypatch, kept_here):
    """requirement: sealed lifecycle arrays are read and decoded once across scopes; full proof
    reads and decodes them again."""
    payload = _payload()
    store, identity = _store_with_arrays(tmp_path / "one", payload)
    load = np.load
    load_calls = 0
    reads = 0
    payload_path = (store.root / "current/lifecycle-arrays" / f"{identity}.bin").resolve()
    open_file = Path.open

    def counted_load(*args, **kwargs):
        nonlocal load_calls
        load_calls += 1
        return load(*args, **kwargs)

    def counted_open(path, mode="r", *args, **kwargs):
        nonlocal reads
        if Path(path).resolve() == payload_path and "r" in mode:
            reads += 1
        return open_file(path, mode, *args, **kwargs)

    monkeypatch.setattr(model_renewal.np, "load", counted_load)
    monkeypatch.setattr(Path, "open", counted_open)
    with content_store.verified_array_read_scope():
        first = model_renewal._arrays(store, identity)
    with content_store.verified_array_read_scope():
        second = model_renewal._arrays(store, identity)

    # Sealed bytes are read once per process where kept; where nothing is kept, each scope reads.
    assert reads == (1 if kept_here else 2)  # sealed bytes are read once per process (EV2)
    assert load_calls == (1 if kept_here else 2)
    if kept_here:
        assert first["features"] is second["features"]
    else:
        assert np.array_equal(first["features"], second["features"])
    with content_store.full_content_proof(), content_store.verified_array_read_scope():
        third = model_renewal._arrays(store, identity)
    assert reads == load_calls == (2 if kept_here else 3)  # full proof reads and decodes again
    assert third["features"] is not second["features"]


def test_process_cache_keeps_unverified_mutable_values_out():
    builds = 0

    def build_mutable():
        nonlocal builds
        builds += 1
        return {"values": []}

    for _ in range(2):
        with content_store.verified_array_read_scope():
            content_store.verified_request_value(
                ("mutable-process", "source"), build_mutable, nbytes=8
            )
    assert builds == 2


def test_failed_semantic_builder_is_never_retained():
    builds = 0

    def build():
        nonlocal builds
        builds += 1
        if builds == 1:
            raise ValueError("fixture validation failed")
        return (np.frombuffer(b"safe", dtype=np.uint8),)

    with (
        pytest.raises(ValueError, match="fixture validation failed"),
        content_store.verified_array_read_scope(),
    ):
        content_store.verified_request_value(("semantic", "exact"), build, nbytes=4)
    with content_store.verified_array_read_scope():
        accepted = content_store.verified_request_value(("semantic", "exact"), build, nbytes=4)
    with content_store.verified_array_read_scope():
        reused = content_store.verified_request_value(("semantic", "exact"), build, nbytes=4)

    assert builds == 2, (
        f"COUNT STOP op=rejected_builder observed={builds} budget=2 "
        "fixture=one-refusal wayon=reduce-work"
    )
    assert accepted is reused


def test_generic_request_cache_never_retains_mutable_shells():
    builds = 0

    def build_mutable():
        nonlocal builds
        builds += 1
        return {"values": []}

    with content_store.verified_array_read_scope():
        first = content_store.verified_request_value(
            ("mutable-fixture", "exact-source"), build_mutable, nbytes=8
        )
        first["values"].append("caller mutation")
        second = content_store.verified_request_value(
            ("mutable-fixture", "exact-source"), build_mutable, nbytes=8
        )
    assert builds == 2
    assert second == {"values": []}


def test_an_opted_in_mutable_value_is_built_once_and_each_caller_gets_its_own_copy():
    builds = 0

    def build_mutable():
        nonlocal builds
        builds += 1
        return {"values": ["verified"]}

    key = ("mutable-copied", "exact-source")
    with content_store.verified_array_read_scope():
        first = content_store.verified_request_value(
            key, build_mutable, nbytes=8, copy_mutable=True
        )
        first["values"].append("caller mutation")
        second = content_store.verified_request_value(
            key, build_mutable, nbytes=8, copy_mutable=True
        )
    with content_store.verified_array_read_scope():
        third = content_store.verified_request_value(
            key, build_mutable, nbytes=8, copy_mutable=True
        )
    assert builds == 1, (
        f"COUNT STOP op=mutable_builder observed={builds} budget=1 "
        "fixture=one-key wayon=reduce-work"
    )
    assert second == third == {"values": ["verified"]}
    assert second is not third and second["values"] is not third["values"]


def test_byte_bounds_rebuild_rejected_and_evicted_values():
    """A rejected size is never cached and an evicted value builds once on its next use."""
    builds: Counter[str] = Counter()
    limit = 512 * 1024 * 1024

    def read(name, size):
        def build():
            builds[name] += 1
            return (name,)

        return content_store.verified_request_value(("byte-bound", name), build, nbytes=size)

    with content_store.verified_array_read_scope():
        for _ in range(2):
            assert read("rejected", limit + 1) == ("rejected",)
        assert read("evicted", limit) == read("evicted", limit) == ("evicted",)
        assert read("replacement", limit) == ("replacement",)
    with content_store.verified_array_read_scope():
        assert read("evicted", limit) == read("evicted", limit) == ("evicted",)
    assert builds == {"rejected": 2, "evicted": 2, "replacement": 1}, (
        f"COUNT STOP op=bounded_builder observed={dict(builds)} budget=2/2/1 "
        "fixture=512MiB-metadata wayon=reduce-work"
    )


def test_discard_only_npz_proof_rehashes_and_binds_each_store(tmp_path, monkeypatch):
    """Control: Discard only npz proof rehashes and binds each store."""
    payload = _payload()
    store, identity = _store_with_arrays(tmp_path / "one", payload)
    other, same_hash = _store_with_arrays(tmp_path / "two", payload)
    load = np.load
    calls = 0

    def counted_load(*args, **kwargs):
        nonlocal calls
        calls += 1
        return load(*args, **kwargs)

    monkeypatch.setattr(model_renewal.np, "load", counted_load)
    with content_store.verified_array_read_scope():
        model_renewal._verify_arrays(store, identity)
        model_renewal._verify_arrays(other, same_hash)
    assert calls == 2
    path = store._path("current/lifecycle-arrays", identity, "bin")
    before = path.stat()
    damaged = bytearray(payload)
    damaged[len(damaged) // 2] ^= 1
    path.write_bytes(damaged)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with (
        content_store.full_content_proof(),
        content_store.verified_array_read_scope(),
        pytest.raises(AlphaDevelopmentArtifactReadbackError, match="content_invalid"),
    ):
        model_renewal._verify_arrays(store, identity)


def test_discard_only_npz_proof_cannot_cache_failed_member_decode(tmp_path, monkeypatch):
    """Control: Discard only npz proof cannot cache failed member decode."""
    stream = BytesIO()
    np.savez(stream, untrusted=np.array([{"object": "must refuse"}], dtype=object))
    store, identity = _store_with_arrays(tmp_path / "one", stream.getvalue())
    load = np.load
    calls = 0

    def counted_load(*args, **kwargs):
        nonlocal calls
        calls += 1
        return load(*args, **kwargs)

    monkeypatch.setattr(model_renewal.np, "load", counted_load)
    for _ in range(2):
        with (
            content_store.verified_array_read_scope(),
            pytest.raises(ValueError, match="Object arrays cannot be loaded"),
        ):
            model_renewal._verify_arrays(store, identity)
    assert calls == 2


def test_prepared_readback_proof_keeps_training_binding_check_and_current_rehash(
    tmp_path, monkeypatch
):
    store, identity = _store_with_arrays(tmp_path / "one", _payload())
    binding = "a" * 64
    prepared = SimpleNamespace(
        content_hash="b" * 64,
        plan=SimpleNamespace(content_hash="c" * 64),
        observations_hash="d" * 64,
        array_file_hash=identity,
        row_axis_hash="e" * 64,
        market_scale=SimpleNamespace(model_dump=lambda **kwargs: {}),
        training_binding_hash=binding,
        row_count=6,
    )
    original_hash = model_renewal.canonical_hash
    checks = 0

    def training_hash(value):
        nonlocal checks
        if isinstance(value, dict) and value.get("weight_rule") == "EQUAL_ROWS":
            checks += 1
            assert value["features"] == model_renewal.alpha_model_array_content_hash(
                np.arange(24, dtype=np.float64).reshape(6, 4)
            )
            return binding
        return original_hash(value)

    monkeypatch.setattr(model_renewal, "canonical_hash", training_hash)
    for _ in range(2):
        with content_store.verified_array_read_scope():
            model_renewal._verify_prepared_values(store, prepared)
    assert checks == 1
    # A caller retaining the declared hash cannot reuse a proof for changed semantics.
    prepared.row_count = 7
    with (
        content_store.verified_array_read_scope(),
        pytest.raises(model_renewal.AlphaLifecycleError, match="prepared_values_invalid"),
    ):
        model_renewal._verify_prepared_values(store, prepared)
    assert checks == 2
    path = store._path("current/lifecycle-arrays", identity, "bin")
    before = path.stat()
    payload = bytearray(path.read_bytes())
    payload[len(payload) // 2] ^= 1
    path.write_bytes(payload)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with (
        content_store.full_content_proof(),
        content_store.verified_array_read_scope(),
        pytest.raises(AlphaDevelopmentArtifactReadbackError, match="content_invalid"),
    ):
        model_renewal._verify_prepared_values(store, prepared)
