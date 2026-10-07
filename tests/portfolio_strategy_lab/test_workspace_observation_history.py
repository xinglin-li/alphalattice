"""Durable observation history keeps exact bits, bounded slot roots and marker-last admission."""

from __future__ import annotations

import json
from datetime import date, timedelta
from hashlib import sha256

import numpy as np
import pyarrow.parquet as pq
import pytest

from alphalattice.investment.alpha_research.publication.artifacts import (
    AlphaCurrentArtifactReadbackError,
    AlphaCurrentArtifactStore,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash


def _sessions(count):
    return tuple(date(2026, 1, 1) + timedelta(days=index) for index in range(count))


def _columns(count):
    values = np.arange(count * 2, dtype=np.float64).reshape(count, 2)
    return {"open": values, "formula::value": values + 1.0}


def _publish(
    store, columns, *, stable, reuse=None, selection="selection", scope="slot", capacity=None
):
    return store.publish_workspace_observation_history(
        scope_hash=canonical_hash(scope),
        selection_hash=canonical_hash(selection),
        dependency_prefix_hash=canonical_hash(
            {name: sha256(value.tobytes()).hexdigest() for name, value in columns.items()}
        ),
        formation_sessions=_sessions(len(next(iter(columns.values())))),
        ordered_listing_ids=("A", "B"),
        stable_session_count=stable,
        columns=columns,
        reuse=reuse,
        capacity=capacity if capacity is not None else lambda _bytes: None,
    )


def _files(store):
    return {path: path.read_bytes() for path in store.root.rglob("*") if path.is_file()}


def test_cold_history_roundtrips_exact_float_bits_as_independently_immutable_columns(tmp_path):
    store = AlphaCurrentArtifactStore(tmp_path / "artifacts")
    assert store.load_workspace_observation_history(canonical_hash("slot")) is None
    bits = np.array(
        [0, 0x8000000000000000, 0x7FF8000000000042, 0x7FF0000000000000], dtype=np.uint64
    ).reshape(2, 2)
    expected_bits = bits.tobytes()
    columns = {"open": bits.view(np.float64), "formula::value": _columns(2)["open"]}
    head = _publish(store, columns, stable=1)
    reopened = AlphaCurrentArtifactStore(tmp_path / "artifacts")
    loaded_head, loaded = reopened.load_workspace_observation_history(canonical_hash("slot"))
    assert loaded_head == head
    assert loaded["open"].shape == (2, 2)
    assert loaded["open"].tobytes() == expected_bits
    assert loaded["formula::value"].tobytes() == columns["formula::value"].tobytes()
    columns["open"][0, 0] = 99.0
    assert loaded["open"].tobytes() == expected_bits
    with pytest.raises(TypeError):
        loaded["other"] = np.zeros((2, 2))
    with pytest.raises(ValueError):
        loaded["open"].setflags(write=True)


def test_stable_complete_segments_append_and_exact_current_is_idempotent(tmp_path):
    store = AlphaCurrentArtifactStore(tmp_path)
    first = _publish(store, _columns(2), stable=1)
    second = _publish(store, _columns(3), stable=2, reuse=first)
    assert second.parts[0] == first.parts[0]
    assert tuple((part.start, part.stop) for part in second.parts) == ((0, 1), (1, 2), (2, 3))
    assert second.predecessor_head_hash == first.head_hash
    before = _files(store)
    admitted = []
    again = _publish(store, _columns(3), stable=2, reuse=second, capacity=admitted.append)
    assert again == second
    assert admitted == []
    assert _files(store) == before
    loaded = store.load_workspace_observation_history(canonical_hash("slot"))[1]
    assert loaded["open"].tobytes() == _columns(3)["open"].tobytes()


def test_source_correction_requires_fresh_publication_and_holds_stable_prefix_parity(tmp_path):
    store = AlphaCurrentArtifactStore(tmp_path)
    first = _publish(store, _columns(3), stable=2)
    corrected = _columns(4)
    corrected["open"][0, 0] = -0.0
    before = _files(store)
    with pytest.raises(AlphaCurrentArtifactReadbackError, match="history_reuse_invalid"):
        _publish(store, corrected, stable=3, reuse=first)
    assert _files(store) == before
    forged = first.model_copy(update={"dependency_prefix_hash": canonical_hash("forged")})
    with pytest.raises(AlphaCurrentArtifactReadbackError, match="history_reuse_invalid"):
        _publish(store, _columns(4), stable=3, reuse=forged)
    fresh = _publish(store, corrected, stable=3)
    assert fresh.parts[0].content_hash != first.parts[0].content_hash
    assert store.load_workspace_observation_history(canonical_hash("slot"))[0] == fresh


def test_semantic_rotations_keep_one_bounded_slot_and_govern_old_head_roots(tmp_path):
    store = AlphaCurrentArtifactStore(tmp_path)
    first = _publish(store, _columns(2), stable=1, selection="model-one")
    renamed = {"renamed": _columns(2)["open"], "formula::value": _columns(2)["formula::value"]}
    second = _publish(store, renamed, stable=1, selection="model-two", reuse=first)
    third = _publish(store, _columns(3), stable=2, selection="model-three", reuse=second)
    other = _publish(store, _columns(2), stable=1, scope="other-slot")
    assert store.load_workspace_observation_history(canonical_hash("slot"))[0] == third
    before = _files(store)
    safe = store.workspace_observation_history_retention()
    roots = {fact.path.stem for fact in safe.roots}
    targets = {fact.path.stem for fact in safe.targets}
    assert {third.head_hash, second.head_hash, other.head_hash} <= roots
    assert first.head_hash in targets
    assert canonical_hash("slot") in roots and canonical_hash("other-slot") in roots
    assert set(safe.head_hashes) == {
        first.head_hash,
        second.head_hash,
        third.head_hash,
        other.head_hash,
    }
    bounded = store.workspace_observation_history_retention(
        referenced_heads=(first.head_hash,), active_scope_hashes=(canonical_hash("slot"),)
    )
    assert first.head_hash in {fact.path.stem for fact in bounded.roots}
    assert canonical_hash("other-slot") in {fact.path.stem for fact in bounded.targets}
    assert not {fact.path for fact in bounded.roots} & {fact.path for fact in bounded.targets}
    for fact in (*bounded.roots, *bounded.targets):
        assert fact.sha256 == sha256(fact.path.read_bytes()).hexdigest()
        assert fact.bytes == fact.path.stat().st_size
    assert _files(store) == before


def test_older_request_does_not_demote_latest_slot(tmp_path):
    store = AlphaCurrentArtifactStore(tmp_path)
    current = _publish(store, _columns(4), stable=3)
    marker = next(
        fact
        for fact in store.workspace_observation_history_retention().roots
        if fact.path.stem == canonical_hash("slot")
    )
    marker_bytes = marker.path.read_bytes()
    historical = _publish(store, _columns(2), stable=1, reuse=current)
    assert historical.formation_sessions == _sessions(2)
    assert marker.path.read_bytes() == marker_bytes
    assert store.load_workspace_observation_history(canonical_hash("slot"))[0] == current
    assert historical.head_hash in {
        fact.path.stem for fact in store.workspace_observation_history_retention().targets
    }


def test_capacity_covers_parts_json_and_marker_staging_and_interrupt_keeps_current(tmp_path):
    store = AlphaCurrentArtifactStore(tmp_path)
    admissions = []
    first = _publish(store, _columns(2), stable=1, capacity=admissions.append)
    assert sorted(admissions) == sorted(len(payload) for payload in _files(store).values())
    assert len(admissions) == 4  # Two Parquet segments, sealed head, then marker.
    before = _files(store)
    admissions.clear()

    def refuse_marker(byte_count):
        admissions.append(byte_count)
        if len(admissions) == 4:
            raise RuntimeError("capacity refused")

    with pytest.raises(RuntimeError, match="capacity refused"):
        # Two new rows make the newly stable segment differ from the old tail,
        # so both parts, the head and the marker each need a staging write.
        _publish(store, _columns(4), stable=3, reuse=first, capacity=refuse_marker)
    assert len(admissions) == 4
    assert store.load_workspace_observation_history(canonical_hash("slot"))[0] == first
    for path, payload in before.items():
        assert path.read_bytes() == payload
    assert not tuple(store.root.rglob("*.tmp"))
    inventory = store.workspace_observation_history_retention()
    assert inventory.targets  # Interrupted publication is unrooted, never silently repaired.


@pytest.mark.parametrize("target", ("marker", "head", "part", "previous"))
def test_named_history_missing_or_corrupt_artifacts_refuse_without_read_repair(tmp_path, target):
    store = AlphaCurrentArtifactStore(tmp_path)
    first = _publish(store, _columns(2), stable=1)
    second = _publish(store, _columns(3), stable=2, reuse=first)
    facts = store.workspace_observation_history_retention().roots
    if target == "marker":
        path = next(f.path for f in facts if f.path.stem == canonical_hash("slot"))
        path.write_bytes(b"{}")
    elif target == "head":
        path = next(f.path for f in facts if f.path.stem == second.head_hash)
        path.unlink()
    elif target == "part":
        path = next(f.path for f in facts if f.path.suffix == ".parquet")
        path.write_bytes(b"corrupt")
    else:
        path = next(f.path for f in facts if f.path.stem == first.head_hash)
        path.unlink()
    before = _files(store)
    with pytest.raises(AlphaCurrentArtifactReadbackError, match="workspace_observation_history_"):
        store.load_workspace_observation_history(canonical_hash("slot"))
    assert _files(store) == before


@pytest.mark.parametrize("change", ("names", "shape"))
def test_part_schema_is_verified_even_when_packed_value_digest_matches(tmp_path, change):
    store = AlphaCurrentArtifactStore(tmp_path)
    _publish(store, _columns(2), stable=2)
    path = next(
        f.path
        for f in store.workspace_observation_history_retention().roots
        if f.path.suffix == ".parquet"
    )
    table = pq.read_table(path)
    if change == "names":
        table = table.rename_columns(["wrong", "open"])
    else:
        table = table.replace_schema_metadata({"shape": json.dumps([1, 4])})
    pq.write_table(table, path, use_dictionary=False)
    with pytest.raises(AlphaCurrentArtifactReadbackError, match="history_invalid"):
        store.load_workspace_observation_history(canonical_hash("slot"))


@pytest.mark.parametrize("invalid", ("dtype", "shape", "capacity"))
def test_invalid_publication_is_refused_before_any_files_are_created(tmp_path, invalid):
    store = AlphaCurrentArtifactStore(tmp_path)
    columns = _columns(2)
    if invalid == "dtype":
        columns["open"] = columns["open"].astype(np.float32)
    elif invalid == "shape":
        columns["open"] = columns["open"].reshape(1, 4)
    with pytest.raises(AlphaCurrentArtifactReadbackError, match="history_invalid"):
        _publish(store, columns, stable=1, capacity=False if invalid == "capacity" else None)
    assert not _files(store)


def test_public_history_writer_requires_explicit_capacity_admission(tmp_path):
    store = AlphaCurrentArtifactStore(tmp_path)
    with pytest.raises(TypeError, match="capacity"):
        store.publish_workspace_observation_history(
            scope_hash=canonical_hash("slot"),
            selection_hash=canonical_hash("selection"),
            dependency_prefix_hash=canonical_hash("source"),
            formation_sessions=_sessions(2),
            ordered_listing_ids=("A", "B"),
            stable_session_count=1,
            columns=_columns(2),
        )
    assert not _files(store)
