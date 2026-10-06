"""Frozen current inputs reuse verified NPZ arrays only inside an open request scope."""

from __future__ import annotations

import os
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from alphalattice.control.workspace_runtime.content_store import verified_model_read_scope
from alphalattice.investment.alpha_research.experiments import development_artifacts
from alphalattice.investment.alpha_research.inputs.frozen_price_volume import (
    FrozenPriceVolumeInputs,
)
from alphalattice.investment.alpha_research.publication.artifacts import (
    AlphaCurrentArtifactReadbackError,
    AlphaCurrentArtifactStore,
)
from alphalattice.investment.alpha_research.scores.model_renewal import (
    verified_lifecycle_admissions,
)
from alphalattice.investment.alpha_research.scores.product_replay import (
    HeterogeneousVintageFeatureSurface,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash


def _source() -> FrozenPriceVolumeInputs:
    shape = (2, 2)
    base = np.arange(4, dtype=np.float64).reshape(shape) + 1.0
    return FrozenPriceVolumeInputs(
        formation_sessions=(date(2026, 1, 2), date(2026, 1, 5)),
        ordered_listing_ids=("QA-A", "QA-B"),
        sector_by_listing_id={"QA-A": "Technology", "QA-B": "Technology"},
        open=base,
        high=base + 0.5,
        low=base - 0.5,
        close=base + 0.25,
        volume=base * 1000.0,
        market_context_values=np.ones((2, 3), dtype=np.float64),
        sector_trend_values=np.ones((2, 1), dtype=np.float64),
        source_binding_hash=canonical_hash("frozen-observation-source"),
        formula_values={"momentum": np.full(shape, 0.02, dtype=np.float64)},
    )


@pytest.fixture
def current_artifacts(tmp_path):
    store = AlphaCurrentArtifactStore(tmp_path / "runtime" / "artifacts")
    observation = store.publish_frozen_observations(_source(), disposition="SYNTHETIC_INPUT_QA")
    feature_source = canonical_hash("frozen-feature-source")
    surfaces = tuple(
        HeterogeneousVintageFeatureSurface.create(
            vintage=vintage,
            ordered_listing_ids=observation.ordered_listing_ids,
            ordered_feature_ids=("value", "quality"),
            features=np.array(
                [[index + 0.1, index + 0.2], [index + 0.3, index + 0.4]],
                dtype=np.float64,
            ),
            source_binding_hash=feature_source,
        )
        for index, vintage in enumerate(("2025-07", "2025-10"))
    )
    preparation = store.publish_frozen_feature_preparation(
        observation=observation,
        authority_hash=canonical_hash("frozen-feature-authority"),
        formation=date(2026, 1, 5),
        surfaces=surfaces,
    )
    return store, observation, preparation


def test_frozen_observations_and_feature_preparation_reuse_npz_with_fresh_shells(
    current_artifacts, monkeypatch
):
    store, observation, preparation = current_artifacts
    load = np.load
    load_calls = 0

    def counted_load(*args, **kwargs):
        nonlocal load_calls
        load_calls += 1
        return load(*args, **kwargs)

    monkeypatch.setattr(np, "load", counted_load)
    with verified_model_read_scope():
        with verified_lifecycle_admissions():
            observations_a = store.load_frozen_observations(observation.snapshot_hash)
            observations_b = store.load_frozen_observations(observation.snapshot_hash)
            preparation_a, surfaces_a = store.load_frozen_feature_preparation(
                preparation.preparation_hash
            )
            preparation_b, surfaces_b = store.load_frozen_feature_preparation(
                preparation.preparation_hash
            )

        assert load_calls == 2  # one archive inflation per exact payload, not per read
        assert observations_a is not observations_b
        assert observations_a.sector_by_listing_id is not observations_b.sector_by_listing_id
        assert observations_a.formula_values is not observations_b.formula_values
        assert observations_a.open is observations_b.open
        observations_a.sector_by_listing_id["QA-A"] = "mutated"
        observations_a.formula_values["momentum"] = np.zeros((2, 2))
        assert observations_b.sector_by_listing_id["QA-A"] == "Technology"
        assert np.all(observations_b.formula_values["momentum"] == 0.02)
        for values in (
            observations_a.open,
            observations_b.open,
            observations_b.formula_values["momentum"],
            surfaces_a[0].features,
        ):
            assert not values.flags.writeable
            with pytest.raises(ValueError):
                values.setflags(write=True)

        assert preparation_a == preparation_b == preparation
        assert surfaces_a is surfaces_b
        assert tuple(item.feature_values_hash for item in surfaces_a) == tuple(
            item.feature_values_hash for item in surfaces_b
        )
        assert all(not item.features.flags.writeable for item in surfaces_b)

    store.load_frozen_observations(observation.snapshot_hash)
    _, surfaces_after_scope = store.load_frozen_feature_preparation(preparation.preparation_hash)
    assert load_calls == 4  # the outer verified model scope releases all cached arrays/surfaces
    assert surfaces_after_scope is not surfaces_a


def test_immutable_artifacts_reuse_across_opted_in_scopes_with_unbroken_os_leases(
    current_artifacts, monkeypatch
):
    store, observation, preparation = current_artifacts
    load = np.load
    load_calls = 0
    watched_paths = {
        (
            store.root
            / "current/frozen-observation-arrays"
            / f"{observation.array_content_hash}.bin"
        ).resolve(): "arrays",
        (
            store.root / "current/frozen-feature-arrays" / f"{preparation.array_content_hash}.bin"
        ).resolve(): "arrays",
        (
            store.root
            / "current/frozen-observation-snapshots"
            / f"{observation.snapshot_hash}.json"
        ).resolve(): "json",
        (
            store.root
            / "current/frozen-feature-preparations"
            / f"{preparation.preparation_hash}.json"
        ).resolve(): "json",
    }
    read_counts = Counter()
    open_file = Path.open

    def counted_load(*args, **kwargs):
        nonlocal load_calls
        load_calls += 1
        return load(*args, **kwargs)

    def counted_open(path, mode="r", *args, **kwargs):
        watched = watched_paths.get(Path(path).resolve())
        if watched and "r" in mode:
            read_counts[watched] += 1
        return open_file(path, mode, *args, **kwargs)

    monkeypatch.setattr(np, "load", counted_load)
    monkeypatch.setattr(Path, "open", counted_open)
    with verified_model_read_scope(reuse_verified=True), verified_lifecycle_admissions():
        observations_a = store.load_frozen_observations(observation.snapshot_hash)
        _, surfaces_a = store.load_frozen_feature_preparation(preparation.preparation_hash)
    with verified_model_read_scope(reuse_verified=True), verified_lifecycle_admissions():
        observations_b = store.load_frozen_observations(observation.snapshot_hash)
        _, surfaces_b = store.load_frozen_feature_preparation(preparation.preparation_hash)

    expected_reads = 2 if os.name == "nt" else 4
    # The complete feature result also verifies JSON after acquiring both source leases.
    assert read_counts["json"] == (3 if os.name == "nt" else 6)
    assert read_counts["arrays"] == expected_reads
    assert load_calls == 2  # one decode per payload survives into the opted-in next scope
    assert observations_a is not observations_b
    assert observations_a.open is observations_b.open
    assert surfaces_a is surfaces_b
    observations_a.sector_by_listing_id["QA-A"] = "changed shell"
    assert observations_b.sector_by_listing_id["QA-A"] == "Technology"
    with pytest.raises(ValueError):
        observations_b.open.setflags(write=True)
    with pytest.raises(ValueError):
        surfaces_b[0].features.setflags(write=True)


def test_publishing_the_same_immutable_artifact_succeeds_after_an_opted_in_read(current_artifacts):
    store, observation, preparation = current_artifacts
    with verified_model_read_scope(reuse_verified=True):
        store.load_frozen_observations(observation.snapshot_hash)
        store.load_frozen_feature_preparation(preparation.preparation_hash)
    repeated = store.publish_frozen_observations(_source(), disposition="SYNTHETIC_INPUT_QA")
    assert repeated == observation
    with verified_model_read_scope(reuse_verified=True):
        assert store.load_frozen_observations(repeated.snapshot_hash).source_binding_hash == (
            observation.snapshot_hash
        )


def test_observation_admission_proof_refuses_current_bytes_tampered_after_verification(
    current_artifacts,
):
    store, observation, _ = current_artifacts
    with verified_model_read_scope(reuse_verified=True):
        store.verify_frozen_observations(observation.snapshot_hash)

    path = store._path("current/frozen-observation-arrays", observation.array_content_hash, "bin")
    before = path.stat()
    damaged = bytearray(path.read_bytes())
    damaged[len(damaged) // 2] ^= 1
    path.write_bytes(damaged)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with (
        verified_model_read_scope(reuse_verified=True),
        pytest.raises(
            development_artifacts.AlphaDevelopmentArtifactReadbackError, match="content_invalid"
        ),
    ):
        store.verify_frozen_observations(observation.snapshot_hash)


def test_observation_admission_proof_skips_unchanged_packed_bytes_in_a_new_scope(
    current_artifacts, monkeypatch
):
    store, observation, _ = current_artifacts
    path = store._path("current/frozen-observation-arrays", observation.array_content_hash, "bin")
    open_file = Path.open
    reads = 0

    def counted_open(candidate, mode="r", *args, **kwargs):
        nonlocal reads
        if Path(candidate) == path and "r" in mode:
            reads += 1
        return open_file(candidate, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted_open)
    for _ in range(2):
        with verified_model_read_scope(reuse_verified=True):
            store.verify_frozen_observations(observation.snapshot_hash)
    assert reads == (1 if os.name == "nt" else 2)


@pytest.mark.parametrize("kind", ("observation", "feature"))
def test_cached_payload_read_still_refuses_tampering_with_restored_mtime(kind, current_artifacts):
    store, observation, preparation = current_artifacts
    category, identity = (
        ("current/frozen-observation-arrays", observation.array_content_hash)
        if kind == "observation"
        else ("current/frozen-feature-arrays", preparation.array_content_hash)
    )
    path = store._path(category, identity, "bin")
    original = path.read_bytes()
    stat = path.stat()
    with verified_model_read_scope(reuse_verified=True), verified_lifecycle_admissions():
        if kind == "observation":
            store.load_frozen_observations(observation.snapshot_hash)

            def read():
                return store.load_frozen_observations(observation.snapshot_hash)
        else:
            store.load_frozen_feature_preparation(preparation.preparation_hash)

            def read():
                return store.load_frozen_feature_preparation(preparation.preparation_hash)

        damaged = bytearray(original)
        damaged[len(damaged) // 2] ^= 0x01
        path.write_bytes(damaged)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with pytest.raises(AlphaCurrentArtifactReadbackError, match="content_invalid"):
            read()

        path.write_bytes(original)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        read()

    path.unlink()
    with verified_model_read_scope(reuse_verified=True), pytest.raises(FileNotFoundError):
        if kind == "observation":
            store.load_frozen_observations(observation.snapshot_hash)
        else:
            store.load_frozen_feature_preparation(preparation.preparation_hash)


@pytest.mark.parametrize("kind", ("observation", "feature"))
def test_json_commitment_is_rechecked_after_a_cached_reconstruction(kind, current_artifacts):
    store, observation, preparation = current_artifacts
    if kind == "observation":
        path = store._path(
            "current/frozen-observation-snapshots", observation.snapshot_hash, "json"
        )
        original = path.read_bytes()
        damaged = original.replace(b'"SYNTHETIC_INPUT_QA"', b'"RECORDED_INPUT_QA"')

        def read():
            return store.load_frozen_observations(observation.snapshot_hash)

        refusal = "Alpha JSON payload hash is invalid"
    else:
        path = store._path(
            "current/frozen-feature-preparations", preparation.preparation_hash, "json"
        )
        original = path.read_bytes()
        damaged = original.replace(b'"2026-01-05"', b'"2026-01-06"')

        def read():
            return store.load_frozen_feature_preparation(preparation.preparation_hash)

        refusal = "Alpha JSON payload hash is invalid"
    assert damaged != original
    stat = path.stat()
    with verified_model_read_scope(reuse_verified=True), verified_lifecycle_admissions():
        read()
        path.write_bytes(damaged)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with pytest.raises(ValueError, match=refusal):
            read()
        path.write_bytes(original)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        read()
