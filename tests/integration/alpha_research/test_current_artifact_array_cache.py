"""Frozen current inputs reuse verified NPZ arrays only inside an open request scope."""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from alphalattice.control.workspace_runtime.content_store import (
    full_content_proof,
    verified_model_read_scope,
)
from alphalattice.investment.alpha_research.experiments import development_artifacts
from alphalattice.investment.alpha_research.inputs.frozen_price_volume import (
    FrozenPriceVolumeInputs,
)
from alphalattice.investment.alpha_research.publication.artifacts import (
    AlphaCurrentArtifactReadbackError,
    AlphaCurrentArtifactStore,
)
from alphalattice.investment.alpha_research.publication.contracts import FrozenFeaturePreparation
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
    current_artifacts, monkeypatch, kept_here
):
    """Control: Frozen arrays share scoped payloads while isolating mutable shells."""
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
            with pytest.raises(ValueError, match="WRITEABLE"):
                values.setflags(write=True)

        assert preparation_a == preparation_b == preparation
        assert surfaces_a is surfaces_b
        assert tuple(item.feature_values_hash for item in surfaces_a) == tuple(
            item.feature_values_hash for item in surfaces_b
        )
        assert all(not item.features.flags.writeable for item in surfaces_b)

    store.load_frozen_observations(observation.snapshot_hash)
    _, surfaces_after_scope = store.load_frozen_feature_preparation(preparation.preparation_hash)
    # The request memo goes with its scope; the process keeps sealed values past it (EV2).
    # Not kept: the second scope inflates both archives again.
    assert load_calls == (2 if kept_here else 4)
    if kept_here:
        assert surfaces_after_scope is surfaces_a
    else:
        assert tuple(item.feature_values_hash for item in surfaces_after_scope) == tuple(
            item.feature_values_hash for item in surfaces_a
        )


@pytest.mark.proves("alpha_research.current_artifacts_read_once_per_process")
def test_immutable_artifacts_are_read_once_across_request_scopes(
    current_artifacts, monkeypatch, kept_here
):
    """requirement: sealed observation and feature artifacts are read and decoded once per process
    across request scopes."""
    store, observation, preparation = current_artifacts
    load = np.load
    load_calls = 0
    watched_paths = {
        store._packed_source_path(
            "frozen-observation-arrays", observation.array_files[-1]
        ).resolve(): "arrays",
        store._packed_source_path(
            "frozen-feature-arrays", preparation.array_content_hash
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
    with verified_model_read_scope(), verified_lifecycle_admissions():
        observations_a = store.load_frozen_observations(observation.snapshot_hash)
        _, surfaces_a = store.load_frozen_feature_preparation(preparation.preparation_hash)
    with verified_model_read_scope(), verified_lifecycle_admissions():
        observations_b = store.load_frozen_observations(observation.snapshot_hash)
        _, surfaces_b = store.load_frozen_feature_preparation(preparation.preparation_hash)

    # Sealed bytes are read once per process on every OS; the next scope answers from memory.
    # Not kept: each scope reads and inflates its sealed bytes again.
    assert read_counts["json"] == (3 if kept_here else 6)
    assert read_counts["arrays"] == (2 if kept_here else 4)
    assert load_calls == (2 if kept_here else 4)
    assert observations_a is not observations_b
    if kept_here:
        assert observations_a.open is observations_b.open
        assert surfaces_a is surfaces_b
    else:
        assert np.array_equal(observations_a.open, observations_b.open)
        assert tuple(item.feature_values_hash for item in surfaces_a) == tuple(
            item.feature_values_hash for item in surfaces_b
        )
    observations_a.sector_by_listing_id["QA-A"] = "changed shell"
    assert observations_b.sector_by_listing_id["QA-A"] == "Technology"
    with pytest.raises(ValueError, match="WRITEABLE"):
        observations_b.open.setflags(write=True)
    with pytest.raises(ValueError, match="WRITEABLE"):
        surfaces_b[0].features.setflags(write=True)


@pytest.mark.proves("alpha_research.current_artifacts_read_once_per_process")
def test_publishing_the_same_immutable_artifact_succeeds_after_a_read(
    current_artifacts, tmp_path, monkeypatch
):
    """requirement: publishing an artifact already kept in memory names the same content and reads
    back."""
    store, observation, preparation = current_artifacts
    from alphalattice.kernel.data_layer import history

    history.held()  # this test's machine, independent of a running Host
    parts = [(digest, "current/frozen-observation-arrays") for digest in observation.array_files]
    parts.append((preparation.array_content_hash, "current/frozen-feature-arrays"))
    for digest, layout in parts:
        body_digest = history.alpha_part_body_digest(digest, layout)
        profile = history.array_profile(body_digest)
        assert profile is not None and profile["kind"] == "packed"
    with verified_model_read_scope():
        store.load_frozen_observations(observation.snapshot_hash)
        store.load_frozen_feature_preparation(preparation.preparation_hash)
    repeated = store.publish_frozen_observations(_source(), disposition="SYNTHETIC_INPUT_QA")
    assert repeated == observation
    with verified_model_read_scope():
        assert store.load_frozen_observations(repeated.snapshot_hash).source_binding_hash == (
            observation.snapshot_hash
        )
    _check_frozen_rollback(tmp_path / "rollback", monkeypatch)
    _check_candidate_score(tmp_path / "score", monkeypatch)
    _check_array_upgrade(tmp_path / "upgrade", monkeypatch)


def _check_frozen_rollback(tmp_path, monkeypatch):
    """A frozen input is readable by the storage mode that wrote it."""
    store = AlphaCurrentArtifactStore(tmp_path / "artifacts")
    payload = b"frozen input"
    category = "frozen-feature-arrays"
    monkeypatch.setenv("ALPHALATTICE_ARRAY_STORE", "legacy")
    digest = store._publish_packed_bytes(category="current/" + category, payload=payload)
    assert store._frozen_payload(category, digest) == payload
    monkeypatch.setenv("ALPHALATTICE_ARRAY_STORE", "door")
    assert store._publish_packed_bytes(category="current/" + category, payload=payload) == digest
    assert store._frozen_payload(category, digest) == payload
    monkeypatch.setenv("ALPHALATTICE_ARRAY_STORE", "legacy")
    assert store._frozen_payload(category, digest) == payload


def _check_candidate_score(tmp_path, monkeypatch):
    """A nullable score and its listing identity survive both readers and the typed relation."""
    from hashlib import sha256

    from alphalattice.kernel.data_layer import history

    store = AlphaCurrentArtifactStore(tmp_path / "artifacts")
    session = date(2026, 1, 2)
    table = pa.table(
        {
            "formation_session": pa.array([session, session], type=pa.date32()),
            "listing_id": ["QA-A", "QA-B"],
            "score": pa.array([1.0, None], type=pa.float64()),
            "availability": ["SCORED", "FEATURE_INCOMPLETE"],
        }
    )
    args = dict(
        request_hash="a" * 64,
        foundation_hash="b" * 64,
        logical_panel_hash="c" * 64,
        candidate_id="candidate",
        estimator_state_hash="d" * 64,
        stability_assessment_hash="e" * 64,
        formation_session=session,
        ordered_listing_ids=("QA-A", "QA-B"),
    )
    monkeypatch.setenv("ALPHALATTICE_ARRAY_STORE", "legacy")
    old = store.publish_current_candidate_score_chunk(table, **args)
    assert store.resolve_current_candidate_score_chunk(old).equals(table)
    monkeypatch.setenv("ALPHALATTICE_ARRAY_STORE", "door")
    new = store.publish_current_candidate_score_chunk(table, **args)
    assert new == old
    assert store.resolve_current_candidate_score_chunk(new).equals(table)
    packed = history.read_alpha_part(new.content_hash, "current/current-candidate-score-chunks")
    digest = sha256(packed).hexdigest()
    assert digest != new.content_hash
    assert history.array_profile(digest)["kind"] == "packed"
    assert store.publish_current_candidate_score_chunk(table, **args) == new
    monkeypatch.setenv("ALPHALATTICE_ARRAY_STORE", "legacy")
    assert store.resolve_current_candidate_score_chunk(old).equals(table)


def _check_array_upgrade(tmp_path, monkeypatch):
    """The upgrade moves a workspace's exact bodies before changing the reader."""
    from types import SimpleNamespace

    from alphalattice.investment.alpha_research.experiments.development_artifacts import (
        alpha_array_files,
        alpha_array_upgrade_family,
    )
    from alphalattice.kernel.data_layer import history

    family = alpha_array_upgrade_family(lambda _workspace: None)
    assert not family.holds(tmp_path / "new-workspace")
    monkeypatch.setenv("ALPHALATTICE_ARRAY_STORE", "legacy")
    store = AlphaCurrentArtifactStore(tmp_path / "artifacts")
    observation = store.publish_frozen_observations(_source(), disposition="SYNTHETIC_INPUT_QA")
    history_values = {"open": np.arange(4, dtype=np.float64).reshape(2, 2)}
    history_head = store.publish_workspace_observation_history(
        scope_hash=canonical_hash("slot"),
        selection_hash=canonical_hash("selection"),
        dependency_prefix_hash=canonical_hash("dependency"),
        formation_sessions=(date(2026, 1, 2), date(2026, 1, 5)),
        ordered_listing_ids=("QA-A", "QA-B"),
        stable_session_count=1,
        columns=history_values,
        capacity=lambda _size: None,
    )
    session = date(2026, 1, 2)
    table = pa.table(
        {
            "formation_session": pa.array([session], type=pa.date32()),
            "listing_id": ["QA-A"],
            "score": pa.array([None], type=pa.float64()),
            "availability": ["FEATURE_INCOMPLETE"],
        }
    )
    score = store.publish_current_candidate_score_chunk(
        table,
        request_hash="a" * 64,
        foundation_hash="b" * 64,
        logical_panel_hash="c" * 64,
        candidate_id="candidate",
        estimator_state_hash="d" * 64,
        stability_assessment_hash="e" * 64,
        formation_session=session,
        ordered_listing_ids=("QA-A",),
    )
    before = {path: path.read_bytes() for path, _ in alpha_array_files(tmp_path)}
    assert before and any(path.suffix == ".bin" for path in before)
    assert any(path.suffix == ".parquet" for path in before)
    log = history.held()
    assert family.holds(tmp_path)
    runner = SimpleNamespace(root=history.data_root(), log=log, workspaces=(tmp_path,))
    family.step(runner)
    assert family.verify(runner)
    assert not family.holds(tmp_path)
    family.switch("new")
    assert store.resolve_current_candidate_score_chunk(score).equals(table)
    assert (
        store.load_frozen_observations(observation.snapshot_hash).source_binding_hash
        == observation.snapshot_hash
    )
    loaded_head, loaded = store.load_workspace_observation_history(canonical_hash("slot"))
    assert loaded_head == history_head
    np.testing.assert_array_equal(loaded["open"], history_values["open"])
    family.switch("old")
    assert store.resolve_current_candidate_score_chunk(score).equals(table)
    assert (
        store.load_frozen_observations(observation.snapshot_hash).source_binding_hash
        == observation.snapshot_hash
    )
    loaded_head, loaded = store.load_workspace_observation_history(canonical_hash("slot"))
    assert loaded_head == history_head
    np.testing.assert_array_equal(loaded["open"], history_values["open"])
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.proves("alpha_research.current_artifacts_read_once_per_process")
def test_complete_immutable_alpha_models_are_validated_once_per_process(
    current_artifacts, monkeypatch, kept_here
):
    """requirement: a complete immutable Alpha model is validated once per process; a later request
    answers from the kept value."""
    store, _, preparation = current_artifacts
    validate = FrozenFeaturePreparation.model_validate
    validations = []

    def counted_validate(cls, value, *args, **kwargs):
        validations.append(value)
        return validate(value, *args, **kwargs)

    monkeypatch.setattr(FrozenFeaturePreparation, "model_validate", classmethod(counted_validate))
    with verified_model_read_scope():
        first, first_surfaces = store.load_frozen_feature_preparation(preparation.preparation_hash)
    first_count = len(validations)
    assert first_count == 1  # both owners prove exact current bytes; the model is validated once
    with verified_model_read_scope():
        second, second_surfaces = store.load_frozen_feature_preparation(
            preparation.preparation_hash
        )
    assert first == second == preparation
    # A sealed model's bytes never change under its name (EV2): a later request answers from
    # the process's kept value, on every OS.
    assert len(validations) == first_count
    assert first is second
    if kept_here:
        assert first_surfaces is second_surfaces
    else:
        assert all(
            np.array_equal(a.features, b.features)
            for a, b in zip(first_surfaces, second_surfaces, strict=True)
        )
    assert tuple(surface.feature_values_hash for surface in first_surfaces) == tuple(
        surface.feature_values_hash for surface in second_surfaces
    )


def test_an_alpha_model_with_nested_mutable_fields_is_always_a_fresh_shell(current_artifacts):
    store, observation, _ = current_artifacts
    with verified_model_read_scope():
        first = store.load_frozen_observation_snapshot(observation.snapshot_hash)
        first.sector_by_listing_id["QA-A"] = "changed by this caller"
        second = store.load_frozen_observation_snapshot(observation.snapshot_hash)
    assert first is not second
    assert first.sector_by_listing_id is not second.sector_by_listing_id
    assert second == observation
    assert second.sector_by_listing_id["QA-A"] == "Technology"


def test_a_valid_content_seal_cannot_replace_the_alpha_models_structure(current_artifacts):
    store, _, preparation = current_artifacts
    payload = preparation.model_dump(mode="json", exclude={"preparation_hash"})
    payload["vintages"] = []
    identity = canonical_hash(payload)
    path = store._path("current/frozen-feature-preparations", identity, "json")
    path.write_text(
        json.dumps(
            {**payload, "preparation_hash": identity}, sort_keys=True, separators=(",", ":")
        ),
        encoding="utf-8",
        newline="\n",
    )
    with verified_model_read_scope():
        for _ in range(2):
            with pytest.raises(ValueError, match="frozen_feature_preparation_invalid"):
                store.load_frozen_feature_preparation(identity)

    from alphalattice.interface.local_application.failure_codes import public_failure

    path.write_text(json.dumps({**payload, "preparation_hash": "0" * 64}), encoding="utf-8")
    with (
        full_content_proof(),
        pytest.raises(development_artifacts.AlphaDevelopmentArtifactReadbackError) as rejected,
    ):
        store.load_frozen_feature_preparation(identity)
    code = public_failure(rejected.value, "research_update.refused")
    assert code == "alpha_research.development_artifact_identity_invalid"
    from alphalattice.interface.local_application.failure_codes import untyped_failure

    assert untyped_failure(
        public_failure(ValueError(str(rejected.value)), "research_update.refused")
    )
    assert untyped_failure(
        public_failure(
            development_artifacts.AlphaDevelopmentArtifactReadbackError("another readback error"),
            "research_update.refused",
        )
    )
    from alphalattice.interface.local_application.cli_contract import refusal_words

    words = refusal_words(code)
    assert words["next_action"] == "RESTORE_THE_WORKSPACE_FROM_A_BACKUP"
    assert "verified workspace backup" in words["detail"]


@pytest.mark.proves("alpha_research.current_artifacts_full_proof_refuses_tamper")
def test_full_proof_refuses_observation_bytes_tampered_after_verification(current_artifacts):
    """tamper: a sealed observation array changed after its proof, size and time restored, is
    refused by name under full proof (an export, the scrub; EV2)."""
    store, observation, _ = current_artifacts
    with verified_model_read_scope():
        store.verify_frozen_observations(observation.snapshot_hash)

    path = store._packed_source_path("frozen-observation-arrays", observation.array_files[-1])
    before = path.stat()
    damaged = bytearray(path.read_bytes())
    damaged[len(damaged) // 2] ^= 1
    path.write_bytes(damaged)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with (
        full_content_proof(),
        verified_model_read_scope(),
        pytest.raises(
            development_artifacts.AlphaDevelopmentArtifactReadbackError, match="content_invalid"
        ),
    ):
        store.verify_frozen_observations(observation.snapshot_hash)


def test_observation_admission_proof_skips_unchanged_packed_bytes_in_a_new_scope(
    current_artifacts, monkeypatch, kept_here
):
    """Control: sealed packed observation bytes are read once across independent scopes."""
    store, observation, _ = current_artifacts
    path = store._packed_source_path("frozen-observation-arrays", observation.array_files[-1])
    open_file = Path.open
    reads = 0

    def counted_open(candidate, mode="r", *args, **kwargs):
        nonlocal reads
        if Path(candidate) == path and "r" in mode:
            reads += 1
        return open_file(candidate, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted_open)
    for _ in range(2):
        with verified_model_read_scope():
            store.verify_frozen_observations(observation.snapshot_hash)
    # Not kept: each new scope takes the packed bytes from disk again.
    assert reads == (1 if kept_here else 2)


@pytest.mark.proves("alpha_research.current_artifacts_full_proof_refuses_tamper")
@pytest.mark.parametrize("kind", ("observation", "feature"), ids=["observation", "feature"])
def test_full_proof_refuses_a_kept_payload_tampered_with_restored_mtime(kind, current_artifacts):
    """tamper: a sealed array changed after a read, its time restored, is refused by name under
    full proof, as is a missing one; the restored bytes read again."""
    store, observation, preparation = current_artifacts
    category, identity = (
        ("current/frozen-observation-arrays", observation.array_files[-1])
        if kind == "observation"
        else ("current/frozen-feature-arrays", preparation.array_content_hash)
    )
    path = store._packed_source_path(category.removeprefix("current/"), identity)
    original = path.read_bytes()
    stat = path.stat()
    from alphalattice.kernel.data_layer import arrays, history

    packed = bytearray(history.read_array(identity, packed=True))
    # Change the packed array, not Parquet bookkeeping that is outside its identity.
    damaged = None
    for offset in range(len(packed)):
        packed[offset] ^= 0x01
        candidate = arrays._packed_body(bytes(packed))
        packed[offset] ^= 0x01
        if len(candidate) == len(original):
            damaged = candidate
            break
    assert damaged is not None and damaged != original
    with verified_model_read_scope(), verified_lifecycle_admissions():
        if kind == "observation":
            store.load_frozen_observations(observation.snapshot_hash)

            def read():
                return store.load_frozen_observations(observation.snapshot_hash)
        else:
            store.load_frozen_feature_preparation(preparation.preparation_hash)

            def read():
                return store.load_frozen_feature_preparation(preparation.preparation_hash)

        path.write_bytes(damaged)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with (
            full_content_proof(),
            pytest.raises(AlphaCurrentArtifactReadbackError, match="content_invalid"),
        ):
            read()

        path.write_bytes(original)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with full_content_proof():
            read()

    path.unlink()
    with full_content_proof(), verified_model_read_scope(), pytest.raises(FileNotFoundError):
        read()


@pytest.mark.proves("alpha_research.current_artifacts_full_proof_refuses_tamper")
@pytest.mark.parametrize("kind", ("observation", "feature"), ids=["observation", "feature"])
def test_full_proof_rechecks_the_json_commitment_after_a_kept_reconstruction(
    kind, current_artifacts
):
    """tamper: a sealed JSON payload changed after a kept reconstruction is refused by its
    commitment under full proof."""
    store, observation, preparation = current_artifacts
    assert not hasattr(store, "publish_current_score_chunk")
    if kind == "observation":
        path = store._path(
            "current/frozen-observation-snapshots", observation.snapshot_hash, "json"
        )
        original = path.read_bytes()
        damaged = original.replace(b'"SYNTHETIC_INPUT_QA"', b'"RECORDED_INPUT_QA"')

        def read():
            return store.load_frozen_observations(observation.snapshot_hash)

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
    with verified_model_read_scope(), verified_lifecycle_admissions():
        read()
        path.write_bytes(damaged)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with full_content_proof(), pytest.raises(ValueError, match=refusal):
            read()
        path.write_bytes(original)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with full_content_proof():
            read()


def test_a_split_day_keeps_earlier_frozen_blocks_and_reads_the_rebase(tmp_path, monkeypatch):
    """requirement: a split day's frozen blocks stay the earlier ones' when only its rows as
    stored and rounding moved, of two as long the first by name, each read once."""
    from alphalattice.investment.alpha_research.publication.contracts import StoredRebase

    store = AlphaCurrentArtifactStore(tmp_path)
    days = tuple(date(2026, 1, 1) + timedelta(days=n) for n in range(43))
    base = np.arange(86, dtype=np.float64).reshape(43, 2) + 10.0
    names = ("open", "high", "low", "close", "volume")

    def source(rows, ohlcv, trend, **extra):  # type: ignore[no-untyped-def]
        return FrozenPriceVolumeInputs(
            formation_sessions=days[:rows], ordered_listing_ids=("QA-A", "QA-B"),
            sector_by_listing_id={"QA-A": "Technology", "QA-B": "Technology"},
            **{n: ohlcv[n][:rows] for n in names}, market_context_values=np.ones((rows, 3)),
            sector_trend_values=trend[:rows], source_binding_hash=canonical_hash("source"),
            formula_values={"momentum": np.full((rows, 2), 0.02)}, **extra,
        )  # fmt: skip

    trend = np.full((43, 1), 0.01)
    before, twin = (
        store.publish_frozen_observations(
            source(42, {n: base for n in names}, trend * scale), disposition="SYNTHETIC_INPUT_QA"
        )
        for scale in (1.0, 1 + 2e-15)
    )
    before = min((before, twin), key=lambda item: item.snapshot_hash)  # the tie's first by name
    listed, reads = Path.glob, []
    monkeypatch.setattr(Path, "glob", lambda self, pattern: reversed(sorted(listed(self, pattern))))
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: reads.append(self.name) or (
        self.read_bytes().decode(*a, **k)))  # fmt: skip
    split = {n: base.copy() for n in names}
    for n in names:
        split[n][:42, 1] *= 3.0 if n == "volume" else 1 / 3
    kept = {"QA-B": (np.ones(42, dtype=np.bool_), {f"ohlcv::{n}": base[:42, 1] for n in names})}
    rebase = StoredRebase(
        listing_id="QA-B", receipt_hash=canonical_hash("split"), price_factor=1 / 3,
        volume_factor=3.0, rows=((0, 42),),
    )  # fmt: skip
    after = store.publish_frozen_observations(
        source(
            43,
            split,
            trend * (1 + 1e-15),
            rebases=(rebase,),
            kept_rows=kept,
            rebase_agreement=(1e-6, 1e-12),
        ),
        disposition="SYNTHETIC_INPUT_QA",
    )
    monkeypatch.undo()
    assert after.array_blocks[:2] == before.array_blocks[:2] and after.rebases == (rebase,)
    earlier = {f"{item.snapshot_hash}.json" for item in (before, twin)}
    assert sorted(name for name in reads if name in earlier) == sorted(earlier)  # each once
    assert (after.kept_blocks, after.kept_agreement) == (
        (0, 1),
        (1e-6, 1e-12),
    )  # named, with its rule
