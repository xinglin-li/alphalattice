from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog
from alphalattice.foundation.feature_engine.inputs.closure_source import (
    FeatureClosureSourceRepository,
)
from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
    PanelClosureArtifactStore,
)
from alphalattice.foundation.feature_engine.panels.closure_contracts import (
    ClosureArtifactRef,
    ExpectedPanelChunk,
    PanelBaseKeyChunk,
    PanelBaseValueChunk,
    PanelBaseValueClosureManifest,
    PanelDerivationRecipe,
    PanelRetentionAssessment,
    SectorRevisionEntry,
    SectorRevisionMap,
)
from alphalattice.foundation.feature_engine.panels.feature_closure_contracts import (
    SectorRevisionMapActivationReceipt,
)
from alphalattice.foundation.feature_engine.panels.feature_closure_coordinator import (
    FeatureBaseClosureCoordinator,
)
from alphalattice.foundation.feature_engine.panels.feature_closure_ledger import (
    FeatureClosureLedger,
    identified,
)
from alphalattice.foundation.feature_engine.panels.recovery_binding import (
    PanelRecoveryBindingPublisher,
)
from alphalattice.foundation.feature_engine.storage.contracts import FeatureMaterializationWrite
from alphalattice.foundation.feature_engine.storage.repositories import (
    FeatureStateRepository,
)
from alphalattice.foundation.market_data_ops.sources.manifest import (
    ManifestListing,
    MarketProfile,
    UniverseManifest,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository

NOW = datetime(2026, 8, 7, 14, tzinfo=UTC)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _manifest() -> UniverseManifest:
    profile = MarketProfile(
        market_profile_id="feature-closure-fixture",
        display_name="Feature closure fixture",
        market="US",
        currency="USD",
        calendar_id="XNYS",
        provider="fixture",
        daily_price_basis="unadjusted",
        manifest_as_of=NOW.date(),
        data_validity_class="CURRENT_UNIVERSE_RESEARCH_ONLY",
    )
    return UniverseManifest(
        manifest_id="feature-closure-fixture",
        profile=profile,
        listings=(
            ManifestListing(
                listing_id="listing-a",
                symbol="AAA",
                mic="XNYS",
                provider_symbol="AAA",
            ),
        ),
        revision_sha256=_hash("manifest"),
        universe_membership_basis="CURRENT_ACTIVE_SURVIVORS",
        is_point_in_time_historical=False,
    )


def _write(
    *,
    catalog: FeatureCatalog,
    session: date,
    identity: str,
    first_value: float,
    rebuilt: tuple[str, ...] | None = None,
    listing_id: str = "listing-a",
) -> FeatureMaterializationWrite:
    values = {factor_id: float(index + 1) for index, factor_id in enumerate(catalog.factor_ids)}
    values[catalog.factor_ids[0]] = first_value
    values[catalog.factor_ids[1]] = -0.0
    values[catalog.factor_ids[2]] = np.nan
    frame = pd.DataFrame(
        (
            {
                "listing_id": listing_id,
                "session_date": session,
                "input_cutoffs_json": json.dumps(
                    {factor_id: None for factor_id in catalog.factor_ids},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                **values,
            },
        )
    )
    return FeatureMaterializationWrite(
        listing_id=listing_id,
        catalog_hash=catalog.binding.catalog_hash,
        rows=frame,
        ineligibility=(),
        raw_input_hash=_hash(f"raw-{session}"),
        action_set_hash_value=_hash("actions"),
        market_reference_revision=_hash("spy"),
        idempotency_key=identity,
        revision_reason="fixture",
        observed_at=NOW,
        factor_ids=rebuilt or catalog.factor_ids,
        rows_are_canonical=True,
    )


def _seed_verified_root(
    *,
    feature_state: FeatureStateRepository,
    artifact_store: PanelClosureArtifactStore,
    ledger: FeatureClosureLedger,
    source: FeatureClosureSourceRepository,
    catalog: FeatureCatalog,
    session: date,
) -> None:
    ref = ClosureArtifactRef(
        kind="fixture",
        content_hash=_hash("artifact-content"),
        physical_sha256=_hash("artifact-physical"),
        byte_count=1,
        row_count=1,
        uri="fixture://closure",
    )
    key = PanelBaseKeyChunk(
        year=session.year,
        key_hash=_hash("key"),
        first_session=session,
        last_session=session,
        listing_count=1,
        session_count=1,
        artifact=ref,
    )
    values = tuple(
        PanelBaseValueChunk(
            year=session.year,
            factor_id=factor_id,
            key_hash=key.key_hash,
            value_hash=_hash(f"value-{factor_id}"),
            null_count=0,
            artifact=ref,
        )
        for factor_id in catalog.factor_ids
    )
    base = identified(
        PanelBaseValueClosureManifest,
        {
            "catalog_hash": catalog.binding.catalog_hash,
            "history_start": session,
            "as_of_session": session,
            "factor_ids": catalog.factor_ids,
            "key_chunks": (key,),
            "value_chunks": values,
        },
        "manifest_hash",
    )
    ledger.publish_model("base-manifests", base.manifest_hash, base)
    snapshot_hash = _hash("active-snapshot")
    sector_revision = _hash("sector")
    sector_map = identified(
        SectorRevisionMap,
        {
            "manifest_revision": _hash("manifest"),
            "sector_revision": sector_revision,
            "entries": (
                SectorRevisionEntry(
                    listing_id="listing-a",
                    provider="fixture",
                    provider_symbol="AAA",
                    sector_name="Technology",
                    sector_key="technology",
                    payload_hash=_hash("sector-payload"),
                    evidence_hash=_hash("sector-evidence"),
                ),
            ),
        },
        "map_hash",
    )
    ledger.publish_sector_map(sector_map)
    recipe = identified(
        PanelDerivationRecipe,
        {
            "snapshot_hash": snapshot_hash,
            "panel_content_hash": _hash("panel-content"),
            "panel_binding_hash": _hash("panel-binding"),
            "schema_hash": _hash("schema"),
            "manifest_revision": _hash("manifest"),
            "sector_revision": sector_revision,
            "sector_map_hash": sector_map.map_hash,
            "catalog_hash": catalog.binding.catalog_hash,
            "policy_hash": _hash("policy"),
            "spy_revision": _hash("spy"),
            "history_start": session,
            "as_of_session": session,
            "listing_ids": ("listing-a",),
            "sessions": (session,),
            "factor_ids": catalog.factor_ids,
            "row_hash_factor_ids": catalog.factor_ids,
            "session_batch_size": 126,
            "base_closure_hash": base.manifest_hash,
            "base_value_catalog_hash": catalog.binding.catalog_hash,
            "availability_closure_hash": _hash("availability"),
            "row_receipt_assignment_hash": _hash("row-receipts"),
            "expected_chunks": (
                ExpectedPanelChunk(
                    year=session.year,
                    first_session=session,
                    last_session=session,
                    row_count=1,
                    chunk_hash=_hash("chunk"),
                    metadata_hash=_hash("metadata"),
                    physical_sha256=_hash("physical"),
                    byte_count=1,
                    uri="fixture://panel/chunk",
                ),
            ),
            "rematerialization_policy": "CURRENT_ENVIRONMENT_BYTE_PARITY_REQUIRED",
        },
        "recipe_hash",
    )
    ledger.publish_model("recipes", recipe.recipe_hash, recipe)
    assessment = identified(
        PanelRetentionAssessment,
        {
            "snapshot_hash": snapshot_hash,
            "recipe_hash": recipe.recipe_hash,
            "disposition": "REMATERIALIZATION_VERIFIED_CURRENT_ENVIRONMENT",
            "existing_panel_bytes": 1,
            "closure_bytes": 1,
            "future_reclaim_candidate_bytes": 1,
            "eviction_authorized": False,
            "verified_at": NOW,
        },
        "assessment_hash",
    )
    ledger.publish_model("retention-assessments", assessment.assessment_hash, assessment)
    failure_hash = _hash("phase-one-failure")
    artifact_store.publish_json(
        category="runs/failures",
        content_hash=failure_hash,
        payload={
            "kind": "PanelStorageClosureFailureReceipt",
            "verified_snapshot_hashes": [snapshot_hash],
            "blocked_snapshot_hashes": [_hash("d117")],
        },
    )
    ledger.admit_catalog_root(
        catalog_hash=catalog.binding.catalog_hash,
        active_snapshot_hash=snapshot_hash,
        phase_one_failure_receipt_hash=failure_hash,
        retention_assessment_hash=assessment.assessment_hash,
        feature_row_hash_digest=source.feature_row_hash_digest(
            catalog_hash=catalog.binding.catalog_hash
        ),
    )


def _fixture(tmp_path):
    catalog = FeatureCatalog.load()
    market_data = MarketDataRepository(tmp_path / "workspace")
    feature_state = FeatureStateRepository(market_data.database, market_data=market_data)
    market_data.bootstrap(_manifest())
    feature_state.ensure_current_storage()
    initial_session = date(2026, 8, 5)
    feature_state.upsert_feature_materialization_batch(
        (_write(catalog=catalog, session=initial_session, identity="root", first_value=1.0),)
    )
    resolver = ArtifactResolver(tmp_path / "artifacts")
    artifact_store = PanelClosureArtifactStore(resolver)
    ledger = FeatureClosureLedger(artifact_store)
    source = FeatureClosureSourceRepository(feature_state.path)
    _seed_verified_root(
        feature_state=feature_state,
        artifact_store=artifact_store,
        ledger=ledger,
        source=source,
        catalog=catalog,
        session=initial_session,
    )
    coordinator = FeatureBaseClosureCoordinator(
        store=feature_state,
        source=source,
        ledger=ledger,
        factor_ids=catalog.factor_ids,
    )
    return catalog, feature_state, artifact_store, ledger, coordinator


def test_verified_feature_source_window_reuses_exact_inputs_and_invalidates_other_writes(tmp_path):
    from dataclasses import replace
    from datetime import timedelta
    from types import SimpleNamespace

    from alphalattice.foundation.feature_engine.panels.materialization_identity import (
        feature_source_values_hash,
    )
    from alphalattice.foundation.feature_engine.producers.base_materializer import _REQUIRED_COLUMNS
    from alphalattice.foundation.feature_engine.runtime.service import FeatureFoundationService
    from alphalattice.foundation.feature_engine.storage.contracts import FeatureSourceWindow

    catalog, store, _artifacts, ledger, persistence = _fixture(tmp_path)
    day = date(2026, 8, 6)
    dates = tuple(day - timedelta(days=i) for i in (2, 1, 0))
    frame = pd.DataFrame(
        {
            name: dates if name == "session_date" else [100.0, 101.0, 102.0]
            for name in _REQUIRED_COLUMNS
        }
    )
    market = frame[["session_date", "provider_adjusted_close"]].copy()
    window = FeatureSourceWindow(
        day,
        dates[0],
        day,
        feature_source_values_hash(frame, _REQUIRED_COLUMNS),
        feature_source_values_hash(market, ("session_date", "provider_adjusted_close")),
    )
    write = replace(
        _write(catalog=catalog, session=day, identity="verified", first_value=2.0),
        source_window=window,
    )
    with store.feature_build_connection() as connection:
        receipt = persistence.persist_batch((write,), connection=connection)[0]
    head = ledger.require_head(catalog.binding.catalog_hash)
    rows = store.feature_rows(
        listing_ids=("listing-a",),
        catalog_hash=catalog.binding.catalog_hash,
        start=day,
        end=day,
        include_values=False,
        include_verification=True,
    )
    assert rows[0]["source_verification_receipt_hash"] == receipt
    assert (
        store.materialization_source_windows(
            listing_id="listing-a",
            catalog_hash=catalog.binding.catalog_hash,
            receipt_hashes=(receipt,),
        )[receipt][0]
        == window
    )
    owner = object.__new__(FeatureFoundationService)
    owner.catalog, owner.feature_state, owner._bounded_warmup_sessions = catalog, store, 2

    def reusable(stock, reference, *, calendar=None):
        with store.feature_build_connection() as connection:
            return owner._verified_source_sessions(
                listing_id="listing-a",
                catalog_hash=catalog.binding.catalog_hash,
                prepared=rows,
                frame=stock,
                market_frame=reference,
                calendar=tuple(stock.session_date) if calendar is None else calendar,
                request=SimpleNamespace(as_of_session=max(stock.session_date)),
                connection=connection,
                market_input_hashes={},
            )

    assert reusable(frame, market) == {day}
    assert reusable(frame.tail(2), market, calendar=dates) == set()
    changed = frame.copy()
    changed.loc[0, "provider_adjusted_close"] += 1
    assert reusable(changed, market) == set()
    changed = frame.copy()
    changed.loc[0, "close_raw"] += 1
    assert reusable(changed, market) == set()
    changed_market = market.copy()
    changed_market.loc[0, "provider_adjusted_close"] += 1
    assert reusable(frame, changed_market) == set()
    next_stock = pd.concat(
        (frame, frame.tail(1).assign(session_date=day + timedelta(days=1))), ignore_index=True
    )
    next_market = pd.concat(
        (market, market.tail(1).assign(session_date=day + timedelta(days=1))), ignore_index=True
    )
    assert reusable(next_stock, next_market) == {day}
    # A new verification can preserve the original numerical row and ledger head.
    with store.feature_build_connection() as connection:
        next_receipt = persistence.persist_batch(
            (replace(write, idempotency_key="verified-again"),), connection=connection
        )[0]
    assert ledger.require_head(catalog.binding.catalog_hash).head_hash == head.head_hash
    with store.feature_build_connection() as connection:
        persistence.persist_batch(
            (replace(write, idempotency_key="partial", factor_ids=(catalog.factor_ids[0],)),),
            connection=connection,
        )
    assert (
        store.feature_rows(
            listing_ids=("listing-a",),
            catalog_hash=catalog.binding.catalog_hash,
            start=day,
            end=day,
            include_values=False,
            include_verification=True,
        )[0]["source_verification_receipt_hash"]
        is None
    )
    with store.feature_build_connection() as connection:
        encoded = connection.execute(
            "SELECT coverage_summary_json FROM feature_materialization_receipt "
            "WHERE receipt_hash=?",
            [next_receipt],
        ).fetchone()[0]
        forged = json.loads(encoded)
        forged["source_window"]["stock_input_hash"] = "f" * 64
        connection.execute(
            "UPDATE feature_materialization_receipt SET coverage_summary_json=? "
            "WHERE receipt_hash=?",
            [json.dumps(forged), next_receipt],
        )
    with pytest.raises(ValueError, match="source_verification_receipt_invalid"):
        store.materialization_source_windows(
            listing_id="listing-a",
            catalog_hash=catalog.binding.catalog_hash,
            receipt_hashes=(next_receipt,),
        )


def test_tail_correction_noop_and_exact_replay_advance_only_real_closure(tmp_path) -> None:
    catalog, feature_state, artifact_store, ledger, coordinator = _fixture(tmp_path)
    tail = _write(
        catalog=catalog,
        session=date(2026, 8, 6),
        identity="tail",
        first_value=2.0,
        rebuilt=(catalog.factor_ids[0],),
    )
    with feature_state.feature_build_connection() as connection:
        receipts = coordinator.persist_batch((tail,), connection=connection)
    head = ledger.require_head(catalog.binding.catalog_hash)
    assert head.transition_cursor == 1
    transition = ledger.load_transition(str(head.last_transition_hash))
    patch = ledger.load_patch(transition.patch_hash)
    table = artifact_store.load_parquet(
        category="feature-ledger/patch-rows", reference=patch.artifact
    )
    assert table.num_rows == 1
    assert table[f"{catalog.factor_ids[1]}__bits"].to_pylist() == [2**63]
    assert table[f"{catalog.factor_ids[1]}__valid"].to_pylist() == [True]
    assert table[f"{catalog.factor_ids[2]}__valid"].to_pylist() == [False]

    with feature_state.feature_build_connection() as connection:
        assert coordinator.persist_batch((tail,), connection=connection) == receipts
    assert ledger.require_head(catalog.binding.catalog_hash).head_hash == head.head_hash

    noop = _write(
        catalog=catalog,
        session=date(2026, 8, 6),
        identity="noop-new-receipt",
        first_value=2.0,
    )
    with feature_state.feature_build_connection() as connection:
        coordinator.persist_batch((noop,), connection=connection)
    assert ledger.require_head(catalog.binding.catalog_hash).head_hash == head.head_hash

    correction = _write(
        catalog=catalog,
        session=date(2026, 8, 6),
        identity="correction",
        first_value=3.0,
    )
    with feature_state.feature_build_connection() as connection:
        coordinator.persist_batch((correction,), connection=connection)
    assert ledger.require_head(catalog.binding.catalog_hash).transition_cursor == 2


def test_a_patch_keeps_a_missing_cell_apart_from_a_nan_one(tmp_path) -> None:
    catalog, feature_state, artifact_store, ledger, coordinator = _fixture(tmp_path)
    write = _write(
        catalog=catalog, session=date(2026, 8, 6), identity="missing-cell", first_value=2.0
    )
    rows = write.rows.copy()
    # A column that is not float64 holds its cells as objects, and a missing one is None.
    rows[catalog.factor_ids[3]] = pd.Series([None], dtype=object)
    with feature_state.feature_build_connection() as connection:
        coordinator.persist_batch((replace(write, rows=rows),), connection=connection)
    transition = ledger.load_transition(
        str(ledger.require_head(catalog.binding.catalog_hash).last_transition_hash)
    )
    table = artifact_store.load_parquet(
        category="feature-ledger/patch-rows",
        reference=ledger.load_patch(transition.patch_hash).artifact,
    )
    nan, missing, present = catalog.factor_ids[2], catalog.factor_ids[3], catalog.factor_ids[4]
    assert table[f"{nan}__bits"].to_pylist() == [int(np.array([np.nan]).view("<u8")[0])]
    assert table[f"{nan}__valid"].to_pylist() == [False]
    assert table[f"{missing}__bits"].to_pylist() == [0]
    assert table[f"{missing}__valid"].to_pylist() == [False]
    assert table[f"{present}__bits"].to_pylist() == [int(np.array([5.0]).view("<u8")[0])]
    assert table[f"{present}__valid"].to_pylist() == [True]


def test_commit_before_marker_is_reconciled_without_recomputing_store(
    tmp_path, monkeypatch
) -> None:
    catalog, feature_state, _artifact_store, ledger, coordinator = _fixture(tmp_path)
    tail = _write(
        catalog=catalog,
        session=date(2026, 8, 6),
        identity="commit-before-marker",
        first_value=2.0,
    )
    original_complete = ledger.complete
    calls = 0

    def fail_once(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("fixture marker crash")
        return original_complete(**kwargs)

    monkeypatch.setattr(ledger, "complete", fail_once)
    with (
        feature_state.feature_build_connection() as connection,
        pytest.raises(RuntimeError, match="marker crash"),
    ):
        coordinator.persist_batch((tail,), connection=connection)
    assert ledger.pending_transition(catalog.binding.catalog_hash) is not None

    store_calls = 0
    original_store = feature_state.upsert_feature_materialization_batch

    def count_store(*args, **kwargs):
        nonlocal store_calls
        store_calls += 1
        return original_store(*args, **kwargs)

    monkeypatch.setattr(feature_state, "upsert_feature_materialization_batch", count_store)
    with feature_state.feature_build_connection() as connection:
        coordinator.reconcile_pending(
            catalog.binding.catalog_hash,
            connection=connection,
        )
    assert store_calls == 0
    assert ledger.pending_transition(catalog.binding.catalog_hash) is None
    assert ledger.require_head(catalog.binding.catalog_hash).transition_cursor == 1


def test_new_listing_is_closed_with_the_complete_factor_axis(tmp_path) -> None:
    catalog, feature_state, artifact_store, ledger, coordinator = _fixture(tmp_path)
    addition = _write(
        catalog=catalog,
        session=date(2026, 8, 6),
        identity="new-listing",
        first_value=4.0,
        listing_id="listing-new",
    )
    with feature_state.feature_build_connection() as connection:
        coordinator.persist_batch((addition,), connection=connection)
    head = ledger.require_head(catalog.binding.catalog_hash)
    transition = ledger.load_transition(str(head.last_transition_hash))
    patch = ledger.load_patch(transition.patch_hash)
    table = artifact_store.load_parquet(
        category="feature-ledger/patch-rows",
        reference=patch.artifact,
    )
    assert patch.listing_ids == ("listing-new",)
    assert patch.factor_ids == catalog.factor_ids
    assert table.num_columns == 3 + 2 * len(catalog.factor_ids)

    panel_table = pa.table(
        {
            "session_date": pa.array((date(2026, 8, 6),), type=pa.date32()),
            "listing_id": pa.array(("listing-new",), type=pa.string()),
            catalog.factor_ids[0]: pa.array((1.0,), type=pa.float64()),
        }
    )
    resolver = ArtifactResolver(tmp_path / "artifacts")
    descriptor = resolver.publish_feature_panel_chunk(
        table=panel_table,
        content_hash=_hash("post-addition-chunk"),
        metadata={"fixture": "post-addition-panel"},
    )
    root = ledger.root_for_head(head)
    recipe = ledger.recipe_for_root(root)
    binding = PanelRecoveryBindingPublisher(ledger=ledger, resolver=resolver).publish(
        snapshot_hash=_hash("post-addition-snapshot"),
        panel_content_hash=_hash("post-addition-content"),
        panel_binding_hash=_hash("post-addition-binding"),
        catalog_hash=catalog.binding.catalog_hash,
        sector_revision=recipe.sector_revision,
        listing_ids=("listing-a", "listing-new"),
        factor_ids=catalog.factor_ids,
        chunks=(
            {
                "uri": descriptor.uri,
                "chunk_hash": descriptor.content_hash,
                "metadata_hash": descriptor.metadata_hash,
            },
        ),
        panel_source_state_hash=_hash("post-addition-source"),
    )
    assert binding.closure_head_hash == head.head_hash
    assert binding.closure_transition_cursor == 1
    assert binding.sector_map_hash == recipe.sector_map_hash


def test_store_failure_leaves_prior_state_and_same_admission_can_retry(
    tmp_path, monkeypatch
) -> None:
    catalog, feature_state, _artifact_store, ledger, coordinator = _fixture(tmp_path)
    tail = _write(
        catalog=catalog,
        session=date(2026, 8, 6),
        identity="retry-prior",
        first_value=2.0,
    )
    original_store = feature_state.upsert_feature_materialization_batch

    def fail_store(*_args, **_kwargs):
        raise RuntimeError("fixture Store failure")

    monkeypatch.setattr(feature_state, "upsert_feature_materialization_batch", fail_store)
    with (
        feature_state.feature_build_connection() as connection,
        pytest.raises(RuntimeError, match="Store failure"),
    ):
        coordinator.persist_batch((tail,), connection=connection)
    assert ledger.pending_transition(catalog.binding.catalog_hash) is not None
    monkeypatch.setattr(feature_state, "upsert_feature_materialization_batch", original_store)
    with feature_state.feature_build_connection() as connection:
        coordinator.reconcile_pending(
            catalog.binding.catalog_hash,
            connection=connection,
        )
        assert ledger.pending_transition(catalog.binding.catalog_hash) is not None
        coordinator.persist_batch((tail,), connection=connection)
    assert ledger.pending_transition(catalog.binding.catalog_hash) is None
    assert ledger.require_head(catalog.binding.catalog_hash).transition_cursor == 1


def test_panel_recovery_binding_requires_a_clean_head_and_exact_sector_map(tmp_path) -> None:
    catalog, _store, _artifact_store, ledger, _coordinator = _fixture(tmp_path)
    resolver = ArtifactResolver(tmp_path / "artifacts")
    session = date(2026, 8, 5)
    table = pa.table(
        {
            "session_date": pa.array((session,), type=pa.date32()),
            "listing_id": pa.array(("listing-a",), type=pa.string()),
            catalog.factor_ids[0]: pa.array((1.0,), type=pa.float64()),
        }
    )
    chunk_hash = _hash("binding-chunk")
    descriptor = resolver.publish_feature_panel_chunk(
        table=table,
        content_hash=chunk_hash,
        metadata={"fixture": "recovery-binding"},
    )
    root = ledger.root_for_head(ledger.require_head(catalog.binding.catalog_hash))
    recipe = ledger.recipe_for_root(root)
    binding = PanelRecoveryBindingPublisher(ledger=ledger, resolver=resolver).publish(
        snapshot_hash=_hash("future-snapshot"),
        panel_content_hash=_hash("future-content"),
        panel_binding_hash=_hash("future-binding"),
        catalog_hash=catalog.binding.catalog_hash,
        sector_revision=recipe.sector_revision,
        listing_ids=("listing-a",),
        factor_ids=catalog.factor_ids,
        chunks=(
            {
                "uri": descriptor.uri,
                "chunk_hash": descriptor.content_hash,
                "metadata_hash": descriptor.metadata_hash,
            },
        ),
        panel_source_state_hash=_hash("source-state"),
    )
    assert binding.sessions == (session,)
    assert ledger.panel_binding(binding.snapshot_hash) == binding


def test_blocked_phase_one_snapshot_cannot_be_admitted_as_catalog_root(tmp_path) -> None:
    catalog, _store, artifact_store, ledger, _coordinator = _fixture(tmp_path)
    blocked_snapshot = _hash("d117")
    assessment_hash = next(
        path.stem for path in (artifact_store.root / "retention-assessments").glob("*.json")
    )
    failure_hash = next(
        path.stem for path in (artifact_store.root / "runs" / "failures").glob("*.json")
    )
    with pytest.raises(ValueError, match=r"feature_closure\.root_unavailable"):
        ledger.admit_catalog_root(
            catalog_hash=catalog.binding.catalog_hash,
            active_snapshot_hash=blocked_snapshot,
            phase_one_failure_receipt_hash=failure_hash,
            retention_assessment_hash=assessment_hash,
            feature_row_hash_digest=_hash("blocked-digest"),
        )


def test_republishing_an_unchanged_sector_activation_leaves_its_pointer_alone(
    tmp_path, monkeypatch
) -> None:
    """Every one-listing update cycle rebinds the same sector evidence.

    The pointer is content the ledger already holds, so republishing it must
    not replace the file: each needless replace is one more instant in which a
    momentary Windows sharing violation can stop the Task (observed once on the
    2026-09-12 QA build as ``feature_closure.sector_map_capture_failed``). A
    changed pointer is still replaced.
    """

    catalog, _store, _artifact_store, ledger, _coordinator = _fixture(tmp_path)
    root = ledger.root_for_head(ledger.require_head(catalog.binding.catalog_hash))
    recipe = ledger.recipe_for_root(root)
    replaced: list[tuple[str, str]] = []
    real_replace = os.replace

    def counted_replace(source, destination):  # type: ignore[no-untyped-def]
        replaced.append((str(source), str(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", counted_replace)

    def receipt(map_hash: str) -> SectorRevisionMapActivationReceipt:
        return identified(
            SectorRevisionMapActivationReceipt,
            {
                "manifest_revision": recipe.manifest_revision,
                "sector_revision": recipe.sector_revision,
                "sector_map_hash": map_hash,
                "store_receipt_hash": _hash("store-receipt"),
                "changed": False,
                "observed_at": NOW,
            },
            "receipt_hash",
        )

    ledger.publish_sector_activation(receipt(recipe.sector_map_hash))
    pointer = [d for _s, d in replaced if d.endswith(f"{recipe.sector_revision}.json")]
    assert len(pointer) == 1
    pointer_path = Path(pointer[0])
    before = pointer_path.read_bytes()

    ledger.publish_sector_activation(receipt(recipe.sector_map_hash))
    assert [d for _s, d in replaced if d.endswith(f"{recipe.sector_revision}.json")] == pointer
    assert pointer_path.read_bytes() == before

    ledger.publish_sector_activation(receipt(_hash("another-map")))
    assert len([d for _s, d in replaced if d.endswith(f"{recipe.sector_revision}.json")]) == 2
    assert pointer_path.read_bytes() != before


def test_changed_pointer_outlives_a_transient_sharing_violation_and_refuses_a_lasting_one(
    tmp_path, monkeypatch
) -> None:
    """requirement: a changed pointer is published atomically or not at all.

    Controlled fault injection on the replace itself: a sharing violation that
    clears is outlived within the durable bound and the pointer is read back
    through the ledger's own reader before success; one that lasts is raised
    as the kernel's typed conflict with the previous pointer untouched and no
    staged file left behind; a readback that does not match what was written
    is refused and never retried; and the next legitimate attempt converges
    without republishing content the store already holds.
    """

    from alphalattice.kernel.shared_kernel import persistence
    from alphalattice.kernel.shared_kernel.domain.errors import WorkspaceConflictError

    catalog, _store, artifact_store, ledger, _coordinator = _fixture(tmp_path)
    root = ledger.root_for_head(ledger.require_head(catalog.binding.catalog_hash))
    recipe = ledger.recipe_for_root(root)
    monkeypatch.setattr(persistence.time, "sleep", lambda _seconds: None)

    def receipt(map_hash: str) -> SectorRevisionMapActivationReceipt:
        return identified(
            SectorRevisionMapActivationReceipt,
            {
                "manifest_revision": recipe.manifest_revision,
                "sector_revision": recipe.sector_revision,
                "sector_map_hash": map_hash,
                "store_receipt_hash": _hash("store-receipt"),
                "changed": False,
                "observed_at": NOW,
            },
            "receipt_hash",
        )

    ledger.publish_sector_activation(receipt(recipe.sector_map_hash))
    pointer = ledger._pointer_path("sector-maps", recipe.sector_revision)
    before = pointer.read_bytes()
    staged_glob = f".{pointer.name}.*.tmp"

    # A violation on the pointer's replace that clears after two refusals
    # (the receipt and marker published before it are content the store
    # already accepts once).
    real_replace = os.replace
    refusals = {"left": 2, "pointer_calls": 0}

    def flaky_replace(source, destination) -> None:
        if str(destination) != str(pointer):
            real_replace(source, destination)
            return
        refusals["pointer_calls"] += 1
        if refusals["left"]:
            refusals["left"] -= 1
            raise PermissionError(32, "The process cannot access the file", str(destination))
        real_replace(source, destination)

    monkeypatch.setattr(persistence.os, "replace", flaky_replace)
    ledger.publish_sector_activation(receipt(_hash("second-map")))
    assert refusals["pointer_calls"] == 3
    assert json.loads(pointer.read_text(encoding="utf-8")) == {
        "sector_map_hash": _hash("second-map")
    }
    assert list(pointer.parent.glob(staged_glob)) == []
    second = pointer.read_bytes()

    # A violation that lasts: typed refusal, previous pointer intact, no staged file.
    def blocked_replace(source, destination) -> None:
        if str(destination) != str(pointer):
            real_replace(source, destination)
            return
        refusals["pointer_calls"] += 1
        raise PermissionError(32, "The process cannot access the file", str(destination))

    refusals["pointer_calls"] = 0
    monkeypatch.setattr(persistence.os, "replace", blocked_replace)
    with pytest.raises(WorkspaceConflictError) as blocked:
        ledger.publish_sector_activation(receipt(_hash("third-map")))
    assert blocked.value.failure.code == "catalog.replace_blocked"
    assert isinstance(blocked.value.__cause__, PermissionError)
    assert refusals["pointer_calls"] == len(persistence.DURABLE_REPLACE_DELAYS) + 1
    assert pointer.read_bytes() == second
    assert list(pointer.parent.glob(staged_glob)) == []

    # A readback the ledger's reader does not accept is refused, not retried.
    refusals["pointer_calls"] = 0

    def counting_replace(source, destination) -> None:
        if str(destination) == str(pointer):
            refusals["pointer_calls"] += 1
        real_replace(source, destination)

    monkeypatch.setattr(persistence.os, "replace", counting_replace)
    monkeypatch.setattr(
        type(ledger), "_read_pointer", staticmethod(lambda _path: {"sector_map_hash": "0" * 64})
    )
    with pytest.raises(ValueError, match="readback"):
        ledger.publish_sector_activation(receipt(_hash("third-map")))
    assert refusals["pointer_calls"] == 1
    monkeypatch.undo()
    monkeypatch.setattr(persistence.time, "sleep", lambda _seconds: None)

    # The next legitimate attempt converges: the receipt and marker the store
    # already holds are not rewritten, only the pointer moves.
    receipts_dir = artifact_store._path("sector-activation-receipts", _hash("x"), ".json").parent
    receipt_files = sorted(p.stat().st_mtime_ns for p in receipts_dir.glob("*.json"))
    ledger.publish_sector_activation(receipt(_hash("third-map")))
    assert json.loads(pointer.read_text(encoding="utf-8")) == {
        "sector_map_hash": _hash("third-map")
    }
    assert sorted(p.stat().st_mtime_ns for p in receipts_dir.glob("*.json")) == receipt_files
    assert list(pointer.parent.glob(staged_glob)) == []
    assert before != second


@pytest.mark.parametrize("kind", ["json", "parquet"])
def test_closure_artifact_replace_failure_cleans_staging_and_retry_reuses_bytes(
    tmp_path, monkeypatch, kind
):
    from alphalattice.kernel.shared_kernel import persistence
    from alphalattice.kernel.shared_kernel.domain.errors import WorkspaceConflictError

    store = PanelClosureArtifactStore(ArtifactResolver(root=tmp_path))
    content_hash = _hash("publication")
    target = store._path("test", content_hash, "." + kind)

    def publish():
        if kind == "json":
            return store.publish_json(category="test", content_hash=content_hash, payload={"a": 1})
        return store.publish_parquet(
            category="test", content_hash=content_hash, kind="test", table=pa.table({"a": [1]})
        )

    with monkeypatch.context() as fault:

        def blocked(*_args):
            raise PermissionError("fixture sharing violation")

        fault.setattr(persistence.os, "replace", blocked)
        fault.setattr(persistence.time, "sleep", lambda _: None)
        with pytest.raises(WorkspaceConflictError):
            publish()
    assert not target.exists()
    assert list(target.parent.iterdir()) == []
    reference = publish()
    before = target.read_bytes(), target.stat().st_mtime_ns
    monkeypatch.setattr(persistence.os, "replace", lambda *_: pytest.fail("rewrote artifact"))
    assert publish() == reference
    assert (target.read_bytes(), target.stat().st_mtime_ns) == before


def test_shared_replace_preserves_the_default_catalog_bound(tmp_path, monkeypatch):
    from alphalattice.kernel.shared_kernel import persistence
    from alphalattice.kernel.shared_kernel.domain.errors import WorkspaceConflictError

    waits = []

    def blocked(*_args):
        raise PermissionError("fixture")

    monkeypatch.setattr(persistence.os, "replace", blocked)
    monkeypatch.setattr(persistence.time, "sleep", waits.append)
    with pytest.raises(WorkspaceConflictError) as failure:
        persistence.replace_with_retry(tmp_path / "staged", tmp_path / "catalog")
    assert waits == [0.01, 0.025, 0.05, 0.1]
    assert failure.value.failure.code == "catalog.replace_blocked"
