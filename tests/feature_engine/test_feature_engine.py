"""Offline Feature Foundation proof: desktop catalog store and sector panel."""

from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from alphalattice.control.observation_runtime.telemetry.progress import WorkProgressUpdate
from alphalattice.control.task_control.child import ChildCalls, ChildRefused
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog
from alphalattice.foundation.feature_engine.catalog.layer import (
    FeatureCatalogLayer,
    restricted_catalog,
)
from alphalattice.foundation.feature_engine.contracts import (
    FeatureBuildRequest,
    FeatureBuildStatus,
    FeatureInvalidation,
)
from alphalattice.foundation.feature_engine.panels.artifacts import (
    PanelArtifactCompositionOwner,
)
from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
    PanelClosureArtifactStore,
)
from alphalattice.foundation.feature_engine.panels.history import (
    derive_feature_panel_history_eligibility,
)
from alphalattice.foundation.feature_engine.producers.base_materializer import (
    BaseFeatureMaterializer,
)
from alphalattice.foundation.feature_engine.producers.cross_section import (
    SectorNeutralPanelMaterializer,
)
from alphalattice.foundation.feature_engine.producers.reference_data import (
    MarketReference,
    MarketReferenceMaintainer,
)
from alphalattice.foundation.feature_engine.publication.current_storage import (
    FeatureRowCatalogAuthorityError,
    assert_rows_carry_installed_catalog,
    ensure_feature_current_schema,
    layered_row_hash,
)
from alphalattice.foundation.feature_engine.publication.sector_map_activation import (
    SectorRevisionMapActivationCoordinator,
)
from alphalattice.foundation.feature_engine.runtime.service import (
    FeatureFoundationService,
)
from alphalattice.foundation.feature_engine.storage.repositories import (
    FeatureStateRepository,
    PanelStateRepository,
)
from alphalattice.foundation.market_data_ops.publication.projection import (
    action_set_hash,
)
from alphalattice.foundation.market_data_ops.sources.contracts import (
    CorporateActionEvent,
)
from alphalattice.foundation.market_data_ops.sources.sanitization import (
    sanitize_payload,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import (
    MarketDataRepository,
)
from alphalattice.kernel.shared_kernel.domain.errors import WorkspaceConflictError
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from tests.feature_engine.panel_workspace import (
    NOW,
    SESSIONS,
    FixtureProvider,
    build_and_publish,
    fixture_manifest,
    fixture_session_authority,
    insert_fixture_bars,
    new_session_invalidations,
    open_closure,
    panel_workspace,
    seed_feature_input_fixture,
)

CASE_ROOT = Path(__file__).resolve().parent
PLAYPEN_ROOT = CASE_ROOT.parents[1]


def test_current_feature_storage_preserves_exact_catalog_axis() -> None:
    connection = duckdb.connect(":memory:")
    try:
        factors = ("factor_z", "factor_a", "factor_m")
        ensure_feature_current_schema(
            connection,
            catalog_hash="a" * 64,
            factor_ids=factors,
            activated_at=datetime(2026, 8, 14, tzinfo=UTC),
        )
        columns = tuple(
            str(row[1])
            for row in connection.execute("PRAGMA table_info('feature_daily_current')").fetchall()
        )
        binding = connection.execute(
            "SELECT factor_count, factor_axis_hash FROM feature_catalog_current"
        ).fetchone()
    finally:
        connection.close()

    # The factor columns follow the row's own identity columns in exact
    # catalog order; columns added after the table existed (the source
    # verification receipt) come after them.
    first_factor = columns.index("updated_at") + 1
    assert columns[first_factor : first_factor + len(factors)] == factors
    assert binding == (
        len(factors),
        canonical_hash({"kind": "FeatureFactorAxis", "factor_ids": factors}),
    )


def test_a_build_after_an_added_column_computes_it_and_carries_every_held_value(tmp_path) -> None:
    """requirement (V92): a workspace whose store holds rows of the shipped catalog, then installs
    one that only adds a formula factor, opens the added column's closure beside the held Panel,
    and its build computes the column over the whole history while every held value stays as it
    was; the column holds what its kernel computes. The Panel merges each held partition with the
    added column: every held cell keeps the batch that measured it, the published origins name
    the earlier catalog for those batches, and the artifact-only rematerializer reproduces every
    chunk byte-exact from the closure."""

    from dataclasses import replace

    from alphalattice.foundation.feature_engine.contracts import FeatureInvalidation
    from alphalattice.foundation.feature_engine.inputs.closure_source import (
        FeatureClosureSourceRepository,
    )
    from alphalattice.foundation.feature_engine.panels.artifacts import (
        PanelArtifactCompositionOwner,
    )
    from alphalattice.foundation.feature_engine.panels.closure import PanelClosurePublisher
    from alphalattice.foundation.feature_engine.panels.closure_source import (
        PanelClosureSourceRepository,
    )
    from alphalattice.foundation.feature_engine.panels.feature_closure_coordinator import (
        FeatureLayerClosures,
    )
    from alphalattice.foundation.feature_engine.panels.logical_identity import (
        PanelLogicalArtifactStore,
        PanelLogicalIdentityPublisher,
    )
    from alphalattice.foundation.feature_engine.panels.recovery_binding import (
        PanelRecoveryBindingPublisher,
    )
    from alphalattice.foundation.feature_engine.panels.rematerialization import (
        ArtifactOnlyPanelRematerializer,
    )
    from alphalattice.foundation.feature_engine.producers.factors.catalog import (
        default_extension_kernel_registry,
    )
    from alphalattice.foundation.feature_engine.producers.factors.formula import (
        formula_specification,
    )
    from alphalattice.foundation.feature_engine.publication.snapshots import (
        FeaturePanelSnapshotPublisher,
    )
    from alphalattice.foundation.feature_engine.runtime.closure_genesis import (
        FeatureClosureGenesisService,
    )
    from alphalattice.kernel.quant.factor_contracts import FactorFamily, FactorSpec, FactorTrack

    manifest = fixture_manifest()
    as_of = SESSIONS[300]
    held = panel_workspace(tmp_path / "held", manifest, end=as_of, spy="a" * 64)
    _outcome, manifest_one = build_and_publish(
        held, manifest, spy="a" * 64, as_of=as_of, observed_at=NOW, refresh_sector=True
    )
    shipped = FeatureCatalog.load()
    listings = tuple(item.listing_id for item in manifest.listings)
    before = held.feature_state.feature_rows(
        listing_ids=listings,
        catalog_hash=shipped.binding.catalog_hash,
        start=SESSIONS[0],
        end=as_of,
        include_lineage=False,
    )
    spec = formula_specification(
        FactorSpec(
            factor_id="formula_reversal_5",
            family=FactorFamily.PRICE_LEVEL_TREND,
            formula_ref="factor.formula",
            formula="close / lag(close, 5) - 1",
            window_sessions=1,
            lag_sessions=0,
            return_convention="as declared",
            required_fields=("close_split_adjusted",),
            literature_sources=("research-local note",),
            minimum_observations=1,
            absolute_tolerance=1e-10,
            relative_tolerance=1e-10,
            track=FactorTrack.MODEL,
        )
    )
    payload = shipped.to_payload()
    payload["factors"] = sorted(
        [*payload["factors"], spec.model_dump(mode="json")],
        key=lambda value: str(value["factor_id"]),
    )
    extended = FeatureCatalog.from_payload(payload)
    layer = FeatureCatalogLayer.over(extended)
    (column,) = layer.columns
    assert layer.base.binding.catalog_hash == shipped.binding.catalog_hash
    source = FeatureClosureSourceRepository(held.market_data.database.path)
    genesis = FeatureClosureGenesisService(
        panel_state=held.panel_state, source=source, ledger=held.ledger
    ).open_genesis(manifest=manifest, catalog=column)
    assert genesis.disposition == "GENESIS_READY"
    feature_state = FeatureStateRepository(
        held.market_data.database,
        market_data=held.market_data,
        installed_catalog=extended,
    )
    service = FeatureFoundationService(
        market_data=held.market_data,
        feature_state=feature_state,
        panel_state=held.panel_state,
        manifest=manifest,
        provider=held.provider,
        mutation_gate=held.gate,
        panel_artifacts=PanelArtifactCompositionOwner(held.resolver),
        feature_persistence=FeatureLayerClosures(
            store=feature_state, source=source, ledger=held.ledger, layer=layer
        ),
        sector_activation=SectorRevisionMapActivationCoordinator(
            store=feature_state, mutation_gate=held.gate, ledger=held.ledger
        ),
        session_authority_resolver=fixture_session_authority,
        installed_catalog=extended,
    )
    publisher = FeaturePanelSnapshotPublisher(
        feature_state=feature_state,
        panel_state=held.panel_state,
        resolver=held.resolver,
        mutation_gate=held.gate,
        recovery_binding=PanelRecoveryBindingPublisher(ledger=held.ledger, resolver=held.resolver),
        logical_identity=PanelLogicalIdentityPublisher(
            resolver=held.resolver,
            store=PanelLogicalArtifactStore(PanelClosureArtifactStore(held.resolver)),
        ),
        catalog=extended,
    )
    _outcome, manifest_two = build_and_publish(
        replace(held, feature_state=feature_state, service=service, publisher=publisher),
        manifest,
        spy="a" * 64,
        as_of=as_of,
        observed_at=NOW + timedelta(hours=1),
        invalidations=(
            FeatureInvalidation("catalog_binding_change", earliest_session=SESSIONS[0]),
        ),
        refresh_sector=False,
    )
    after = {
        (row["listing_id"], row["session_date"]): row
        for row in feature_state.feature_rows(
            listing_ids=listings,
            catalog_hash=extended.binding.catalog_hash,
            start=SESSIONS[0],
            end=as_of,
            include_lineage=False,
        )
    }
    assert len(after) == len(before)
    for row in before:
        carried = after[(row["listing_id"], row["session_date"])]
        for factor_id in shipped.factor_ids:
            old, new = row[factor_id], carried[factor_id]
            assert (old is None and new is None) or old == new or (old != old and new != new), (
                factor_id,
                row["listing_id"],
            )
    registry = default_extension_kernel_registry()
    for listing_id in listings[:3]:
        frame, _raw, _actions = feature_state.projected_feature_frame(
            manifest, listing_id=listing_id, through=as_of
        )
        frame = pd.DataFrame(frame).assign(listing_id=listing_id)
        expected = registry.compute(frame, spec).to_numpy(dtype=float)
        stored = np.array(
            [
                np.nan
                if after[(listing_id, day)]["formula_reversal_5"] is None
                else after[(listing_id, day)]["formula_reversal_5"]
                for day in frame["session_date"]
            ],
            dtype=float,
        )
        np.testing.assert_allclose(stored, expected, rtol=0.0, atol=0.0)

    # The Panel computed the added column alone: every held cell, recorded under the new
    # catalog, keeps the binding of the build that measured it.
    binding_one = str(manifest_one["panel_binding_hash"])
    binding_two = str(manifest_two["panel_binding_hash"])
    connection = duckdb.connect(str(held.market_data.path), read_only=True)
    try:
        stamped = {
            (str(factor_id), str(binding)): int(count)
            for factor_id, binding, count in connection.execute(
                "SELECT factor_id, panel_binding_hash, count(*) "
                "FROM panel_cross_section_availability WHERE catalog_hash = ? GROUP BY 1, 2",
                [extended.binding.catalog_hash],
            ).fetchall()
        }
    finally:
        connection.close()
    assert {binding for (factor_id, binding) in stamped if factor_id in shipped.factor_ids} == {
        binding_one
    }
    assert {binding for (factor_id, binding) in stamped if factor_id == "formula_reversal_5"} == {
        binding_two
    }
    origins = manifest_two["safe_summary"]["lineage"]["partition_origins"]
    assert origins[binding_one]["catalog_hash"] == shipped.binding.catalog_hash
    assert "catalog_hash" not in origins[binding_two]

    # Its closure replays each batch under the binding that ran it, the carried ones under the
    # earlier catalog's, and reproduces every chunk byte-exact.
    capture = PanelClosurePublisher(
        resolver=held.resolver,
        source=PanelClosureSourceRepository(
            database_path=held.market_data.path, resolver=held.resolver
        ),
        store=PanelClosureArtifactStore(held.resolver),
    )
    closure = capture.publish(
        tuple(
            item
            for item in capture.inventory()
            if item.snapshot_hash == str(manifest_two["snapshot_hash"])
        )
    )
    recipe = closure.recipes[str(manifest_two["snapshot_hash"])]
    assert {
        (item.panel_binding_hash, item.catalog_hash) for item in recipe.partition_origins or ()
    } == {(binding_one, shipped.binding.catalog_hash)}
    result = ArtifactOnlyPanelRematerializer(
        resolver=held.resolver, store=PanelClosureArtifactStore(held.resolver)
    ).rematerialize(recipe.recipe_hash)
    assert result.logical_parity and result.physical_parity
    assert [item.chunk_hash for item in result.chunks] == [
        chunk["chunk_hash"] for chunk in manifest_two["chunks"]
    ]


def test_an_interrupted_first_load_is_discarded_and_its_column_built_again(
    tmp_path, monkeypatch
) -> None:
    """requirement (V92): an added column's first build writes its rows with no per-key
    transition and moves its closure head to their digest once. A build interrupted before the
    head moved leaves the first load pending: the catalog reports recovery pending and refuses
    work, and the next build discards the column's rows, loads it again and completes it, the
    rows the same as the interrupted attempt wrote."""

    from alphalattice.foundation.feature_engine.inputs.closure_source import (
        FeatureClosureSourceRepository,
    )
    from alphalattice.foundation.feature_engine.panels.feature_closure_coordinator import (
        FeatureLayerClosures,
    )
    from alphalattice.foundation.feature_engine.producers.factors.formula import (
        formula_specification,
    )
    from alphalattice.foundation.feature_engine.runtime.closure_genesis import (
        FeatureClosureGenesisService,
    )
    from alphalattice.kernel.quant.factor_contracts import FactorFamily, FactorSpec, FactorTrack

    manifest = fixture_manifest()
    as_of = SESSIONS[300]
    held = panel_workspace(tmp_path / "held", manifest, end=as_of, spy="a" * 64)
    build_and_publish(
        held, manifest, spy="a" * 64, as_of=as_of, observed_at=NOW, refresh_sector=True
    )
    shipped = FeatureCatalog.load()
    spec = formula_specification(
        FactorSpec(
            factor_id="formula_reversal_5",
            family=FactorFamily.PRICE_LEVEL_TREND,
            formula_ref="factor.formula",
            formula="close / lag(close, 5) - 1",
            window_sessions=1,
            lag_sessions=0,
            return_convention="as declared",
            required_fields=("close_split_adjusted",),
            literature_sources=("research-local note",),
            minimum_observations=1,
            absolute_tolerance=1e-10,
            relative_tolerance=1e-10,
            track=FactorTrack.MODEL,
        )
    )
    payload = shipped.to_payload()
    payload["factors"] = sorted(
        [*payload["factors"], spec.model_dump(mode="json")],
        key=lambda value: str(value["factor_id"]),
    )
    extended = FeatureCatalog.from_payload(payload)
    layer = FeatureCatalogLayer.over(extended)
    (column,) = layer.columns
    column_hash = column.binding.catalog_hash
    source = FeatureClosureSourceRepository(held.market_data.database.path)
    FeatureClosureGenesisService(
        panel_state=held.panel_state, source=source, ledger=held.ledger
    ).open_genesis(manifest=manifest, catalog=column)
    genesis = held.ledger.require_head(column_hash)
    feature_state = FeatureStateRepository(
        held.market_data.database,
        market_data=held.market_data,
        installed_catalog=extended,
    )
    persistence = FeatureLayerClosures(
        store=feature_state, source=source, ledger=held.ledger, layer=layer
    )
    service = FeatureFoundationService(
        market_data=held.market_data,
        feature_state=feature_state,
        panel_state=held.panel_state,
        manifest=manifest,
        provider=held.provider,
        mutation_gate=held.gate,
        panel_artifacts=PanelArtifactCompositionOwner(held.resolver),
        feature_persistence=persistence,
        sector_activation=SectorRevisionMapActivationCoordinator(
            store=feature_state, mutation_gate=held.gate, ledger=held.ledger
        ),
        session_authority_resolver=fixture_session_authority,
        installed_catalog=extended,
    )
    request = FeatureBuildRequest.create(
        manifest_revision=manifest.revision_sha256,
        catalog=extended.binding,
        spy_revision="a" * 64,
        history_start=SESSIONS[0],
        as_of_session=as_of,
        invalidations=(
            FeatureInvalidation("catalog_binding_change", earliest_session=SESSIONS[0]),
        ),
    )
    listings = tuple(item.listing_id for item in manifest.listings)
    catalog_hash = extended.binding.catalog_hash

    def column_rows() -> dict[tuple[str, date], str]:
        rows = feature_state.feature_rows(
            listing_ids=listings,
            catalog_hash=column_hash,
            start=SESSIONS[0],
            end=as_of,
        )
        return {(row["listing_id"], row["session_date"]): str(row["row_hash"]) for row in rows}

    def interrupted(self, catalog_hashes, *, connection):
        raise RuntimeError("simulated interruption before the first load completes")

    with monkeypatch.context() as patch:
        patch.setattr(FeatureLayerClosures, "complete_first_loads", interrupted)
        outcome = service.build(request, observed_at=NOW + timedelta(hours=1), refresh_sector=False)
    assert outcome.status is FeatureBuildStatus.BLOCKED
    assert held.ledger.require_head(column_hash) == genesis
    assert held.ledger.pending_first_load(column_hash) is not None
    written = column_rows()
    assert written
    assert (
        persistence.closure_disposition(catalog_hash, expected_listing_ids=listings)
        == "TRANSITION_RECOVERY_PENDING"
    )
    with pytest.raises(ValueError, match=r"feature_closure\.recovery_required"):
        persistence.assert_ready(catalog_hash)

    discarded: list[int] = []
    discard = FeatureStateRepository.discard_part_rows

    def counted(self, part_hash, *, _connection):
        discarded.append(discard(self, part_hash, _connection=_connection))
        return discarded[-1]

    monkeypatch.setattr(FeatureStateRepository, "discard_part_rows", counted)
    outcome = service.build(request, observed_at=NOW + timedelta(hours=2), refresh_sector=False)
    assert outcome.status is FeatureBuildStatus.COMPLETED, outcome
    assert discarded == [len(written)]
    assert held.ledger.pending_first_load(column_hash) is None
    head = held.ledger.require_head(column_hash)
    assert head.transition_cursor == 1 and head.predecessor_head_hash == genesis.head_hash
    assert head.feature_row_hash_digest == source.feature_row_hash_digest(catalog_hash=column_hash)
    assert column_rows() == written
    assert (
        persistence.closure_disposition(catalog_hash, expected_listing_ids=listings)
        == "MEMBERSHIP_COMPLETE"
    )


def test_a_carried_availability_cell_is_recorded_as_an_upsert_records_it(tmp_path) -> None:
    """requirement (V92): a Panel cell carried to a catalog that only adds columns is recorded
    under that catalog exactly as an upsert under it records the same cell, with its values,
    binding, receipt and availability hash; a cell the catalog already holds is kept."""

    from alphalattice.foundation.feature_engine.panels.availability import (
        PanelAvailabilityRepository,
    )

    sessions = {"2024-01-02": "c" * 64, "2024-01-03": "d" * 64}
    cells = [
        {
            "session_date": session,
            "factor_id": factor_id,
            "universe_size": 12,
            "computed_count": 11 - index,
            "coverage": (11 - index) / 12,
            "sector_counts": {"Energy": 5, "Technology": 6 - index},
            "winsor_lower": None if index else -2.5,
            "winsor_upper": None if index else 2.5,
            "residual_median": 0.125 * index,
            "residual_mad": 1.0 / 3.0,
            "status": "available",
            "reason": None,
            "panel_binding_hash": "e" * 64,
            "small_sector_warning": bool(index),
            "small_sector_names": ("Technology",) if index else (),
        }
        for session in sessions
        for index, factor_id in enumerate(("mom_21", "rev_5"))
    ]
    held = {**cells[0], "coverage": 0.5, "panel_binding_hash": "f" * 64}

    def database(name: str) -> duckdb.DuckDBPyConnection:
        market = MarketDataRepository(tmp_path / name)
        market.bootstrap(fixture_manifest())
        return duckdb.connect(str(market.path))

    repository = PanelAvailabilityRepository()
    carried, direct = database("carried"), database("direct")
    try:
        common = {"cross_section_by_session": sessions, "policy_hash": "p" * 64, "observed_at": NOW}
        repository.upsert(
            carried,
            catalog_hash="a" * 64,
            availability=cells,
            materialization_receipt_hash="r" * 64,
            **common,
        )
        repository.upsert(
            carried,
            catalog_hash="b" * 64,
            availability=[held],
            materialization_receipt_hash="s" * 64,
            **common,
        )
        assert (
            repository.carry(
                carried,
                from_catalog_hash="a" * 64,
                to_catalog_hash="b" * 64,
                factor_ids=("mom_21", "rev_5"),
                **common,
            )
            == len(cells) - 1
        )
        repository.upsert(
            direct,
            catalog_hash="b" * 64,
            availability=cells[1:],
            materialization_receipt_hash="r" * 64,
            **common,
        )
        repository.upsert(
            direct,
            catalog_hash="b" * 64,
            availability=[held],
            materialization_receipt_hash="s" * 64,
            **common,
        )

        def recorded(connection: duckdb.DuckDBPyConnection) -> list[tuple[object, ...]]:
            return connection.execute(
                "SELECT * FROM panel_cross_section_availability WHERE catalog_hash = ? "
                "ORDER BY ALL",
                ["b" * 64],
            ).fetchall()

        assert recorded(carried) == recorded(direct)
        assert len(recorded(carried)) == len(cells)
    finally:
        carried.close()
        direct.close()


def test_a_catalog_that_gains_a_factor_gains_its_column_and_the_earlier_rows_stay_its_own() -> None:
    """regression (EX, found by the fork's V92 work, 2026-10-01): after a person activates a
    formula factor, the store the workspace already holds must take the new catalog; it refused
    with a binder error, since its table had no column for the factor, so the next data update
    failed. A row the earlier catalog computed holds nothing in the new column and keeps naming
    its own catalog."""

    connection = duckdb.connect(":memory:")
    try:
        ensure_feature_current_schema(
            connection,
            catalog_hash="a" * 64,
            factor_ids=("factor_a", "factor_m"),
            activated_at=datetime(2026, 8, 14, tzinfo=UTC),
        )
        connection.execute(
            "INSERT INTO feature_input_cutoff_set VALUES ('c', '{}', 2)",
        )
        connection.execute(
            "INSERT INTO feature_daily_current (listing_id, session_date, catalog_hash, "
            "raw_input_hash, action_set_hash, market_reference_revision, cutoff_set_hash, "
            "row_hash, updated_at, factor_a, factor_m) VALUES ('L', DATE '2026-08-13', ?, 'r', "
            "'x', 'm', 'c', 'h', TIMESTAMP '2026-08-14 00:00:00', 1.0, 2.0)",
            ["a" * 64],
        )
        ensure_feature_current_schema(
            connection,
            catalog_hash="b" * 64,
            factor_ids=("factor_a", "factor_m", "formula_reversal_5"),
            activated_at=datetime(2026, 8, 15, tzinfo=UTC),
        )
        row = connection.execute(
            "SELECT catalog_hash, factor_a, factor_m, formula_reversal_5 FROM feature_daily_runtime"
        ).fetchone()
    finally:
        connection.close()
    assert row == ("a" * 64, 1.0, 2.0, None)


def _formula_factor(factor_id: str, formula: str) -> dict[str, object]:
    from alphalattice.foundation.feature_engine.producers.factors.formula import (
        formula_specification,
    )
    from alphalattice.kernel.quant.factor_contracts import FactorFamily, FactorSpec, FactorTrack

    spec = formula_specification(
        FactorSpec(
            factor_id=factor_id,
            family=FactorFamily.PRICE_LEVEL_TREND,
            formula_ref="factor.formula",
            formula=formula,
            window_sessions=1,
            lag_sessions=0,
            return_convention="as declared",
            required_fields=("close_split_adjusted",),
            literature_sources=("research-local note",),
            minimum_observations=1,
            absolute_tolerance=1e-10,
            relative_tolerance=1e-10,
            track=FactorTrack.MODEL,
        )
    )
    return spec.model_dump(mode="json")


def _catalog_with(catalog: FeatureCatalog, *factors: dict[str, object]) -> FeatureCatalog:
    payload = catalog.to_payload()
    payload["factors"] = sorted(
        [*payload["factors"], *factors], key=lambda value: str(value["factor_id"])
    )
    return FeatureCatalog.from_payload(payload)


def test_an_activated_catalog_is_layered_on_the_shipped_one() -> None:
    """requirement (V92): a catalog that only adds factors to the shipped one is the shipped
    catalog and one column catalog per added factor, each binding only what decides its own
    values, so a second activation leaves the first column's identity where it was; a catalog
    that changes anything else is its own base."""

    shipped = FeatureCatalog.load()
    reversal = _formula_factor("formula_reversal_5", "close / lag(close, 5) - 1")
    extended = _catalog_with(shipped, reversal)
    layer = FeatureCatalogLayer.over(extended)
    assert layer.layered
    assert layer.base.binding.catalog_hash == shipped.binding.catalog_hash
    assert [column.factor_ids for column in layer.columns] == [("formula_reversal_5",)]
    assert layer.part_hashes == (
        shipped.binding.catalog_hash,
        restricted_catalog(extended, ("formula_reversal_5",)).binding.catalog_hash,
    )
    assert layer.factor_parts()["formula_reversal_5"] == layer.part_hashes[1]
    assert layer.factor_parts()[shipped.factor_ids[0]] == layer.part_hashes[0]

    second = _catalog_with(
        extended, _formula_factor("formula_drift_10", "close / lag(close, 10) - 1")
    )
    grown = FeatureCatalogLayer.over(second)
    assert grown.base.binding.catalog_hash == shipped.binding.catalog_hash
    assert {column.factor_ids[0]: column.binding.catalog_hash for column in grown.columns} == {
        "formula_drift_10": restricted_catalog(second, ("formula_drift_10",)).binding.catalog_hash,
        "formula_reversal_5": layer.part_hashes[1],
    }

    assert not FeatureCatalogLayer.over(shipped).layered
    payload = extended.to_payload()
    payload["factors"] = [item for item in payload["factors"] if item["factor_id"] != "max_21"]
    narrowed = FeatureCatalog.from_payload(payload)
    assert FeatureCatalogLayer.over(narrowed).parts == (narrowed,)


def test_a_layered_catalog_reads_its_rows_composed_from_its_parts() -> None:
    """requirement (V92): rows are keyed by the catalog that computed them too -- a table an
    earlier release keyed by listing and session alone is re-keyed, every row kept -- and a
    layered catalog owns no rows: its row is its parts' where every part holds one, each value
    and cutoff its factor's part's and its identity ``layered_row_hash`` of theirs. A row an
    earlier build stored under the layered catalog itself is not presented beside it and
    refuses publication until retired."""

    base, column, layered = "a" * 64, "c" * 64, "b" * 64
    connection = duckdb.connect(":memory:")
    try:
        connection.execute(
            "CREATE TABLE feature_daily_current (listing_id VARCHAR NOT NULL, "
            "session_date DATE NOT NULL, catalog_hash VARCHAR NOT NULL, "
            "raw_input_hash VARCHAR NOT NULL, action_set_hash VARCHAR NOT NULL, "
            "market_reference_revision VARCHAR NOT NULL, cutoff_set_hash VARCHAR NOT NULL, "
            "row_hash VARCHAR NOT NULL, updated_at TIMESTAMP NOT NULL, factor_a DOUBLE, "
            "factor_m DOUBLE, PRIMARY KEY (listing_id, session_date))"
        )
        connection.execute(
            "INSERT INTO feature_daily_current VALUES ('L', DATE '2026-08-12', ?, 'r', 'x', "
            "'m', 'cb', 'h0', TIMESTAMP '2026-08-13 00:00:00', 0.5, 0.25)",
            [base],
        )
        ensure_feature_current_schema(
            connection,
            catalog_hash=base,
            factor_ids=("factor_a", "factor_m"),
            activated_at=datetime(2026, 8, 14, tzinfo=UTC),
        )
        assert connection.execute(
            "SELECT constraint_column_names FROM duckdb_constraints() "
            "WHERE table_name = 'feature_daily_current' AND constraint_type = 'PRIMARY KEY'"
        ).fetchone() == (["listing_id", "session_date", "catalog_hash"],)
        base_cutoffs = json.dumps(
            {"factor_a": "2026-08-13", "factor_m": None}, sort_keys=True, separators=(",", ":")
        )
        column_cutoffs = json.dumps({"formula_x": "2026-08-12"}, separators=(",", ":"))
        connection.execute(
            "INSERT INTO feature_input_cutoff_set VALUES ('cb', ?, 2), ('cx', ?, 1)",
            [base_cutoffs, column_cutoffs],
        )
        connection.execute(
            "INSERT INTO feature_daily_current (listing_id, session_date, catalog_hash, "
            "raw_input_hash, action_set_hash, market_reference_revision, cutoff_set_hash, "
            "row_hash, updated_at, factor_a, factor_m) VALUES ('L', DATE '2026-08-13', ?, 'r', "
            "'x', 'm', 'cb', 'hb', TIMESTAMP '2026-08-14 00:00:00', 1.0, 2.0)",
            [base],
        )
        ensure_feature_current_schema(
            connection,
            catalog_hash=layered,
            factor_ids=("factor_a", "factor_m", "formula_x"),
            activated_at=datetime(2026, 8, 15, tzinfo=UTC),
            parts=((base, ("factor_a", "factor_m")), (column, ("formula_x",))),
        )
        connection.execute(
            "INSERT INTO feature_daily_current (listing_id, session_date, catalog_hash, "
            "raw_input_hash, action_set_hash, market_reference_revision, cutoff_set_hash, "
            "row_hash, updated_at, formula_x) VALUES ('L', DATE '2026-08-13', ?, 'r2', 'x', "
            "'m', 'cx', 'hx', TIMESTAMP '2026-08-15 00:00:00', 3.0)",
            [column],
        )
        connection.execute(
            "INSERT INTO feature_daily_current (listing_id, session_date, catalog_hash, "
            "raw_input_hash, action_set_hash, market_reference_revision, cutoff_set_hash, "
            "row_hash, updated_at, factor_a, factor_m, formula_x) VALUES ('L', "
            "DATE '2026-08-13', ?, 'r', 'x', 'm', 'cb', 'whole', "
            "TIMESTAMP '2026-08-12 00:00:00', 9.0, 9.0, 9.0)",
            [layered],
        )
        composed = connection.execute(
            "SELECT session_date, factor_a, factor_m, formula_x, input_cutoffs_json, row_hash, "
            "updated_at, raw_input_hash, source_verification_receipt_hash "
            "FROM feature_daily_runtime WHERE catalog_hash = ?",
            [layered],
        ).fetchall()
        assert composed == [
            (
                date(2026, 8, 13),
                1.0,
                2.0,
                3.0,
                json.dumps(
                    {"factor_a": "2026-08-13", "factor_m": None, "formula_x": "2026-08-12"},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                layered_row_hash(layered, ("hb", "hx")),
                datetime(2026, 8, 15),
                "r",
                None,
            )
        ]
        assert connection.execute(
            "SELECT catalog_hash, count(*) FROM feature_daily_runtime "
            "WHERE catalog_hash <> ? GROUP BY 1 ORDER BY 1",
            [layered],
        ).fetchall() == [(base, 2), (column, 1)]
        with pytest.raises(FeatureRowCatalogAuthorityError, match="identity_mismatch"):
            assert_rows_carry_installed_catalog(
                connection, catalog_hash=layered, parts=(base, column)
            )
        connection.execute("DELETE FROM feature_daily_current WHERE catalog_hash = ?", [layered])
        assert_rows_carry_installed_catalog(connection, catalog_hash=layered, parts=(base, column))
    finally:
        connection.close()


def test_current_feature_storage_rejects_duplicate_axis() -> None:
    connection = duckdb.connect(":memory:")
    try:
        with pytest.raises(
            ValueError,
            match="current Feature factor axis must be ordered and unique",
        ):
            ensure_feature_current_schema(
                connection,
                catalog_hash="a" * 64,
                factor_ids=("factor_a", "factor_a"),
                activated_at=datetime(2026, 8, 14, tzinfo=UTC),
            )
    finally:
        connection.close()


def test_panel_history_eligibility_exposes_warmup_without_dropping_partial_history() -> None:
    factor_ids = ("factor-a", "factor-b")
    availability = [
        {
            "session_date": "2024-01-02",
            "factor_id": factor_id,
            "status": "unavailable",
            "coverage": 0.0,
        }
        for factor_id in factor_ids
    ]
    availability.extend(
        {
            "session_date": "2024-01-03",
            "factor_id": factor_id,
            "status": "available",
            "coverage": 1.0,
        }
        for factor_id in factor_ids
    )

    eligibility = derive_feature_panel_history_eligibility(
        availability,
        factor_ids=factor_ids,
    )

    assert eligibility.panel_history_start == date(2024, 1, 2)
    assert eligibility.model_eligible_history_start == date(2024, 1, 3)
    assert eligibility.warmup_session_count == 1
    assert eligibility.observed_session_count == 2
    assert eligibility.safe_summary()["eligibility_hash"] == eligibility.eligibility_hash


class FailingFixturePanelRecoveryBinding:
    def publish(self, **_kwargs) -> None:
        raise ValueError("feature_closure.panel_binding_incomplete")


class RecordingReferenceProvider(FixtureProvider):
    def __init__(self) -> None:
        super().__init__(("SPY",))
        self.action_ranges: list[tuple[date, date]] = []
        self.daily_ranges: list[tuple[date, date]] = []

    def fetch_daily(self, symbols, *, start, end):
        self.daily_ranges.append((start, end))
        return super().fetch_daily(symbols, start=start, end=end)

    def fetch_action_history(self, *, listing_id, provider_symbol, start, end):
        del listing_id, provider_symbol
        self.action_ranges.append((start, end))
        return ()


def test_spy_reference_uses_bounded_overlap_after_full_anchor(tmp_path: Path) -> None:
    manifest = fixture_manifest()
    market_data = MarketDataRepository(tmp_path / "workspace")
    feature_state = FeatureStateRepository(market_data.database, market_data=market_data)
    PanelStateRepository(market_data.database, market_data=market_data)
    provider = RecordingReferenceProvider()
    market_data.bootstrap(manifest)
    asset_provider = FixtureProvider(tuple(item.symbol for item in manifest.listings))
    payload = asset_provider.fetch_daily(
        tuple(item.symbol for item in manifest.listings),
        start=SESSIONS[0],
        end=SESSIONS[-1],
    )
    market_data.apply_validated_batch(
        manifest,
        sanitize_payload(manifest, asset_provider.name, payload, tuple(payload)),
        ingestion_id="parent-history",
        observed_at=NOW,
    )
    maintainer = MarketReferenceMaintainer(
        market_data=market_data,
        feature_state=feature_state,
        parent_manifest=manifest,
        provider=provider,
    )

    anchored = maintainer.refresh(as_of_session=SESSIONS[-2], observed_at=NOW)
    incremental = maintainer.refresh(
        as_of_session=SESSIONS[-1],
        observed_at=NOW + timedelta(minutes=1),
    )
    calls_after_incremental = (len(provider.daily_ranges), len(provider.action_ranges))
    exact = maintainer.refresh(
        as_of_session=SESSIONS[-1],
        observed_at=NOW + timedelta(days=2),
    )

    assert anchored.status == "completed"
    assert incremental.status == "completed"
    assert exact.status == "completed"
    assert (len(provider.daily_ranges), len(provider.action_ranges)) == calls_after_incremental
    assert provider.action_ranges[0][0] == SESSIONS[0]
    assert provider.action_ranges[1][0] == max(SESSIONS[0], SESSIONS[-2] - timedelta(days=45))


def test_bounded_feature_projection_ignores_actions_before_its_source_window(
    tmp_path: Path,
) -> None:
    manifest = fixture_manifest()
    provider = FixtureProvider(tuple(item.symbol for item in manifest.listings))
    market_data = MarketDataRepository(tmp_path / "workspace")
    feature_state = FeatureStateRepository(market_data.database, market_data=market_data)
    PanelStateRepository(market_data.database, market_data=market_data)
    seed_feature_input_fixture(market_data, feature_state, manifest, provider)
    listing_id = manifest.listings[0].listing_id
    action = CorporateActionEvent(
        listing_id=listing_id,
        provider=provider.name,
        effective_date=SESSIONS[10],
        action_kind="CASH_DIVIDEND",
        cash_amount=0.25,
        provenance="fixture-old-action",
    )
    connection = duckdb.connect(str(market_data.path))
    try:
        connection.execute(
            """
            INSERT INTO corporate_action_current VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                action.listing_id,
                action.provider,
                action.effective_date,
                action.action_kind,
                action.new_shares_per_old_share,
                action.cash_amount,
                action.provisional,
                action.provenance,
                "fixture-old-action-hash",
                "ACTIVE",
                NOW.replace(tzinfo=None),
            ],
        )
    finally:
        connection.close()

    rows, _raw_hash, full_action_hash = feature_state.projected_feature_frame(
        manifest,
        listing_id=listing_id,
        start=SESSIONS[100],
        through=SESSIONS[-1],
    )

    assert rows[0]["session_date"] == SESSIONS[100]
    assert all(float(row["cash_dividend"]) == 0.0 for row in rows)
    assert full_action_hash == action_set_hash((action,))


def test_vectorized_feature_block_has_bounded_local_contract() -> None:
    dates = pd.bdate_range("2018-01-02", periods=1_600)
    rng = np.random.default_rng(8)

    def frame(seed: int) -> pd.DataFrame:
        close = 100.0 * np.cumprod(
            1.0 + np.random.default_rng(seed).normal(0.0002, 0.01, len(dates))
        )
        return pd.DataFrame(
            {
                "session_date": dates,
                "open_raw": close * 0.997,
                "high_raw": close * 1.01,
                "low_raw": close * 0.99,
                "close_raw": close,
                "volume_raw": rng.integers(1_000_000, 2_000_000, len(dates)),
                "open_split_adjusted": close * 0.997,
                "high_split_adjusted": close * 1.01,
                "low_split_adjusted": close * 0.99,
                "close_split_adjusted": close,
                "provider_adjusted_close": close,
                "close_total_return_adjusted": close,
            }
        )

    asset, market = frame(2), frame(3)
    catalog = FeatureCatalog.load()
    materializer = BaseFeatureMaterializer(catalog)
    assert len(catalog.factor_ids) == len(catalog.factors)
    assert "mom_21" not in catalog.factor_ids
    assert "mom_63" not in catalog.factor_ids
    assert {"rev_21", "rev_63", "mom_126_21", "mom_252_21"}.issubset(catalog.factor_ids)
    block = materializer.materialize_listing(
        listing_id="fixture", projected_bars=asset, market_bars=market
    )
    latest = block.values.iloc[-1]
    # Formula-level independent oracles and invariance properties live in the
    # desktop-economics test module.  This case locks the block/catalog shape,
    # finite mature outputs and persisted lag cutoffs.
    for factor_id in catalog.factor_ids:
        assert math.isfinite(float(latest[factor_id]))
    assert not block.ineligibility.empty
    assert "input_cutoffs_json" in block.values
    # Per-Formula cutoffs remain explicit durable evidence, and each states the
    # source event the Formula actually consumed rather than a uniform offset.
    factor_by_id = {spec.factor_id: spec for spec in catalog.factors}
    ordered = pd.DatetimeIndex(dates)
    for position in (300, 800, len(block.values) - 1):
        row = block.values.iloc[position]
        recorded = json.loads(row["input_cutoffs_json"])
        for factor_id in ("rev_21", "amihud_21", "beta_63"):
            skip = factor_by_id[factor_id].lag_sessions
            expected = dates[position - skip].date().isoformat() if position >= skip else None
            assert recorded[factor_id] == expected
        # seasonality_12m selects a calendar month twelve back; its newest source
        # event is that month's final close, which is neither the observation
        # session nor any fixed offset from it.
        selected = ordered.to_period("M")[position] - 12
        members = [index for index, value in enumerate(ordered.to_period("M")) if value == selected]
        assert recorded["seasonality_12m"] == (
            dates[members[-1]].date().isoformat() if members else None
        )


def test_amihud_averages_the_sessions_with_volume_and_a_zero_volume_bar_never_blanks_a_name() -> (
    None
):
    """requirement (V517, Amihud 2002): the illiquidity ratio is the mean |return| per dollar
    traded over the window's sessions with positive volume, given that at least 80% of the
    window has one (17 of 21, 202 of 252). A zero-volume bar -- a real range reported with no
    volume, or a flat placeholder -- is a session without the ratio, never an infinite one that
    blanks the name for its whole window; a window without one is the plain mean, unchanged."""

    dates = pd.bdate_range("2018-01-02", periods=700)
    close = 100.0 * np.cumprod(1.0 + np.random.default_rng(5).normal(0.0002, 0.01, len(dates)))
    volume = np.random.default_rng(6).integers(1_000_000, 2_000_000, len(dates)).astype(float)
    # A real range reported with no volume, and a flat placeholder at the prior close.
    volume[400] = 0.0
    volume[450] = 0.0
    close[450] = close[449]
    # Five zero-volume sessions inside one 21-session window, and fifty-one inside a year.
    volume[[600, 602, 604, 606, 608]] = 0.0
    volume[50:101] = 0.0

    def bars(prices: np.ndarray, volumes: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "session_date": dates,
                "open_raw": prices * 0.997,
                "high_raw": prices * 1.01,
                "low_raw": prices * 0.99,
                "close_raw": prices,
                "volume_raw": volumes,
                "open_split_adjusted": prices * 0.997,
                "high_split_adjusted": prices * 1.01,
                "low_split_adjusted": prices * 0.99,
                "close_split_adjusted": prices,
                "provider_adjusted_close": prices,
                "close_total_return_adjusted": prices,
            }
        )

    market = 100.0 * np.cumprod(1.0 + np.random.default_rng(7).normal(0.0002, 0.01, len(dates)))
    block = BaseFeatureMaterializer(FeatureCatalog.load()).materialize_listing(
        listing_id="fixture",
        projected_bars=bars(close, volume),
        market_bars=bars(market, np.full(len(dates), 1_500_000.0)),
    )
    values = block.values.set_index("session_date")
    prices = pd.Series(close)
    traded = (np.log(prices / prices.shift(1)).abs() / (prices * volume)).where(volume > 0.0)
    ratio = traded.to_numpy()

    def oracle(end: int, window: int) -> float:
        observed = ratio[end - window + 1 : end + 1]
        observed = observed[np.isfinite(observed)]
        return float(observed.mean()) if observed.size >= math.ceil(0.8 * window) else math.nan

    def published(end: int, factor_id: str) -> float:
        return float(values.loc[dates[end].date(), factor_id])

    # Each zero-volume bar leaves the names' windows that hold it computed over the others.
    for end in (400, 410, 420, 450, 460, 470, 650):
        for factor_id, window in (("amihud_21", 21), ("amihud_252", 252)):
            assert published(end, factor_id) == pytest.approx(oracle(end, window), rel=1e-12)
            assert math.isfinite(published(end, factor_id)), (end, factor_id)
    # A window with no zero-volume bar is the plain mean of its sessions, as before.
    assert published(399, "amihud_21") == pytest.approx(float(np.mean(ratio[379:400])), rel=1e-12)
    # Below the minimum share the name is missing: 16 of 21, and 201 of 252.
    assert math.isnan(published(620, "amihud_21"))
    assert math.isfinite(published(625, "amihud_21"))
    assert math.isnan(published(301, "amihud_252"))
    assert math.isfinite(published(302, "amihud_252"))


def test_missing_daily_market_observation_keeps_the_other_members_usable(tmp_path: Path) -> None:
    """A source absence is not an uncomputed row or a permanent membership exit."""
    from dataclasses import replace

    from alphalattice.foundation.feature_engine.panels.closure import PanelClosurePublisher
    from alphalattice.foundation.feature_engine.panels.closure_source import (
        PanelClosureSourceRepository,
    )
    from alphalattice.foundation.feature_engine.panels.rematerialization import (
        ArtifactOnlyPanelRematerializer,
    )
    from alphalattice.foundation.market_data_ops.sources.manifest import ManifestListing

    manifest = replace(
        fixture_manifest(),
        listings=tuple(
            ManifestListing(f"listing-{index:02d}", f"T{index:02d}", "XNYS", f"T{index:02d}")
            for index in range(60)
        ),
    )
    prior, today = SESSIONS[400:402]
    workspace = panel_workspace(tmp_path / "partial", manifest, end=prior, spy="a" * 64)
    workspace.provider.sectors = {
        item.symbol: f"Sector-{index // 20}" for index, item in enumerate(manifest.listings)
    }
    workspace.provider.sectors["SPY"] = "Reference"
    _first, original = build_and_publish(
        workspace, manifest, spy="a" * 64, as_of=prior, observed_at=NOW, refresh_sector=True
    )
    missing = manifest.listings[-1].listing_id
    usable = replace(manifest, listings=manifest.listings[:-1])
    insert_fixture_bars(workspace.market_data, usable, workspace.provider, start=today, end=today)
    workspace.feature_state.upsert_market_reference(
        reference_id="SPY",
        listing_id=MarketReference.spy(manifest).listing_id,
        provider=workspace.provider.name,
        symbol="SPY",
        revision_hash="b" * 64,
        action_audit_receipt_hash=None,
        latest_session=today,
        observed_at=NOW + timedelta(days=1),
    )
    outcome, partial = build_and_publish(
        workspace,
        manifest,
        spy="b" * 64,
        as_of=today,
        observed_at=NOW + timedelta(days=1),
        invalidations=new_session_invalidations(manifest, today),
        refresh_sector=False,
    )
    assert outcome.status is FeatureBuildStatus.COMPLETED
    assert partial["safe_summary"]["membership"]["as_of_member_count"] == 60
    table = pq.read_table(
        workspace.resolver.resolve_feature_panel_chunk_ref(
            uri=partial["chunks"][-1]["uri"],
            content_hash=partial["chunks"][-1]["chunk_hash"],
            metadata_hash=partial["chunks"][-1]["metadata_hash"],
        )
    ).to_pandas()
    selected = table.loc[table["session_date"] == today]
    assert len(selected) == 60
    factor = "mom_126_21"
    assert selected.loc[selected["listing_id"] == missing, factor].isna().all()
    assert selected.loc[selected["listing_id"] != missing, factor].notna().sum() == 59
    assert partial["chunks"][:-1] == original["chunks"][:-1]
    kwargs = dict(
        listing_ids=tuple(item.listing_id for item in manifest.listings),
        catalog_hash=workspace.service.catalog.binding.catalog_hash,
        as_of_session=today,
    )
    # Historical strict callers still refuse the absent row; the new policy
    # commits the verified raw absence rather than silently dropping its stock.
    with pytest.raises(ValueError, match="missing an active as-of feature row"):
        workspace.panel_state.panel_source_state_hash(**kwargs)
    partial_hash = workspace.panel_state.panel_source_state_hash(
        **kwargs, allow_missing_market_observations=True
    )
    closure = PanelClosurePublisher(
        resolver=workspace.resolver,
        source=PanelClosureSourceRepository(
            database_path=workspace.market_data.path,
            resolver=workspace.resolver,
        ),
        store=PanelClosureArtifactStore(workspace.resolver),
    ).publish()
    partial_recipe = closure.recipes[str(partial["snapshot_hash"])]
    base_manifest = next(iter(closure.base_manifests.values()))
    assert base_manifest.absent_source_keys is not None
    assert base_manifest.absent_source_keys.row_count == 1
    recovered = replace(manifest, listings=(manifest.listings[-1],))
    # This time SPY already exists; insert only the missing listing's source
    # through the real validated-batch owner instead of duplicating that bar.
    payload = workspace.provider.fetch_daily(
        (manifest.listings[-1].symbol,), start=today, end=today
    )
    workspace.market_data.apply_validated_batch(
        recovered,
        sanitize_payload(recovered, workspace.provider.name, payload, tuple(payload)),
        ingestion_id="fixture-recovered-daily-bar",
        observed_at=NOW + timedelta(days=1),
    )
    with pytest.raises(ValueError, match="panel_member_row_not_materialized"):
        workspace.panel_state.panel_source_state_hash(
            **kwargs, allow_missing_market_observations=True
        )
    workspace.market_data.complete_action_audit(
        recovered,
        listing_id=missing,
        provider=workspace.provider.name,
        observed_actions=(),
        observed_adjusted_closes=workspace.provider.fetch_adjusted_close_history(
            listing_id=missing,
            provider_symbol=recovered.listings[0].provider_symbol,
            start=today,
            end=today,
        ),
        history_start=today,
        history_end=today,
        requested_as_of=today,
        observed_at=NOW + timedelta(days=1),
    )
    _resumed, complete = build_and_publish(
        workspace,
        manifest,
        spy="b" * 64,
        as_of=today,
        observed_at=NOW + timedelta(days=1),
        invalidations=new_session_invalidations(recovered, today),
        refresh_sector=False,
    )
    complete_hash = workspace.panel_state.panel_source_state_hash(**kwargs)
    assert complete_hash != partial_hash
    assert complete_hash == workspace.panel_state.panel_source_state_hash(
        **kwargs, allow_missing_market_observations=True
    )
    assert complete["safe_summary"]["membership"] == partial["safe_summary"]["membership"]
    chunk = complete["chunks"][-1]
    restored = pq.read_table(
        workspace.resolver.resolve_feature_panel_chunk_ref(
            uri=chunk["uri"],
            content_hash=chunk["chunk_hash"],
            metadata_hash=chunk["metadata_hash"],
        )
    ).to_pandas()
    assert restored.loc[restored["session_date"] == today, factor].notna().sum() == 60
    reproduced = ArtifactOnlyPanelRematerializer(
        resolver=workspace.resolver,
        store=PanelClosureArtifactStore(workspace.resolver),
    ).rematerialize(partial_recipe.recipe_hash)
    assert reproduced.logical_parity and reproduced.physical_parity
    from alphalattice.control.product_host.storage.retention import _artifact_references

    store = PanelClosureArtifactStore(workspace.resolver)
    absent_path = store.physical_path(
        category="base-keys", reference=base_manifest.absent_source_keys
    )
    # The retention walker recognizes this declared descriptor once the normal
    # storage-closure operation has registered the recipe as a recovery root.
    assert base_manifest.absent_source_keys.content_hash in _artifact_references(
        base_manifest.model_dump(mode="json")
    )
    original_absence = absent_path.read_bytes()
    try:
        absent_path.write_bytes(
            original_absence[:4] + bytes([original_absence[4] ^ 1]) + original_absence[5:]
        )
        with pytest.raises(ValueError, match="physical hash"):
            ArtifactOnlyPanelRematerializer(resolver=workspace.resolver, store=store).rematerialize(
                partial_recipe.recipe_hash
            )
    finally:
        absent_path.write_bytes(original_absence)

    # A quality restriction must mask even valid, already-materialized values.
    # It belongs to this snapshot's source scope, not to its nominal member list.
    from alphalattice.foundation.feature_engine.inputs.contracts import ListingQuarantine

    quarantine = ListingQuarantine.create(
        listing_id=missing,
        reason_codes=("SOURCE_QUALITY_UNAVAILABLE",),
        evidence_hash="d" * 64,
        execution_receipt_hash="e" * 64,
        recheck_after_at=NOW + timedelta(days=2),
    )
    workspace.panel_state.record_listing_quarantines(
        candidate_manifest_revision=manifest.revision_sha256,
        quarantines=(quarantine,),
        observed_at=NOW + timedelta(days=1),
        effective_session=today,
    )
    _masked, restricted = build_and_publish(
        workspace,
        manifest,
        spy="b" * 64,
        as_of=today,
        observed_at=NOW + timedelta(days=1),
        invalidations=(
            FeatureInvalidation(
                "panel_binding_change", source_receipt_hash=quarantine.quarantine_hash
            ),
        ),
        refresh_sector=False,
    )
    assert restricted["safe_summary"]["membership"]["as_of_member_count"] == 60
    assert restricted["safe_summary"]["membership"]["source_exclusions"]
    restricted_closure = PanelClosurePublisher(
        resolver=workspace.resolver,
        source=PanelClosureSourceRepository(
            database_path=workspace.market_data.path, resolver=workspace.resolver
        ),
        store=store,
    ).publish()
    restricted_recipe = restricted_closure.recipes[str(restricted["snapshot_hash"])]
    assert restricted_recipe.membership.source_exclusions
    workspace.panel_state.clear_listing_quarantine(
        quarantine_hash=quarantine.quarantine_hash,
        qualification_receipt_hash="f" * 64,
        cleared_at=NOW + timedelta(days=2),
        effective_session=today,
    )
    preserved = ArtifactOnlyPanelRematerializer(
        resolver=workspace.resolver, store=store
    ).rematerialize(restricted_recipe.recipe_hash)
    assert preserved.logical_parity and preserved.physical_parity


def _chunk_row_hashes(resolver: ArtifactResolver, entry: dict[str, object]) -> list[str]:
    path = resolver.resolve_feature_panel_chunk_ref(
        uri=str(entry["uri"]),
        content_hash=str(entry["chunk_hash"]),
        metadata_hash=str(entry["metadata_hash"]),
    )
    return pq.read_table(path, columns=["row_hash"]).column("row_hash").to_pylist()


def _clipping_evidence(resolver: ArtifactResolver, manifest_payload: dict[str, object]):
    from alphalattice.foundation.feature_engine.producers.preprocessing.contracts import (
        PanelClippingEvidence,
    )
    from alphalattice.foundation.feature_engine.publication.snapshots import (
        resolve_panel_preprocessing_lineage,
    )

    lineage = resolve_panel_preprocessing_lineage(
        resolver, panel_content_hash=str(manifest_payload["panel_content_hash"])
    )
    assert lineage is not None
    return PanelClippingEvidence.model_validate(
        resolver.load_panel_clipping_evidence(
            resolver.panel_clipping_evidence_uri(str(lineage["clipping_evidence_hash"]))
        )
    )


def test_daily_append_reuses_closed_years_and_keeps_provenance_recovery_and_retention(
    tmp_path,
) -> None:
    """requirement: a new session must not rewrite the closed years' partitions.

    Day one builds and publishes the Panel through the penultimate fixture
    session. Day two appends the last session under a new SPY revision: every
    closed year is reused under its origin binding, only the current year is
    composed, and the manifest records every origin. The reused rows are
    identical to a from-scratch build of the same data, the folded clipping
    evidence matches that build's counts, the mixed-origin snapshot reads,
    freezes into a closure and rematerializes byte-exact, the residue owner
    keeps the reused years' availability rows, and a correction reaching back
    into a closed year recomposes exactly that year and the ones after it.
    """

    from alphalattice.foundation.feature_engine.panels.closure import PanelClosurePublisher
    from alphalattice.foundation.feature_engine.panels.closure_source import (
        PanelClosureSourceRepository,
    )
    from alphalattice.foundation.feature_engine.panels.reader import (
        FeaturePanelReader,
        FeaturePanelReadRequest,
    )
    from alphalattice.foundation.feature_engine.panels.rematerialization import (
        ArtifactOnlyPanelRematerializer,
    )
    from alphalattice.foundation.feature_engine.panels.retention_residue import (
        PanelRetentionResidueOwner,
    )

    manifest = fixture_manifest()
    day_one, day_two = SESSIONS[-2], SESSIONS[-1]
    spy_one, spy_two = "a" * 64, "b" * 64
    workspace = panel_workspace(tmp_path / "daily", manifest, end=day_one, spy=spy_one)
    _first, manifest_one = build_and_publish(
        workspace, manifest, spy=spy_one, as_of=day_one, observed_at=NOW, refresh_sector=True
    )
    binding_one = str(manifest_one["panel_binding_hash"])
    assert all(chunk["origin_binding_hash"] == binding_one for chunk in manifest_one["chunks"])
    lineage_one = manifest_one["safe_summary"]["lineage"]
    assert set(lineage_one["partition_origins"]) == {binding_one}
    assert lineage_one["partition_origins"][binding_one]["spy_revision"] == spy_one
    assert manifest_one["safe_summary"]["partition_reuse"] == {
        "reused_partition_count": 0,
        "composed_partition_count": len(manifest_one["chunks"]),
    }

    # Day two: the last session arrives and the market reference moves on.
    insert_fixture_bars(
        workspace.market_data, manifest, workspace.provider, start=day_two, end=day_two
    )
    workspace.feature_state.upsert_market_reference(
        reference_id="SPY",
        listing_id=MarketReference.spy(manifest).listing_id,
        provider=workspace.provider.name,
        symbol="SPY",
        revision_hash=spy_two,
        action_audit_receipt_hash=None,
        latest_session=day_two,
        observed_at=NOW + timedelta(days=1),
    )
    second, manifest_two = build_and_publish(
        workspace,
        manifest,
        spy=spy_two,
        as_of=day_two,
        observed_at=NOW + timedelta(days=1),
        invalidations=new_session_invalidations(manifest, day_two),
        refresh_sector=False,
    )
    binding_two = str(manifest_two["panel_binding_hash"])
    assert binding_two != binding_one
    years = [int(chunk["year"]) for chunk in manifest_two["chunks"]]
    assert second.coverage_summary["panel_partitions_reused"] == len(years) - 1
    assert second.coverage_summary["panel_partitions_written"] == 1
    assert second.coverage_summary["panel_partitions_reused_years"] == years[:-1]
    by_year_one = {int(chunk["year"]): chunk for chunk in manifest_one["chunks"]}
    for chunk in manifest_two["chunks"][:-1]:
        assert chunk == by_year_one[int(chunk["year"])]
        assert chunk["origin_binding_hash"] == binding_one
    current = manifest_two["chunks"][-1]
    assert current["origin_binding_hash"] == binding_two
    assert current["chunk_hash"] != by_year_one[int(current["year"])]["chunk_hash"]
    assert current["last_session"] == day_two.isoformat()
    origins_two = manifest_two["safe_summary"]["lineage"]["partition_origins"]
    assert set(origins_two) == {binding_one, binding_two}
    assert origins_two[binding_one]["spy_revision"] == spy_one
    assert origins_two[binding_two]["spy_revision"] == spy_two
    assert origins_two[binding_two]["materialization_receipt_hash"] == second.receipt_hash
    assert manifest_two["safe_summary"]["partition_reuse"] == {
        "reused_partition_count": len(years) - 1,
        "composed_partition_count": 1,
    }
    # Day one's snapshot, under yesterday's binding, is no longer the active
    # Panel: the projection the reader admits by says so at once, and says
    # what the database says, instead of listing both days as ACTIVE until
    # the next workspace open re-evaluated it.
    lifecycles = {
        item["snapshot_hash"]: (item["lifecycle"], item["reason"])
        for item in workspace.panel_state.feature_panel_snapshot_lifecycles()
    }
    assert lifecycles[str(manifest_one["snapshot_hash"])] == (
        "SUPERSEDED",
        "not_current_active_panel",
    )
    assert lifecycles[str(manifest_two["snapshot_hash"])] == ("ACTIVE", None)
    for snapshot_hash, (lifecycle, reason) in lifecycles.items():
        projected = workspace.resolver.feature_panel_snapshot_lifecycle(
            workspace.resolver.feature_panel_manifest_uri(snapshot_hash)
        )
        assert (projected["lifecycle"], projected["reason"]) == (lifecycle, reason)

    # The reused partitions are exactly what a build from scratch computes.
    reference = panel_workspace(tmp_path / "reference", manifest, end=day_two, spy=spy_two)
    _reference_build, reference_manifest = build_and_publish(
        reference, manifest, spy=spy_two, as_of=day_two, observed_at=NOW, refresh_sector=True
    )
    assert (
        reference_manifest["safe_summary"]["lineage"]["sector_revision"]
        == (manifest_two["safe_summary"]["lineage"]["sector_revision"])
    )
    for reused, rebuilt in zip(manifest_two["chunks"], reference_manifest["chunks"], strict=True):
        assert reused["year"] == rebuilt["year"]
        assert _chunk_row_hashes(workspace.resolver, reused) == _chunk_row_hashes(
            reference.resolver, rebuilt
        )
    evidence_two = _clipping_evidence(workspace.resolver, manifest_two)
    evidence_reference = _clipping_evidence(reference.resolver, reference_manifest)
    assert evidence_two.panel_binding_hash == binding_two
    assert evidence_two.total_clipped_count == evidence_reference.total_clipped_count
    # Cell by cell on the calendar, not only in total: the fold joins each
    # cell to the batch that measured it, so the reused years' counts sit in
    # calendar order beside the appended session's.
    assert [
        (item.factor_id, item.per_session_clipped_counts, item.per_session_clipped_fractions)
        for item in evidence_two.factor_records
    ] == [
        (item.factor_id, item.per_session_clipped_counts, item.per_session_clipped_fractions)
        for item in evidence_reference.factor_records
    ]
    assert evidence_two.total_clipped_count > 0

    # The mixed-origin snapshot qualifies whole through the governed reader's
    # chunk validation (each partition under its own origin); the fixture has
    # no Gateway admission, which is the reader's separate, unchanged refusal.
    manifest_two_uri = workspace.resolver.feature_panel_manifest_uri(
        str(manifest_two["snapshot_hash"])
    )
    qualified = FeaturePanelReader(workspace.resolver)._qualified_chunk_paths(
        manifest_two,
        FeaturePanelReadRequest(
            manifest_ref=manifest_two_uri,
            start_session=SESSIONS[0],
            end_session=day_two,
            factor_columns=("rev_21",),
        ),
    )
    assert len(qualified) == len(years)
    identity = ds.dataset(qualified, format="parquet").to_table(columns=["session_date"])
    assert identity.num_rows == len(SESSIONS) * len(manifest.listings)
    assert sorted(set(identity.column("session_date").to_pylist())) == list(SESSIONS)

    # It freezes into a closure whose recipe records the origins, and the
    # artifact-only rematerializer reproduces every chunk byte-exact.
    closure = PanelClosurePublisher(
        resolver=workspace.resolver,
        source=PanelClosureSourceRepository(
            database_path=workspace.market_data.path, resolver=workspace.resolver
        ),
        store=PanelClosureArtifactStore(workspace.resolver),
    ).publish()
    assert not closure.blocked_snapshots
    recipe = closure.recipes[str(manifest_two["snapshot_hash"])]
    assert {item.panel_binding_hash for item in recipe.partition_origins or ()} == {binding_one}
    assert [item.origin_binding_hash for item in recipe.expected_chunks] == [
        binding_one for _ in years[:-1]
    ] + [binding_two]
    result = ArtifactOnlyPanelRematerializer(
        resolver=workspace.resolver, store=PanelClosureArtifactStore(workspace.resolver)
    ).rematerialize(recipe.recipe_hash)
    assert result.logical_parity and result.physical_parity
    assert [item.chunk_hash for item in result.chunks] == [
        chunk["chunk_hash"] for chunk in manifest_two["chunks"]
    ]

    # Residue cleanup rooted at the day-two snapshot keeps the reused years'
    # availability rows, which were stamped by day one's build.
    def availability_bindings() -> dict[str, int]:
        connection = duckdb.connect(str(workspace.market_data.path), read_only=True)
        try:
            return {
                str(row[0]): int(row[1])
                for row in connection.execute(
                    "SELECT panel_binding_hash, count(*) "
                    "FROM panel_cross_section_availability GROUP BY 1"
                ).fetchall()
            }
        finally:
            connection.close()

    before = availability_bindings()
    assert set(before) == {binding_one, binding_two}
    receipt = PanelRetentionResidueOwner(
        workspace=workspace.market_data.workspace, resolver=workspace.resolver
    ).close(
        protected_panel_snapshot_hashes=(str(manifest_two["snapshot_hash"]),),
        cutover_acceptance_marker_hash="c" * 64,
        completed_at=NOW + timedelta(days=1, hours=1),
    )
    assert availability_bindings() == before
    assert {item.panel_binding_hash for item in receipt.retained_generations} == set(before)
    assert receipt.retired_generations == ()
    # Reachability counts the shared partitions once; day one's own current-year
    # chunk is the only file the day-two root does not reach.
    reachability = workspace.resolver.feature_panel_reachability(
        root_manifest_uris=[manifest_two_uri]
    )
    assert reachability.unreferenced == (
        workspace.resolver.feature_panel_chunk_uri(str(by_year_one[years[-1]]["chunk_hash"])),
        workspace.resolver.feature_panel_manifest_uri(str(manifest_one["snapshot_hash"])),
    )

    # A correction reaching back into the previous year (same market
    # reference, so the binding is day two's) recomposes that year and the
    # one after it, and reuses the rest.
    correction_session = max(session for session in SESSIONS if session.year == years[-2])
    third, manifest_three = build_and_publish(
        workspace,
        manifest,
        spy=spy_two,
        as_of=day_two,
        observed_at=NOW + timedelta(days=2),
        invalidations=(
            FeatureInvalidation(
                "adjusted_return_correction",
                listing_id=manifest.listings[0].listing_id,
                affected_sessions=(correction_session,),
                source_fields=("provider_adjusted_close",),
                factor_ids=("rev_21",),
                source_receipt_hash="e" * 64,
            ),
        ),
        refresh_sector=False,
    )
    assert third.coverage_summary["panel_partitions_reused_years"] == years[:-2]
    assert third.coverage_summary["panel_partitions_written"] == 2
    assert str(manifest_three["panel_binding_hash"]) == binding_two
    origins_three = [chunk["origin_binding_hash"] for chunk in manifest_three["chunks"]]
    assert origins_three == [binding_one] * (len(years) - 2) + [binding_two, binding_two]
    # The previous year was day one's partition; recomposed under day two's
    # binding it is a new file even though the fixture correction changed no
    # value. The current year was already day two's: recomposed to identical
    # rows under the same binding it names the same file, which is kept, not
    # rewritten (an immutable partition is never overwritten).
    assert manifest_three["chunks"][-2]["chunk_hash"] != manifest_two["chunks"][-2]["chunk_hash"]
    assert manifest_three["chunks"][-1] == manifest_two["chunks"][-1]
    assert _chunk_row_hashes(workspace.resolver, manifest_three["chunks"][-2]) == (
        _chunk_row_hashes(workspace.resolver, manifest_two["chunks"][-2])
    )
    assert set(manifest_three["safe_summary"]["lineage"]["partition_origins"]) == {
        binding_one,
        binding_two,
    }
    # The corrected cells are the only ones this build measured; the evidence
    # still describes the whole Panel, folded from every batch that owns cells.
    evidence_three = _clipping_evidence(workspace.resolver, manifest_three)
    assert evidence_three.panel_binding_hash == binding_two
    assert {item.factor_id for item in evidence_three.factor_records} == {
        item.factor_id for item in evidence_reference.factor_records
    }
    assert all(
        len(item.per_session_clipped_counts) == len(SESSIONS)
        for item in evidence_three.factor_records
    )
    # The correction left day two's batches owning only the cells it did not
    # recompute. Their receipts cannot re-derive from those cells alone; the
    # cells are verified one by one and the chunks still reproduce byte-exact.
    closure_three = PanelClosurePublisher(
        resolver=workspace.resolver,
        source=PanelClosureSourceRepository(
            database_path=workspace.market_data.path, resolver=workspace.resolver
        ),
        store=PanelClosureArtifactStore(workspace.resolver),
    ).publish()
    recipe_three = closure_three.recipes[str(manifest_three["snapshot_hash"])]
    rematerializer = ArtifactOnlyPanelRematerializer(
        resolver=workspace.resolver, store=PanelClosureArtifactStore(workspace.resolver)
    )
    replayed = rematerializer.rematerialize(recipe_three.recipe_hash)
    assert replayed.logical_parity and replayed.physical_parity
    assert [item.chunk_hash for item in replayed.chunks] == [
        chunk["chunk_hash"] for chunk in manifest_three["chunks"]
    ]
    # Day two shares day three's binding, so its live availability now
    # carries day three's batch and cannot reproduce day two's content
    # identity; the capture recovers day two's earlier recipe instead of
    # sealing one no replay could satisfy, and that recipe still replays.
    assert str(manifest_two["snapshot_hash"]) not in closure_three.blocked_snapshots
    assert closure_three.recipes[str(manifest_two["snapshot_hash"])] == recipe
    assert rematerializer.rematerialize(recipe.recipe_hash).physical_parity


def test_feature_materialization_failure_keeps_bounded_diagnostics(tmp_path, monkeypatch) -> None:
    manifest = fixture_manifest()
    provider = FixtureProvider(tuple(item.symbol for item in manifest.listings))
    market_data = MarketDataRepository(tmp_path / "workspace")
    feature_state = FeatureStateRepository(market_data.database, market_data=market_data)
    panel_state = PanelStateRepository(market_data.database, market_data=market_data)
    spy_revision = seed_feature_input_fixture(market_data, feature_state, manifest, provider)
    resolver = ArtifactResolver(tmp_path / "artifacts")
    gate = WorkspaceMutationGate()
    ledger, feature_persistence = open_closure(
        market_data=market_data,
        panel_state=panel_state,
        feature_state=feature_state,
        resolver=resolver,
        manifest=manifest,
    )
    service = FeatureFoundationService(
        market_data=market_data,
        feature_state=feature_state,
        panel_state=panel_state,
        manifest=manifest,
        provider=provider,
        mutation_gate=gate,
        panel_artifacts=PanelArtifactCompositionOwner(resolver),
        feature_persistence=feature_persistence,
        sector_activation=SectorRevisionMapActivationCoordinator(
            store=feature_state, mutation_gate=gate, ledger=ledger
        ),
        session_authority_resolver=fixture_session_authority,
    )

    # A listing computes in a Host worker (W10), whose ValueError answers as a refusal
    # carrying the name of what it raised.
    def fail_materialization(_calls):
        raise ChildRefused("fixture materialization detail", type_name="ValueError")

    monkeypatch.setattr(ChildCalls, "answer", fail_materialization)
    request = FeatureBuildRequest.create(
        manifest_revision=manifest.revision_sha256,
        catalog=service.catalog.binding,
        spy_revision=spy_revision,
        history_start=SESSIONS[0],
        as_of_session=SESSIONS[-1],
    )
    outcome = service.build(request, observed_at=NOW, refresh_sector=False)
    assert outcome.status is FeatureBuildStatus.BLOCKED
    assert outcome.failure_code == "feature.materialization_failed"
    failure = outcome.coverage_summary["materialization_failure"]
    assert failure == {
        "stage": "base_feature_materialization",
        "exception_type": "ValueError",
        "listing_id": manifest.listings[0].listing_id,
        "first_target_session": SESSIONS[0].isoformat(),
        "last_target_session": SESSIONS[-1].isoformat(),
        "detail": "fixture materialization detail",
    }

    # A closure file the ledger could not place on disk within its retry
    # bound (the typed conflict the shared replace raises) is named as the
    # retryable publication refusal, not as a materialization defect.
    def blocked_publication(_calls):
        raise WorkspaceConflictError(
            "replacement of pending.json is blocked by an open handle",
            code="catalog.replace_blocked",
        )

    monkeypatch.setattr(ChildCalls, "answer", blocked_publication)
    outcome = service.build(request, observed_at=NOW, refresh_sector=False)
    assert outcome.status is FeatureBuildStatus.BLOCKED
    assert outcome.failure_code == "feature_closure.publication_blocked"
    failure = outcome.coverage_summary["materialization_failure"]
    assert failure["exception_type"] == "WorkspaceConflictError"
    assert failure["detail"] == "replacement of pending.json is blocked by an open handle"


def test_a_build_writes_the_same_bytes_whatever_its_workers(tmp_path) -> None:
    """requirement (PF #1, W10, LAWS PA3): a build's listings compute in the Host's kept
    workers and are written in listing order, so one worker and several write the same Feature
    rows, cutoff sets, receipts, revisions, ineligibility runs and closure head. The count is an
    execution parameter: recorded beside the build, in nothing it wrote."""

    written: dict[int, tuple[str, ...]] = {}
    for workers in (1, 3):
        manifest = fixture_manifest()
        provider = FixtureProvider(tuple(item.symbol for item in manifest.listings))
        market_data = MarketDataRepository(tmp_path / f"workers-{workers}" / "workspace")
        feature_state = FeatureStateRepository(market_data.database, market_data=market_data)
        panel_state = PanelStateRepository(market_data.database, market_data=market_data)
        spy_revision = seed_feature_input_fixture(market_data, feature_state, manifest, provider)
        resolver = ArtifactResolver(tmp_path / f"workers-{workers}" / "artifacts")
        gate = WorkspaceMutationGate()
        ledger, feature_persistence = open_closure(
            market_data=market_data,
            panel_state=panel_state,
            feature_state=feature_state,
            resolver=resolver,
            manifest=manifest,
        )
        updates: list[WorkProgressUpdate] = []
        service = FeatureFoundationService(
            market_data=market_data,
            feature_state=feature_state,
            panel_state=panel_state,
            manifest=manifest,
            provider=provider,
            mutation_gate=gate,
            panel_artifacts=PanelArtifactCompositionOwner(resolver),
            feature_persistence=feature_persistence,
            sector_activation=SectorRevisionMapActivationCoordinator(
                store=feature_state, mutation_gate=gate, ledger=ledger
            ),
            progress_sink=updates.append,
            session_authority_resolver=fixture_session_authority,
            feature_workers=workers,
        )
        request = FeatureBuildRequest.create(
            manifest_revision=manifest.revision_sha256,
            catalog=service.catalog.binding,
            spy_revision=spy_revision,
            history_start=SESSIONS[0],
            as_of_session=SESSIONS[-1],
        )
        service.build(request, observed_at=NOW, refresh_sector=False)
        (finished,) = (
            update
            for update in updates
            if update.stage_id == "base_feature_materialization" and update.status == "SUCCEEDED"
        )
        assert finished.counters["workers"] == workers
        head = ledger.current_head(service.catalog.binding.catalog_hash)
        assert head is not None
        connection = duckdb.connect(str(market_data.path), read_only=True)
        try:
            tables = tuple(
                json.dumps(
                    connection.execute(f"SELECT * FROM {table} ORDER BY ALL").fetchall(),
                    default=str,
                )
                for table in (
                    "feature_daily_current",
                    "feature_input_cutoff_set",
                    "feature_materialization_receipt",
                    "feature_daily_revision",
                    "feature_ineligibility_run",
                )
            )
        finally:
            connection.close()
        written[workers] = (head.head_hash, head.feature_row_hash_digest, *tables)
    assert written[1] == written[3]


def test_a_panel_is_the_same_whatever_its_workers(tmp_path) -> None:
    """requirement (PF #7b, W10, LAWS PA3): a build's Panel chunks compute in the Host's kept
    workers and are staged, published and recorded in chunk order, so one worker and several
    write the same Panel availability and chunk files and publish the same snapshot. The count
    is an execution parameter: recorded beside the build, in nothing it wrote."""
    from dataclasses import replace

    from alphalattice.foundation.market_data_ops.sources.manifest import ManifestListing

    manifest = replace(
        fixture_manifest(),
        listings=tuple(
            ManifestListing(f"listing-{index:02d}", f"T{index:02d}", "XNYS", f"T{index:02d}")
            for index in range(60)
        ),
    )
    as_of = SESSIONS[400]
    written: dict[int, tuple[object, ...]] = {}
    for workers in (1, 3):
        workspace = panel_workspace(
            tmp_path / f"workers-{workers}", manifest, end=as_of, spy="a" * 64
        )
        workspace.provider.sectors = {
            item.symbol: f"Sector-{index // 20}" for index, item in enumerate(manifest.listings)
        }
        workspace.provider.sectors["SPY"] = "Reference"
        workspace.service.feature_workers = workers
        outcome, published = build_and_publish(
            workspace, manifest, spy="a" * 64, as_of=as_of, observed_at=NOW, refresh_sector=True
        )
        chunks = published["chunks"]
        assert isinstance(chunks, list) and chunks
        chunk_bytes = tuple(
            workspace.resolver.resolve_feature_panel_chunk_ref(
                uri=str(item["uri"]),
                content_hash=str(item["chunk_hash"]),
                metadata_hash=str(item["metadata_hash"]),
            ).read_bytes()
            for item in chunks
        )
        connection = duckdb.connect(str(workspace.market_data.path), read_only=True)
        try:
            availability = connection.execute(
                "SELECT * FROM panel_cross_section_availability ORDER BY ALL"
            ).fetchall()
        finally:
            connection.close()
        written[workers] = (
            outcome.receipt_hash,
            published["panel_content_hash"],
            json.dumps(chunks, sort_keys=True, default=str),
            chunk_bytes,
            json.dumps(availability, default=str),
        )
    assert written[1] == written[3]


def test_panel_fails_closed_below_coverage_and_small_sector() -> None:
    catalog = FeatureCatalog.load()
    materializer = SectorNeutralPanelMaterializer(catalog)
    rows = []
    for index in range(8):
        rows.append(
            {
                "listing_id": f"l{index}",
                "session_date": "2026-08-03",
                **{factor: float(index) for factor in catalog.factor_ids},
            }
        )
    panel = materializer.materialize(
        feature_rows=rows,
        active_listing_ids=tuple(f"l{index}" for index in range(8)),
        manifest_revision="a" * 64,
        sector_revision="s" * 64,
        sector_by_listing_id={
            f"l{index}": "sector-a" if index < 4 else "sector-b" for index in range(8)
        },
        spy_revision="p" * 64,
    )
    assert not panel.complete
    assert not panel.admission.research_admissible
    assert "active_sector_sample_below_5" in panel.admission.structural_failure_reasons
    assert {item["status"] for item in panel.availability} == {"unavailable"}
    assert {item["reason"] for item in panel.availability} == {"sector_sample_below_5"}


def test_sector_token_budget_defers_without_sleep(tmp_path) -> None:
    manifest = fixture_manifest()
    market_data = MarketDataRepository(tmp_path / "workspace")
    feature_state = FeatureStateRepository(market_data.database, market_data=market_data)
    PanelStateRepository(market_data.database, market_data=market_data)
    market_data.bootstrap(manifest)
    # 20 available at first use, then no token exists at the same controlled timestamp.
    for _ in range(20):
        assert feature_state.consume_workspace_tokens(bucket_id="fixture", requested=1, now=NOW)[0]
    granted, retry_after = feature_state.consume_workspace_tokens(
        bucket_id="fixture", requested=1, now=NOW
    )
    assert not granted
    assert retry_after == NOW + pd.Timedelta(seconds=2).to_pytimedelta()


def test_panel_fails_closed_at_97_point_5_percent_coverage() -> None:
    catalog = FeatureCatalog.load()
    active = tuple(f"l{index:03d}" for index in range(200))
    rows = [
        {
            "listing_id": listing_id,
            "session_date": "2026-08-03",
            **{factor: (float(index) if index < 195 else None) for factor in catalog.factor_ids},
        }
        for index, listing_id in enumerate(active)
    ]
    panel = SectorNeutralPanelMaterializer(catalog).materialize(
        feature_rows=rows,
        active_listing_ids=active,
        manifest_revision="m" * 64,
        sector_revision="s" * 64,
        sector_by_listing_id={
            listing_id: f"sector-{index % 5}" for index, listing_id in enumerate(active)
        },
        spy_revision="p" * 64,
    )
    assert {item["reason"] for item in panel.availability} == {"coverage_below_98_percent"}


def test_a_stale_panel_is_judged_by_feature_alone_and_its_projection_follows(
    tmp_path: Path,
) -> None:
    """regression (V176): Market Data's bootstrap marked Panels SUPERSEDED by its own SQL, so
    after a publication that failed once its binding moved the database read SUPERSEDED while
    the projection research reads said ACTIVE; the bootstrap moves no lifecycle, and Feature's
    one function moves the rows and the projection together."""

    from alphalattice.foundation.feature_engine.publication.snapshots import (
        reconcile_feature_panel_lifecycles,
    )

    manifest = fixture_manifest()
    as_of = SESSIONS[-1]
    workspace = panel_workspace(tmp_path / "stale", manifest, end=as_of, spy="a" * 64)
    _built, published = build_and_publish(
        workspace, manifest, spy="a" * 64, as_of=as_of, observed_at=NOW, refresh_sector=True
    )
    snapshot = str(published["snapshot_hash"])

    def lifecycle() -> tuple[tuple[object, object], tuple[object, object]]:
        rows = {
            item["snapshot_hash"]: (item["lifecycle"], item["reason"])
            for item in workspace.panel_state.feature_panel_snapshot_lifecycles()
        }
        projected = workspace.resolver.feature_panel_snapshot_lifecycle(
            workspace.resolver.feature_panel_manifest_uri(snapshot)
        )
        return rows[snapshot], (projected["lifecycle"], projected["reason"])

    # The binding moved and no snapshot of the new one was published.
    connection = workspace.market_data._connect()
    try:
        connection.execute("DELETE FROM active_feature_panel_binding")
    finally:
        connection.close()
    workspace.market_data.bootstrap(manifest)
    assert lifecycle() == (("ACTIVE", None), ("ACTIVE", None))

    reconcile_feature_panel_lifecycles(
        panel_state=workspace.panel_state,
        resolver=workspace.resolver,
        mutation_gate=workspace.gate,
        observed_at=NOW,
    )
    assert lifecycle() == (("SUPERSEDED", "not_current_active_panel"),) * 2


def test_an_activation_holds_the_base_and_retires_rows_outside_its_layer(tmp_path) -> None:
    """regression (V398, found timing V92 on a real workspace, 2026-10-01): the store held rows
    an earlier catalog computed for listings an earlier manifest held and a manifest transition
    dropped; no build reaches them again, so after a person activated a formula factor every
    publication refused (`feature_storage.row_catalog_identity_mismatch`). The build retires
    every row a catalog outside its layer computed once each part holds the axis, and the Panel
    publishes. The activation's layer (V92) holds the shipped catalog's rows as they were and
    writes its column catalog's beside them."""

    from dataclasses import replace

    from alphalattice.foundation.feature_engine.contracts import FeatureInvalidation
    from alphalattice.foundation.feature_engine.inputs.closure_source import (
        FeatureClosureSourceRepository,
    )
    from alphalattice.foundation.feature_engine.panels.artifacts import (
        PanelArtifactCompositionOwner,
    )
    from alphalattice.foundation.feature_engine.panels.feature_closure_coordinator import (
        FeatureLayerClosures,
    )
    from alphalattice.foundation.feature_engine.panels.logical_identity import (
        PanelLogicalArtifactStore,
        PanelLogicalIdentityPublisher,
    )
    from alphalattice.foundation.feature_engine.panels.recovery_binding import (
        PanelRecoveryBindingPublisher,
    )
    from alphalattice.foundation.feature_engine.producers.factors.formula import (
        formula_specification,
    )
    from alphalattice.foundation.feature_engine.publication.snapshots import (
        FeaturePanelSnapshotPublisher,
    )
    from alphalattice.foundation.feature_engine.runtime.closure_genesis import (
        FeatureClosureGenesisService,
    )
    from alphalattice.kernel.quant.factor_contracts import FactorFamily, FactorSpec, FactorTrack

    manifest = fixture_manifest()
    as_of = SESSIONS[300]
    held = panel_workspace(tmp_path / "held", manifest, end=as_of, spy="a" * 64)
    build_and_publish(
        held, manifest, spy="a" * 64, as_of=as_of, observed_at=NOW, refresh_sector=True
    )
    shipped = FeatureCatalog.load()
    connection = duckdb.connect(str(held.market_data.path))
    try:
        held_rows = dict(
            connection.execute(
                "SELECT listing_id || session_date, row_hash FROM feature_daily_current"
            ).fetchall()
        )
        connection.execute(
            "INSERT INTO feature_daily_current SELECT * REPLACE ('dropped-listing' AS "
            "listing_id, ? AS catalog_hash) FROM feature_daily_current WHERE listing_id = ?",
            ["e" * 64, manifest.listings[0].listing_id],
        )
    finally:
        connection.close()
    spec = formula_specification(
        FactorSpec(
            factor_id="formula_reversal_5",
            family=FactorFamily.PRICE_LEVEL_TREND,
            formula_ref="factor.formula",
            formula="close / lag(close, 5) - 1",
            window_sessions=1,
            lag_sessions=0,
            return_convention="as declared",
            required_fields=("close_split_adjusted",),
            literature_sources=("research-local note",),
            minimum_observations=1,
            absolute_tolerance=1e-10,
            relative_tolerance=1e-10,
            track=FactorTrack.MODEL,
        )
    )
    payload = shipped.to_payload()
    payload["factors"] = sorted(
        [*payload["factors"], spec.model_dump(mode="json")],
        key=lambda value: str(value["factor_id"]),
    )
    extended = FeatureCatalog.from_payload(payload)
    layer = FeatureCatalogLayer.over(extended)
    (column,) = layer.columns
    source = FeatureClosureSourceRepository(held.market_data.database.path)
    FeatureClosureGenesisService(
        panel_state=held.panel_state, source=source, ledger=held.ledger
    ).open_genesis(manifest=manifest, catalog=column)
    feature_state = FeatureStateRepository(
        held.market_data.database, market_data=held.market_data, installed_catalog=extended
    )
    service = FeatureFoundationService(
        market_data=held.market_data,
        feature_state=feature_state,
        panel_state=held.panel_state,
        manifest=manifest,
        provider=held.provider,
        mutation_gate=held.gate,
        panel_artifacts=PanelArtifactCompositionOwner(held.resolver),
        feature_persistence=FeatureLayerClosures(
            store=feature_state, source=source, ledger=held.ledger, layer=layer
        ),
        sector_activation=SectorRevisionMapActivationCoordinator(
            store=feature_state, mutation_gate=held.gate, ledger=held.ledger
        ),
        session_authority_resolver=fixture_session_authority,
        installed_catalog=extended,
    )
    publisher = FeaturePanelSnapshotPublisher(
        feature_state=feature_state,
        panel_state=held.panel_state,
        resolver=held.resolver,
        mutation_gate=held.gate,
        recovery_binding=PanelRecoveryBindingPublisher(ledger=held.ledger, resolver=held.resolver),
        logical_identity=PanelLogicalIdentityPublisher(
            resolver=held.resolver,
            store=PanelLogicalArtifactStore(PanelClosureArtifactStore(held.resolver)),
        ),
        catalog=extended,
    )
    _outcome, published = build_and_publish(
        replace(held, feature_state=feature_state, service=service, publisher=publisher),
        manifest,
        spy="a" * 64,
        as_of=as_of,
        observed_at=NOW + timedelta(hours=1),
        invalidations=(
            FeatureInvalidation("catalog_binding_change", earliest_session=SESSIONS[0]),
        ),
        refresh_sector=False,
    )
    assert published["safe_summary"]["lineage"]["catalog_hash"] == extended.binding.catalog_hash
    connection = duckdb.connect(str(held.market_data.path), read_only=True)
    try:
        assert (
            dict(
                connection.execute(
                    "SELECT listing_id || session_date, row_hash FROM feature_daily_current "
                    "WHERE catalog_hash = ?",
                    [shipped.binding.catalog_hash],
                ).fetchall()
            )
            == held_rows
        )
        assert dict(
            connection.execute(
                "SELECT catalog_hash, count(*) FROM feature_daily_current GROUP BY 1"
            ).fetchall()
        ) == {
            shipped.binding.catalog_hash: len(held_rows),
            column.binding.catalog_hash: len(held_rows),
        }
        assert not connection.execute(
            "SELECT count(*) FROM feature_daily_current WHERE listing_id = 'dropped-listing'"
        ).fetchone()[0]
    finally:
        connection.close()
