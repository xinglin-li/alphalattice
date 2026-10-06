from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.feature_engine.contracts import FeaturePanelBinding, canonical_hash
from alphalattice.foundation.feature_engine.panels.artifacts import (
    PanelCompositionBinding,
    PreparedPanelChunk,
    panel_content_identity,
)
from alphalattice.foundation.feature_engine.panels.closure import PanelClosurePublisher
from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
    PanelClosureArtifactStore,
    ordered_key_hash,
    ordered_value_hash,
)
from alphalattice.foundation.feature_engine.panels.closure_contracts import (
    ClosureArtifactRef,
    PanelBaseKeyChunk,
    PanelBaseValueChunk,
    PanelBaseValueClosureManifest,
    PanelDerivationRecipe,
)
from alphalattice.foundation.feature_engine.panels.closure_source import (
    PanelClosureSourceRepository,
)
from alphalattice.foundation.feature_engine.panels.identity import (
    hash_panel_rows,
    panel_chunk_hash,
    panel_schema_hash,
)
from alphalattice.foundation.feature_engine.panels.rematerialization import (
    ArtifactOnlyPanelRematerializer,
)
from alphalattice.foundation.feature_engine.producers.preprocessing.robust_cross_section import (
    PanelCrossSectionKernel,
)
from alphalattice.foundation.market_data_ops.returns.sector_revision_identity import (
    sector_revision_hash,
)


def test_key_and_value_identity_preserve_order_bits_and_validity() -> None:
    sessions = (date(2026, 1, 2), date(2026, 1, 2))
    listings = ("A", "B")
    key_hash = ordered_key_hash(sessions, listings)
    assert key_hash != ordered_key_hash(sessions, reversed(listings))
    values = np.array([0.0, -0.0, np.nan], dtype="<f8")
    valid = np.array([True, True, False])
    original = ordered_value_hash(
        factor_id="factor_a", key_hash=key_hash, values=values, valid=valid
    )
    changed_sign = values.copy()
    changed_sign[1] = 0.0
    assert original != ordered_value_hash(
        factor_id="factor_a", key_hash=key_hash, values=changed_sign, valid=valid
    )
    assert original != ordered_value_hash(
        factor_id="factor_a",
        key_hash=key_hash,
        values=values,
        valid=np.array([True, True, True]),
    )


def test_sector_revision_identity_is_order_stable_and_semantically_sensitive() -> None:
    observations = [
        {
            "listing_id": "B",
            "provider": "YAHOO",
            "provider_symbol": "B",
            "sector_name": "Financials",
            "sector_key": "financials",
            "payload_hash": "1" * 64,
            "evidence_hash": "2" * 64,
        },
        {
            "listing_id": "A",
            "provider": "YAHOO",
            "provider_symbol": "A",
            "sector_name": "Technology",
            "sector_key": "technology",
            "payload_hash": "3" * 64,
            "evidence_hash": "4" * 64,
        },
    ]
    original = sector_revision_hash(
        manifest_revision="a" * 64,
        observations=observations,
    )
    assert original == sector_revision_hash(
        manifest_revision="a" * 64,
        observations=reversed(observations),
    )
    changed = [dict(item) for item in observations]
    changed[0]["sector_name"] = "Industrials"
    assert original != sector_revision_hash(
        manifest_revision="a" * 64,
        observations=changed,
    )
    assert original != sector_revision_hash(
        manifest_revision="b" * 64,
        observations=observations,
    )


def test_base_manifest_preserves_nonlexical_factor_order() -> None:
    artifact = ClosureArtifactRef(
        kind="fixture",
        content_hash="2" * 64,
        physical_sha256="3" * 64,
        byte_count=1,
        row_count=1,
        uri="artifact://fixture",
    )
    key_chunk = PanelBaseKeyChunk(
        year=2026,
        key_hash="4" * 64,
        first_session=date(2026, 1, 2),
        last_session=date(2026, 1, 2),
        listing_count=1,
        session_count=1,
        artifact=artifact,
    )
    value_chunks = tuple(
        PanelBaseValueChunk(
            year=2026,
            factor_id=factor_id,
            key_hash="4" * 64,
            value_hash=value_hash * 64,
            null_count=0,
            artifact=artifact,
        )
        for factor_id, value_hash in (("z_factor", "5"), ("a_factor", "6"))
    )
    values = {
        "catalog_hash": "1" * 64,
        "history_start": date(2026, 1, 1),
        "as_of_session": date(2026, 1, 2),
        "factor_ids": ("z_factor", "a_factor"),
        "key_chunks": (key_chunk,),
        "value_chunks": value_chunks,
    }
    provisional = PanelBaseValueClosureManifest.model_construct(**values, manifest_hash="0" * 64)
    manifest = PanelBaseValueClosureManifest(
        **values,
        manifest_hash=canonical_hash(
            provisional.model_dump(mode="json", exclude={"manifest_hash"})
        ),
    )
    assert manifest.factor_ids == ("z_factor", "a_factor")
    # The source-absence extension must not add a null key to historical payloads.
    payload = manifest.model_dump(mode="json")
    assert "absent_source_keys" not in payload
    assert PanelBaseValueClosureManifest.model_validate(payload).model_dump(mode="json") == payload


def test_artifact_only_rematerializer_reproduces_fixture_bytes(tmp_path: Path) -> None:
    resolver, database, snapshot_hash = _fixture_workspace(tmp_path)
    source = PanelClosureSourceRepository(database_path=database, resolver=resolver)
    store = PanelClosureArtifactStore(resolver)
    publication = PanelClosurePublisher(resolver=resolver, source=source, store=store).publish()

    assert tuple(publication.recipes) == (snapshot_hash,)
    recipe = publication.recipes[snapshot_hash]
    result = ArtifactOnlyPanelRematerializer(resolver=resolver, store=store).rematerialize(
        recipe.recipe_hash
    )

    assert result.snapshot_hash == snapshot_hash
    assert result.logical_parity is True
    assert result.physical_parity is True
    assert len(result.chunks) == 1

    # A replay that cannot fail proves nothing. Perturb one value in the durable
    # base closure -- the *input* side, not the published chunk it is compared
    # against -- and the replay must refuse rather than reproduce the expectation.
    #
    # The refusal arrives from the closure store rather than from the numerical
    # comparison downstream, and that is the design rather than a weaker check:
    # every child is content-addressed, so the store re-derives each Parquet's
    # hash on load and a single flipped bit cannot survive far enough to be
    # compared. Asserting the message keeps this honest about *which* layer
    # caught it.
    base = store.load_model(
        category="base-manifests",
        content_hash=recipe.base_closure_hash,
        model=PanelBaseValueClosureManifest,
    )
    descriptor = base.value_chunks[0]
    frozen = store.load_parquet(category="base-values", reference=descriptor.artifact)
    bits = frozen.column("value_bits").combine_chunks().to_pylist()
    # One bit of one IEEE-754 payload: the smallest edit that changes a number.
    bits[0] = int(bits[0]) ^ 1
    tampered = frozen.set_column(
        frozen.column_names.index("value_bits"),
        "value_bits",
        pa.array(bits, type=frozen.schema.field("value_bits").type),
    )
    pq.write_table(
        tampered,
        store.physical_path(category="base-values", reference=descriptor.artifact),
        compression="zstd",
    )

    with pytest.raises(ValueError, match="physical hash does not match its reference"):
        ArtifactOnlyPanelRematerializer(resolver=resolver, store=store).rematerialize(
            recipe.recipe_hash
        )


def test_base_probe_judges_partial_groups_by_retained_cells_and_proven_scope(
    tmp_path: Path,
) -> None:
    """regression: a partially superseded batch could vouch for a wrong base source.

    The probe skipped the receipt check of a partial group and compared
    nothing in its place, and took a matching cell count as proof that the
    retained rows belonged to the batch. Every candidate artifact here is
    separately valid: the right base closure (the values the snapshot was
    built from), a wrong one (the same cells after a base value moved without
    the Panel cell being recomputed), a clip record that makes the one
    frozen batch partial, and one whose scope does not cover the frozen cell.
    """

    import duckdb

    from alphalattice.foundation.feature_engine.panels.closure_contracts import (
        PanelDerivationRecipe,
    )
    from alphalattice.foundation.feature_engine.producers.preprocessing.contracts import (
        PanelClipObservationRecord,
        PanelFactorClipObservationRecord,
    )

    resolver, database, snapshot_hash = _fixture_workspace(tmp_path)
    source = PanelClosureSourceRepository(database_path=database, resolver=resolver)
    store = PanelClosureArtifactStore(resolver)
    rematerializer = ArtifactOnlyPanelRematerializer(resolver=resolver, store=store)
    right = PanelClosurePublisher(resolver=resolver, source=source, store=store).publish()
    recipe_right = right.recipes[snapshot_hash]
    assert rematerializer.base_closure_probe_matches(recipe_right) is True

    connection = duckdb.connect(str(database), read_only=True)
    try:
        receipt = str(
            connection.execute(
                "SELECT DISTINCT materialization_receipt_hash FROM panel_factor_availability"
            ).fetchone()[0]
        )
    finally:
        connection.close()

    def clip_record(sessions: tuple[str, ...]) -> PanelClipObservationRecord:
        return PanelClipObservationRecord.seal(
            {
                "receipt_hash": receipt,
                "panel_binding_hash": str(right.snapshots[0].manifest["panel_binding_hash"]),
                "sessions": sessions,
                "ordered_sessions_hash": canonical_hash(list(sessions)),
                "ordered_listing_ids_hash": canonical_hash(list(right.snapshots[0].listing_ids)),
                "factor_ids": ("factor_a",),
                "raw_input_identity": canonical_hash([receipt, "raw"]),
                "transformed_identity": canonical_hash([receipt, "transformed"]),
                "observations": (
                    PanelFactorClipObservationRecord(
                        factor_id="factor_a",
                        finite_input_count=10 * len(sessions),
                        per_session_finite_counts=(10,) * len(sessions),
                        per_session_clipped_counts=(0,) * len(sessions),
                        boundary_identity=canonical_hash([receipt, "bounds"]),
                    ),
                ),
            }
        )

    def publish_record(record: PanelClipObservationRecord) -> None:
        target = resolver.root / "feature-panel" / "clip-observations" / f"{receipt}.json"
        target.unlink(missing_ok=True)
        resolver.publish_panel_clip_observation(
            payload=record.model_dump(mode="json"), receipt_hash=receipt
        )

    # The batch measured two sessions; the snapshot retains one of its cells.
    publish_record(clip_record(("2026-01-02", "2026-01-05")))
    assert rematerializer.base_closure_probe_matches(recipe_right) is True

    # A base value moves after the fact without the Panel cell moving: the
    # current base closure is a wrong source for this snapshot.
    connection = duckdb.connect(str(database))
    try:
        connection.execute(
            "UPDATE feature_daily_current SET factor_a = factor_a * 3.0 WHERE listing_id = 'L03'"
        )
    finally:
        connection.close()
    later = PanelClosurePublisher(resolver=resolver, source=source, store=store).publish()
    wrong_base = next(
        manifest.manifest_hash
        for manifest in later.base_manifests.values()
        if manifest.manifest_hash != recipe_right.base_closure_hash
    )
    recipe_wrong = PanelDerivationRecipe.seal(
        {
            **recipe_right.model_dump(exclude={"recipe_hash"}),
            "expected_chunks": tuple(recipe_right.expected_chunks),
            "partition_origins": recipe_right.partition_origins,
            "base_closure_hash": wrong_base,
        }
    )
    assert rematerializer.base_closure_probe_matches(recipe_wrong) is False
    # The capture selected the valid alternative -- the earlier recipe -- and
    # that recipe is actually recoverable.
    assert later.recipes[snapshot_hash] == recipe_right
    assert snapshot_hash not in later.blocked_snapshots
    assert rematerializer.rematerialize(recipe_right.recipe_hash).physical_parity is True

    # Cardinality is not membership: a record whose scope does not cover the
    # frozen cell cannot claim it, whatever the count.
    publish_record(clip_record(("2026-01-05",)))
    assert rematerializer.base_closure_probe_matches(recipe_right) is False


def test_legacy_partial_receipt_recovers_the_original_epoch_layout(tmp_path, monkeypatch):
    from alphalattice.foundation.feature_engine.producers.preprocessing.contracts import (
        PanelClipObservationRecord,
    )

    resolver, database, snapshot = _fixture_workspace(
        tmp_path, listing_count=60, original_batch_sessions=2
    )
    store = PanelClosureArtifactStore(resolver)
    source = PanelClosureSourceRepository(database_path=database, resolver=resolver)
    captured = PanelClosurePublisher(resolver=resolver, source=source, store=store).publish()
    assert snapshot in captured.recipes, captured.blocked_snapshots
    recipe = captured.recipes[snapshot]
    reader = ArtifactOnlyPanelRematerializer(resolver=resolver, store=store)
    assert reader.base_closure_probe_matches(recipe)
    rebuilt = reader.rematerialize(recipe.recipe_hash)
    assert rebuilt.logical_parity and rebuilt.physical_parity
    original = resolver.load_panel_clip_observation

    def wrong_axis(receipt):
        payload = original(receipt)
        record = PanelClipObservationRecord.seal(
            {
                **{key: value for key, value in payload.items() if key != "record_hash"},
                "ordered_listing_ids_hash": "f" * 64,
            }
        )
        return record.model_dump(mode="json")

    with monkeypatch.context() as patch:
        patch.setattr(resolver, "load_panel_clip_observation", wrong_axis)
        assert not reader.base_closure_probe_matches(recipe)
        with pytest.raises(ValueError, match="legacy_replay_layout_unproven"):
            reader.rematerialize(recipe.recipe_hash)
    with monkeypatch.context() as patch:
        patch.setattr(resolver, "load_panel_clip_observation", lambda _: None)
        assert not reader.base_closure_probe_matches(recipe)


def test_a_recipe_published_before_partition_reuse_republishes_its_exact_bytes(
    tmp_path: Path,
) -> None:
    """regression: a legacy recipe's identity survived, its serialization did not.

    The recipe model gained optional origin fields whose identity rule
    excludes absent values, so a pre-reuse recipe re-derives its recorded
    hash. The writer then serialized those absent fields as ``null`` and
    published different bytes under the historical hash; the content store
    refused the retry. The published shape must be the identity's shape.
    """

    resolver, database, snapshot_hash = _fixture_workspace(tmp_path)
    source = PanelClosureSourceRepository(database_path=database, resolver=resolver)
    store = PanelClosureArtifactStore(resolver)
    recipe = (
        PanelClosurePublisher(resolver=resolver, source=source, store=store)
        .publish()
        .recipes[snapshot_hash]
    )
    # The bytes the pre-reuse writer published: the model of that time had no
    # origin fields at all, and the fixture snapshot predates partition reuse.
    legacy = recipe.model_dump(mode="json")
    del legacy["partition_origins"]
    del legacy["row_identity_basis"]
    del legacy["membership"]
    for chunk in legacy["expected_chunks"]:
        del chunk["origin_binding_hash"]
        del chunk["cross_sections"]
    assert (
        canonical_hash({key: value for key, value in legacy.items() if key != "recipe_hash"})
        == recipe.recipe_hash
    )
    legacy_bytes = json.dumps(
        legacy, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    target = store.root / "recipes" / f"{recipe.recipe_hash}.json"
    target.write_bytes(legacy_bytes)

    republished = PanelClosurePublisher(resolver=resolver, source=source, store=store).publish()

    assert republished.recipes[snapshot_hash] == recipe
    assert target.read_bytes() == legacy_bytes
    assert (
        store.load_model(
            category="recipes", content_hash=recipe.recipe_hash, model=PanelDerivationRecipe
        )
        == recipe
    )


def test_closure_source_scope_skips_unavailable_historical_manifests(
    tmp_path: Path,
) -> None:
    resolver, database, snapshot_hash = _fixture_workspace(tmp_path)
    unavailable_hash = "f" * 64
    unavailable = resolver.root / "feature-panel" / "manifests" / f"{unavailable_hash}.json"
    unavailable.write_text("not-readable-legacy-evidence", encoding="utf-8")

    source = PanelClosureSourceRepository(
        database_path=database,
        resolver=resolver,
        snapshot_hashes=frozenset({snapshot_hash}),
    )

    inventory = source.inventory_snapshots()

    assert tuple(item.snapshot_hash for item in inventory) == (snapshot_hash,)


def _fixture_workspace(
    tmp_path: Path, *, listing_count: int = 10, original_batch_sessions: int = 1
) -> tuple[ArtifactResolver, Path, str]:
    artifact_root = tmp_path / "artifacts"
    database = tmp_path / "market-data.duckdb"
    resolver = ArtifactResolver(artifact_root)
    session = date(2026, 1, 2)
    listings = tuple(f"L{index:02d}" for index in range(listing_count))
    sectors = {
        listing: "A" if index < listing_count // 2 else "B"
        for index, listing in enumerate(listings)
    }
    manifest_revision = "1" * 64
    catalog_hash = "2" * 64
    policy_hash = "3" * 64
    spy_revision = "4" * 64
    entries = tuple(
        {
            "listing_id": listing,
            "provider": "YAHOO",
            "provider_symbol": listing,
            "sector_name": sectors[listing],
            "sector_key": sectors[listing].lower(),
            "payload_hash": canonical_hash([listing, "payload"]),
            "evidence_hash": canonical_hash([listing, "evidence"]),
        }
        for listing in listings
    )
    sector_revision = canonical_hash(
        {"source": "YAHOO_CURRENT_SECTOR", "manifest": manifest_revision, "items": entries}
    )
    binding = FeaturePanelBinding.create(
        manifest_revision=manifest_revision,
        sector_revision=sector_revision,
        catalog_hash=catalog_hash,
        spy_revision=spy_revision,
        policy_hash=policy_hash,
    )
    base_rows = [
        {
            "session_date": session,
            "listing_id": listing,
            "factor_a": float(index + (index // 5) * 0.25),
        }
        for index, listing in enumerate(listings)
    ]
    original_rows = base_rows
    if original_batch_sessions > 1:
        from dataclasses import replace

        from alphalattice.foundation.feature_engine.producers.cross_section import (
            clip_observation_record,
        )

        rng = np.random.default_rng(71)
        original_rows = [
            {
                "session_date": session + timedelta(days=3 * day),
                "listing_id": name,
                "factor_a": float(rng.normal()),
            }
            for day in range(original_batch_sessions)
            for name in listings
        ]
        base_rows = original_rows[:listing_count]
    materialization = PanelCrossSectionKernel().materialize(
        feature_rows=original_rows,
        active_listing_ids=listings,
        sector_by_listing_id=sectors,
        factor_ids=("factor_a",),
        binding=binding,
    )
    if original_batch_sessions > 1:
        record = clip_observation_record(materialization)
        resolver.publish_panel_clip_observation(
            payload=record.model_dump(mode="json"), receipt_hash=record.receipt_hash
        )
        materialization = replace(
            materialization,
            rows=materialization.rows.loc[
                materialization.rows["session_date"] == session.isoformat()
            ],
            availability=[
                row
                for row in materialization.availability
                if row["session_date"] == session.isoformat()
            ],
        )
    raw = pa.Table.from_pydict(
        {
            "session_date": pa.array([session] * len(listings), type=pa.date32()),
            "listing_id": pa.array(listings),
            "materialization_receipt_hash": pa.array(
                [materialization.receipt_hash] * len(listings)
            ),
            "factor_a": pa.array(materialization.rows["factor_a"].to_numpy()),
        }
    )
    panel = hash_panel_rows(
        raw,
        manifest_revision=manifest_revision,
        sector_revision=sector_revision,
        catalog_hash=catalog_hash,
        policy_hash=policy_hash,
        factor_ids=("factor_a",),
    )
    schema_hash = panel_schema_hash(panel.schema)
    chunk_hash = panel_chunk_hash(panel, panel_binding_hash=binding.panel_binding_hash, year=2026)
    chunk_descriptor = resolver.publish_feature_panel_chunk(
        table=panel,
        content_hash=chunk_hash,
        metadata={
            "panel_binding_hash": binding.panel_binding_hash,
            "calendar_year": "2026",
            "schema_hash": schema_hash,
        },
    )
    availability = []
    for item in materialization.availability:
        availability_hash = canonical_hash(item)
        availability.append({**item, "availability_hash": availability_hash})
    prepared = PreparedPanelChunk(
        year=2026,
        first_session=session,
        last_session=session,
        row_count=panel.num_rows,
        chunk_hash=chunk_hash,
        metadata_hash=chunk_descriptor.metadata_hash,
        uri=chunk_descriptor.uri,
    )
    content = panel_content_identity(
        binding=PanelCompositionBinding(
            manifest_revision=manifest_revision,
            sector_revision=sector_revision,
            catalog_hash=catalog_hash,
            policy_hash=policy_hash,
            panel_binding_hash=binding.panel_binding_hash,
            history_start=session,
            as_of_session=session,
            factor_ids=("factor_a",),
        ),
        chunks=(prepared,),
        load_table=lambda _chunk: panel,
        availability=availability,
    )
    manifest_identity = {
        "kind": "FeaturePanelSnapshotManifest",
        "panel_binding_hash": binding.panel_binding_hash,
        "panel_content_hash": content.panel_content_hash,
        "history_start": session.isoformat(),
        "as_of_session": session.isoformat(),
        "knowledge_cutoff_at": "2026-01-03T00:00:00+00:00",
        "temporal_identity_hash": "5" * 64,
        "active_listing_count": len(listings),
        "listing_set_hash": "6" * 64,
        "schema_hash": schema_hash,
        "chunks": [
            {
                "year": 2026,
                "first_session": session.isoformat(),
                "last_session": session.isoformat(),
                "row_count": panel.num_rows,
                "chunk_hash": chunk_hash,
                "metadata_hash": chunk_descriptor.metadata_hash,
                "uri": chunk_descriptor.uri,
            }
        ],
        "safe_summary": {
            "row_count": panel.num_rows,
            "availability_count": 1,
            "factor_catalog_summary": {"factor_a": {}},
            "lineage": {
                "catalog_hash": catalog_hash,
                "manifest_revision": manifest_revision,
                "panel_binding_hash": binding.panel_binding_hash,
                "panel_content_hash": content.panel_content_hash,
                "policy_hash": policy_hash,
                "sector_revision": sector_revision,
                "spy_revision": spy_revision,
            },
        },
    }
    snapshot_hash = canonical_hash(manifest_identity)
    resolver.publish_feature_panel_manifest(
        payload={**manifest_identity, "snapshot_hash": snapshot_hash},
        snapshot_hash=snapshot_hash,
    )
    lifecycle_identity = {
        "kind": "FeaturePanelSnapshotLifecycleProjection",
        "snapshots": [{"lifecycle": "ACTIVE", "reason": None, "snapshot_hash": snapshot_hash}],
    }
    lifecycle = {**lifecycle_identity, "projection_hash": canonical_hash(lifecycle_identity)}
    lifecycle_path = artifact_root / "feature-panel" / "snapshot-lifecycle.json"
    lifecycle_path.write_text(json.dumps(lifecycle), encoding="utf-8")
    _populate_database(
        database=database,
        session=session,
        listings=listings,
        entries=entries,
        sector_revision=sector_revision,
        manifest_revision=manifest_revision,
        catalog_hash=catalog_hash,
        policy_hash=policy_hash,
        panel_binding_hash=binding.panel_binding_hash,
        receipt_hash=materialization.receipt_hash,
        base_rows=base_rows,
        availability=availability,
    )
    return resolver, database, snapshot_hash


def _populate_database(
    *,
    database: Path,
    session: date,
    listings: tuple[str, ...],
    entries: tuple[dict[str, object], ...],
    sector_revision: str,
    manifest_revision: str,
    catalog_hash: str,
    policy_hash: str,
    panel_binding_hash: str,
    receipt_hash: str,
    base_rows: list[dict[str, object]],
    availability: list[dict[str, object]],
) -> None:
    connection = duckdb.connect(str(database))
    try:
        connection.execute(
            """
            CREATE TABLE feature_daily_current (
                listing_id VARCHAR, session_date DATE, catalog_hash VARCHAR, factor_a DOUBLE
            );
            CREATE VIEW feature_daily_runtime AS
            SELECT * FROM feature_daily_current;
            CREATE TABLE sector_classification_current (
                listing_id VARCHAR, provider VARCHAR, provider_symbol VARCHAR,
                sector_name VARCHAR, sector_key VARCHAR, payload_hash VARCHAR,
                evidence_hash VARCHAR, sector_revision VARCHAR, retrieved_at TIMESTAMP
            );
            CREATE TABLE sector_classification_revision (
                revision_id VARCHAR, listing_id VARCHAR, provider VARCHAR,
                prior_payload_hash VARCHAR, next_payload_hash VARCHAR,
                sector_revision VARCHAR, revision_kind VARCHAR, retrieved_at TIMESTAMP
            );
            CREATE TABLE panel_factor_availability (
                manifest_revision VARCHAR, sector_revision VARCHAR, catalog_hash VARCHAR,
                policy_hash VARCHAR, session_date DATE, factor_id VARCHAR,
                universe_size INTEGER, computed_count INTEGER, coverage DOUBLE,
                sector_counts_json VARCHAR, winsor_lower DOUBLE, winsor_upper DOUBLE,
                residual_median DOUBLE, residual_mad DOUBLE, status VARCHAR, reason VARCHAR,
                panel_hash VARCHAR, panel_binding_hash VARCHAR, small_sector_warning BOOLEAN,
                small_sector_names_json VARCHAR, availability_hash VARCHAR,
                materialization_receipt_hash VARCHAR
            );
            CREATE TABLE universe_manifest (
                manifest_id VARCHAR, revision_sha256 VARCHAR
            );
            CREATE TABLE universe_manifest_listing (
                manifest_id VARCHAR, listing_id VARCHAR
            );
            CREATE TABLE listing (
                listing_id VARCHAR, display_symbol VARCHAR
            );
            """
        )
        connection.execute(
            "INSERT INTO universe_manifest VALUES ('fixture-manifest', ?)",
            [manifest_revision],
        )
        connection.executemany(
            "INSERT INTO universe_manifest_listing VALUES ('fixture-manifest', ?)",
            [(listing,) for listing in listings],
        )
        connection.executemany(
            "INSERT INTO listing VALUES (?, ?)",
            [(listing, listing) for listing in listings],
        )
        connection.executemany(
            "INSERT INTO feature_daily_current VALUES (?, ?, ?, ?)",
            [(row["listing_id"], session, catalog_hash, row["factor_a"]) for row in base_rows],
        )
        connection.executemany(
            "INSERT INTO sector_classification_current VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    item["listing_id"],
                    item["provider"],
                    item["provider_symbol"],
                    item["sector_name"],
                    item["sector_key"],
                    item["payload_hash"],
                    item["evidence_hash"],
                    sector_revision,
                    "2026-01-03",
                )
                for item in entries
            ],
        )
        connection.executemany(
            "INSERT INTO sector_classification_revision "
            "VALUES (?, ?, ?, NULL, ?, ?, 'INSERTED', ?)",
            [
                (
                    canonical_hash([item["listing_id"], "revision"]),
                    item["listing_id"],
                    item["provider"],
                    item["payload_hash"],
                    sector_revision,
                    "2026-01-03",
                )
                for item in entries
            ],
        )
        item = availability[0]
        connection.execute(
            """
            INSERT INTO panel_factor_availability VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                manifest_revision,
                sector_revision,
                catalog_hash,
                policy_hash,
                session,
                "factor_a",
                item["universe_size"],
                item["computed_count"],
                item["coverage"],
                json.dumps(item["sector_counts"], sort_keys=True),
                item["winsor_lower"],
                item["winsor_upper"],
                item["residual_median"],
                item["residual_mad"],
                item["status"],
                item["reason"],
                "7" * 64,
                panel_binding_hash,
                item["small_sector_warning"],
                json.dumps(item["small_sector_names"]),
                item["availability_hash"],
                receipt_hash,
            ],
        )
    finally:
        connection.close()


def test_installed_preprocessing_catalog_and_clipping_evidence_round_trip(tmp_path) -> None:
    """The Panel's transformation is an installed method with a durable receipt.

    Evidence is measured while transforming, so it can say what the clip did;
    the assertions below check that it survives a durable round trip and that a
    tampered receipt is refused rather than read back as fact.
    """

    import json

    from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
    from alphalattice.foundation.feature_engine.panels.development_input import (
        DevelopmentFeatureOverlayColumn,
        DevelopmentFeatureOverlayManifest,
    )
    from alphalattice.foundation.feature_engine.producers.preprocessing.catalog import (
        JOINT_PRIMARY_RELATIVE_FACTOR_STD_Z,
        JOINT_PRIMARY_STATE_INTERACTION_BLOCK,
        ROBUST_SECTOR_NEUTRAL_Z,
        ROBUST_UNIVERSE_Z,
        STATE_INTERACTION_BLOCK,
        TIME_SERIES_ABSOLUTE_STATE_ROBUST,
        build_installed_panel_preprocessing_catalog,
    )
    from alphalattice.foundation.feature_engine.producers.preprocessing.contracts import (
        PanelPreprocessingError,
    )

    catalog = build_installed_panel_preprocessing_catalog()
    # Six recipes are installed for development overlays, and exactly one of
    # them is admitted for the active Panel. Installation breadth and activation
    # breadth are separate scopes; asserting only the first would let a
    # development-only recipe reach a production Panel unnoticed.
    assert set(catalog.recipe_ids) == {
        ROBUST_SECTOR_NEUTRAL_Z,
        ROBUST_UNIVERSE_Z,
        TIME_SERIES_ABSOLUTE_STATE_ROBUST,
        STATE_INTERACTION_BLOCK,
        JOINT_PRIMARY_RELATIVE_FACTOR_STD_Z,
        JOINT_PRIMARY_STATE_INTERACTION_BLOCK,
    }
    assert tuple(
        recipe_id
        for recipe_id in catalog.recipe_ids
        if catalog.capability(recipe_id).admitted_for_active_panel
    ) == (ROBUST_SECTOR_NEUTRAL_Z,)
    recipe = catalog.resolve(ROBUST_SECTOR_NEUTRAL_Z)
    # Constants are read from the numerical owner, never respelled, so the
    # installed method cannot describe a transformation the code does not run.
    assert recipe.sequence == (
        "median_mad_winsor",
        "equal_sector_demean",
        "global_robust_zscore",
    )
    with pytest.raises(PanelPreprocessingError, match="NOT_INSTALLED"):
        catalog.resolve("SOME_OTHER_METHOD")

    resolver = ArtifactResolver(tmp_path)
    payload = {
        "kind": "PanelClippingEvidence",
        "preprocessing_binding_hash": "a" * 64,
        "recipe_id": ROBUST_SECTOR_NEUTRAL_Z,
        "recipe_hash": recipe.recipe_hash,
        "panel_binding_hash": "b" * 64,
        "ordered_sessions_hash": "c" * 64,
        "ordered_listing_ids_hash": "d" * 64,
        "ordered_factor_ids": ["alpha_factor"],
        "raw_input_identity": "e" * 64,
        "transformed_identity": "f" * 64,
        "factor_records": [],
        "total_clipped_count": 0,
    }
    from alphalattice.kernel.shared_kernel.identity import canonical_hash

    evidence_hash = canonical_hash(payload)
    descriptor = resolver.publish_panel_clipping_evidence(
        payload={**payload, "evidence_hash": evidence_hash}, evidence_hash=evidence_hash
    )
    assert resolver.load_panel_clipping_evidence(descriptor.uri)["recipe_id"] == (
        ROBUST_SECTOR_NEUTRAL_Z
    )

    target = tmp_path / "feature-panel" / "clipping-evidence" / f"{evidence_hash}.json"
    tampered = json.loads(target.read_text(encoding="utf-8"))
    tampered["total_clipped_count"] = 99
    target.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="identity is invalid"):
        resolver.load_panel_clipping_evidence(descriptor.uri)

    column = DevelopmentFeatureOverlayColumn(
        factor_id="alpha_factor",
        formula_specification_hash="1" * 64,
        implementation_hash="2" * 64,
        admission_receipt_hash="3" * 64,
        preprocessing_recipe_id=ROBUST_SECTOR_NEUTRAL_Z,
        preprocessing_recipe_hash=recipe.recipe_hash,
        preprocessing_implementation_hash="4" * 64,
        raw_child_identity="5" * 64,
        preprocessing_child_identity="6" * 64,
    )
    manifest_values = {
        "kind": "DevelopmentFeatureOverlayManifest",
        "scope": "DEVELOPMENT_ONLY",
        "base_panel_snapshot_hash": "7" * 64,
        "base_panel_binding_hash": "8" * 64,
        "source_identity": "9" * 64,
        "installed_kernel_capability_hash": "a" * 64,
        "factor_catalog_revision_hash": "b" * 64,
        "preprocessing_catalog_hash": catalog.catalog_hash,
        "ordered_session_axis": ["2026-01-02"],
        "ordered_listing_axis": ["L00"],
        "ordered_candidate_axis": ["alpha_factor"],
        "columns": [column.model_dump(mode="json", exclude_none=True)],
        "parquet_sha256": "c" * 64,
        "parquet_relative_path": "values.parquet",
        "raw_parquet_sha256": "d" * 64,
        "raw_parquet_relative_path": "raw_values.parquet",
        "row_count": 1,
    }
    assert (
        DevelopmentFeatureOverlayManifest(
            **manifest_values, overlay_hash=canonical_hash(manifest_values)
        ).raw_parquet_relative_path
        == "raw_values.parquet"
    )


def _clip_record(
    receipt: str, sessions: tuple[str, ...], clipped: tuple[int, ...], *, finite: int = 10
):
    from alphalattice.foundation.feature_engine.producers.preprocessing.contracts import (
        PanelClipObservationRecord,
        PanelFactorClipObservationRecord,
    )

    return PanelClipObservationRecord.seal(
        {
            "receipt_hash": receipt,
            "panel_binding_hash": "b" * 64,
            "sessions": sessions,
            "ordered_sessions_hash": canonical_hash(list(sessions)),
            "ordered_listing_ids_hash": canonical_hash(["L00", "L01"]),
            "factor_ids": ("factor_a",),
            "raw_input_identity": canonical_hash([receipt, "raw"]),
            "transformed_identity": canonical_hash([receipt, "transformed"]),
            "observations": (
                PanelFactorClipObservationRecord(
                    factor_id="factor_a",
                    finite_input_count=finite * len(sessions),
                    per_session_finite_counts=(finite,) * len(sessions),
                    per_session_clipped_counts=clipped,
                    boundary_identity=canonical_hash([receipt, "bounds"]),
                ),
            ),
        }
    )


_D1, _D2, _D3, _D4 = "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"


def test_clipping_evidence_joins_cells_by_session_and_keeps_calendar_order() -> None:
    """regression: a replaced middle session lands in its own position.

    The original batch measured three sessions, a later batch recomputed only
    the middle one. Concatenating what each batch still owned sealed
    (1, 3, 9); the evidence must read (1, 9, 3), each count beside its own
    session's finite count, with the batch identities folded in the order
    the batches first appear on the axis. One batch owning every cell is the
    fold every earlier complete build produced.
    """

    from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog
    from alphalattice.foundation.feature_engine.producers.cross_section import (
        SectorNeutralPanelMaterializer,
    )

    materializer = SectorNeutralPanelMaterializer(FeatureCatalog.load())
    original = _clip_record("1" * 64, (_D1, _D2, _D3), (1, 2, 3))
    replacement = _clip_record("2" * 64, (_D2,), (9,), finite=20)

    whole = materializer.clipping_evidence_from_records(
        (original,),
        panel_binding_hash="b" * 64,
        owned={("1" * 64, "factor_a"): frozenset((_D1, _D2, _D3))},
        sessions=(_D1, _D2, _D3),
    )
    assert whole.factor_records[0].per_session_clipped_counts == (1, 2, 3)
    assert whole.ordered_sessions_hash == canonical_hash([original.ordered_sessions_hash])
    assert whole.factor_records[0].boundary_identity == canonical_hash(
        [original.observations[0].boundary_identity]
    )

    folded = materializer.clipping_evidence_from_records(
        # Record order is not evidence order: the replacement comes first here.
        (replacement, original),
        panel_binding_hash="b" * 64,
        owned={
            ("1" * 64, "factor_a"): frozenset((_D1, _D3)),
            ("2" * 64, "factor_a"): frozenset((_D2,)),
        },
        sessions=(_D1, _D2, _D3),
    )
    record = folded.factor_records[0]
    assert record.per_session_clipped_counts == (1, 9, 3)
    assert record.per_session_clipped_fractions == (0.1, 0.45, 0.3)
    assert (record.clipped_count, record.finite_input_count) == (13, 40)
    assert folded.ordered_sessions_hash == canonical_hash(
        [original.ordered_sessions_hash, replacement.ordered_sessions_hash]
    )
    assert folded.raw_input_identity == canonical_hash(
        [original.raw_input_identity, replacement.raw_input_identity]
    )
    assert record.boundary_identity == canonical_hash(
        [original.observations[0].boundary_identity, replacement.observations[0].boundary_identity]
    )


@pytest.mark.parametrize(
    ("owned", "sessions", "code"),
    [
        pytest.param(
            {("1" * 64, "factor_a"): frozenset((_D1, _D2, _D3, _D4))},
            (_D1, _D2, _D3, _D4),
            "CELL_NOT_OBSERVED",
            id="ownership-names-a-session-the-record-never-measured",
        ),
        pytest.param(
            {
                ("1" * 64, "factor_a"): frozenset((_D1, _D2, _D3)),
                ("2" * 64, "factor_a"): frozenset((_D2,)),
            },
            (_D1, _D2, _D3),
            "CELL_OWNED_TWICE",
            id="one-cell-attributed-to-two-batches",
        ),
        pytest.param(
            {("1" * 64, "factor_a"): frozenset((_D1, _D3))},
            (_D1, _D2, _D3),
            "CELL_UNOWNED",
            id="a-cell-of-the-panel-no-batch-owns",
        ),
        pytest.param(
            {("1" * 64, "factor_a"): frozenset((_D1, _D2, _D3))},
            (_D1, _D2),
            "CELL_OUTSIDE_AXIS",
            id="ownership-outside-the-panel-calendar",
        ),
    ],
)
def test_clipping_evidence_refuses_ownership_the_records_do_not_substantiate(
    owned, sessions, code
) -> None:
    """regression: evidence was sealed for four dates from a record covering three."""

    from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog
    from alphalattice.foundation.feature_engine.producers.cross_section import (
        SectorNeutralPanelMaterializer,
    )
    from alphalattice.foundation.feature_engine.producers.preprocessing.contracts import (
        PanelPreprocessingError,
    )

    materializer = SectorNeutralPanelMaterializer(FeatureCatalog.load())
    records = (
        _clip_record("1" * 64, (_D1, _D2, _D3), (1, 2, 3)),
        _clip_record("2" * 64, (_D2,), (9,)),
    )
    with pytest.raises(PanelPreprocessingError, match=code):
        materializer.clipping_evidence_from_records(
            records, panel_binding_hash="b" * 64, owned=owned, sessions=sessions
        )


def test_development_overlay_reuses_base_and_feeds_exact_dynamic_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement: inert Formula extensions reach Alpha without rebuilding the base Panel."""

    from types import MappingProxyType, SimpleNamespace

    from alphalattice.control.product_host.research_authoring import panel_methodology_sources
    from alphalattice.control.product_host.research_authoring.panel_methodology_sources import (
        _development_overlay_raw_values,
        _formation_observation_dollar_volume,
    )
    from alphalattice.foundation.feature_engine.panels.development_overlay import (
        DevelopmentFeatureOverlayService,
        read_raw_development_feature_overlay,
    )
    from alphalattice.foundation.feature_engine.producers.factors.registry import (
        FeatureKernelRegistry,
    )
    from alphalattice.foundation.feature_engine.producers.factors.specifications import (
        admit_factor_development_capabilities,
    )
    from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
    from alphalattice.investment.alpha_research.inputs.panel_feature_materialization import (
        PanelFeatureBoundaryError,
        materialize_panel_feature_projection,
        preflight_panel_feature_plan,
    )
    from alphalattice.investment.alpha_research.inputs.panel_feature_views import (
        SPARSE_SESSION_AMPLITUDE_REQUIRED_FORMULA_IDS,
        PanelFeatureSourceArrays,
        assemble_panel_context_arrays,
    )
    from alphalattice.protocols.research_authoring.contracts import AuthoringError

    sessions = tuple(pd.bdate_range("2024-01-02", periods=300).date)
    listings = tuple(f"L{value:02d}" for value in range(20))
    rows: list[dict[str, object]] = []
    for session_position, session in enumerate(sessions):
        for listing_position, listing in enumerate(listings):
            trend = 100.0 * np.exp(
                0.0004 * session_position
                + 0.01 * np.sin(session_position / 9.0 + listing_position / 7.0)
            )
            open_value = trend * (1.0 + 0.002 * np.sin(session_position + listing_position))
            rows.append(
                {
                    "session_date": session,
                    "listing_id": listing,
                    "open_split_adjusted": open_value,
                    "high_split_adjusted": max(open_value, trend) * 1.01,
                    "low_split_adjusted": min(open_value, trend) * 0.99,
                    "close_split_adjusted": trend,
                    "close_raw": trend,
                    "volume_raw": float(1_000 + session_position + listing_position),
                }
            )
    source = pd.DataFrame(rows)
    base_rows = source.loc[:, ["session_date", "listing_id"]].assign(mom_21=1.0)
    binding = FeaturePanelBinding.create(
        manifest_revision="1" * 64,
        sector_revision="2" * 64,
        catalog_hash="3" * 64,
        spy_revision="4" * 64,
        policy_hash="5" * 64,
    )
    receipts = tuple(
        value
        for value in admit_factor_development_capabilities()
        if value.factor_id in set(SPARSE_SESSION_AMPLITUDE_REQUIRED_FORMULA_IDS)
    )
    assert len(receipts) == len(SPARSE_SESSION_AMPLITUDE_REQUIRED_FORMULA_IDS)
    assert all(value.disposition == "ADMITTED" for value in receipts)
    assert any(value.factor_id == "session_dollar_volume" for value in receipts)
    service = DevelopmentFeatureOverlayService()
    gap_receipt = tuple(value for value in receipts if value.factor_id == "gap")
    first = service.materialize(
        source_rows=source,
        base_panel_rows=base_rows,
        base_panel_snapshot_hash="6" * 64,
        source_identity="7" * 64,
        panel_binding=binding,
        active_listing_ids=listings,
        sector_by_listing_id={
            value: ("A" if position < 10 else "B") for position, value in enumerate(listings)
        },
        output_root=tmp_path,
        admission_receipts=gap_receipt,
    )
    assert first.kind == "DevelopmentFeatureOverlayManifest"
    reusable = service.find_reusable(
        output_root=tmp_path,
        base_panel_snapshot_hash="6" * 64,
        source_identity="7" * 64,
        panel_binding=binding,
    )
    assert reusable is not None and reusable.overlay_hash == first.overlay_hash
    called: list[str] = []
    original = FeatureKernelRegistry.compute

    def counted(self: FeatureKernelRegistry, frame: pd.DataFrame, spec: object) -> pd.Series:
        called.append(str(spec.factor_id))  # type: ignore[attr-defined]
        return original(self, frame, spec)  # type: ignore[arg-type]

    monkeypatch.setattr(FeatureKernelRegistry, "compute", counted)
    final = service.materialize(
        source_rows=source,
        base_panel_rows=base_rows,
        base_panel_snapshot_hash="6" * 64,
        source_identity="7" * 64,
        panel_binding=binding,
        active_listing_ids=listings,
        sector_by_listing_id={
            value: ("A" if position < 10 else "B") for position, value in enumerate(listings)
        },
        output_root=tmp_path,
        admission_receipts=receipts,
        previous_overlay=reusable,
    )
    assert final.kind == "DevelopmentFeatureOverlayManifest"
    assert final.overlay_hash != first.overlay_hash
    assert final.base_panel_snapshot_hash == first.base_panel_snapshot_hash == "6" * 64
    assert final.base_panel_binding_hash == first.base_panel_binding_hash
    assert final.ordered_candidate_axis == tuple(
        sorted(SPARSE_SESSION_AMPLITUDE_REQUIRED_FORMULA_IDS)
    )
    assert tuple(value.factor_id for value in final.columns) == final.ordered_candidate_axis
    assert "gap" not in called
    assert set(called) == set(SPARSE_SESSION_AMPLITUDE_REQUIRED_FORMULA_IDS) - {"gap"}
    exact = service.find_exact(
        output_root=tmp_path,
        base_panel_snapshot_hash="6" * 64,
        source_identity="7" * 64,
        panel_binding=binding,
        admission_receipts=receipts,
    )
    assert exact is not None and exact.overlay_hash == final.overlay_hash
    assert (
        service.find_exact(
            output_root=tmp_path,
            base_panel_snapshot_hash="6" * 64,
            source_identity="8" * 64,
            panel_binding=binding,
            admission_receipts=receipts,
        )
        is None
    )
    _manifest, overlay_rows = read_raw_development_feature_overlay(
        output_root=tmp_path, overlay_hash=final.overlay_hash
    )
    dollar_volume = overlay_rows["session_dollar_volume"].to_numpy(dtype=float)
    expected_dollar_volume = source["close_raw"].to_numpy(dtype=float) * source[
        "volume_raw"
    ].to_numpy(dtype=float)
    np.testing.assert_array_equal(dollar_volume, expected_dollar_volume)
    authority = SimpleNamespace(
        panel_snapshot_hash="6" * 64,
        source_watermark_hash="7" * 64,
        ordered_listing_ids=listings,
        sessions=sessions,
    )
    recipe = SimpleNamespace(
        manifest_revision="1" * 64,
        sector_revision="2" * 64,
        catalog_hash="3" * 64,
        spy_revision="4" * 64,
        policy_hash="5" * 64,
        panel_binding_hash=binding.panel_binding_hash,
    )
    combined, resolved_overlay_hash = _development_overlay_raw_values(
        roots=SimpleNamespace(development_overlay_artifact_root=tmp_path),
        authority=authority,
        recipe=recipe,
        base_rows=base_rows,
        development_overlay_method_id="SESSION_OBSERVATION_FORMULA_OVERLAY",
    )
    assert resolved_overlay_hash == final.overlay_hash
    with pytest.raises(AuthoringError, match="panel_development_overlay_unresolved"):
        _development_overlay_raw_values(
            roots=SimpleNamespace(development_overlay_artifact_root=tmp_path),
            authority=SimpleNamespace(**{**vars(authority), "source_watermark_hash": "8" * 64}),
            recipe=recipe,
            base_rows=base_rows,
            development_overlay_method_id="SESSION_OBSERVATION_FORMULA_OVERLAY",
        )
    factor_ids = ("mom_21", *SPARSE_SESSION_AMPLITUDE_REQUIRED_FORMULA_IDS)
    factor_values = np.ascontiguousarray(
        combined.loc[:, list(factor_ids)].to_numpy(float).reshape(300, 20, len(factor_ids))
    )
    target = np.ascontiguousarray(np.tile(np.linspace(-1.0, 1.0, 20), (300, 1)), dtype=np.float64)
    log_returns = np.ascontiguousarray(target * 0.001, dtype=np.float64)
    simple_returns = np.ascontiguousarray(np.expm1(log_returns), dtype=np.float64)
    sector_context = np.ascontiguousarray(
        np.stack(
            [np.linspace(0.0, 1.0, 300)[:, None] + value for value in range(10)], axis=2
        ).reshape(300, 2, 5),
        dtype=np.float64,
    )
    market_context = np.ascontiguousarray(
        np.column_stack([np.sin(np.arange(300) / (value + 2.0)) for value in range(17)]),
        dtype=np.float64,
    )
    for value in (
        factor_values,
        target,
        log_returns,
        simple_returns,
        sector_context,
        market_context,
    ):
        value.setflags(write=False)
    plan = preflight_panel_feature_plan(
        source=PanelFeatureSourceArrays(
            formation_sessions=sessions,
            holding_end_sessions=tuple(value + timedelta(days=1) for value in sessions),
            ordered_listing_ids=listings,
            ordered_factor_ids=factor_ids,
            absolute_state_factor_ids=("mom_21",),
            ordered_sector_ids=("A", "B"),
            sector_by_listing_id=MappingProxyType(
                {value: ("A" if position < 10 else "B") for position, value in enumerate(listings)}
            ),
            raw_formula_values=factor_values,
            total_return_target_z=target,
            raw_log_execution_returns=log_returns,
            raw_simple_execution_returns=simple_returns,
            sector_context_values=sector_context,
            market_context_values=market_context,
            source_identity_hashes=MappingProxyType(
                {"base_panel": "6" * 64, "development_overlay": final.overlay_hash}
            ),
        ),
        selected_method_ids=(
            "RELATIVE_CONTROL",
            "SPARSE_SESSION_AMPLITUDE",
        ),
        maximum_aggregation_span=1,
    )
    sparse_view = plan.catalog.resolve("SPARSE_SESSION_AMPLITUDE")
    relative_role = next(
        value for value in sparse_view.roles if value.role_id == "RELATIVE_STOCK_CROSS_SECTION"
    )
    non_neutral_role = next(
        value for value in sparse_view.roles if value.role_id == "NON_NEUTRAL_STOCK_CROSS_SECTION"
    )
    assert non_neutral_role.source_ids == (
        "at_own_high_share_63",
        "close_to_close",
        "gap",
        "gap_amplitude",
        "previous_close_excursion_intraday_adjusted_square",
        "high_extension",
        "intraday",
        "intraday_amplitude",
        "low_extension",
        "previous_close_excursion_scaled_span_square",
        "range_position",
        "return_run_length_21",
        "previous_close_excursion_intraday_cross",
        "session_span",
        "sessions_since_252_high",
    )
    dollar_volume_position = relative_role.source_ids.index("session_dollar_volume")
    assert relative_role.source_transform_ids[dollar_volume_position] == ("current",)
    assert "session_dollar_volume" not in non_neutral_role.source_ids
    control_role = plan.catalog.resolve("RELATIVE_CONTROL").roles[0]
    assert "session_dollar_volume" not in control_role.source_ids
    raw_dollar_volume_position = factor_ids.index("session_dollar_volume")
    np.testing.assert_allclose(
        plan.formula_surface[:, :, raw_dollar_volume_position],
        factor_values[:, :, raw_dollar_volume_position],
    )
    assert plan.preflight.feature_count_by_method["SPARSE_SESSION_AMPLITUDE"] > len(factor_ids)
    sparse_feature_count = plan.preflight.feature_count_by_method["SPARSE_SESSION_AMPLITUDE"]
    assert len(plan.preflight.ordered_feature_axis_hash_by_method) == 2
    baseline_projection = materialize_panel_feature_projection(
        plan=plan,
        method_id="SPARSE_SESSION_AMPLITUDE",
        program_hash="7" * 64,
        fold_index=0,
        boundary_id="OUTER",
        training_sessions=sessions[252:280],
        transform_sessions=sessions[280:300],
    )
    assert len(baseline_projection.ordered_feature_ids) == sparse_feature_count
    assert base_rows.columns.tolist() == ["session_date", "listing_id", "mom_21"]
    assert all("::" not in value for value in overlay_rows.columns)
    transport_sessions = (date(2024, 1, 2), date(2024, 1, 3))
    transport_workspace = tmp_path / "transport-workspace"
    transport_repository = MarketDataRepository(transport_workspace)
    connection = transport_repository._connect()
    try:
        connection.execute(
            """
            CREATE TABLE raw_daily_bar_current (
                listing_id VARCHAR,
                provider VARCHAR,
                session_date DATE,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT
            )
            """
        )
        connection.executemany(
            "INSERT INTO raw_daily_bar_current VALUES (?, 'FIXTURE', ?, ?, ?, ?, ?, ?)",
            [
                ("B", transport_sessions[1], 10.0, 10.0, 10.0, 10.0, 1),
                ("A", transport_sessions[0], 10.0, 10.0, 10.0, 10.0, 1),
                ("B", transport_sessions[0], 30.0, 30.0, 30.0, 30.0, 1),
                ("A", transport_sessions[1], 20.0, 20.0, 20.0, 20.0, 2),
            ],
        )
    finally:
        connection.close()
    oracle = tuple(
        tuple(
            float(bar.close) * float(bar.volume)
            for bar in transport_repository.raw_bars(
                listing_id,
                start=transport_sessions[0],
                through=transport_sessions[-1],
            )
        )
        for listing_id in ("A", "B")
    )
    bulk_rows = transport_repository.raw_close_volume_observations(
        ("B", "A"), start=transport_sessions[0], through=transport_sessions[-1]
    )
    assert tuple(
        zip(
            bulk_rows.column("session_date").to_pylist(),
            bulk_rows.column("listing_id").to_pylist(),
            strict=True,
        )
    ) == (
        (transport_sessions[0], "A"),
        (transport_sessions[0], "B"),
        (transport_sessions[1], "A"),
        (transport_sessions[1], "B"),
    )

    class _CountingRepository(MarketDataRepository):
        open_count = 0

        def _connect(self, *, read_only: bool = False) -> duckdb.DuckDBPyConnection:
            type(self).open_count += 1
            return super()._connect(read_only=read_only)

    monkeypatch.setattr(panel_methodology_sources, "MarketDataRepository", _CountingRepository)
    arithmetic_calls = 0
    arithmetic_owner = panel_methodology_sources.raw_session_dollar_volume

    def counted_arithmetic(raw_close: np.ndarray, raw_volume: np.ndarray) -> np.ndarray:
        nonlocal arithmetic_calls
        arithmetic_calls += 1
        return arithmetic_owner(raw_close, raw_volume)

    monkeypatch.setattr(
        panel_methodology_sources,
        "raw_session_dollar_volume",
        counted_arithmetic,
    )
    before_stat = transport_repository.path.stat()
    transport_values, transport_hash = _formation_observation_dollar_volume(
        workspace=transport_workspace,
        sessions=transport_sessions,
        listing_ids=("A", "B"),
    )
    after_stat = transport_repository.path.stat()
    assert _CountingRepository.open_count == 1
    assert arithmetic_calls == 1
    assert (before_stat.st_size, before_stat.st_mtime_ns) == (
        after_stat.st_size,
        after_stat.st_mtime_ns,
    )
    np.testing.assert_array_equal(transport_values, np.asarray(oracle).T)
    np.testing.assert_array_equal(transport_values, ((10.0, 30.0), (40.0, 10.0)))
    reversed_values, reversed_axis_hash = _formation_observation_dollar_volume(
        workspace=transport_workspace,
        sessions=transport_sessions,
        listing_ids=("B", "A"),
    )
    np.testing.assert_array_equal(reversed_values, transport_values[:, ::-1])
    assert reversed_axis_hash != transport_hash

    def transport_table(
        *,
        listings: list[str] | None = None,
        sessions: list[date] | None = None,
        close: list[float] | None = None,
        volume: list[int] | None = None,
    ) -> pa.Table:
        return pa.table(
            {
                "listing_id": listings or ["A", "B", "A", "B"],
                "session_date": sessions
                or [
                    transport_sessions[0],
                    transport_sessions[0],
                    transport_sessions[1],
                    transport_sessions[1],
                ],
                "close": close or [10.0, 30.0, 20.0, 10.0],
                "volume": volume or [1, 1, 2, 1],
            }
        )

    complete_rows = transport_table()

    class _MalformedTransportRepository:
        rows: pa.Table = complete_rows

        def __init__(self, _workspace: Path) -> None:
            pass

        def raw_close_volume_observations(
            self, _listing_ids: tuple[str, ...], **_bounds: object
        ) -> pa.Table:
            return type(self).rows

    monkeypatch.setattr(
        panel_methodology_sources,
        "MarketDataRepository",
        _MalformedTransportRepository,
    )
    malformed_rows = (
        complete_rows.slice(0, 3),
        transport_table(listings=["A", "B", "A", "A"]),
        transport_table(
            sessions=[
                transport_sessions[0],
                transport_sessions[0],
                transport_sessions[1],
                date(2024, 1, 4),
            ]
        ),
        transport_table(listings=["A", "C", "A", "B"]),
    )
    for malformed in malformed_rows:
        _MalformedTransportRepository.rows = malformed
        with pytest.raises(AuthoringError, match="observation_dollar_volume_axis"):
            _formation_observation_dollar_volume(
                workspace=transport_workspace,
                sessions=transport_sessions,
                listing_ids=("A", "B"),
            )
    for changed in (
        transport_table(close=[11.0, 30.0, 20.0, 10.0]),
        transport_table(volume=[1, 1, 3, 1]),
    ):
        _MalformedTransportRepository.rows = changed
        _changed_values, changed_hash = _formation_observation_dollar_volume(
            workspace=transport_workspace,
            sessions=transport_sessions,
            listing_ids=("A", "B"),
        )
        assert changed_hash != transport_hash
    _MalformedTransportRepository.rows = complete_rows
    _sector_transport, market_transport = assemble_panel_context_arrays(
        formation_sessions=transport_sessions,
        holding_end_sessions=(date(2024, 1, 4), date(2024, 1, 5)),
        ordered_listing_ids=("A", "B"),
        ordered_sector_ids=("S",),
        sector_by_listing_id=MappingProxyType({"A": "S", "B": "S"}),
        raw_log_execution_returns=np.zeros((2, 2), dtype=np.float64),
        raw_simple_execution_returns=np.zeros((2, 2), dtype=np.float64),
        sector_state_values=np.zeros((2, 1, 4), dtype=np.float64),
        market_interaction_state_values=np.zeros((2, 2), dtype=np.float64),
        observation_returns=np.asarray(((0.1, -0.1), (-0.2, 0.3)), dtype=np.float64),
        observation_dollar_volume=np.asarray(((10.0, 30.0), (40.0, 10.0))),
    )
    np.testing.assert_allclose(market_transport[:, -1], (0.25, 0.2), rtol=0.0, atol=0.0)
    missing_formula_values = np.ascontiguousarray(factor_values[:, :, :-1])
    missing_formula_values.setflags(write=False)
    with pytest.raises(
        PanelFeatureBoundaryError,
        match="sparse_session_amplitude_required_formula_missing",
    ):
        preflight_panel_feature_plan(
            source=PanelFeatureSourceArrays(
                formation_sessions=sessions,
                holding_end_sessions=tuple(value + timedelta(days=1) for value in sessions),
                ordered_listing_ids=listings,
                ordered_factor_ids=factor_ids[:-1],
                absolute_state_factor_ids=("mom_21",),
                ordered_sector_ids=("A", "B"),
                sector_by_listing_id=plan.source.sector_by_listing_id,
                raw_formula_values=missing_formula_values,
                total_return_target_z=target,
                raw_log_execution_returns=log_returns,
                raw_simple_execution_returns=simple_returns,
                sector_context_values=sector_context,
                market_context_values=market_context,
                source_identity_hashes=plan.source.source_identity_hashes,
            ),
            selected_method_ids=("RELATIVE_CONTROL", "SPARSE_SESSION_AMPLITUDE"),
            maximum_aggregation_span=1,
        )


def test_panel_snapshot_identity_binds_the_preprocessing_method(tmp_path) -> None:
    """From a Panel, the method that built it is reachable and verified.

    The marker is the edge that stops the clipping receipt being an orphan, and
    the lineage it yields is what a snapshot binds -- so two Panels whose
    numbers coincide are still distinguishable when different methods produced
    them. A marker whose evidence describes another Panel is refused rather
    than bound.
    """

    from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
    from alphalattice.foundation.feature_engine.producers.preprocessing.catalog import (
        ROBUST_SECTOR_NEUTRAL_Z,
        build_installed_panel_preprocessing_catalog,
    )
    from alphalattice.foundation.feature_engine.producers.preprocessing.contracts import (
        PanelClippingEvidence,
        PanelFactorClippingRecord,
        PanelPreprocessingBinding,
        build_panel_preprocessing_seal_marker,
    )
    from alphalattice.foundation.feature_engine.publication.snapshots import (
        resolve_panel_preprocessing_lineage,
    )
    from alphalattice.kernel.shared_kernel.identity import canonical_hash

    catalog = build_installed_panel_preprocessing_catalog()
    recipe, adapter = catalog.resolve_executable(ROBUST_SECTOR_NEUTRAL_Z)
    panel_binding_hash = "a" * 64

    binding_values = {
        "kind": "PanelPreprocessingBinding",
        "recipe_id": recipe.recipe_id,
        "recipe_hash": recipe.recipe_hash,
        "implementation_id": adapter.implementation_id,
        "implementation_binding_hash": (
            catalog.implementation_binding(ROBUST_SECTOR_NEUTRAL_Z).implementation_binding_hash
        ),
        "implementation": catalog.implementation_binding(ROBUST_SECTOR_NEUTRAL_Z).model_dump(
            mode="json"
        ),
        "catalog_hash": catalog.catalog_hash,
        "policy_hash": "b" * 64,
        "panel_binding_hash": panel_binding_hash,
    }
    binding = PanelPreprocessingBinding(
        **{
            **binding_values,
            "implementation": catalog.implementation_binding(ROBUST_SECTOR_NEUTRAL_Z),
        },
        binding_hash=canonical_hash(binding_values),
    )
    record_values = {
        "kind": "PanelFactorClippingRecord",
        "factor_id": "alpha_factor",
        "finite_input_count": 10,
        "clipped_count": 1,
        "clipped_fraction": 0.1,
        "per_session_clipped_counts": (1,),
        "per_session_clipped_fractions": (0.1,),
        "boundary_identity": "c" * 64,
    }
    record = PanelFactorClippingRecord(**record_values, record_hash=canonical_hash(record_values))
    evidence_values = {
        "kind": "PanelClippingEvidence",
        "preprocessing_binding_hash": binding.binding_hash,
        "recipe_id": recipe.recipe_id,
        "recipe_hash": recipe.recipe_hash,
        "panel_binding_hash": panel_binding_hash,
        "ordered_sessions_hash": "d" * 64,
        "ordered_listing_ids_hash": "e" * 64,
        "ordered_factor_ids": ("alpha_factor",),
        "raw_input_identity": "0" * 64,
        "transformed_identity": "1" * 64,
        "factor_records": tuple(value.model_dump(mode="json") for value in (record,)),
        "total_clipped_count": 1,
    }
    evidence = PanelClippingEvidence(
        **{**evidence_values, "factor_records": (record,)},
        evidence_hash=canonical_hash(evidence_values),
    )
    panel_content_hash = "9" * 64
    marker = build_panel_preprocessing_seal_marker(
        binding=binding, evidence=evidence, panel_content_hash=panel_content_hash
    )

    resolver = ArtifactResolver(tmp_path)

    # Before the binding is durable, the marker and the receipt still agree with
    # each other about its hash -- they were built from the same object. That
    # agreement is exactly what used to pass for verification, so resolution must
    # refuse here rather than accept a graph whose middle node does not exist.
    resolver.publish_panel_clipping_evidence(
        payload=evidence.model_dump(mode="json"), evidence_hash=evidence.evidence_hash
    )
    resolver.publish_panel_preprocessing_marker(
        payload=marker.model_dump(mode="json"), panel_content_hash=panel_content_hash
    )
    with pytest.raises(ValueError, match="binding named by the marker is missing"):
        resolve_panel_preprocessing_lineage(resolver, panel_content_hash=panel_content_hash)

    resolver.publish_panel_preprocessing_binding(
        payload=binding.model_dump(mode="json"), binding_hash=binding.binding_hash
    )
    lineage = resolve_panel_preprocessing_lineage(resolver, panel_content_hash=panel_content_hash)
    assert lineage is not None
    assert lineage["recipe_id"] == ROBUST_SECTOR_NEUTRAL_Z
    assert lineage["implementation_id"] == adapter.implementation_id
    assert lineage["clipping_evidence_hash"] == evidence.evidence_hash
    assert lineage["preprocessing_binding_hash"] == binding.binding_hash
    assert lineage["policy_hash"] == "b" * 64

    # A binding that exists but describes a different Panel is refused too: the
    # edge is checked, not merely present.
    foreign_values = {**binding_values, "panel_binding_hash": "8" * 64}
    foreign = PanelPreprocessingBinding(
        **{
            **foreign_values,
            "implementation": catalog.implementation_binding(ROBUST_SECTOR_NEUTRAL_Z),
        },
        binding_hash=canonical_hash(foreign_values),
    )
    foreign_marker_values = {
        key: value for key, value in marker.model_dump(mode="json").items() if key != "marker_hash"
    }
    foreign_marker_values["preprocessing_binding_hash"] = foreign.binding_hash
    foreign_marker_values["panel_content_hash"] = "3" * 64
    resolver.publish_panel_preprocessing_binding(
        payload=foreign.model_dump(mode="json"), binding_hash=foreign.binding_hash
    )
    resolver.publish_panel_preprocessing_marker(
        payload={
            **foreign_marker_values,
            "marker_hash": canonical_hash(foreign_marker_values),
        },
        panel_content_hash="3" * 64,
    )
    with pytest.raises(ValueError, match="contradicts the marker"):
        resolve_panel_preprocessing_lineage(resolver, panel_content_hash="3" * 64)

    # A second build of the same content may carry a different clipping receipt,
    # but it may not claim a different method: that is the one disagreement the
    # store refuses.
    contradicting = marker.model_dump(mode="json")
    contradicting["recipe_id"] = "SOME_OTHER_METHOD"
    contradicting["marker_hash"] = canonical_hash(
        {key: value for key, value in contradicting.items() if key != "marker_hash"}
    )
    with pytest.raises(ValueError, match="contradicts the sealed method"):
        resolver.publish_panel_preprocessing_marker(
            payload=contradicting, panel_content_hash=panel_content_hash
        )

    # A Panel built before the seam existed carries no preprocessing identity and
    # does not acquire one by being read.
    assert resolve_panel_preprocessing_lineage(resolver, panel_content_hash="f" * 64) is None


def test_preprocessing_lineage_terminates_in_resolvable_implementation_facts() -> None:
    """The implementation is a document a reader can open, not a leaf hash.

    A binding that carries only ``implementation_binding_hash`` lets a reader
    confirm two documents quote the same string and nothing else: not which
    modules were hashed, not which versions were pinned, not whether the closure
    named has anything to do with this Panel. So the child travels with the
    binding, and a substituted one is refused rather than followed.
    """

    from alphalattice.foundation.feature_engine.producers.preprocessing.catalog import (
        ROBUST_SECTOR_NEUTRAL_Z,
        build_installed_panel_preprocessing_catalog,
    )
    from alphalattice.foundation.feature_engine.producers.preprocessing.contracts import (
        PanelPreprocessingBinding,
        PanelPreprocessingImplementationBinding,
    )

    catalog = build_installed_panel_preprocessing_catalog()
    recipe = catalog.resolve(ROBUST_SECTOR_NEUTRAL_Z)
    installed = catalog.implementation_binding(ROBUST_SECTOR_NEUTRAL_Z)
    values = {
        "kind": "PanelPreprocessingBinding",
        "recipe_id": recipe.recipe_id,
        "recipe_hash": recipe.recipe_hash,
        "implementation_id": installed.implementation_id,
        "implementation_binding_hash": installed.implementation_binding_hash,
        "implementation": installed.model_dump(mode="json"),
        "catalog_hash": catalog.catalog_hash,
        "policy_hash": "b" * 64,
        "panel_binding_hash": "a" * 64,
    }
    binding = PanelPreprocessingBinding(
        **{**values, "implementation": installed}, binding_hash=canonical_hash(values)
    )
    # The facts are reachable from the binding alone.
    assert binding.implementation.implementation_owners
    assert binding.implementation.implementation_content_hash == (
        installed.implementation_content_hash
    )
    # The environment is provenance, never part of this identity (LAWS.md ID6).
    assert binding.implementation.numerical_environment_hash is None

    # A child whose content was swapped while the quoted hash stayed put is the
    # substitution this edge exists to catch.
    substituted = PanelPreprocessingImplementationBinding.create(
        implementation_id=installed.implementation_id,
        implementation_owners=installed.implementation_owners,
        implementation_content_hash="7" * 64,
    )
    swapped = {**values, "implementation": substituted.model_dump(mode="json")}
    # Raised inside a model validator, so pydantic wraps it; the code is what
    # matters and it names the substituted child rather than a generic failure.
    with pytest.raises(ValueError, match="IMPLEMENTATION_CHILD_MISMATCH"):
        PanelPreprocessingBinding(
            **{**swapped, "implementation": substituted},
            binding_hash=canonical_hash(values),
        )
