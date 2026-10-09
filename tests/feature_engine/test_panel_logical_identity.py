from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
    PanelClosureArtifactStore,
)
from alphalattice.foundation.feature_engine.panels.logical_contracts import PanelLogicalRevision
from alphalattice.foundation.feature_engine.panels.logical_identity import (
    PanelLogicalArtifactStore,
    PanelLogicalIdentityPublisher,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash


def test_physical_encoding_and_provenance_do_not_change_logical_identity(
    tmp_path: Path,
) -> None:
    resolver = ArtifactResolver(tmp_path / "artifacts")
    first = _publish_fixture(
        resolver,
        suffix="first",
        values=(0.0, -0.0, 1.0, np.nan),
        valid=(True, True, True, True),
        metadata_note="physical-a",
    )
    second = _publish_fixture(
        resolver,
        suffix="second",
        values=(0.0, -0.0, 1.0, np.nan),
        valid=(True, True, True, True),
        metadata_note="physical-b",
    )
    store = PanelLogicalArtifactStore(PanelClosureArtifactStore(resolver))
    publisher = PanelLogicalIdentityPublisher(resolver=resolver, store=store)

    first_marker = publisher.publish_snapshot(first, published_at=_now())
    second_marker = publisher.publish_snapshot(second, published_at=_now())

    assert first_marker.logical_panel_hash == second_marker.logical_panel_hash
    assert first_marker.physical_materialization_hash != second_marker.physical_materialization_hash
    assert first_marker.retention_disposition == "PHYSICAL_RETENTION_REQUIRED"
    assert publisher.publish_snapshot(first, published_at=_later()) == first_marker


@pytest.mark.parametrize(
    ("changed_values", "changed_valid"),
    [
        ((-0.0, -0.0, 1.0, np.nan), (True, True, True, True)),
        ((0.0, -0.0, 1.0, np.nan), (True, True, True, False)),
        ((0.0, -0.0, 2.0, np.nan), (True, True, True, True)),
    ],
)
def test_signed_zero_null_mask_and_values_rotate_logical_identity(
    tmp_path: Path,
    changed_values: tuple[float, ...],
    changed_valid: tuple[bool, ...],
) -> None:
    resolver = ArtifactResolver(tmp_path / "artifacts")
    baseline = _publish_fixture(
        resolver,
        suffix="baseline",
        values=(0.0, -0.0, 1.0, np.nan),
        valid=(True, True, True, True),
    )
    changed = _publish_fixture(
        resolver,
        suffix="changed",
        values=changed_values,
        valid=changed_valid,
    )
    publisher = PanelLogicalIdentityPublisher(
        resolver=resolver,
        store=PanelLogicalArtifactStore(PanelClosureArtifactStore(resolver)),
    )

    assert publisher.publish_snapshot(baseline, published_at=_now()).logical_panel_hash != (
        publisher.publish_snapshot(changed, published_at=_now()).logical_panel_hash
    )


def test_policy_rotates_revision_without_changing_logical_content_manifest(
    tmp_path: Path,
) -> None:
    resolver = ArtifactResolver(tmp_path / "artifacts")
    first = _publish_fixture(
        resolver,
        suffix="policy-a",
        values=(1.0, 2.0, 3.0, 4.0),
        valid=(True, True, True, True),
        policy_hash="4" * 64,
    )
    second = _publish_fixture(
        resolver,
        suffix="policy-b",
        values=(1.0, 2.0, 3.0, 4.0),
        valid=(True, True, True, True),
        policy_hash="5" * 64,
    )
    store = PanelLogicalArtifactStore(PanelClosureArtifactStore(resolver))
    publisher = PanelLogicalIdentityPublisher(resolver=resolver, store=store)
    first_marker = publisher.publish_snapshot(first, published_at=_now())
    second_marker = publisher.publish_snapshot(second, published_at=_now())
    first_revision = store.load_model(
        "revisions", first_marker.logical_panel_hash, PanelLogicalRevision
    )
    second_revision = store.load_model(
        "revisions", second_marker.logical_panel_hash, PanelLogicalRevision
    )

    assert first_revision.logical_manifest_hash == second_revision.logical_manifest_hash
    assert first_revision.logical_panel_hash != second_revision.logical_panel_hash


def test_logical_semantic_index_binds_panel_factor_axis_and_selected_sessions(
    tmp_path: Path,
) -> None:
    resolver = ArtifactResolver(tmp_path / "artifacts")
    manifest = _publish_fixture(
        resolver,
        suffix="index",
        values=(1.0, 2.0, 3.0, 4.0),
        valid=(True, True, True, True),
    )
    store = PanelLogicalArtifactStore(PanelClosureArtifactStore(resolver))
    publisher = PanelLogicalIdentityPublisher(resolver=resolver, store=store)
    marker = publisher.publish_snapshot(manifest, published_at=_now())
    index = publisher.semantic_index(marker)

    selected = (date(2026, 1, 2),)
    assert index.slice_hash(selected) == index.slice_hash(selected)
    assert index.logical_panel_hash == marker.logical_panel_hash
    assert index.factor_ids == ("factor_a",)
    with pytest.raises(ValueError, match="outside"):
        index.slice_hash((date(2025, 1, 2),))


def test_legacy_safe_summary_may_be_subset_of_physical_factor_axis(
    tmp_path: Path,
) -> None:
    resolver = ArtifactResolver(tmp_path / "artifacts")
    manifest = _publish_fixture(
        resolver,
        suffix="legacy-summary-subset",
        values=(1.0, 2.0, 3.0, 4.0),
        valid=(True, True, True, True),
        include_legacy_extra_factor=True,
    )
    publisher = PanelLogicalIdentityPublisher(
        resolver=resolver,
        store=PanelLogicalArtifactStore(PanelClosureArtifactStore(resolver)),
    )

    marker = publisher.publish_snapshot(manifest, published_at=_now())
    index = publisher.semantic_index(marker)

    assert index.factor_ids == ("factor_a", "legacy_factor")


def test_current_pointer_compare_and_swap_is_fail_closed(tmp_path: Path) -> None:
    resolver = ArtifactResolver(tmp_path / "artifacts")
    first = _publish_fixture(
        resolver,
        suffix="one",
        values=(1.0, 2.0, 3.0, 4.0),
        valid=(True, True, True, True),
    )
    second = _publish_fixture(
        resolver,
        suffix="two",
        values=(1.0, 2.0, 3.0, 5.0),
        valid=(True, True, True, True),
    )
    store = PanelLogicalArtifactStore(PanelClosureArtifactStore(resolver))
    publisher = PanelLogicalIdentityPublisher(resolver=resolver, store=store)
    first_marker = publisher.publish_snapshot(first, published_at=_now())
    second_marker = publisher.publish_snapshot(second, published_at=_now())

    store.compare_and_swap_current(expected_marker_hash=None, next_marker=first_marker)
    with pytest.raises(ValueError, match="identity_migration_cas_failed"):
        store.compare_and_swap_current(expected_marker_hash="f" * 64, next_marker=second_marker)
    assert store.current_marker() == first_marker


def _publish_fixture(
    resolver: ArtifactResolver,
    *,
    suffix: str,
    values: tuple[float, ...],
    valid: tuple[bool, ...],
    metadata_note: str = "same",
    policy_hash: str = "4" * 64,
    catalog_hash: str = "3" * 64,
    include_legacy_extra_factor: bool = False,
) -> str:
    sessions = (date(2026, 1, 2), date(2026, 1, 5))
    listings = ("L01", "L02")
    factor_values = pa.array(values, mask=np.logical_not(valid), type=pa.float64())
    columns: dict[str, object] = {
        "manifest_revision": ["1" * 64] * 4,
        "sector_revision": ["2" * 64] * 4,
        "catalog_hash": [catalog_hash] * 4,
        "policy_hash": [policy_hash] * 4,
        "session_date": pa.array(
            [sessions[0], sessions[0], sessions[1], sessions[1]], type=pa.date32()
        ),
        "listing_id": [listings[0], listings[1], listings[0], listings[1]],
        "row_hash": [canonical_hash([suffix, index]) for index in range(4)],
        "materialization_receipt_hash": ["6" * 64] * 4,
        "factor_a": factor_values,
    }
    if include_legacy_extra_factor:
        columns["legacy_factor"] = pa.array((4.0, 3.0, 2.0, 1.0), type=pa.float64())
    table = pa.Table.from_pydict(columns)
    chunk_hash = canonical_hash(["chunk", suffix])
    descriptor = resolver.publish_feature_panel_chunk(
        table=table,
        content_hash=chunk_hash,
        metadata={
            "panel_binding_hash": canonical_hash(["binding", suffix]),
            "calendar_year": "2026",
            "schema_hash": "7" * 64,
            "note": metadata_note,
        },
    )
    safe_summary = {
        "availability_count": 2,
        "factor_catalog_summary": {
            "factor_a": {
                "available_session_count": 2,
                "coverage_failure_count": 0,
                "null_ratio": 0.0,
                "small_sector_warning_count": 0,
                "unavailable_session_count": 0,
                "zero_mad_count": 0,
            }
        },
        "lineage": {
            "manifest_revision": "1" * 64,
            "sector_revision": "2" * 64,
            "catalog_hash": catalog_hash,
            "policy_hash": policy_hash,
            "spy_revision": "8" * 64,
        },
    }
    identity = {
        "kind": "FeaturePanelSnapshotManifest",
        "panel_binding_hash": canonical_hash(["binding", suffix]),
        "panel_content_hash": canonical_hash(["legacy-content", suffix]),
        "history_start": sessions[0].isoformat(),
        "as_of_session": sessions[-1].isoformat(),
        "knowledge_cutoff_at": _now().isoformat(),
        "temporal_identity_hash": "9" * 64,
        "active_listing_count": 2,
        "listing_set_hash": canonical_hash(listings),
        "schema_hash": "7" * 64,
        "chunks": [
            {
                "year": 2026,
                "first_session": sessions[0].isoformat(),
                "last_session": sessions[-1].isoformat(),
                "row_count": 4,
                "chunk_hash": chunk_hash,
                "metadata_hash": descriptor.metadata_hash,
                "uri": descriptor.uri,
            }
        ],
        "safe_summary": safe_summary,
    }
    snapshot_hash = canonical_hash(identity)
    descriptor = resolver.publish_feature_panel_manifest(
        payload={**identity, "snapshot_hash": snapshot_hash}, snapshot_hash=snapshot_hash
    )
    return descriptor.uri


def _now() -> datetime:
    return datetime(2026, 8, 7, tzinfo=UTC)


def _later() -> datetime:
    return datetime(2026, 8, 7, 0, 1, tzinfo=UTC)


def test_the_write_batch_does_not_enter_a_panels_derivation_binding(tmp_path: Path) -> None:
    """Regression: the Feature closure's head is decided by how many listings each
    transition writes, an execution parameter, and it entered every Panel's derivation binding;
    two closures of one Panel that differ only in their heads publish one binding, without one."""

    from alphalattice.foundation.feature_engine.panels.logical_contracts import (
        PanelDerivationBinding,
    )

    def published(head: str, root: Path) -> PanelDerivationBinding:
        resolver = ArtifactResolver(root / "artifacts")
        manifest = _publish_fixture(resolver, suffix="batch", values=(1.0,) * 4, valid=(True,) * 4)
        ledger = SimpleNamespace(
            panel_binding=lambda _snapshot: SimpleNamespace(
                catalog_hash="3" * 64, sector_map_hash=None, closure_head_hash=head
            )
        )
        store = PanelLogicalArtifactStore(PanelClosureArtifactStore(resolver))
        publisher = PanelLogicalIdentityPublisher(
            resolver=resolver,
            store=store,
            closure_ledger=ledger,  # type: ignore[arg-type]
        )
        marker = publisher.publish_snapshot(manifest, published_at=_now())
        binding: PanelDerivationBinding = store.load_model(
            "derivation-bindings", marker.derivation_binding_hash, PanelDerivationBinding
        )
        return binding

    eight, sixteen = published("1" * 64, tmp_path / "eight"), published("2" * 64, tmp_path / "16")
    assert eight.closure_head_hash is None
    assert eight == sixteen
