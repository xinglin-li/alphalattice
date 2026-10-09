from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import pytest

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.feature_engine.panels.retention_residue import (
    PanelRetentionResidueOwner,
)
from alphalattice.foundation.feature_engine.storage.repositories import PanelStateRepository
from alphalattice.kernel.shared_kernel.identity import canonical_hash

NOW = datetime(2026, 8, 7, 15, 0, tzinfo=UTC)


def test_residue_cleanup_keeps_only_rooted_generation_and_reports_physical_truth(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "runtime"
    artifact_root = workspace / "artifacts"
    workspace.mkdir(parents=True)
    resolver = ArtifactResolver(artifact_root)
    active_identity: dict[str, object] = {"chunks": []}
    active_hash = canonical_hash(active_identity)
    evicted_hash = canonical_hash("evicted-panel")
    missing_hash = canonical_hash("missing-panel")
    active_manifest = artifact_root / "feature-panel" / "manifests" / f"{active_hash}.json"
    active_manifest.parent.mkdir(parents=True)
    active_manifest.write_text(
        json.dumps(
            {**active_identity, "snapshot_hash": active_hash},
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    eviction_plan_hash = canonical_hash("eviction-plan")
    availability = artifact_root / "storage-governance" / "availability"
    availability.mkdir(parents=True)
    (availability / f"{eviction_plan_hash}.json").write_text(
        json.dumps(
            {
                "eviction_plan_hash": eviction_plan_hash,
                "entries": [
                    {
                        "relative_path": (f"artifacts/feature-panel/manifests/{evicted_hash}.json"),
                        "availability": "EVICTED_BY_RETENTION",
                    }
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    _seed_database(
        workspace / "market-data.duckdb",
        active_hash=active_hash,
        evicted_hash=evicted_hash,
        missing_hash=missing_hash,
        resolver=resolver,
    )
    owner = PanelRetentionResidueOwner(workspace=workspace, resolver=resolver)

    receipt = owner.close(
        protected_panel_snapshot_hashes=(active_hash,),
        cutover_acceptance_marker_hash=canonical_hash("acceptance"),
        completed_at=NOW,
    )

    assert len(receipt.retained_generations) == 1
    assert receipt.retained_generations[0].panel_binding_hash == canonical_hash("active-binding")
    assert len(receipt.retired_generations) == 1
    states = {str(value["snapshot_hash"]): value for value in owner.observe_snapshot_states()}
    assert states[active_hash]["physical_availability"] == "AVAILABLE"
    assert states[evicted_hash] == {
        "snapshot_hash": evicted_hash,
        "lifecycle": "SUPERSEDED",
        "reason": "retained as semantic history",
        "physical_availability": "EVICTED_BY_RETENTION",
        "eviction_plan_hash": eviction_plan_hash,
    }
    assert states[missing_hash]["physical_availability"] == "MISSING_OR_TAMPERED"
    assert states[missing_hash]["eviction_plan_hash"] is None
    assert owner.receipt_for_acceptance(receipt.cutover_acceptance_marker_hash) == receipt


def _seed_database(
    path: Path,
    *,
    active_hash: str,
    evicted_hash: str,
    missing_hash: str,
    resolver: ArtifactResolver,
) -> None:
    connection = duckdb.connect(str(path))
    try:
        connection.execute(
            """
            CREATE TABLE panel_factor_availability (
                manifest_revision VARCHAR,
                sector_revision VARCHAR,
                catalog_hash VARCHAR,
                policy_hash VARCHAR,
                session_date DATE,
                factor_id VARCHAR,
                panel_binding_hash VARCHAR,
                availability_hash VARCHAR
            )
            """
        )
        for suffix, binding in (
            ("active", canonical_hash("active-binding")),
            ("stale", canonical_hash("stale-binding")),
        ):
            connection.execute(
                "INSERT INTO panel_factor_availability VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    canonical_hash([suffix, "manifest"]),
                    canonical_hash([suffix, "sector"]),
                    canonical_hash([suffix, "catalog"]),
                    canonical_hash([suffix, "policy"]),
                    date(2026, 8, 6),
                    "factor_a",
                    binding,
                    canonical_hash([suffix, "availability"]),
                ],
            )
        connection.execute(
            """
            CREATE TABLE feature_panel_snapshot_manifest (
                snapshot_hash VARCHAR PRIMARY KEY,
                manifest_uri VARCHAR,
                panel_binding_hash VARCHAR,
                lifecycle VARCHAR,
                lifecycle_reason VARCHAR
            )
            """
        )
        rows = (
            (
                active_hash,
                resolver.feature_panel_manifest_uri(active_hash),
                canonical_hash("active-binding"),
                "ACTIVE",
                None,
            ),
            (
                evicted_hash,
                resolver.feature_panel_manifest_uri(evicted_hash),
                canonical_hash("stale-binding"),
                "SUPERSEDED",
                "retained as semantic history",
            ),
            (
                missing_hash,
                resolver.feature_panel_manifest_uri(missing_hash),
                canonical_hash("stale-binding"),
                "QUARANTINED",
                "tamper investigation",
            ),
        )
        connection.executemany(
            "INSERT INTO feature_panel_snapshot_manifest VALUES (?, ?, ?, ?, ?)", rows
        )
    finally:
        connection.close()


@pytest.mark.parametrize("store", ("absent", "corrupt"))
def test_retention_decision_discovery_is_empty_only_before_a_store_exists(
    tmp_path: Path, store
) -> None:
    "regression: optional first-use discovery never hides an existing unreadable store."
    import duckdb

    panel = PanelStateRepository(tmp_path / "workspace")
    assert not panel.path.exists()
    if store == "absent":
        assert panel.feature_input_raw_retention_decisions() == ()
        assert not panel.path.exists()
    else:
        panel.path.write_bytes(b"synthetic invalid database")
        with pytest.raises(duckdb.IOException):
            panel.feature_input_raw_retention_decisions()
