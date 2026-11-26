"""Post-cutover cleanup for operational Panel availability and physical truth."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import duckdb
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.control.workspace_runtime.database import open_workspace_database
from alphalattice.foundation.feature_engine.contracts import PANEL_ROW_IDENTITY_BY_CROSS_SECTION
from alphalattice.foundation.feature_engine.panels.artifacts import manifest_partition_origins
from alphalattice.kernel.shared_kernel.identity import canonical_hash


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class PanelAvailabilityGeneration(_Contract):
    """The availability rows one build stamped, under one key.

    Rows under the binding rule are keyed by the build's manifest and sector
    revisions; rows under the session-cross-section rule by the cross-section
    each session's members define, so the revisions are absent and the rule
    is named. Both are excluded from the receipt identity when absent, so a
    receipt written before the second table re-derives its recorded hash.
    """

    manifest_revision: str | None = None
    sector_revision: str | None = None
    catalog_hash: str
    policy_hash: str
    panel_binding_hash: str
    row_count: int = Field(ge=0)
    ordered_availability_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    availability_basis: str | None = None


class PanelRetentionResidueReceipt(_Contract):
    """Record retained and retired availability generations after cutover."""

    kind: Literal["PanelRetentionResidueReceipt"] = "PanelRetentionResidueReceipt"
    cutover_acceptance_marker_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    protected_panel_snapshot_hashes: tuple[str, ...]
    retained_generations: tuple[PanelAvailabilityGeneration, ...]
    retired_generations: tuple[PanelAvailabilityGeneration, ...]
    snapshot_state_count: int = Field(ge=1)
    lifecycle_projection_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    completed_at: datetime
    receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_receipt(self) -> PanelRetentionResidueReceipt:
        """Verify the receipt clock, protected roots, and content identity."""
        if self.completed_at.tzinfo is None or self.completed_at.utcoffset() is None:
            raise ValueError("Panel residue receipt clock must be timezone-aware")
        if self.protected_panel_snapshot_hashes != tuple(
            sorted(set(self.protected_panel_snapshot_hashes))
        ):
            raise ValueError("Panel residue protected roots must be unique and sorted")
        if self.receipt_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"receipt_hash"}, exclude_none=True)
        ):
            raise ValueError("Panel residue receipt identity is invalid")
        return self


def _table_exists(connection: duckdb.DuckDBPyConnection, name: str) -> bool:
    row = connection.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [name]
    ).fetchone()
    return bool(row and int(row[0]))


class PanelRetentionResidueOwner:
    """Mutate only operational residue after an accepted logical cutover."""

    def __init__(self, *, workspace: Path, resolver: ArtifactResolver) -> None:
        """Locate operational residue in the selected workspace."""
        self.workspace = workspace.resolve()
        self.database_path = self.workspace / "market-data.duckdb"
        self.artifact_root = self.workspace / "artifacts"
        self.resolver = resolver

    def inventory_generations(self) -> tuple[PanelAvailabilityGeneration, ...]:
        """List availability generations currently present in DuckDB."""
        connection = open_workspace_database(self.database_path, read_only=True)
        try:
            rows = connection.execute(
                """
                SELECT manifest_revision, sector_revision, catalog_hash, policy_hash,
                       panel_binding_hash, count(*) AS row_count,
                       list(availability_hash ORDER BY session_date, factor_id) AS hashes
                FROM panel_factor_availability
                GROUP BY ALL
                ORDER BY manifest_revision, sector_revision, catalog_hash, policy_hash,
                         panel_binding_hash
                """
            ).fetchall()
            # A workspace opened read-only before any build under the
            # cross-section rule ran may not hold the second table yet.
            cross_section_rows = (
                connection.execute(
                    """
                    SELECT catalog_hash, policy_hash, panel_binding_hash, count(*) AS row_count,
                           list(availability_hash
                                ORDER BY session_date, factor_id, cross_section_identity) AS hashes
                    FROM panel_cross_section_availability
                    GROUP BY ALL
                    ORDER BY catalog_hash, policy_hash, panel_binding_hash
                    """
                ).fetchall()
                if _table_exists(connection, "panel_cross_section_availability")
                else []
            )
        finally:
            connection.close()
        return tuple(
            PanelAvailabilityGeneration(
                manifest_revision=str(row[0]),
                sector_revision=str(row[1]),
                catalog_hash=str(row[2]),
                policy_hash=str(row[3]),
                panel_binding_hash=str(row[4]),
                row_count=int(row[5]),
                ordered_availability_hash=canonical_hash(tuple(str(value) for value in row[6])),
            )
            for row in rows
        ) + tuple(
            PanelAvailabilityGeneration(
                catalog_hash=str(row[0]),
                policy_hash=str(row[1]),
                panel_binding_hash=str(row[2]),
                row_count=int(row[3]),
                ordered_availability_hash=canonical_hash(tuple(str(value) for value in row[4])),
                availability_basis=PANEL_ROW_IDENTITY_BY_CROSS_SECTION,
            )
            for row in cross_section_rows
        )

    def receipt_for_acceptance(
        self, acceptance_marker_hash: str
    ) -> PanelRetentionResidueReceipt | None:
        """Return the sealed residue receipt for an acceptance marker, if any."""
        root = self.artifact_root / "storage-governance" / "panel-residue-receipts"
        matches = tuple(
            receipt
            for path in (sorted(root.glob("*.json")) if root.is_dir() else ())
            if (
                receipt := PanelRetentionResidueReceipt.model_validate_json(path.read_bytes())
            ).cutover_acceptance_marker_hash
            == acceptance_marker_hash
        )
        if len(matches) > 1:
            raise ValueError("storage.eviction_readback_failed")
        return matches[0] if matches else None

    def observe_snapshot_states(self) -> tuple[dict[str, object], ...]:
        """Observe each snapshot's lifecycle and physical availability."""
        connection = open_workspace_database(self.database_path, read_only=True)
        try:
            rows = connection.execute(
                """
                SELECT snapshot_hash, manifest_uri, lifecycle, lifecycle_reason
                FROM feature_panel_snapshot_manifest
                ORDER BY snapshot_hash
                """
            ).fetchall()
        finally:
            connection.close()
        evicted = self._evicted_paths()
        values: list[dict[str, object]] = []
        for snapshot_hash, manifest_uri, lifecycle, reason in rows:
            physical, plan_hash = self._classify_snapshot(
                snapshot_hash=str(snapshot_hash),
                manifest_uri=str(manifest_uri),
                evicted=evicted,
            )
            values.append(
                {
                    "snapshot_hash": str(snapshot_hash),
                    "lifecycle": str(lifecycle),
                    "reason": str(reason) if reason is not None else None,
                    "physical_availability": physical,
                    "eviction_plan_hash": plan_hash,
                }
            )
        return tuple(values)

    def close(
        self,
        *,
        protected_panel_snapshot_hashes: Iterable[str],
        cutover_acceptance_marker_hash: str,
        completed_at: datetime,
    ) -> PanelRetentionResidueReceipt:
        """Close operational residue under a protected cutover acceptance."""
        protected_snapshots = tuple(sorted(set(protected_panel_snapshot_hashes)))
        if not protected_snapshots:
            raise ValueError("storage.retention_root_mismatch")
        before = self.inventory_generations()
        connection = open_workspace_database(self.database_path, read_only=False)
        try:
            self._ensure_physical_state_columns(connection)
            placeholders = ",".join("?" for _ in protected_snapshots)
            binding_rows = connection.execute(
                f"""
                SELECT DISTINCT panel_binding_hash
                FROM feature_panel_snapshot_manifest
                WHERE snapshot_hash IN ({placeholders})
                """,
                list(protected_snapshots),
            ).fetchall()
            bindings = {str(row[0]) for row in binding_rows if row[0]}
            # A protected snapshot's availability rows are stamped by the
            # builds that computed its cells, which -- once a build reuses
            # partitions -- are not only its own. Its manifest records every
            # such origin; rows of those bindings are part of the snapshot's
            # operational truth and are kept with it.
            for snapshot_hash in protected_snapshots:
                bindings.update(self._manifest_origin_bindings(snapshot_hash))
            protected_bindings = tuple(sorted(bindings))
            if not protected_bindings:
                raise ValueError("storage.retention_root_mismatch")
            binding_placeholders = ",".join("?" for _ in protected_bindings)
            connection.execute("BEGIN TRANSACTION")
            for table in ("panel_factor_availability", "panel_cross_section_availability"):
                if not _table_exists(connection, table):
                    continue
                connection.execute(
                    f"DELETE FROM {table} WHERE panel_binding_hash NOT IN ({binding_placeholders})",
                    list(protected_bindings),
                )
            self._reconcile_snapshot_states(connection)
            connection.execute("COMMIT")
        except Exception:
            with suppress(Exception):
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
        after = self.inventory_generations()
        after_by_binding = {value.panel_binding_hash: value for value in after}
        for generation in before:
            if (
                generation.panel_binding_hash in protected_bindings
                and after_by_binding.get(generation.panel_binding_hash) != generation
            ):
                raise ValueError("storage.eviction_readback_failed")
        snapshots = self._snapshot_projection()
        projection_hash = self._publish_snapshot_projection(snapshots)
        retained_bindings = {value.panel_binding_hash for value in after}
        retired = tuple(
            value for value in before if value.panel_binding_hash not in retained_bindings
        )
        values: dict[str, object] = {
            "cutover_acceptance_marker_hash": cutover_acceptance_marker_hash,
            "protected_panel_snapshot_hashes": protected_snapshots,
            "retained_generations": after,
            "retired_generations": retired,
            "snapshot_state_count": len(snapshots),
            "lifecycle_projection_hash": projection_hash,
            "completed_at": completed_at.astimezone(UTC),
        }
        provisional = PanelRetentionResidueReceipt.model_construct(**values, receipt_hash="0" * 64)
        receipt = PanelRetentionResidueReceipt.model_validate(
            {
                **values,
                "receipt_hash": canonical_hash(
                    provisional.model_dump(mode="json", exclude={"receipt_hash"}, exclude_none=True)
                ),
            }
        )
        self._publish_receipt(receipt)
        return cast(PanelRetentionResidueReceipt, receipt)

    def _manifest_origin_bindings(self, snapshot_hash: str) -> set[str]:
        manifest = self.resolver.load_feature_panel_manifest(
            self.resolver.feature_panel_manifest_uri(snapshot_hash)
        )
        return set(manifest_partition_origins(manifest))

    @staticmethod
    def _ensure_physical_state_columns(connection: duckdb.DuckDBPyConnection) -> None:
        """Upgrade the operational projection before its first truthful reconciliation."""
        connection.execute(
            "ALTER TABLE feature_panel_snapshot_manifest "
            "ADD COLUMN IF NOT EXISTS physical_availability VARCHAR DEFAULT 'AVAILABLE'"
        )
        connection.execute(
            "ALTER TABLE feature_panel_snapshot_manifest "
            "ADD COLUMN IF NOT EXISTS eviction_plan_hash VARCHAR"
        )
        connection.execute(
            "UPDATE feature_panel_snapshot_manifest "
            "SET physical_availability = 'AVAILABLE' WHERE physical_availability IS NULL"
        )

    def reconcile_physical_availability(self) -> None:
        """Project approved file eviction without changing logical lifecycle or deleting rows."""
        with open_workspace_database(self.database_path, read_only=False) as connection:
            self._ensure_physical_state_columns(connection)
            connection.begin()
            try:
                self._reconcile_snapshot_states(connection)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        self._publish_snapshot_projection(self._snapshot_projection())

    def _publish_snapshot_projection(self, snapshots: tuple[dict[str, object], ...]) -> str:
        identity = self.resolver.publish_feature_panel_lifecycle_projection(snapshots=snapshots)
        if not all(
            self.resolver.feature_panel_snapshot_lifecycle(
                self.resolver.feature_panel_manifest_uri(str(v["snapshot_hash"]))
            )
            == v
            for v in snapshots
        ):
            raise ValueError("storage.eviction_readback_failed")
        return str(identity)

    def _reconcile_snapshot_states(self, connection: duckdb.DuckDBPyConnection) -> None:
        evicted = self._evicted_paths()
        rows = connection.execute(
            "SELECT snapshot_hash, manifest_uri, lifecycle FROM feature_panel_snapshot_manifest"
        ).fetchall()
        for snapshot_hash, manifest_uri, lifecycle in rows:
            snapshot = str(snapshot_hash)
            availability, eviction_plan_hash = self._classify_snapshot(
                snapshot_hash=snapshot,
                manifest_uri=str(manifest_uri),
                evicted=evicted,
            )
            if str(lifecycle) == "ACTIVE" and availability != "AVAILABLE":
                raise ValueError("storage.eviction_readback_failed")
            connection.execute(
                """
                UPDATE feature_panel_snapshot_manifest
                SET physical_availability = ?, eviction_plan_hash = ?
                WHERE snapshot_hash = ?
                """,
                [availability, eviction_plan_hash, snapshot],
            )

    def _classify_snapshot(
        self,
        *,
        snapshot_hash: str,
        manifest_uri: str,
        evicted: dict[Path, str],
    ) -> tuple[str, str | None]:
        manifest_path = self.artifact_root / "feature-panel" / "manifests" / f"{snapshot_hash}.json"
        expected_paths = {manifest_path.resolve()}
        if manifest_path.is_file():
            manifest = self.resolver.load_feature_panel_manifest(manifest_uri)
            for item in cast(list[dict[str, object]], manifest.get("chunks", [])):
                expected_paths.add(
                    (
                        self.artifact_root
                        / "feature-panel"
                        / "chunks"
                        / f"{item['chunk_hash']}.parquet"
                    ).resolve()
                )
        missing = tuple(path for path in expected_paths if not path.is_file())
        if not missing:
            return "AVAILABLE", None
        plans = {evicted[path] for path in missing if path in evicted}
        if all(path in evicted for path in missing) and len(plans) == 1:
            return "EVICTED_BY_RETENTION", plans.pop()
        return "MISSING_OR_TAMPERED", None

    def _snapshot_projection(self) -> tuple[dict[str, object], ...]:
        connection = open_workspace_database(self.database_path, read_only=True)
        try:
            rows = connection.execute(
                """
                SELECT snapshot_hash, lifecycle, lifecycle_reason,
                       physical_availability, eviction_plan_hash
                FROM feature_panel_snapshot_manifest
                ORDER BY snapshot_hash
                """
            ).fetchall()
        finally:
            connection.close()
        return tuple(
            {
                "snapshot_hash": str(row[0]),
                "lifecycle": str(row[1]),
                "reason": str(row[2]) if row[2] is not None else None,
                "physical_availability": str(row[3]),
                "eviction_plan_hash": str(row[4]) if row[4] is not None else None,
            }
            for row in rows
        )

    def _evicted_paths(self) -> dict[Path, str]:
        values: dict[Path, str] = {}
        root = self.artifact_root / "storage-governance" / "availability"
        for path in sorted(root.glob("*.json")) if root.is_dir() else ():
            payload = json.loads(path.read_text(encoding="utf-8"))
            plan_hash = str(payload.get("eviction_plan_hash", ""))
            for entry in cast(list[dict[str, object]], payload.get("entries", [])):
                if entry.get("availability") != "EVICTED_BY_RETENTION":
                    continue
                target = (self.workspace / str(entry["relative_path"])).resolve()
                values[target] = plan_hash
        return values

    def _publish_receipt(self, receipt: PanelRetentionResidueReceipt) -> None:
        target = (
            self.artifact_root
            / "storage-governance"
            / "panel-residue-receipts"
            / f"{receipt.receipt_hash}.json"
        )
        serialized = json.dumps(
            receipt.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        if target.exists() and target.read_bytes() != serialized:
            raise ValueError("storage.eviction_readback_failed")
        if not target.exists():
            staged.write_bytes(serialized)
            os.replace(staged, target)
        staged.unlink(missing_ok=True)


__all__ = [
    "PanelAvailabilityGeneration",
    "PanelRetentionResidueOwner",
    "PanelRetentionResidueReceipt",
]
