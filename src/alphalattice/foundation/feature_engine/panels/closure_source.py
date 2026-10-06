"""Read-only legacy source adapter for one-time Panel closure publication.

The clean-room rematerializer does not depend on this module.  It exists only
to freeze currently reachable mutable inputs into immutable closure artifacts.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.control.workspace_runtime.reader_threads import connect_duckdb
from alphalattice.foundation.feature_engine.contracts import (
    PANEL_ROW_IDENTITY_BY_BINDING,
    PANEL_ROW_IDENTITY_BY_CROSS_SECTION,
    PanelCrossSectionRange,
    canonical_hash,
)
from alphalattice.foundation.feature_engine.panels.artifacts import manifest_row_identity_basis
from alphalattice.foundation.feature_engine.panels.identity import PANEL_IDENTITY_COLUMNS


@dataclass(frozen=True)
class PanelSnapshotSource:
    """Carry a sealed Panel manifest and its source lifecycle for closure."""

    snapshot_hash: str
    lifecycle: str
    manifest: dict[str, object]
    factor_ids: tuple[str, ...]
    listing_ids: tuple[str, ...]
    """The calculation axis, in calculation order: every listing the rows hold."""

    sessions: tuple[date, ...]
    row_identity_basis: str = PANEL_ROW_IDENTITY_BY_BINDING
    member_listing_ids: tuple[str, ...] = ()
    """The manifest's members, in calculation order; the whole axis under the binding rule."""

    members_by_session: Mapping[date, tuple[str, ...]] | None = field(default=None, repr=False)
    """Each session's members (sorted), read from the rows; None under the binding rule."""

    @property
    def lineage(self) -> dict[str, str]:
        """Return the manifest's safe lineage identifiers."""
        safe_summary = self.manifest["safe_summary"]
        assert isinstance(safe_summary, dict)
        lineage = safe_summary["lineage"]
        assert isinstance(lineage, dict)
        return {
            str(key): str(value)
            for key, value in lineage.items()
            if not isinstance(value, (dict, list))
        }

    def members(self, session: date) -> tuple[str, ...]:
        """The listings whose rows the session holds, sorted."""
        if self.members_by_session is None:
            return tuple(sorted(self.listing_ids))
        return self.members_by_session[session]

    @property
    def cross_sections(self) -> tuple[PanelCrossSectionRange, ...]:
        """Return cross-section ranges represented by the manifest chunks."""
        chunks = self.manifest.get("chunks")
        assert isinstance(chunks, list)
        return tuple(
            PanelCrossSectionRange.from_payload(item)
            for chunk in chunks
            if isinstance(chunk, dict)
            for item in chunk.get("cross_sections") or ()
        )


class PanelClosureSourceRepository:
    """Freeze closure inputs without extending the central market-data store."""

    def __init__(
        self,
        *,
        database_path: Path,
        resolver: ArtifactResolver,
        snapshot_hashes: frozenset[str] | None = None,
    ) -> None:
        """Bind the source database, resolver, and optional snapshot scope."""
        self._database_path = database_path.resolve()
        self._resolver = resolver
        self._snapshot_hashes = snapshot_hashes

    def inventory_snapshots(self) -> tuple[PanelSnapshotSource, ...]:
        """List admitted Panel snapshots and their durable manifests."""
        manifest_root = Path(self._resolver.root) / "feature-panel" / "manifests"
        lifecycle_path = Path(self._resolver.root) / "feature-panel" / "snapshot-lifecycle.json"
        lifecycle_payload = json.loads(lifecycle_path.read_text(encoding="utf-8"))
        lifecycle_rows = lifecycle_payload.get("snapshots", [])
        lifecycle = {
            str(item["snapshot_hash"]): str(item["lifecycle"])
            for item in lifecycle_rows
            if isinstance(item, dict)
        }
        snapshots: list[PanelSnapshotSource] = []
        for path in sorted(manifest_root.glob("*.json")):
            if self._snapshot_hashes is not None and path.stem not in self._snapshot_hashes:
                continue
            manifest = self._resolver.load_feature_panel_manifest(
                self._resolver.feature_panel_manifest_uri(path.stem)
            )
            chunks = manifest.get("chunks")
            if not isinstance(chunks, list) or not chunks:
                raise ValueError("Panel closure source manifest has no chunks")
            first = chunks[0]
            if not isinstance(first, dict):
                raise ValueError("Panel closure source chunk descriptor is invalid")
            first_path = self._resolver.resolve_feature_panel_chunk_ref(
                uri=str(first["uri"]),
                content_hash=str(first["chunk_hash"]),
                metadata_hash=str(first["metadata_hash"]),
            )
            schema = pq.read_schema(first_path)
            factor_ids = tuple(
                column for column in schema.names if column not in PANEL_IDENTITY_COLUMNS
            )
            basis = manifest_row_identity_basis(manifest)
            listing_ids: set[str] = set()
            sessions: set[date] = set()
            members_by_session: dict[date, tuple[str, ...]] = {}
            previous_members: tuple[str, ...] = ()
            for raw_chunk in sorted(
                (item for item in chunks if isinstance(item, dict)),
                key=lambda item: int(str(item["year"])),
            ):
                chunk_path = self._resolver.resolve_feature_panel_chunk_ref(
                    uri=str(raw_chunk["uri"]),
                    content_hash=str(raw_chunk["chunk_hash"]),
                    metadata_hash=str(raw_chunk["metadata_hash"]),
                )
                axes = pq.read_table(chunk_path, columns=["session_date", "listing_id"])
                listing_ids.update(str(value) for value in axes.column("listing_id").to_pylist())
                sessions.update(axes.column("session_date").to_pylist())
                if basis == PANEL_ROW_IDENTITY_BY_CROSS_SECTION:
                    # Each session's members are its rows. Consecutive sessions
                    # with the same members share one tuple, so a decade of
                    # membership costs the memory of its epochs, not its rows.
                    frame = axes.to_pandas()
                    for session, group in frame.groupby("session_date", sort=True):
                        members = tuple(sorted(str(value) for value in group["listing_id"]))
                        if members == previous_members:
                            members = previous_members
                        previous_members = members
                        members_by_session[session] = members
            if len(chunks) != len([item for item in chunks if isinstance(item, dict)]):
                raise ValueError("Panel closure source chunk descriptor is invalid")
            physical_listing_ids = frozenset(listing_ids)
            lineage = manifest.get("safe_summary", {}).get("lineage", {})
            if not isinstance(lineage, dict):
                raise ValueError("Panel closure source manifest has no lineage")
            member_listing_ids = self._read_manifest_listing_axis(
                manifest_revision=str(lineage["manifest_revision"]),
            )
            if basis == PANEL_ROW_IDENTITY_BY_CROSS_SECTION:
                # The rows hold every listing some session's cross-section
                # held; the manifest's members are those of the latest one.
                # The axis is read in calculation order from the listing
                # table, exactly as a build orders it.
                if not frozenset(member_listing_ids) <= physical_listing_ids:
                    raise ValueError(
                        "Panel closure manifest members are not all held by physical rows"
                    )
                ordered_listing_ids = self._read_calculation_axis(physical_listing_ids)
                if canonical_hash(sorted(ordered_listing_ids)) != str(manifest["listing_set_hash"]):
                    raise ValueError(
                        "Panel closure manifest listing axis does not match physical rows"
                    )
            else:
                if frozenset(member_listing_ids) != physical_listing_ids:
                    raise ValueError(
                        "Panel closure manifest calculation axis does not match physical rows"
                    )
                ordered_listing_ids = member_listing_ids
            snapshots.append(
                PanelSnapshotSource(
                    snapshot_hash=path.stem,
                    lifecycle=lifecycle.get(path.stem, "UNKNOWN"),
                    manifest=manifest,
                    factor_ids=factor_ids,
                    listing_ids=ordered_listing_ids,
                    sessions=tuple(sorted(sessions)),
                    row_identity_basis=basis,
                    member_listing_ids=member_listing_ids,
                    members_by_session=(
                        members_by_session if basis == PANEL_ROW_IDENTITY_BY_CROSS_SECTION else None
                    ),
                )
            )
        return tuple(snapshots)

    def _read_calculation_axis(self, listing_ids: frozenset[str]) -> tuple[str, ...]:
        """The calculation order of an arbitrary listing set: by display symbol.

        The same rule ``_read_manifest_listing_axis`` applies to a manifest,
        over the listings a cross-section Panel's rows actually hold, which
        include members that have since left the manifest.
        """
        marks = ", ".join("?" for _ in listing_ids)
        connection = connect_duckdb(self._database_path, read_only=True)
        try:
            rows = connection.execute(
                f"""
                SELECT listing_id FROM listing
                WHERE listing_id IN ({marks})
                ORDER BY display_symbol, listing_id
                """,
                sorted(listing_ids),
            ).fetchall()
        finally:
            connection.close()
        ordered = tuple(str(row[0]) for row in rows)
        if set(ordered) != set(listing_ids):
            raise ValueError("calculation axis listings are absent from the listing table")
        return ordered

    def _read_manifest_listing_axis(self, *, manifest_revision: str) -> tuple[str, ...]:
        """Recover the numerical axis used by ``load_universe_manifest``.

        Panel Parquet rows are ordered by opaque listing ID for stable physical
        identity.  Cross-sectional arithmetic instead consumes the manifest's
        display-symbol order.  Reconstructing from Parquet would therefore
        preserve membership but lose the floating-point operation order.
        """
        connection = connect_duckdb(self._database_path, read_only=True)
        try:
            rows = connection.execute(
                """
                SELECT l.listing_id
                FROM universe_manifest AS manifest
                JOIN universe_manifest_listing AS member
                  ON member.manifest_id = manifest.manifest_id
                JOIN listing AS l ON l.listing_id = member.listing_id
                WHERE manifest.revision_sha256 = ?
                ORDER BY l.display_symbol
                """,
                [manifest_revision],
            ).fetchall()
        finally:
            connection.close()
        listing_ids = tuple(str(row[0]) for row in rows)
        if not listing_ids or len(listing_ids) != len(set(listing_ids)):
            raise ValueError("manifest calculation listing axis is incomplete or ambiguous")
        return listing_ids

    def read_base_values(
        self,
        *,
        catalog_hash: str,
        factor_ids: tuple[str, ...],
        required_keys: pa.Table,
        permit_observed_absences: bool = False,
    ) -> pa.Table:
        """Read base Formula values for a Panel's requested source scope."""
        if required_keys.num_rows == 0 or not factor_ids:
            raise ValueError("base closure source scope cannot be empty")
        if required_keys.column_names != ["session_date", "listing_id"]:
            raise ValueError("base closure source keys have an invalid schema")
        connection = connect_duckdb(self._database_path, read_only=True)
        try:
            connection.register("panel_required_keys", required_keys)
            table = connection.execute(
                f"""
                SELECT CAST(k.session_date AS DATE) AS session_date,
                       CAST(k.listing_id AS VARCHAR) AS listing_id,
                       f.listing_id IS NOT NULL AS source_row_present,
                       {", ".join(f'f."{factor_id}"' for factor_id in factor_ids)}
                FROM panel_required_keys AS k
                LEFT JOIN feature_daily_runtime AS f
                  ON f.catalog_hash = ?
                 AND f.session_date = k.session_date
                 AND f.listing_id = k.listing_id
                ORDER BY k.session_date, k.listing_id
                """,
                [catalog_hash],
            ).to_arrow_table()
            present = table.column("source_row_present").to_pylist()
            if table.num_rows != required_keys.num_rows:
                raise ValueError("base closure source has an invalid key count")
            if not all(present):
                if not permit_observed_absences:
                    raise ValueError(
                        "base closure source is incomplete: "
                        f"expected {required_keys.num_rows}, "
                        f"got {sum(bool(item) for item in present)}"
                    )
                missing = table.filter(pa.array([not item for item in present])).select(
                    ["session_date", "listing_id"]
                )
                connection.register("panel_missing_base_keys", missing)
                if connection.execute(
                    """SELECT count(*) FROM panel_missing_base_keys k
                    JOIN raw_daily_bar_current r USING (session_date, listing_id)"""
                ).fetchone()[0]:
                    raise ValueError("feature_panel.base_row_not_materialized")
        finally:
            connection.close()
        return table

    def read_sector_rows(self, *, listing_ids: tuple[str, ...]) -> tuple[dict[str, object], ...]:
        """Read sector revision rows for the selected listings."""
        marks = ", ".join("?" for _ in listing_ids)
        connection = connect_duckdb(self._database_path, read_only=True)
        try:
            result = connection.execute(
                f"""
                SELECT listing_id, provider, provider_symbol, sector_name, sector_key,
                       payload_hash, evidence_hash, sector_revision
                FROM sector_classification_current
                WHERE listing_id IN ({marks})
                ORDER BY listing_id
                """,
                list(listing_ids),
            )
            names = [str(column[0]) for column in result.description]
            rows = tuple(dict(zip(names, row, strict=True)) for row in result.fetchall())
        finally:
            connection.close()
        if len(rows) != len(listing_ids):
            raise ValueError("sector revision map source is incomplete")
        return rows

    def sector_revision_ledger_counts(self) -> tuple[int, int]:
        """Count sector revision and membership rows in the source ledger."""
        connection = connect_duckdb(self._database_path, read_only=True)
        try:
            row = connection.execute(
                """
                SELECT count(*), count(*) FILTER (WHERE revision_kind <> 'INSERTED')
                FROM sector_classification_revision
                """
            ).fetchone()
        finally:
            connection.close()
        assert row is not None
        return int(row[0]), int(row[1])

    def read_availability(self, *, snapshot: PanelSnapshotSource) -> pa.Table:
        """The snapshot's availability rows, under the key its rows were computed with."""
        lineage = snapshot.lineage
        connection = connect_duckdb(self._database_path, read_only=True)
        try:
            if snapshot.row_identity_basis == PANEL_ROW_IDENTITY_BY_CROSS_SECTION:
                ranges = snapshot.cross_sections
                if not ranges:
                    raise ValueError("availability closure source has no cross-section ranges")
                predicate = " OR ".join(
                    "(cross_section_identity = ? AND session_date BETWEEN ? AND ?)" for _ in ranges
                )
                parameters: list[object] = [lineage["catalog_hash"], lineage["policy_hash"]]
                for item in ranges:
                    parameters.extend(
                        [item.cross_section_identity, item.first_session, item.last_session]
                    )
                table = connection.execute(
                    f"""
                    SELECT CAST(session_date AS DATE) AS session_date,
                           CAST(factor_id AS VARCHAR) AS factor_id,
                           universe_size, computed_count, coverage, sector_counts_json,
                           winsor_lower, winsor_upper, residual_median, residual_mad,
                           status, reason, panel_binding_hash AS panel_hash,
                           panel_binding_hash, small_sector_warning, small_sector_names_json,
                           availability_hash, materialization_receipt_hash,
                           cross_section_identity
                    FROM panel_cross_section_availability
                    WHERE catalog_hash = ? AND policy_hash = ? AND ({predicate})
                    ORDER BY session_date, factor_id
                    """,
                    parameters,
                ).to_arrow_table()
            else:
                table = connection.execute(
                    """
                    SELECT CAST(session_date AS DATE) AS session_date,
                           CAST(factor_id AS VARCHAR) AS factor_id,
                           universe_size, computed_count, coverage, sector_counts_json,
                           winsor_lower, winsor_upper, residual_median, residual_mad,
                           status, reason, panel_hash, panel_binding_hash,
                           small_sector_warning, small_sector_names_json,
                           availability_hash, materialization_receipt_hash
                    FROM panel_factor_availability
                    WHERE manifest_revision = ? AND sector_revision = ?
                      AND catalog_hash = ? AND policy_hash = ?
                      AND session_date BETWEEN ? AND ?
                    ORDER BY session_date, factor_id
                    """,
                    [
                        lineage["manifest_revision"],
                        lineage["sector_revision"],
                        lineage["catalog_hash"],
                        lineage["policy_hash"],
                        snapshot.sessions[0],
                        snapshot.sessions[-1],
                    ],
                ).to_arrow_table()
        finally:
            connection.close()
        safe_summary = snapshot.manifest.get("safe_summary")
        if not isinstance(safe_summary, dict):
            raise ValueError("availability closure source has no safe summary")
        expected = int(safe_summary.get("availability_count", -1))
        factor_summary = safe_summary.get("factor_catalog_summary")
        if not isinstance(factor_summary, dict):
            raise ValueError("availability closure source has no factor summary")
        expected_factors = {str(value) for value in factor_summary}
        observed_factors = {str(value) for value in table.column("factor_id").to_pylist()}
        if table.num_rows != expected:
            raise ValueError(
                "availability closure source is incomplete: "
                f"expected {expected}, got {table.num_rows}"
            )
        if observed_factors != expected_factors:
            raise ValueError("availability closure factor axis does not match the snapshot summary")
        return table


__all__ = ["PanelClosureSourceRepository", "PanelSnapshotSource"]
