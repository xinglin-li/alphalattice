"""Publish immutable inputs required to rematerialize historical Panels."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pydantic import BaseModel

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.feature_engine.contracts import (
    PANEL_ROW_IDENTITY_BY_CROSS_SECTION,
    PanelSourceExclusion,
)
from alphalattice.foundation.feature_engine.panels.artifacts import (
    PanelCompositionBinding,
    PreparedPanelChunk,
    manifest_partition_origins,
    panel_content_identity,
    prepared_chunk,
)
from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
    PanelClosureArtifactStore,
    arrow_table_logical_hash,
    ordered_key_hash,
    ordered_value_hash,
)
from alphalattice.foundation.feature_engine.panels.closure_contracts import (
    ClosureArtifactRef,
    ExpectedPanelChunk,
    PanelAvailabilityClosure,
    PanelBaseKeyChunk,
    PanelBaseValueChunk,
    PanelBaseValueClosureManifest,
    PanelCrossSectionRecord,
    PanelDerivationRecipe,
    PanelMembershipBasisRecord,
    PanelMembershipEpochRecord,
    PanelMembershipRecord,
    PanelPartitionOrigin,
    PanelRowReceiptAssignment,
    PanelSectorReclassification,
    SectorRevisionBackfillReceipt,
    SectorRevisionEntry,
    SectorRevisionMap,
    durable_payload,
)
from alphalattice.foundation.feature_engine.panels.closure_source import (
    PanelClosureSourceRepository,
    PanelSnapshotSource,
)
from alphalattice.foundation.feature_engine.panels.rematerialization import (
    ArtifactOnlyPanelRematerializer,
    availability_rows,
)
from alphalattice.foundation.feature_engine.producers.preprocessing.robust_cross_section import (
    PANEL_SESSION_BATCH_SIZE,
    allows_missing_source_rows,
)
from alphalattice.foundation.market_data_ops.returns.sector_revision_identity import (
    sector_revision_hash,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash


@dataclass(frozen=True)
class PanelClosurePublication:
    """Collect immutable closure children and publication accounting."""

    snapshots: tuple[PanelSnapshotSource, ...]
    sector_maps: dict[str, SectorRevisionMap]
    sector_receipts: tuple[SectorRevisionBackfillReceipt, ...]
    base_manifests: dict[str, PanelBaseValueClosureManifest]
    availability: dict[str, PanelAvailabilityClosure]
    row_receipts: dict[str, PanelRowReceiptAssignment]
    recipes: dict[str, PanelDerivationRecipe]
    blocked_snapshots: dict[str, str]
    newly_written_bytes: int
    content_reused_artifact_count: int


class PanelClosurePublisher:
    """Freeze current legacy inputs into immutable, readback-verified children."""

    def __init__(
        self,
        *,
        resolver: ArtifactResolver,
        source: PanelClosureSourceRepository,
        store: PanelClosureArtifactStore,
        progress: Callable[[str, int, int, str | None], None] | None = None,
    ) -> None:
        """Bind the source, resolver, store, and optional progress observer."""
        self._resolver = resolver
        self._source = source
        self._store = store
        self._progress = progress
        self._new_bytes = 0
        self._reused = 0

    def inventory(self) -> tuple[PanelSnapshotSource, ...]:
        """List source snapshots eligible for closure publication."""
        snapshots: tuple[PanelSnapshotSource, ...] = self._source.inventory_snapshots()
        return snapshots

    def publish(
        self, snapshots: tuple[PanelSnapshotSource, ...] | None = None
    ) -> PanelClosurePublication:
        """Seal and verify immutable closure children for the selected snapshots."""
        snapshots = snapshots if snapshots is not None else self.inventory()
        if not snapshots:
            raise ValueError("Panel closure publication requires at least one snapshot")
        sector_maps, sector_receipts = self._publish_sector_maps(snapshots)
        base_manifests = self._publish_base_manifests(snapshots)
        availability, superseded = self._publish_availability(snapshots)
        row_receipts = self._publish_row_receipts(snapshots)
        recipes, blocked_snapshots = self._publish_recipes(
            snapshots=snapshots,
            sector_maps=sector_maps,
            base_manifests=base_manifests,
            availability=availability,
            superseded=superseded,
            row_receipts=row_receipts,
        )
        return PanelClosurePublication(
            snapshots=snapshots,
            sector_maps=sector_maps,
            sector_receipts=sector_receipts,
            base_manifests=base_manifests,
            availability=availability,
            row_receipts=row_receipts,
            recipes=recipes,
            blocked_snapshots=blocked_snapshots,
            newly_written_bytes=self._new_bytes,
            content_reused_artifact_count=self._reused,
        )

    def _publish_sector_maps(
        self, snapshots: tuple[PanelSnapshotSource, ...]
    ) -> tuple[dict[str, SectorRevisionMap], tuple[SectorRevisionBackfillReceipt, ...]]:
        by_revision: dict[str, list[PanelSnapshotSource]] = defaultdict(list)
        for snapshot in snapshots:
            by_revision[snapshot.lineage["sector_revision"]].append(snapshot)
        if not by_revision:
            raise ValueError("Panel closure publication has no sector revision")
        ledger_rows, correction_rows = self._source.sector_revision_ledger_counts()
        maps: dict[str, SectorRevisionMap] = {}
        receipts: list[SectorRevisionBackfillReceipt] = []
        for revision, revision_snapshots in sorted(by_revision.items()):
            # The revision is derived over the members of the manifest it was
            # activated for. Under the cross-section rule later manifests keep
            # a sector revision until the map is next refreshed, so the
            # snapshots under one revision may carry several manifests; the
            # deriving one is the one that reproduces the revision. The map
            # holds every listing any snapshot under it computed with, which
            # includes members that have since left and still sit in the
            # sessions before their exit.
            axis = tuple(
                sorted({listing for item in revision_snapshots for listing in item.listing_ids})
            )
            rows = self._source.read_sector_rows(listing_ids=axis)
            entries = tuple(
                SectorRevisionEntry(
                    listing_id=str(row["listing_id"]),
                    provider=str(row["provider"]),
                    provider_symbol=str(row["provider_symbol"]),
                    sector_name=str(row["sector_name"]),
                    sector_key=str(row["sector_key"]) if row["sector_key"] is not None else None,
                    payload_hash=str(row["payload_hash"]),
                    evidence_hash=str(row["evidence_hash"]),
                )
                for row in rows
            )
            deriving_manifest: str | None = None
            for candidate in sorted(
                revision_snapshots, key=lambda item: item.lineage["manifest_revision"]
            ):
                members = set(candidate.member_listing_ids or candidate.listing_ids)
                safe_items = tuple(
                    item.model_dump(mode="python") for item in entries if item.listing_id in members
                )
                derived_revision = sector_revision_hash(
                    manifest_revision=candidate.lineage["manifest_revision"],
                    observations=safe_items,
                )
                if derived_revision == revision:
                    deriving_manifest = candidate.lineage["manifest_revision"]
                    break
            if deriving_manifest is None:
                raise ValueError("feature_panel.sector_revision_map_unrecoverable")
            sector_map_values: dict[str, Any] = {
                "manifest_revision": deriving_manifest,
                "sector_revision": revision,
                "entries": entries,
            }
            sector_map = _identified(SectorRevisionMap, sector_map_values, "map_hash")
            self._publish_model("sector-maps", sector_map.map_hash, sector_map)
            basis = (
                "CURRENT_ROWS_EXACT"
                if all(str(row["sector_revision"]) == revision for row in rows)
                else "CURRENT_ROWS_WITH_ZERO_CORRECTION_LEDGER"
            )
            receipt = _identified(
                SectorRevisionBackfillReceipt,
                {
                    "sector_map_hash": sector_map.map_hash,
                    "revision_ledger_row_count": ledger_rows,
                    "correction_row_count": correction_rows,
                    "reconstruction_basis": basis,
                },
                "receipt_hash",
            )
            self._publish_model("sector-map-receipts", receipt.receipt_hash, receipt)
            maps[revision] = sector_map
            receipts.append(receipt)
            self._report("sector_maps", len(receipts), len(by_revision), revision[:12])
        return maps, tuple(receipts)

    def _publish_base_manifests(
        self, snapshots: tuple[PanelSnapshotSource, ...]
    ) -> dict[str, PanelBaseValueClosureManifest]:
        groups: dict[str, list[PanelSnapshotSource]] = defaultdict(list)
        for snapshot in snapshots:
            groups[snapshot.lineage["catalog_hash"]].append(snapshot)
        factors_by_catalog = {
            catalog_hash: tuple(
                sorted({factor for source in sources for factor in source.factor_ids})
            )
            for catalog_hash, sources in groups.items()
        }
        manifests: dict[str, PanelBaseValueClosureManifest] = {}
        total_chunks = sum(
            len({session.year for source in sources for session in source.sessions})
            * (1 + len(factors_by_catalog[catalog_hash]))
            for catalog_hash, sources in groups.items()
        )
        completed_chunks = 0
        for catalog_hash, sources in sorted(groups.items()):
            factors = factors_by_catalog[catalog_hash]
            sessions = tuple(sorted({value for source in sources for value in source.sessions}))
            key_chunks: list[PanelBaseKeyChunk] = []
            value_chunks: list[PanelBaseValueChunk] = []
            absent_tables: list[pa.Table] = []
            for year in sorted({session.year for session in sessions}):
                consumed_keys = sorted(
                    {
                        (session, listing_id)
                        for source in sources
                        for session in source.sessions
                        if session.year == year
                        for listing_id in source.members(session)
                    }
                )
                year_sessions = tuple(sorted({session for session, _listing in consumed_keys}))
                year_listings = tuple(sorted({listing for _session, listing in consumed_keys}))
                key_table = pa.table(
                    {
                        "session_date": pa.array(
                            (session for session, _listing in consumed_keys), type=pa.date32()
                        ),
                        "listing_id": pa.array(
                            (listing for _session, listing in consumed_keys), type=pa.string()
                        ),
                    }
                )
                table = self._source.read_base_values(
                    catalog_hash=catalog_hash,
                    factor_ids=factors,
                    required_keys=key_table,
                    permit_observed_absences=any(
                        allows_missing_source_rows(source.lineage["policy_hash"])
                        for source in sources
                    ),
                )
                absent = table.filter(pc.invert(table["source_row_present"])).select(
                    ["session_date", "listing_id"]
                )
                for row in absent.to_pylist():
                    if any(
                        not allows_missing_source_rows(source.lineage["policy_hash"])
                        and row["session_date"] in source.sessions
                        and row["listing_id"] in source.members(row["session_date"])
                        for source in sources
                    ):
                        raise ValueError("feature_panel.legacy_base_source_missing")
                if absent.num_rows:
                    absent_tables.append(absent)
                key_hash = ordered_key_hash(
                    key_table.column("session_date").to_pylist(),
                    (str(value) for value in key_table.column("listing_id").to_pylist()),
                )
                key_ref = self._publish_table(
                    category="base-keys",
                    content_hash=key_hash,
                    kind="PanelBaseKeyChunk",
                    table=key_table,
                )
                key_chunks.append(
                    PanelBaseKeyChunk(
                        year=year,
                        key_hash=key_hash,
                        first_session=year_sessions[0],
                        last_session=year_sessions[-1],
                        listing_count=len(year_listings),
                        session_count=len(year_sessions),
                        artifact=key_ref,
                    )
                )
                completed_chunks += 1
                self._report(
                    "base_closure_chunks",
                    completed_chunks,
                    total_chunks,
                    f"{catalog_hash[:8]}/{year}/keys",
                )
                for factor_id in factors:
                    column = table.column(factor_id).combine_chunks()
                    valid = pc.is_valid(column).to_numpy(zero_copy_only=False)
                    filled = pc.fill_null(column, np.nan).to_numpy(zero_copy_only=False)
                    values = np.asarray(filled, dtype="<f8")
                    value_hash = ordered_value_hash(
                        factor_id=factor_id,
                        key_hash=key_hash,
                        values=values,
                        valid=np.asarray(valid, dtype=np.bool_),
                    )
                    value_table = pa.table(
                        {
                            "value_bits": pa.array(values.view("<u8"), type=pa.uint64()),
                            "is_valid": pa.array(valid, type=pa.bool_()),
                        }
                    )
                    value_ref = self._publish_table(
                        category="base-values",
                        content_hash=value_hash,
                        kind="PanelBaseValueChunk",
                        table=value_table,
                    )
                    value_chunks.append(
                        PanelBaseValueChunk(
                            year=year,
                            factor_id=factor_id,
                            key_hash=key_hash,
                            value_hash=value_hash,
                            null_count=int((~np.asarray(valid, dtype=np.bool_)).sum()),
                            artifact=value_ref,
                        )
                    )
                    completed_chunks += 1
                    self._report(
                        "base_closure_chunks",
                        completed_chunks,
                        total_chunks,
                        f"{catalog_hash[:8]}/{year}/{factor_id}",
                    )
            manifest_values: dict[str, Any] = {
                "catalog_hash": catalog_hash,
                "history_start": sessions[0],
                "as_of_session": sessions[-1],
                "factor_ids": factors,
                "key_chunks": tuple(key_chunks),
                "value_chunks": tuple(value_chunks),
            }
            if absent_tables:
                absent = pa.concat_tables(absent_tables)
                manifest_values["absent_source_keys"] = self._publish_table(
                    category="base-keys",
                    kind="PanelBaseKeyChunk",
                    content_hash=ordered_key_hash(
                        absent["session_date"].to_pylist(), absent["listing_id"].to_pylist()
                    ),
                    table=absent,
                )
            manifest = _identified(
                PanelBaseValueClosureManifest,
                manifest_values,
                "manifest_hash",
            )
            self._publish_model("base-manifests", manifest.manifest_hash, manifest)
            manifests[catalog_hash] = manifest
        return manifests

    def _publish_availability(
        self, snapshots: tuple[PanelSnapshotSource, ...]
    ) -> tuple[dict[str, PanelAvailabilityClosure], frozenset[str]]:
        """Freeze each snapshot's availability, or say the live rows no longer describe it.

        The availability table is operational: a later build that recomputes
        a cell restamps it with its own binding and receipt. Every cell of a
        snapshot was written by a build its manifest records as an origin,
        so a row stamped by any other binding is a cell that moved after the
        snapshot was published -- and the exact test is the snapshot's own
        content identity, which the rows must reproduce. Judged per snapshot:
        two snapshots under one binding (a same-day rebuild) read the same
        live rows, and those rows describe at most the later one. Freezing
        rows that do not describe a snapshot would seal a recipe whose chunks
        no replay can reproduce; such a snapshot is reported as superseded,
        and the closure captured before the later build -- if one was -- is
        the one that describes it.
        """
        closures: dict[str, PanelAvailabilityClosure] = {}
        superseded: set[str] = set()
        for snapshot in snapshots:
            binding = str(snapshot.manifest["panel_binding_hash"])
            table = self._source.read_availability(snapshot=snapshot)
            stamped = {str(value) for value in table.column("panel_binding_hash").to_pylist()}
            if not stamped <= set(
                manifest_partition_origins(snapshot.manifest)
            ) or not self._availability_describes(snapshot, table):
                superseded.add(snapshot.snapshot_hash)
                continue
            content_hash = arrow_table_logical_hash(table)
            reference = self._publish_table(
                category="availability",
                content_hash=content_hash,
                kind="PanelAvailabilityClosureRows",
                table=table,
            )
            closure = _identified(
                PanelAvailabilityClosure,
                {
                    "panel_binding_hash": binding,
                    "history_start": snapshot.sessions[0],
                    "as_of_session": snapshot.sessions[-1],
                    "availability_count": table.num_rows,
                    "artifact": reference,
                },
                "closure_hash",
            )
            self._publish_model("availability-manifests", closure.closure_hash, closure)
            closures[snapshot.snapshot_hash] = closure
        return closures, frozenset(superseded)

    def _availability_describes(self, snapshot: PanelSnapshotSource, table: pa.Table) -> bool:
        """Whether the live availability rows reproduce the snapshot's content identity.

        The origin rule above is a cheap necessary condition; this is the
        exact one. A snapshot's ``panel_content_hash`` digests its chunks'
        row identities and every availability row's hash, so availability
        that a later build restamped -- under another binding or, for a
        same-day build, under the same one -- cannot reproduce it. Only
        session, listing and row-hash columns of the chunks are read.
        """
        lineage = snapshot.lineage
        chunks = snapshot.manifest["chunks"]
        assert isinstance(chunks, list)
        prepared = [
            prepared_chunk(item, default_origin=str(snapshot.manifest["panel_binding_hash"]))
            for item in chunks
            if isinstance(item, dict)
        ]

        def load_identity_columns(chunk: PreparedPanelChunk) -> pa.Table:
            path = self._resolver.resolve_feature_panel_chunk_ref(
                uri=chunk.uri, content_hash=chunk.chunk_hash, metadata_hash=chunk.metadata_hash
            )
            return pq.read_table(path, columns=["session_date", "listing_id", "row_hash"])

        try:
            content = panel_content_identity(
                binding=PanelCompositionBinding(
                    manifest_revision=lineage["manifest_revision"],
                    sector_revision=lineage["sector_revision"],
                    catalog_hash=lineage["catalog_hash"],
                    policy_hash=lineage["policy_hash"],
                    panel_binding_hash=str(snapshot.manifest["panel_binding_hash"]),
                    history_start=date.fromisoformat(str(snapshot.manifest["history_start"])),
                    as_of_session=date.fromisoformat(str(snapshot.manifest["as_of_session"])),
                    factor_ids=tuple(sorted(snapshot.factor_ids)),
                    row_identity_basis=snapshot.row_identity_basis,
                ),
                chunks=tuple(prepared),
                load_table=load_identity_columns,
                availability=availability_rows(table),
            )
        except ValueError:
            return False
        return bool(content.panel_content_hash == str(snapshot.manifest["panel_content_hash"]))

    def _publish_row_receipts(
        self, snapshots: tuple[PanelSnapshotSource, ...]
    ) -> dict[str, PanelRowReceiptAssignment]:
        assignments: dict[str, PanelRowReceiptAssignment] = {}
        for snapshot in snapshots:
            content_hash = str(snapshot.manifest["panel_content_hash"])
            if content_hash in assignments:
                continue
            tables: list[pa.Table] = []
            chunks = snapshot.manifest["chunks"]
            assert isinstance(chunks, list)
            for item in chunks:
                assert isinstance(item, dict)
                path = self._resolver.resolve_feature_panel_chunk_ref(
                    uri=str(item["uri"]),
                    content_hash=str(item["chunk_hash"]),
                    metadata_hash=str(item["metadata_hash"]),
                )
                tables.append(
                    pq.read_table(
                        path,
                        columns=["session_date", "listing_id", "materialization_receipt_hash"],
                    )
                )
            table = pa.concat_tables(tables).combine_chunks()
            assignment_content = arrow_table_logical_hash(table)
            reference = self._publish_table(
                category="row-receipts",
                content_hash=assignment_content,
                kind="PanelRowReceiptAssignmentRows",
                table=table,
            )
            assignment = _identified(
                PanelRowReceiptAssignment,
                {
                    "panel_content_hash": content_hash,
                    "row_count": table.num_rows,
                    "artifact": reference,
                },
                "assignment_hash",
            )
            self._publish_model("row-receipt-manifests", assignment.assignment_hash, assignment)
            assignments[content_hash] = assignment
        return assignments

    def _publish_recipes(
        self,
        *,
        snapshots: tuple[PanelSnapshotSource, ...],
        sector_maps: dict[str, SectorRevisionMap],
        base_manifests: dict[str, PanelBaseValueClosureManifest],
        availability: dict[str, PanelAvailabilityClosure],
        superseded: frozenset[str],
        row_receipts: dict[str, PanelRowReceiptAssignment],
    ) -> tuple[dict[str, PanelDerivationRecipe], dict[str, str]]:
        recipes: dict[str, PanelDerivationRecipe] = {}
        blocked: dict[str, str] = {}
        for snapshot in snapshots:
            if snapshot.snapshot_hash in superseded:
                # The live tables no longer describe this snapshot. A recipe
                # captured while they still did is the valid alternative: its
                # children are immutable, so it stays exactly as recoverable
                # as the day it was sealed.
                earlier = self._earlier_recipe(snapshot)
                if earlier is not None:
                    recipes[snapshot.snapshot_hash] = earlier
                    self._report(
                        "derivation_recipes",
                        len(recipes) + len(blocked),
                        len(snapshots),
                        f"{snapshot.snapshot_hash[:12]}/EARLIER_CAPTURE",
                    )
                    continue
                blocked[snapshot.snapshot_hash] = "feature_panel.availability_superseded"
                self._report(
                    "derivation_recipes",
                    len(recipes) + len(blocked),
                    len(snapshots),
                    f"{snapshot.snapshot_hash[:12]}/AVAILABILITY_SUPERSEDED",
                )
                continue
            lineage = snapshot.lineage
            expected: list[ExpectedPanelChunk] = []
            chunks = snapshot.manifest["chunks"]
            assert isinstance(chunks, list)
            for item in chunks:
                assert isinstance(item, dict)
                path = self._resolver.resolve_feature_panel_chunk_ref(
                    uri=str(item["uri"]),
                    content_hash=str(item["chunk_hash"]),
                    metadata_hash=str(item["metadata_hash"]),
                )
                origin = item.get("origin_binding_hash")
                recorded_ranges = item.get("cross_sections") or None
                expected.append(
                    ExpectedPanelChunk(
                        year=int(item["year"]),
                        first_session=item["first_session"],
                        last_session=item["last_session"],
                        row_count=int(item["row_count"]),
                        chunk_hash=str(item["chunk_hash"]),
                        metadata_hash=str(item["metadata_hash"]),
                        physical_sha256=_file_sha256(path),
                        byte_count=path.stat().st_size,
                        uri=str(item["uri"]),
                        origin_binding_hash=str(origin) if origin is not None else None,
                        cross_sections=(
                            tuple(
                                PanelCrossSectionRecord.model_validate(entry)
                                for entry in recorded_ranges
                            )
                            if isinstance(recorded_ranges, list)
                            else None
                        ),
                    )
                )
            panel_content_hash = str(snapshot.manifest["panel_content_hash"])
            panel_binding_hash = str(snapshot.manifest["panel_binding_hash"])
            sealed_lineage = cast(dict[str, object], safe_summary_lineage(snapshot.manifest))
            recorded_origins = sealed_lineage.get("partition_origins")
            cross_section_rule = snapshot.row_identity_basis == PANEL_ROW_IDENTITY_BY_CROSS_SECTION
            partition_origins = (
                tuple(
                    PanelPartitionOrigin(
                        panel_binding_hash=binding_hash,
                        spy_revision=str(entry["spy_revision"]),
                        # Recorded only under the cross-section rule, where an
                        # origin's manifest and sector revisions may differ
                        # from the recipe's; a binding-rule origin shares them.
                        manifest_revision=(
                            str(entry["manifest_revision"]) if cross_section_rule else None
                        ),
                        sector_revision=(
                            str(entry["sector_revision"]) if cross_section_rule else None
                        ),
                        # A catalog this one only adds columns to (V92).
                        catalog_hash=(
                            str(entry["catalog_hash"])
                            if entry.get("catalog_hash") is not None
                            else None
                        ),
                    )
                    for binding_hash, entry in sorted(
                        manifest_partition_origins(snapshot.manifest).items()
                    )
                    if binding_hash != panel_binding_hash
                )
                if isinstance(recorded_origins, dict)
                else None
            )
            safe_summary = snapshot.manifest.get("safe_summary")
            if not isinstance(safe_summary, dict):
                raise ValueError("Panel recipe source has no safe summary")
            factor_summary = safe_summary.get("factor_catalog_summary")
            if not isinstance(factor_summary, dict):
                raise ValueError("Panel recipe source has no factor summary")
            row_hash_factor_ids = tuple(sorted(str(value) for value in factor_summary))
            if not set(row_hash_factor_ids).issubset(snapshot.factor_ids):
                raise ValueError("Panel recipe row-hash factor axis is outside physical schema")
            values = {
                "snapshot_hash": snapshot.snapshot_hash,
                "panel_content_hash": panel_content_hash,
                "panel_binding_hash": panel_binding_hash,
                "schema_hash": str(snapshot.manifest["schema_hash"]),
                "manifest_revision": lineage["manifest_revision"],
                "sector_revision": lineage["sector_revision"],
                "sector_map_hash": sector_maps[lineage["sector_revision"]].map_hash,
                "catalog_hash": lineage["catalog_hash"],
                "policy_hash": lineage["policy_hash"],
                "spy_revision": lineage["spy_revision"],
                "history_start": date.fromisoformat(str(snapshot.manifest["history_start"])),
                "as_of_session": date.fromisoformat(str(snapshot.manifest["as_of_session"])),
                "listing_ids": snapshot.listing_ids,
                "sessions": snapshot.sessions,
                "factor_ids": snapshot.factor_ids,
                "row_hash_factor_ids": row_hash_factor_ids,
                "session_batch_size": PANEL_SESSION_BATCH_SIZE,
                "availability_closure_hash": availability[snapshot.snapshot_hash].closure_hash,
                "row_receipt_assignment_hash": row_receipts[panel_content_hash].assignment_hash,
                "expected_chunks": tuple(expected),
                "rematerialization_policy": "CURRENT_ENVIRONMENT_BYTE_PARITY_REQUIRED",
                "partition_origins": partition_origins,
            }
            if cross_section_rule:
                values["row_identity_basis"] = snapshot.row_identity_basis
                values["membership"] = _membership_record(snapshot, safe_summary)
            reclassified = lineage.get("sector_reclassifications")
            if reclassified:
                values["sector_reclassifications"] = tuple(
                    PanelSectorReclassification.model_validate(item) for item in reclassified
                )
            verifier = ArtifactOnlyPanelRematerializer(resolver=self._resolver, store=self._store)
            matching: list[tuple[PanelBaseValueClosureManifest, PanelDerivationRecipe]] = []
            for candidate in sorted(
                base_manifests.values(),
                key=lambda item: (item.catalog_hash, item.manifest_hash),
            ):
                if not set(snapshot.factor_ids).issubset(candidate.factor_ids):
                    continue
                candidate_values = {
                    **values,
                    "base_closure_hash": candidate.manifest_hash,
                    "base_value_catalog_hash": candidate.catalog_hash,
                }
                candidate_recipe = PanelDerivationRecipe.seal(candidate_values)
                if verifier.base_closure_probe_matches(candidate_recipe):
                    matching.append((candidate, candidate_recipe))
            if not matching:
                # The live base values no longer reproduce this snapshot's
                # batches (a later correction moved them); a recipe an earlier
                # capture sealed over the values of its day is still exact.
                earlier = self._earlier_recipe(snapshot)
                if earlier is not None:
                    recipes[snapshot.snapshot_hash] = earlier
                    self._report(
                        "derivation_recipes",
                        len(recipes) + len(blocked),
                        len(snapshots),
                        f"{snapshot.snapshot_hash[:12]}/EARLIER_CAPTURE",
                    )
                    continue
                blocked[snapshot.snapshot_hash] = "feature_panel.base_closure_incomplete"
                self._report(
                    "derivation_recipes",
                    len(recipes) + len(blocked),
                    len(snapshots),
                    f"{snapshot.snapshot_hash[:12]}/BLOCKED_UNRECOVERABLE_INPUT",
                )
                continue
            same_catalog = [
                item for item in matching if item[0].catalog_hash == lineage["catalog_hash"]
            ]
            _selected_base, recipe = (same_catalog or matching)[0]
            self._publish_model("recipes", recipe.recipe_hash, recipe)
            recipes[snapshot.snapshot_hash] = recipe
            self._report(
                "derivation_recipes",
                len(recipes) + len(blocked),
                len(snapshots),
                snapshot.snapshot_hash[:12],
            )
        return recipes, blocked

    def _earlier_recipe(self, snapshot: PanelSnapshotSource) -> PanelDerivationRecipe | None:
        """The newest recipe an earlier capture sealed for this snapshot, if its children exist."""
        candidates = self._store.find_models(
            category="recipes",
            model=PanelDerivationRecipe,
            matches=lambda recipe: (
                recipe.snapshot_hash == snapshot.snapshot_hash
                and recipe.panel_content_hash == str(snapshot.manifest["panel_content_hash"])
            ),
        )
        for recipe in sorted(candidates, key=lambda item: item.recipe_hash):
            try:
                self._store.load_model(
                    category="base-manifests",
                    content_hash=recipe.base_closure_hash,
                    model=PanelBaseValueClosureManifest,
                )
                self._store.load_model(
                    category="availability-manifests",
                    content_hash=recipe.availability_closure_hash,
                    model=PanelAvailabilityClosure,
                )
            except (FileNotFoundError, ValueError):
                continue
            return recipe
        return None

    def _publish_table(
        self, *, category: str, content_hash: str, kind: str, table: pa.Table
    ) -> ClosureArtifactRef:
        target = self._store.root / category / f"{content_hash}.parquet"
        existed = target.exists()
        reference: ClosureArtifactRef = self._store.publish_parquet(
            category=category, content_hash=content_hash, kind=kind, table=table
        )
        if existed:
            self._reused += 1
        else:
            self._new_bytes += reference.byte_count
        return reference

    def _publish_model(self, category: str, content_hash: str, model: BaseModel) -> None:
        target = self._store.root / category / f"{content_hash}.json"
        existed = target.exists()
        self._store.publish_json(
            category=category,
            content_hash=content_hash,
            payload=durable_payload(model),
        )
        if existed:
            self._reused += 1
        else:
            self._new_bytes += target.stat().st_size

    def _report(self, stage: str, completed: int, total: int, item: str | None) -> None:
        if self._progress is not None:
            self._progress(stage, completed, total, item)


def _membership_record(
    snapshot: PanelSnapshotSource, safe_summary: Mapping[str, object]
) -> PanelMembershipRecord:
    """The membership a recipe records: the rows' members, the manifest's promises.

    The epochs come from the rows themselves (which listings each session
    holds), so the recipe describes what was computed; the basis ranges and
    the Universe record reference come from the sealed manifest.
    """
    membership = safe_summary.get("membership")
    if not isinstance(membership, Mapping):
        raise ValueError("Panel recipe source has no membership record")
    axis = snapshot.listing_ids
    epochs: list[PanelMembershipEpochRecord] = []
    for session in snapshot.sessions:
        members = set(snapshot.members(session))
        absent = tuple(listing_id for listing_id in axis if listing_id not in members)
        if epochs and epochs[-1].absent_listing_ids == absent:
            epochs[-1] = PanelMembershipEpochRecord(
                first_session=epochs[-1].first_session,
                last_session=session,
                absent_listing_ids=absent,
            )
        else:
            epochs.append(
                PanelMembershipEpochRecord(
                    first_session=session, last_session=session, absent_listing_ids=absent
                )
            )
    basis_ranges = membership.get("basis_ranges")
    if not isinstance(basis_ranges, list) or not basis_ranges:
        raise ValueError("Panel recipe source membership has no basis ranges")
    record_hash = membership.get("bootstrap_record_hash")
    t0 = membership.get("bootstrap_t0_session")
    return PanelMembershipRecord(
        basis_ranges=tuple(
            PanelMembershipBasisRecord(
                first_session=date.fromisoformat(
                    str(cast(Mapping[str, object], item)["first_session"])
                ),
                last_session=date.fromisoformat(
                    str(cast(Mapping[str, object], item)["last_session"])
                ),
                basis=str(cast(Mapping[str, object], item)["basis"]),
            )
            for item in basis_ranges
        ),
        epochs=tuple(epochs),
        journal_sequence=int(str(membership.get("journal_sequence", 0))),
        bootstrap_record_hash=str(record_hash) if record_hash is not None else None,
        bootstrap_t0_session=date.fromisoformat(str(t0)) if t0 is not None else None,
        source_exclusions=(
            tuple(
                PanelSourceExclusion.from_payload(item) for item in membership["source_exclusions"]
            )
            if membership.get("source_exclusions")
            else None
        ),
    )


def safe_summary_lineage(manifest: Mapping[str, object]) -> Mapping[str, object]:
    """The sealed lineage mapping of a snapshot manifest, values untouched."""
    summary = manifest.get("safe_summary")
    lineage = summary.get("lineage") if isinstance(summary, Mapping) else None
    if not isinstance(lineage, Mapping):
        raise ValueError("Panel recipe source has no lineage")
    return lineage


def _identified[ContractT: BaseModel](
    model: type[ContractT], values: dict[str, Any], hash_field: str
) -> ContractT:
    provisional = model.model_construct(**values, **{hash_field: "0" * 64})
    identity = durable_payload(provisional, exclude={hash_field})
    return cast(
        ContractT,
        model.model_validate({**values, hash_field: canonical_hash(identity)}),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "PanelClosurePublication",
    "PanelClosurePublisher",
]
