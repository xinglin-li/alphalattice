"""Artifact-only clean-room rematerialization for historical Feature Panels."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.feature_engine.contracts import (
    PANEL_ROW_IDENTITY_BY_BINDING,
    PANEL_ROW_IDENTITY_BY_CROSS_SECTION,
    FeaturePanelBinding,
)
from alphalattice.foundation.feature_engine.panels.artifacts import (
    PanelCompositionBinding,
    PreparedPanelChunk,
    panel_content_identity,
)
from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
    PanelClosureArtifactStore,
    ordered_key_hash,
    ordered_value_hash,
)
from alphalattice.foundation.feature_engine.panels.closure_contracts import (
    PanelAvailabilityClosure,
    PanelBaseValueClosureManifest,
    PanelDerivationRecipe,
    PanelRowReceiptAssignment,
    SectorRevisionMap,
)
from alphalattice.foundation.feature_engine.panels.identity import (
    CROSS_SECTION_IDENTITY_COLUMN,
    hash_panel_rows_with_recorded_axes,
    panel_chunk_hash,
    panel_schema_hash,
)
from alphalattice.foundation.feature_engine.producers.preprocessing.catalog import (
    ROBUST_SECTOR_NEUTRAL_Z,
)
from alphalattice.foundation.feature_engine.producers.preprocessing.contracts import (
    PanelClipObservationRecord,
)
from alphalattice.foundation.feature_engine.producers.preprocessing.robust_cross_section import (
    PanelCrossSectionKernel,
    PanelMaterialization,
    allows_missing_source_rows,
    panel_ordered_members_identity,
)
from alphalattice.kernel.quant.sector_history import SectorHistory, SectorReclassification


class PanelRematerializationMismatch(ValueError):
    """A closure cannot reproduce its admitted Panel exactly."""


@dataclass(frozen=True)
class RematerializedChunkEvidence:
    """Record logical and physical parity evidence for one rebuilt chunk."""

    year: int
    row_count: int
    chunk_hash: str
    metadata_hash: str
    physical_sha256: str
    byte_count: int


@dataclass(frozen=True)
class PanelRematerializationResult:
    """Report reconstructed Panel content and chunk parity."""

    snapshot_hash: str
    panel_content_hash: str
    chunks: tuple[RematerializedChunkEvidence, ...]
    logical_parity: bool
    physical_parity: bool


class ArtifactOnlyPanelRematerializer:
    """Rebuild a Panel using immutable closure artifacts and in-memory math only.

    Deliberately pinned to the frozen legacy transformation rather than resolved
    through the installed preprocessing catalog. Rematerialization exists to
    reproduce a Panel that already exists, so it must run the method that Panel
    was built by -- not whichever method is installed today. Resolving here
    would silently rebuild historical evidence under a newer recipe and call the
    result a reproduction.

    This is therefore *not* a general new-method executor, and a new
    preprocessing recipe must not be routed through it. New Panels are built by
    ``SectorNeutralPanelMaterializer``, which resolves the catalog.
    """

    LEGACY_FROZEN_RECIPE_ID = ROBUST_SECTOR_NEUTRAL_Z

    def __init__(self, *, resolver: ArtifactResolver, store: PanelClosureArtifactStore) -> None:
        """Read immutable closure artifacts through the resolver and store."""
        self._resolver = resolver
        self._store = store
        self._kernel = PanelCrossSectionKernel()

    def read_base_formula_values(
        self,
        recipe_hash: str,
        *,
        sessions: tuple[date, ...],
        listing_ids: tuple[str, ...],
        factor_ids: tuple[str, ...],
    ) -> pa.Table:
        """Read an identity-checked subset of the frozen pre-Panel Formula closure.

        The closure already is the durable owner of the Formula values used to
        materialize a Panel.  Downstream development methods sometimes need
        those values *before* the Panel's cross-sectional preprocessing (for
        example, to install a different fold-owned final scale).  Serving that
        surface here avoids making either Alpha or a Campaign a second decoder
        for ``base-values`` bit patterns and validity masks.

        This is read-only and deliberately takes a recipe identity rather than
        a filesystem path.  The recipe binds the catalog, axes and base closure;
        every value chunk is re-hashed by ``_load_base_year`` before release.
        """
        recipe = self._store.load_model(
            category="recipes", content_hash=recipe_hash, model=PanelDerivationRecipe
        )
        base = self._store.load_model(
            category="base-manifests",
            content_hash=recipe.base_closure_hash,
            model=PanelBaseValueClosureManifest,
        )
        if (
            base.catalog_hash != recipe.base_value_catalog_hash
            or not sessions
            or sessions != tuple(sorted(set(sessions)))
            or not set(sessions).issubset(recipe.sessions)
            or not listing_ids
            or listing_ids != tuple(sorted(set(listing_ids)))
            or set(listing_ids) != set(recipe.listing_ids)
            or not factor_ids
            or factor_ids != tuple(sorted(set(factor_ids)))
            or not set(factor_ids).issubset(recipe.factor_ids)
        ):
            raise PanelRematerializationMismatch("feature_panel.base_formula_request_invalid")
        frames = tuple(
            self._load_base_year(
                base=base,
                year=year,
                sessions=tuple(value for value in sessions if value.year == year),
                listing_ids=listing_ids,
                factor_ids=factor_ids,
                recipe=recipe,
            )
            for year in sorted({value.year for value in sessions})
        )
        frame = pd.concat(frames, ignore_index=True).sort_values(
            ["session_date", "listing_id"], kind="mergesort"
        )
        if len(frame) != recipe.expected_row_count(sessions):
            raise PanelRematerializationMismatch("feature_panel.base_closure_incomplete")
        return pa.Table.from_pandas(frame.reset_index(drop=True), preserve_index=False)

    def rematerialize(
        self,
        recipe_hash: str,
        *,
        progress: Callable[[int, int, int], object] | None = None,
    ) -> PanelRematerializationResult:
        """Rebuild a Panel from its frozen recipe and verify chunk parity."""
        recipe = self._store.load_model(
            category="recipes", content_hash=recipe_hash, model=PanelDerivationRecipe
        )
        base = self._store.load_model(
            category="base-manifests",
            content_hash=recipe.base_closure_hash,
            model=PanelBaseValueClosureManifest,
        )
        sector_map = self._store.load_model(
            category="sector-maps",
            content_hash=recipe.sector_map_hash,
            model=SectorRevisionMap,
        )
        availability = self._store.load_model(
            category="availability-manifests",
            content_hash=recipe.availability_closure_hash,
            model=PanelAvailabilityClosure,
        )
        receipts = self._store.load_model(
            category="row-receipt-manifests",
            content_hash=recipe.row_receipt_assignment_hash,
            model=PanelRowReceiptAssignment,
        )
        self._validate_children(recipe, base, sector_map, availability, receipts)
        availability_table = self._store.load_parquet(
            category="availability", reference=availability.artifact
        ).replace_schema_metadata(None)
        receipt_table = self._store.load_parquet(
            category="row-receipts", reference=receipts.artifact
        ).replace_schema_metadata(None)
        sector_history = recipe_sector_history(recipe, sector_map)
        binding = FeaturePanelBinding.create(
            manifest_revision=recipe.manifest_revision,
            sector_revision=recipe.sector_revision,
            catalog_hash=recipe.catalog_hash,
            spy_revision=recipe.spy_revision,
            policy_hash=recipe.policy_hash,
        )
        if binding.panel_binding_hash != recipe.panel_binding_hash:
            raise PanelRematerializationMismatch("feature_panel.recipe_input_mismatch")
        prepared: list[PreparedPanelChunk] = []
        evidence: list[RematerializedChunkEvidence] = []
        temporary_paths: dict[int, Path] = {}
        base_cache: OrderedDict[int, pd.DataFrame] = OrderedDict()
        with tempfile.TemporaryDirectory(prefix="panel-rematerialization-") as directory:
            temp_root = Path(directory)
            for expected in recipe.expected_chunks:
                rows, actual_availability = self._rematerialize_year(
                    recipe=recipe,
                    base=base,
                    year=expected.year,
                    sector_history=sector_history,
                    availability_table=availability_table,
                    base_cache=base_cache,
                )
                self._verify_availability(
                    expected_year=expected.year,
                    actual=actual_availability,
                    frozen=availability_table,
                )
                raw = self._attach_frozen_receipts(
                    rows=rows,
                    receipt_table=receipt_table,
                    year=expected.year,
                    factor_ids=recipe.factor_ids,
                )
                if recipe.identity_basis == PANEL_ROW_IDENTITY_BY_CROSS_SECTION:
                    # Each row binds its session's recorded cross-section; the
                    # provenance columns are the writer's, which a reused
                    # partition records as its origin binding's revisions.
                    origin = expected.origin_binding_hash or recipe.panel_binding_hash
                    manifest_revision, sector_revision, _spy = recipe.origin_revisions(origin)
                    raw = raw.append_column(
                        CROSS_SECTION_IDENTITY_COLUMN,
                        pa.array(
                            [
                                recipe.cross_section_identity(cast(date, session))
                                for session in raw.column("session_date").to_pylist()
                            ],
                            pa.string(),
                        ),
                    )
                else:
                    manifest_revision = recipe.manifest_revision
                    sector_revision = recipe.sector_revision
                hashed = hash_panel_rows_with_recorded_axes(
                    raw,
                    manifest_revision=manifest_revision,
                    sector_revision=sector_revision,
                    catalog_hash=recipe.catalog_hash,
                    policy_hash=recipe.policy_hash,
                    row_hash_factor_ids=recipe.row_hash_factor_ids,
                    output_factor_ids=recipe.factor_ids,
                    identity_basis=recipe.identity_basis,
                )
                if hashed.num_rows != expected.row_count:
                    raise PanelRematerializationMismatch(
                        "feature_panel.rematerialization_content_mismatch"
                    )
                schema_hash = panel_schema_hash(hashed.schema)
                # The chunk was hashed and written by the build whose binding
                # the recipe records as its origin -- the snapshot's own unless
                # the snapshot reused the partition from an earlier build.
                origin_binding = expected.origin_binding_hash or recipe.panel_binding_hash
                chunk_hash = panel_chunk_hash(
                    hashed, panel_binding_hash=origin_binding, year=expected.year
                )
                if schema_hash != recipe.schema_hash or chunk_hash != expected.chunk_hash:
                    raise PanelRematerializationMismatch(
                        "feature_panel.rematerialization_content_mismatch"
                    )
                metadata = {
                    b"alphalattice.snapshot_kind": b"FeaturePanelChunk",
                    b"alphalattice.chunk_hash": chunk_hash.encode("ascii"),
                    b"alphalattice.panel_binding_hash": origin_binding.encode("ascii"),
                    b"alphalattice.calendar_year": str(expected.year).encode("ascii"),
                    b"alphalattice.schema_hash": schema_hash.encode("ascii"),
                }
                output = temp_root / f"{expected.year}.parquet"
                pq.write_table(hashed.replace_schema_metadata(metadata), output, compression="zstd")
                physical_sha = _file_sha256(output)
                metadata_hash = _metadata_hash(output)
                if metadata_hash != expected.metadata_hash:
                    raise PanelRematerializationMismatch(
                        "feature_panel.rematerialization_physical_mismatch"
                    )
                if (
                    physical_sha != expected.physical_sha256
                    or output.stat().st_size != expected.byte_count
                ):
                    raise PanelRematerializationMismatch(
                        "feature_panel.rematerialization_physical_mismatch"
                    )
                temporary_paths[expected.year] = output
                prepared.append(
                    PreparedPanelChunk(
                        year=expected.year,
                        first_session=expected.first_session,
                        last_session=expected.last_session,
                        row_count=expected.row_count,
                        chunk_hash=chunk_hash,
                        metadata_hash=metadata_hash,
                        uri=f"temporary://{expected.year}",
                    )
                )
                evidence.append(
                    RematerializedChunkEvidence(
                        year=expected.year,
                        row_count=hashed.num_rows,
                        chunk_hash=chunk_hash,
                        metadata_hash=metadata_hash,
                        physical_sha256=physical_sha,
                        byte_count=output.stat().st_size,
                    )
                )
                if progress is not None:
                    progress(len(evidence), len(recipe.expected_chunks), expected.year)
            content = panel_content_identity(
                binding=PanelCompositionBinding(
                    manifest_revision=recipe.manifest_revision,
                    sector_revision=recipe.sector_revision,
                    catalog_hash=recipe.catalog_hash,
                    policy_hash=recipe.policy_hash,
                    panel_binding_hash=recipe.panel_binding_hash,
                    history_start=recipe.history_start,
                    as_of_session=recipe.as_of_session,
                    # The current composition contract has a sorted factor
                    # axis. Historical physical order was already replayed for
                    # row hashes, and factor IDs are not part of this legacy
                    # panel-content digest preamble.
                    factor_ids=tuple(sorted(recipe.factor_ids)),
                    row_identity_basis=recipe.identity_basis,
                ),
                chunks=prepared,
                load_table=lambda chunk: pq.read_table(
                    temporary_paths[chunk.year]
                ).replace_schema_metadata(None),
                availability=availability_rows(availability_table),
            )
            if content.panel_content_hash != recipe.panel_content_hash:
                raise PanelRematerializationMismatch(
                    "feature_panel.rematerialization_content_mismatch"
                )
        return PanelRematerializationResult(
            snapshot_hash=recipe.snapshot_hash,
            panel_content_hash=recipe.panel_content_hash,
            chunks=tuple(evidence),
            logical_parity=True,
            physical_parity=True,
        )

    def base_closure_probe_matches(self, recipe: PanelDerivationRecipe) -> bool:
        """Check bounded frozen receipts before admitting a recipe's base source.

        A catalog hash alone cannot recover values that were overwritten by a
        later correction within the same generation. The probe is artifact-only
        and replays the earliest frozen materialization receipt group and
        every group that wrote a cell of the latest session; a whole group
        must re-derive its receipt, a partially superseded one must match
        its retained cells one by one. Full chunk parity remains the
        publication authority.
        """
        base = self._store.load_model(
            category="base-manifests",
            content_hash=recipe.base_closure_hash,
            model=PanelBaseValueClosureManifest,
        )
        sector_map = self._store.load_model(
            category="sector-maps",
            content_hash=recipe.sector_map_hash,
            model=SectorRevisionMap,
        )
        availability = self._store.load_model(
            category="availability-manifests",
            content_hash=recipe.availability_closure_hash,
            model=PanelAvailabilityClosure,
        )
        if base.catalog_hash != recipe.base_value_catalog_hash:
            raise PanelRematerializationMismatch("feature_panel.recipe_input_mismatch")
        frozen_rows = availability_rows(
            self._store.load_parquet(
                category="availability", reference=availability.artifact
            ).replace_schema_metadata(None)
        )
        by_receipt: dict[str, list[dict[str, object]]] = {}
        for item in frozen_rows:
            by_receipt.setdefault(str(item["materialization_receipt_hash"]), []).append(item)
        ordered = sorted(
            by_receipt.items(),
            key=lambda item: (
                min(str(row["session_date"]) for row in item[1]),
                item[0],
            ),
        )
        # The earliest group, plus every group holding a cell of the latest
        # session: a correction's lookback lands there, so the batch that
        # wrote the newest cells is checked whether or not it also owns the
        # latest first session, and a partially superseded batch does not
        # stand in for the batch that superseded it.
        latest_session = max(str(row["session_date"]) for row in frozen_rows)
        selected = [ordered[0]] + [
            group
            for group in ordered[1:]
            if any(str(row["session_date"]) == latest_session for row in group[1])
        ]
        sector_history = recipe_sector_history(recipe, sector_map)
        cache: OrderedDict[int, pd.DataFrame] = OrderedDict()
        try:
            for receipt_hash, receipt_rows in selected:
                materialization = self._materialize_receipt(
                    recipe=recipe,
                    base=base,
                    receipt_rows=receipt_rows,
                    sector_history=sector_history,
                    binding=self._receipt_binding(recipe, receipt_rows),
                    base_cache=cache,
                )
                self._verify_receipt_group(
                    receipt_hash,
                    receipt_rows,
                    materialization=materialization,
                    whole=self._group_is_whole(receipt_hash, receipt_rows),
                )
        except PanelRematerializationMismatch:
            return False
        return True

    def _group_is_whole(self, receipt_hash: str, receipt_rows: list[dict[str, object]]) -> bool:
        """Whether the closure attributes to a receipt every cell its batch measured.

        A batch receipt hashes the availability of the whole batch, so it
        re-derives only from the whole batch. A later build that recomputed
        some of a batch's cells (a correction reaching a few factors of a few
        sessions in a reused year) left the receipt owning the rest, and the
        rest cannot re-derive a receipt that also covered what was replaced;
        those cells are verified one by one against the frozen availability
        and through the chunk's row identities instead. The batch's scope is
        its clip-observation record, persisted under the receipt by the build
        that ran it: every retained cell must lie inside that scope -- a row
        the batch never measured cannot belong to it, whatever the count --
        and the group is whole exactly when the scope is fully retained. A
        receipt without a record predates partition reuse; nothing then
        superseded part of a batch except a same-day retry, and such a group
        must re-derive its receipt as it always had to.
        """
        record = self._receipt_observation(receipt_hash, receipt_rows)
        if record is None:
            return True
        return len(receipt_rows) == len(record.sessions) * len(record.factor_ids)

    def _receipt_observation(
        self, receipt_hash: str, receipt_rows: list[dict[str, object]]
    ) -> PanelClipObservationRecord | None:
        """One reader verifies that every attributed cell belongs to its batch."""
        payload = self._resolver.load_panel_clip_observation(receipt_hash)
        if payload is None:
            return None
        record = PanelClipObservationRecord.model_validate(payload)
        if record.receipt_hash != receipt_hash:
            raise PanelRematerializationMismatch("feature_panel.recipe_input_mismatch")
        scope = {
            (session, factor_id) for session in record.sessions for factor_id in record.factor_ids
        }
        retained = {(str(item["session_date"]), str(item["factor_id"])) for item in receipt_rows}
        if len(retained) != len(receipt_rows) or not retained <= scope:
            raise PanelRematerializationMismatch("feature_panel.recipe_input_mismatch")
        return record

    def _verify_receipt_group(
        self,
        receipt_hash: str,
        receipt_rows: list[dict[str, object]],
        *,
        materialization: PanelMaterialization,
        whole: bool,
    ) -> None:
        """A whole group re-derives its receipt; a partial one matches cell by cell."""
        if whole:
            if materialization.receipt_hash != receipt_hash:
                raise PanelRematerializationMismatch(
                    "feature_panel.rematerialization_content_mismatch"
                )
            return
        self._verifyavailability_rows(expected=receipt_rows, actual=materialization.availability)

    @staticmethod
    def _receipt_binding(
        recipe: PanelDerivationRecipe, receipt_rows: list[dict[str, object]]
    ) -> FeaturePanelBinding:
        """The exact binding one materialization batch ran under.

        A batch receipt hashes the binding and the availability it produced,
        and every availability row carries that binding. A snapshot that
        reuses partitions holds batches of several builds, so the binding is
        read from the group's own rows and rebuilt from the recipe's recorded
        origins, under the catalog the origin records when it is another;
        a group whose rows disagree, or whose binding no origin
        explains, cannot be replayed.
        """
        binding_hashes = {str(item["panel_binding_hash"]) for item in receipt_rows}
        if len(binding_hashes) != 1:
            raise PanelRematerializationMismatch("feature_panel.recipe_input_mismatch")
        binding_hash = binding_hashes.pop()
        try:
            manifest_revision, sector_revision, spy_revision = recipe.origin_revisions(binding_hash)
            catalog_hash = recipe.origin_catalog_hash(binding_hash)
        except ValueError as exc:
            raise PanelRematerializationMismatch("feature_panel.recipe_input_mismatch") from exc
        binding = FeaturePanelBinding.create(
            manifest_revision=manifest_revision,
            sector_revision=sector_revision,
            catalog_hash=catalog_hash,
            spy_revision=spy_revision,
            policy_hash=recipe.policy_hash,
        )
        if binding.panel_binding_hash != binding_hash:
            raise PanelRematerializationMismatch("feature_panel.recipe_input_mismatch")
        return binding

    def _rematerialize_year(
        self,
        *,
        recipe: PanelDerivationRecipe,
        base: PanelBaseValueClosureManifest,
        year: int,
        sector_history: SectorHistory,
        availability_table: pa.Table,
        base_cache: OrderedDict[int, pd.DataFrame],
    ) -> tuple[pd.DataFrame, list[dict[str, object]]]:
        year_sessions = tuple(session for session in recipe.sessions if session.year == year)
        # One row per member of each session, in physical order; a Panel under
        # the binding rule holds the whole axis on every session.
        keys = [
            (session, listing_id)
            for session in year_sessions
            for listing_id in sorted(recipe.members(session))
        ]
        positions = {key: index for index, key in enumerate(keys)}
        factor_values: dict[str, np.ndarray] = {
            factor_id: np.full(len(keys), np.nan, dtype=float) for factor_id in recipe.factor_ids
        }
        frozen_availability = availability_rows(availability_table)
        by_receipt: dict[str, list[dict[str, object]]] = {}
        for item in frozen_availability:
            receipt_hash = str(item["materialization_receipt_hash"])
            by_receipt.setdefault(receipt_hash, []).append(item)
        actual_availability: list[dict[str, object]] = []
        for receipt_hash, receipt_rows in sorted(by_receipt.items()):
            receipt_sessions = tuple(
                sorted({date.fromisoformat(str(item["session_date"])) for item in receipt_rows})
            )
            selected = tuple(session for session in receipt_sessions if session.year == year)
            if not selected:
                continue
            receipt_factors = tuple(sorted({str(item["factor_id"]) for item in receipt_rows}))
            # The cells the closure attributes to this receipt. A group
            # another batch partially superseded is replayed over its own
            # sessions and factors, and only the cells it still owns are
            # taken from that replay; the rest belong to the later batch.
            owned = {
                (date.fromisoformat(str(item["session_date"])), str(item["factor_id"]))
                for item in receipt_rows
            }
            materialization = self._materialize_receipt(
                recipe=recipe,
                base=base,
                receipt_rows=receipt_rows,
                sector_history=sector_history,
                binding=self._receipt_binding(recipe, receipt_rows),
                base_cache=base_cache,
            )
            self._verify_receipt_group(
                receipt_hash,
                receipt_rows,
                materialization=materialization,
                whole=self._group_is_whole(receipt_hash, receipt_rows),
            )
            row_sessions = [
                date.fromisoformat(str(value)) for value in materialization.rows["session_date"]
            ]
            row_listings = [str(value) for value in materialization.rows["listing_id"]]
            targets = np.array(
                [
                    positions.get((session, listing_id), -1)
                    for session, listing_id in zip(row_sessions, row_listings, strict=True)
                ],
                dtype=int,
            )
            in_year = np.array([session in selected for session in row_sessions], dtype=bool)
            for factor_id in receipt_factors:
                owned_sessions = {session for session in selected if (session, factor_id) in owned}
                if not owned_sessions:
                    continue
                take = in_year & np.array(
                    [session in owned_sessions for session in row_sessions], dtype=bool
                )
                if (targets[take] < 0).any():
                    raise PanelRematerializationMismatch("feature_panel.recipe_input_mismatch")
                factor_values[factor_id][targets[take]] = materialization.rows[
                    factor_id
                ].to_numpy()[take]
            actual_availability.extend(
                item
                for item in materialization.availability
                if date.fromisoformat(str(item["session_date"])).year == year
                and (date.fromisoformat(str(item["session_date"])), str(item["factor_id"])) in owned
            )
        rows = pd.DataFrame(
            {
                "listing_id": np.asarray([listing_id for _s, listing_id in keys], dtype=object),
                "session_date": np.asarray(
                    [session.isoformat() for session, _l in keys], dtype=object
                ),
                **factor_values,
            }
        )
        if not actual_availability:
            raise PanelRematerializationMismatch("feature_panel.base_closure_incomplete")
        return rows, actual_availability

    def _materialize_receipt(
        self,
        *,
        recipe: PanelDerivationRecipe,
        base: PanelBaseValueClosureManifest,
        receipt_rows: list[dict[str, object]],
        sector_history: SectorHistory,
        binding: FeaturePanelBinding,
        base_cache: OrderedDict[int, pd.DataFrame],
    ) -> PanelMaterialization:
        receipt_sessions = tuple(
            sorted({date.fromisoformat(str(item["session_date"])) for item in receipt_rows})
        )
        # A build materializes one run of one Sector map at a time, so each receipt's
        # sessions read one map; a receipt across a reclassification is not one this build made.
        runs = sector_history.runs(receipt_sessions)
        if len(runs) != 1:
            raise PanelRematerializationMismatch("feature_panel.recipe_input_mismatch")
        sector_by_listing_id = runs[0][1]
        receipt_factors = tuple(sorted({str(item["factor_id"]) for item in receipt_rows}))
        legacy_layout = self._legacy_replay_layout(recipe, receipt_rows, binding)
        frame_parts: list[pd.DataFrame] = []
        for batch_year in sorted({session.year for session in receipt_sessions}):
            batch_sessions = tuple(
                session for session in receipt_sessions if session.year == batch_year
            )
            source = self._cached_base_year(
                cache=base_cache,
                base=base,
                recipe=recipe,
                year=batch_year,
            )
            frame_parts.append(source.loc[source["session_date"].isin(batch_sessions)])
        return self._kernel.materialize(
            feature_rows=pd.concat(frame_parts, ignore_index=True),
            active_listing_ids=recipe.listing_ids,
            sector_by_listing_id=sector_by_listing_id,
            factor_ids=receipt_factors,
            binding=binding,
            legacy_epoch_session_counts=legacy_layout,
            members_by_session=(
                {session: recipe.members(session) for session in receipt_sessions}
                if recipe.membership is not None
                else None
            ),
            source_exclusions_by_session=(
                {
                    session: tuple(
                        sorted(
                            {
                                item.listing_id
                                for item in recipe.membership.source_exclusions
                                if item.first_session <= session <= item.last_session
                                and item.listing_id in recipe.members(session)
                            }
                        )
                    )
                    for session in receipt_sessions
                }
                if recipe.membership is not None and recipe.membership.source_exclusions
                else None
            ),
        )

    def _legacy_replay_layout(
        self,
        recipe: PanelDerivationRecipe,
        rows: list[dict[str, object]],
        binding: FeaturePanelBinding,
    ) -> dict[date, int] | None:
        """Recover numerical row layout from a bound batch, not the retained slice."""
        if allows_missing_source_rows(binding.policy_hash):
            return None
        receipts = {str(row["materialization_receipt_hash"]) for row in rows}
        if len(receipts) != 1:
            raise PanelRematerializationMismatch("feature_panel.recipe_input_mismatch")
        receipt = receipts.pop()
        record = self._receipt_observation(receipt, rows)
        if record is None:
            return None  # Historical whole-receipt verification still must pass.
        if record.panel_binding_hash != binding.panel_binding_hash:
            raise PanelRematerializationMismatch("feature_panel.recipe_input_mismatch")
        try:
            dates = tuple(date.fromisoformat(day) for day in record.sessions)
        except ValueError as exc:
            raise PanelRematerializationMismatch(
                "feature_panel.legacy_replay_layout_unproven"
            ) from exc
        retained = {date.fromisoformat(str(row["session_date"])) for row in rows}
        try:
            members = [recipe.members(day) for day in dates]
        except KeyError as exc:
            raise PanelRematerializationMismatch(
                "feature_panel.legacy_replay_layout_unproven"
            ) from exc
        epochs: list[tuple[int, int, tuple[str, ...]]] = []
        start = 0
        for end in range(1, len(dates) + 1):
            if end == len(dates) or members[end] != members[start]:
                epochs.append((start, end, members[start]))
                start = end
        # A later recipe can have a wider coverage union. Reconstruct possible
        # historical unions from its recorded epochs and require an exact match
        # to the batch's axis identity. No numerical trial or guessed roster.
        axes = {recipe.listing_ids}
        if len(epochs) == 1:
            axes.add(members[0])
        if recipe.membership is not None:
            union: set[str] = set()
            for epoch in recipe.membership.epochs:
                union.update(recipe.members(epoch.first_session))
                axes.add(tuple(name for name in recipe.listing_ids if name in union))
        identities = set()
        for axis in axes:
            positions = {name: i for i, name in enumerate(axis)}
            if any(set(names) - positions.keys() for _, _, names in epochs):
                continue
            identities.add(
                panel_ordered_members_identity(
                    axis,
                    dates,
                    tuple(
                        (a, b, np.asarray([positions[name] for name in names], dtype=np.int64))
                        for a, b, names in epochs
                    ),
                    uniform=all(names == axis for _, _, names in epochs),
                )
            )
        if record.ordered_listing_ids_hash not in identities:
            raise PanelRematerializationMismatch("feature_panel.legacy_replay_layout_unproven")
        return {
            day: stop - begin
            for begin, stop, _ in epochs
            for day in dates[begin:stop]
            if day in retained
        }

    def _cached_base_year(
        self,
        *,
        cache: OrderedDict[int, pd.DataFrame],
        base: PanelBaseValueClosureManifest,
        recipe: PanelDerivationRecipe,
        year: int,
    ) -> pd.DataFrame:
        if year in cache:
            frame = cache.pop(year)
            cache[year] = frame
            return frame
        sessions = tuple(session for session in recipe.sessions if session.year == year)
        frame = self._load_base_year(
            base=base,
            year=year,
            sessions=sessions,
            listing_ids=recipe.listing_ids,
            factor_ids=recipe.factor_ids,
            recipe=recipe,
        )
        frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.date
        cache[year] = frame
        while len(cache) > 3:
            cache.popitem(last=False)
        return frame

    def _load_base_year(
        self,
        *,
        base: PanelBaseValueClosureManifest,
        year: int,
        sessions: tuple[date, ...],
        listing_ids: tuple[str, ...],
        factor_ids: tuple[str, ...],
        recipe: PanelDerivationRecipe,
    ) -> pd.DataFrame:
        key_descriptor = next((item for item in base.key_chunks if item.year == year), None)
        if key_descriptor is None:
            raise PanelRematerializationMismatch("feature_panel.base_closure_incomplete")
        keys = self._store.load_parquet(
            category="base-keys", reference=key_descriptor.artifact
        ).replace_schema_metadata(None)
        if (
            ordered_key_hash(
                keys.column("session_date").to_pylist(),
                (str(value) for value in keys.column("listing_id").to_pylist()),
            )
            != key_descriptor.key_hash
        ):
            raise PanelRematerializationMismatch("feature_panel.base_closure_incomplete")
        mask = pc.and_(
            pc.is_in(keys.column("session_date"), value_set=pa.array(sessions, type=pa.date32())),
            pc.is_in(keys.column("listing_id"), value_set=pa.array(listing_ids)),
        )
        filtered_keys = keys.filter(mask)
        # The closure holds each session's members (the dense grid under the
        # binding rule); the requested listings must be the recipe's axis.
        expected_rows = recipe.expected_row_count(sessions)
        if set(listing_ids) != set(recipe.listing_ids) or filtered_keys.num_rows != expected_rows:
            raise PanelRematerializationMismatch("feature_panel.base_closure_incomplete")
        frame = filtered_keys.to_pandas()
        absent_positions: np.ndarray = np.zeros(len(frame), dtype=bool)
        if base.absent_source_keys is not None:
            absent = self._store.load_parquet(
                category="base-keys", reference=base.absent_source_keys
            ).replace_schema_metadata(None)
            if absent.column_names != ["session_date", "listing_id"]:
                raise PanelRematerializationMismatch("feature_panel.base_absence_axis_invalid")
            absent_keys = tuple(
                zip(
                    absent["session_date"].to_pylist(),
                    absent["listing_id"].to_pylist(),
                    strict=True,
                )
            )
            if absent_keys != tuple(sorted(set(absent_keys))) or any(
                session.year not in {item.year for item in base.key_chunks}
                for session, _listing in absent_keys
            ):
                raise PanelRematerializationMismatch("feature_panel.base_absence_axis_invalid")
            year_absences = absent.filter(pc.equal(pc.year(absent["session_date"]), year))
            if year_absences.join(
                keys, keys=["session_date", "listing_id"], join_type="left anti"
            ).num_rows:
                raise PanelRematerializationMismatch("feature_panel.base_absence_axis_invalid")
            if (
                ordered_key_hash(
                    absent["session_date"].to_pylist(), absent["listing_id"].to_pylist()
                )
                != base.absent_source_keys.content_hash
            ):
                raise PanelRematerializationMismatch("feature_panel.base_absence_identity_mismatch")
            absent_index = pd.MultiIndex.from_frame(absent.to_pandas())
            absent_positions = np.asarray(pd.MultiIndex.from_frame(frame).isin(absent_index))
            if absent_positions.any() and not allows_missing_source_rows(recipe.policy_hash):
                raise PanelRematerializationMismatch("feature_panel.legacy_base_source_missing")
        for factor_id in factor_ids:
            descriptor = next(
                (
                    item
                    for item in base.value_chunks
                    if item.year == year and item.factor_id == factor_id
                ),
                None,
            )
            if descriptor is None or descriptor.key_hash != key_descriptor.key_hash:
                raise PanelRematerializationMismatch("feature_panel.base_closure_incomplete")
            table = self._store.load_parquet(
                category="base-values", reference=descriptor.artifact
            ).replace_schema_metadata(None)
            bits = table.column("value_bits").combine_chunks().to_numpy(zero_copy_only=False)
            valid = table.column("is_valid").combine_chunks().to_numpy(zero_copy_only=False)
            values = np.asarray(bits, dtype="<u8").view("<f8")
            if (
                ordered_value_hash(
                    factor_id=factor_id,
                    key_hash=descriptor.key_hash,
                    values=values,
                    valid=np.asarray(valid, dtype=np.bool_),
                )
                != descriptor.value_hash
            ):
                raise PanelRematerializationMismatch("feature_panel.base_closure_incomplete")
            filtered_values = values[np.asarray(mask.to_numpy(zero_copy_only=False), dtype=bool)]
            filtered_valid = np.asarray(valid, dtype=bool)[
                np.asarray(mask.to_numpy(zero_copy_only=False), dtype=bool)
            ]
            if filtered_valid[absent_positions].any():
                raise PanelRematerializationMismatch("feature_panel.base_absence_has_value")
            restored = filtered_values.copy()
            restored[~filtered_valid] = np.nan
            frame[factor_id] = restored
        return cast(pd.DataFrame, frame)

    @staticmethod
    def _attach_frozen_receipts(
        *,
        rows: pd.DataFrame,
        receipt_table: pa.Table,
        year: int,
        factor_ids: tuple[str, ...],
    ) -> pa.Table:
        mask = pc.equal(pc.year(receipt_table.column("session_date")), pa.scalar(year))
        receipts = receipt_table.filter(mask).to_pandas()
        receipts["session_date"] = pd.to_datetime(receipts["session_date"]).dt.date
        frame = rows.copy()
        frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.date
        merged = frame.merge(
            receipts,
            on=["session_date", "listing_id"],
            how="left",
            validate="one_to_one",
        )
        if merged["materialization_receipt_hash"].isna().any():
            raise PanelRematerializationMismatch("feature_panel.recipe_input_mismatch")
        columns = [
            "session_date",
            "listing_id",
            "materialization_receipt_hash",
            *factor_ids,
        ]
        return pa.Table.from_pandas(merged.loc[:, columns], preserve_index=False)

    @staticmethod
    def _verify_availability(
        *, expected_year: int, actual: list[dict[str, object]], frozen: pa.Table
    ) -> None:
        mask = pc.equal(pc.year(frozen.column("session_date")), pa.scalar(expected_year))
        ArtifactOnlyPanelRematerializer._verifyavailability_rows(
            expected=availability_rows(frozen.filter(mask)), actual=actual
        )

    @staticmethod
    def _verifyavailability_rows(
        *, expected: list[dict[str, object]], actual: list[dict[str, object]]
    ) -> None:
        """Every frozen availability row must be reproduced, field by field, by the replay."""
        actual_by_key = {
            (str(item["session_date"]), str(item["factor_id"])): item for item in actual
        }
        fields = (
            "universe_size",
            "computed_count",
            "coverage",
            "winsor_lower",
            "winsor_upper",
            "residual_median",
            "residual_mad",
            "status",
            "reason",
            "small_sector_warning",
        )
        expected_keys = {(str(item["session_date"]), str(item["factor_id"])) for item in expected}
        if len(expected_keys) != len(expected) or not expected_keys.issubset(actual_by_key):
            raise PanelRematerializationMismatch("feature_panel.rematerialization_content_mismatch")
        for item in expected:
            key = (str(item["session_date"]), str(item["factor_id"]))
            observed = actual_by_key.get(key)
            if observed is None or any(observed[field] != item[field] for field in fields):
                raise PanelRematerializationMismatch(
                    "feature_panel.rematerialization_content_mismatch"
                )
            if observed["sector_counts"] != json.loads(str(item["sector_counts_json"])):
                raise PanelRematerializationMismatch(
                    "feature_panel.rematerialization_content_mismatch"
                )
            observed_names = cast(tuple[str, ...], observed["small_sector_names"])
            if list(observed_names) != json.loads(str(item["small_sector_names_json"])):
                raise PanelRematerializationMismatch(
                    "feature_panel.rematerialization_content_mismatch"
                )

    @staticmethod
    def _validate_children(
        recipe: PanelDerivationRecipe,
        base: PanelBaseValueClosureManifest,
        sector_map: SectorRevisionMap,
        availability: PanelAvailabilityClosure,
        receipts: PanelRowReceiptAssignment,
    ) -> None:
        # The map's manifest is the one its revision was derived for. Under
        # the binding rule that is the recipe's; under the cross-section rule
        # a snapshot keeps its sector revision across manifests until the map
        # is next refreshed, so only the revision must agree.
        if (
            base.catalog_hash != recipe.base_value_catalog_hash
            or sector_map.sector_revision != recipe.sector_revision
            or (
                recipe.identity_basis == PANEL_ROW_IDENTITY_BY_BINDING
                and sector_map.manifest_revision != recipe.manifest_revision
            )
            or availability.panel_binding_hash != recipe.panel_binding_hash
            or receipts.panel_content_hash != recipe.panel_content_hash
        ):
            raise PanelRematerializationMismatch("feature_panel.recipe_input_mismatch")


def availability_rows(table: pa.Table) -> list[dict[str, object]]:
    names = table.column_names
    columns = [table.column(name).to_pylist() for name in names]
    return [dict(zip(names, row, strict=True)) for row in zip(*columns, strict=True)]


def _metadata_hash(path: Path) -> str:
    metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
    normalized = {
        key.decode("utf-8"): value.decode("utf-8") for key, value in sorted(metadata.items())
    }
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recipe_sector_history(
    recipe: PanelDerivationRecipe, sector_map: SectorRevisionMap
) -> SectorHistory:
    """The Sector each session of a recipe read: its map, then its reclassifications.

    Args:
        recipe: The Panel's derivation recipe; its reclassifications are its record.
        sector_map: The map the recipe names.

    Returns:
        The history over the map's listings.
    """
    return SectorHistory(
        current_revision=recipe.sector_revision,
        current={item.listing_id: item.sector_name for item in sector_map.entries},
        reclassifications=tuple(
            SectorReclassification(
                listing_id=item.listing_id,
                effective_session=item.effective_session,
                prior_sector=item.prior_sector,
                sector=item.sector,
            )
            for item in recipe.sector_reclassifications or ()
        ),
    )


__all__ = [
    "ArtifactOnlyPanelRematerializer",
    "PanelRematerializationMismatch",
    "PanelRematerializationResult",
    "RematerializedChunkEvidence",
    "recipe_sector_history",
]
