"""Marker-last publication of causal execution outcomes."""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from time import perf_counter
from typing import Literal, cast

import pyarrow as pa

from alphalattice.control.observation_runtime.telemetry.progress import WorkProgressUpdate
from alphalattice.control.task_control.child import ChildStartFailed
from alphalattice.control.workspace_runtime.artifacts import ArtifactDescriptor, ArtifactResolver
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.foundation.feature_engine.contracts import panel_source_manifest_revision
from alphalattice.foundation.feature_engine.panels.reader import FeaturePanelReader
from alphalattice.foundation.feature_engine.panels.semantic_index import (
    FeaturePanelSemanticIndexService,
)
from alphalattice.foundation.market_data_ops.sources.contracts import CorporateActionEvent
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.kernel.data.calendar import materialize_calendar_schedule
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.validation.enums import RebalanceFrequency
from alphalattice.kernel.validation.schedule_contracts import cadence_threshold_profile

from .artifacts import (
    DEVELOPMENT_ONLY_MANIFEST_CATEGORY,
    DEVELOPMENT_ONLY_MARKER_CATEGORY,
    METHOD_BINDING_CATEGORY,
    METHOD_SEAL_MARKER_CATEGORY,
    _ExecutionOutcomeArtifactStore,
)
from .compile import (
    ListingExecutionRows,
    ListingRowDerivation,
    _daily_session_axis,
    _derive_in_worker,
    _initialize_derivation_worker,
)
from .contracts import (
    _ACTION_ID,
    CausalExecutionOutcomeChunk,
    CausalExecutionOutcomeManifest,
    CausalExecutionOutcomeMarker,
    CausalExecutionOutcomeReceipt,
    DevelopmentOnlyExecutionOutcomeManifest,
    DevelopmentOnlyExecutionOutcomeMarker,
)
from .methods import (
    ONE_SESSION_RECIPE_ID,
    ExecutionOutcomeMethodBinding,
    ExecutionOutcomeMethodCatalog,
    ExecutionOutcomeMethodError,
    ExecutionOutcomeMethodRecipe,
    ExecutionOutcomeMethodSealMarker,
    ExecutionOutcomePublicationPolicy,
    ExecutionOutcomePublicationPolicyCatalog,
    SealableOutcomeManifest,
    action_lineage_hash,
    build_execution_outcome_method_binding,
    build_execution_outcome_method_seal_marker,
    build_installed_execution_outcome_method_catalog,
    build_installed_execution_outcome_publication_policy_catalog,
    resolve_schedule_points,
    verify_execution_outcome_method_seal_marker,
)


@dataclass(frozen=True, slots=True)
class _PublicationSurface:
    """Where one publication scope's artifacts live and how its marker is shaped.

    The publisher reads sources, derives rows and seals a method exactly once;
    only the *output contract* differs between the two admitted scopes. Holding
    that difference in one resolved object keeps the marker-last transaction --
    the most safety-critical code here -- a single implementation rather than
    two that must be kept in step.
    """

    manifest_category: str
    marker_category: str

    @classmethod
    def for_policy(cls, policy: ExecutionOutcomePublicationPolicy) -> _PublicationSurface:
        if policy.publication_scope == "DEVELOPMENT_ONLY":
            return cls(
                manifest_category=DEVELOPMENT_ONLY_MANIFEST_CATEGORY,
                marker_category=DEVELOPMENT_ONLY_MARKER_CATEGORY,
            )
        return cls(manifest_category="manifests", marker_category="markers")

    @property
    def development_only(self) -> bool:
        return self.manifest_category == DEVELOPMENT_ONLY_MANIFEST_CATEGORY

    def build_marker(self, *, manifest: SealableOutcomeManifest, manifest_ref: str) -> _MarkerLike:
        values: dict[str, object] = {
            "kind": (
                "DevelopmentOnlyExecutionOutcomeMarker"
                if self.development_only
                else "CausalExecutionOutcomeMarker"
            ),
            "snapshot_hash": manifest.snapshot_hash,
            "manifest_ref": manifest_ref,
            "schedule_hash": manifest.schedule_hash,
            "listing_set_hash": manifest.listing_set_hash,
        }
        if self.development_only:
            values["allowed_scope"] = "DEVELOPMENT_ONLY"
            return DevelopmentOnlyExecutionOutcomeMarker(
                **values, marker_hash=canonical_hash(values)
            )
        return CausalExecutionOutcomeMarker(**values, marker_hash=canonical_hash(values))

    def load_marker(self, store: _ExecutionOutcomeArtifactStore, uri: str) -> _MarkerLike:
        payload = store.load_json(category=self.marker_category, uri=uri)
        if self.development_only:
            return cast(
                DevelopmentOnlyExecutionOutcomeMarker,
                DevelopmentOnlyExecutionOutcomeMarker.model_validate(payload),
            )
        return cast(
            CausalExecutionOutcomeMarker, CausalExecutionOutcomeMarker.model_validate(payload)
        )

    def load_manifest(self, store: _ExecutionOutcomeArtifactStore, uri: str) -> _ManifestLike:
        payload = store.load_json(category=self.manifest_category, uri=uri)
        if self.development_only:
            return cast(
                DevelopmentOnlyExecutionOutcomeManifest,
                DevelopmentOnlyExecutionOutcomeManifest.model_validate(payload),
            )
        return cast(
            CausalExecutionOutcomeManifest, CausalExecutionOutcomeManifest.model_validate(payload)
        )

    @staticmethod
    def published_chunks(manifest: _ManifestLike) -> tuple[CausalExecutionOutcomeChunk, ...]:
        """Every chunk the manifest claims, whichever contract it is."""
        if isinstance(manifest, DevelopmentOnlyExecutionOutcomeManifest):
            return manifest.development_chunks
        return (*manifest.development_chunks, *manifest.sealed_holdout_chunks)


type _ManifestLike = CausalExecutionOutcomeManifest | DevelopmentOnlyExecutionOutcomeManifest
type _MarkerLike = CausalExecutionOutcomeMarker | DevelopmentOnlyExecutionOutcomeMarker


@dataclass(frozen=True)
class PublishedCausalExecutionOutcome:
    """Return a published snapshot with its artifacts and method seal."""

    manifest: _ManifestLike
    manifest_artifact: ArtifactDescriptor
    marker: _MarkerLike
    marker_artifact: ArtifactDescriptor
    receipt: CausalExecutionOutcomeReceipt
    receipt_artifact: ArtifactDescriptor
    method_binding: ExecutionOutcomeMethodBinding
    method_binding_artifact: ArtifactDescriptor
    method_seal_marker: ExecutionOutcomeMethodSealMarker
    method_seal_marker_artifact: ArtifactDescriptor


def _method_authority(binding: ExecutionOutcomeMethodBinding) -> dict[str, object]:
    """Select seal fields that must agree before exact reuse.

    Only the source watermark is excluded: it is a counter over source
    revisions and moves without any consumed row changing. The action lineage
    stays in the comparison even though the shared verifier also treats it as
    a derivation input -- a re-derivation that saw different admitted actions
    inside the schedule's reach is refused rather than filed under the first
    seal, because the receipt written below records only the new watermark and
    would not preserve that changed lineage.
    """
    return cast(
        dict[str, object],
        binding.model_dump(mode="json", exclude={"binding_hash", "source_watermark_hash"}),
    )


# Below this many rows a publication derives its rows in this process; above
# it, worker processes derive listings in parallel (each import of the
# derivation costs a worker a second or two, which a small publication does
# not earn back).
PARALLEL_DERIVATION_ROW_THRESHOLD = 200_000

# The most worker processes a publication uses, whatever the allowance. On
# the 466-name, 2,513-session recording four workers reach the same wall time
# as eight (28.3 s either way, against 47.9 s in this process) because the
# parent's own store reads, unpickling and chunk assembly bound the rest;
# eight only cost 11 more CPU seconds and 290 MB more resident memory at the
# aligned peak.
DERIVATION_WORKER_CEILING = 4

# How many listings may be handed to the workers ahead of consumption. Each
# submitted unit holds that listing's bars in this process until a worker
# takes it and its rows until this process consumes them, in listing order;
# a bounded window keeps that holding near what the serial path holds.
DERIVATION_WINDOW_PER_WORKER = 4


def _derivation_workers(configured: int | None) -> int:
    """How many worker processes a large publication may use.

    The ceiling over the processors this process is allowed: the rows are
    cheap per listing, and the parent's own reads, unpickling and chunk
    assembly stay serial. Worker processes inherit this process's affinity,
    so a Host that bound itself to part of the machine keeps that bound; a
    Host that knows its allowance passes it as ``derivation_workers`` and
    never gets more workers than that allowance.
    """
    allowed = configured if configured is not None else (os.cpu_count() or 1)
    return max(1, min(DERIVATION_WORKER_CEILING, allowed))


class CausalExecutionOutcomePublisher:
    """Publish DAILY causal outcomes without exposing DuckDB downstream."""

    def __init__(
        self,
        *,
        store: MarketDataRepository,
        resolver: ArtifactResolver,
        artifact_root: Path,
        mutation_gate: WorkspaceMutationGate,
        progress_sink: Callable[[WorkProgressUpdate], object] | None = None,
        derivation_workers: int | None = None,
    ) -> None:
        """Install the store, artifact owner, mutation gate, and work allowance.

        Args:
            store: Source market-data repository.
            resolver: Artifact resolver for publication dependencies.
            artifact_root: Root for immutable outcome artifacts.
            mutation_gate: Workspace authority for publication.
            progress_sink: Optional consumer of bounded work progress.
            derivation_workers: Maximum allowed derivation workers, if fixed.

        """
        self.store = store
        self.resolver = resolver
        self.artifacts = _ExecutionOutcomeArtifactStore(artifact_root)
        self.index_service = FeaturePanelSemanticIndexService(resolver)
        self.mutation_gate = mutation_gate
        self.progress_sink = progress_sink
        # ``None`` sizes the pool from the allowed processors; ``1`` derives
        # every listing in this process whatever the size.
        self.derivation_workers = derivation_workers

    def _publish_progress(
        self,
        *,
        operation_id: str,
        stage_id: str,
        completed: int,
        total: int,
        unit_name: str,
        current_item: str | None = None,
        counters: dict[str, int] | None = None,
        status: Literal["RUNNING", "SUCCEEDED", "FAILED", "REUSED_EXACT"] = "RUNNING",
        failure_code: str | None = None,
    ) -> None:
        if self.progress_sink is None:
            return
        self.progress_sink(
            WorkProgressUpdate(
                operation_id=operation_id,
                stage_id=stage_id,
                status=status,
                completed_units=completed,
                total_units=total,
                unit_name=unit_name,
                current_item=current_item,
                counters=counters or {},
                failure_code=failure_code,
            )
        )

    def publish_daily(
        self,
        *,
        panel_manifest_ref: str,
        completed_at: datetime,
        recipe_id: str = ONE_SESSION_RECIPE_ID,
    ) -> PublishedCausalExecutionOutcome:
        """Publish one DAILY outcome snapshot under one installed method.

        The method arrives as a stable id, never as a recipe object, so a caller
        cannot present a method that wears an installed id while carrying
        different offsets: the Host resolves the id in its own catalog and
        re-derives the recipe's identity before anything else happens.
        """
        started = perf_counter()
        if completed_at.tzinfo is None or completed_at.utcoffset() is None:
            raise ValueError("causal execution publication clock must be timezone-aware")
        catalog = build_installed_execution_outcome_method_catalog()
        recipe = catalog.resolve(recipe_id)
        catalog_binding = catalog.binding
        policies = build_installed_execution_outcome_publication_policy_catalog()
        # Admission before any source read: an installed-but-unadmitted method
        # must not cause a single bar or action row to be touched. Resolution is
        # by the method's own id, so which admission applies is a fact about the
        # installed catalog rather than a choice this call makes.
        policy = policies.resolve(recipe.recipe_id)
        policy.admit(recipe)
        surface = _PublicationSurface.for_policy(policy)
        # The Panel's admission (available, ACTIVE, temporally consistent,
        # Gateway-admitted) is the axis reader's, and comes before the index
        # or a source row is read: an index already on disk admits nothing.
        listing_ids = FeaturePanelReader(self.resolver).listing_axis(panel_manifest_ref)
        panel = self.resolver.load_feature_panel_manifest(panel_manifest_ref)
        index, _index_ref, _backfilled = self.index_service.obtain(panel_manifest_ref)
        market_as_of = date.fromisoformat(str(panel["as_of_session"]))
        manifest = self.store.load_universe_manifest_revision(panel_source_manifest_revision(panel))
        listings = self.store.listing_scope(manifest, listing_ids=listing_ids)
        # Outcomes are observations over the coverage axis. The paired Panel's
        # per-session rows decide which observations are research samples.
        listing_set_hash = canonical_hash(listing_ids)
        summary = panel.get("safe_summary")
        membership = summary.get("membership") if isinstance(summary, dict) else None
        universe_limits = (
            (
                *dict.fromkeys(str(item["basis"]) for item in membership["basis_ranges"]),
                "OUTCOME_COVERAGE_NOT_FORMATION_ELIGIBILITY",
            )
            if isinstance(membership, dict)
            else ("CURRENT_ACTIVE_SURVIVORS",)
        )
        sessions = tuple(value.session_date for value in index.sessions)
        anchors = _daily_session_axis(sessions, minimum_sessions=recipe.minimum_axis_sessions)
        calendar = materialize_calendar_schedule(
            ("XNYS", "XNAS"),
            start=anchors[0],
            end=anchors[-1],
            as_of_timestamp=completed_at,
        )
        by_session: dict[date, dict[str, object]] = {}
        venue_rows: dict[date, dict[str, dict[str, object]]] = {}
        for row in calendar.to_pylist():
            venue_rows.setdefault(row["session_date"], {})[row["calendar_id"]] = row
        for session, venues in sorted(venue_rows.items()):
            if set(venues) != {"XNYS", "XNAS"}:
                continue
            xnys, xnas = venues["XNYS"], venues["XNAS"]
            if (
                xnys["session_open_timestamp"] != xnas["session_open_timestamp"]
                or xnys["session_close_timestamp"] != xnas["session_close_timestamp"]
            ):
                raise ValueError("common execution calendar venue clocks disagree")
            by_session[session] = xnys
        if tuple(by_session) != anchors:
            raise ValueError("causal execution calendar differs from the frozen panel axis")
        points = resolve_schedule_points(
            recipe=recipe,
            ordered_sessions=anchors,
            session_clocks=by_session,
        )
        schedule_identity = tuple(point.model_dump(mode="json") for point in points)
        # Deliberately the same three keys as before the method seam existed,
        # with the formula supplied by the recipe. Folding recipe_hash in here
        # would rotate every published one-session schedule identity for no
        # gain: two recipes cannot share a formula and a point sequence.
        schedule_hash = canonical_hash(
            {
                "research_cadence": "DAILY",
                "points": schedule_identity,
                "formula": recipe.return_formula_identity,
            }
        )
        watermark = self.store.execution_source_watermark(
            manifest, through=market_as_of, listing_ids=listing_ids
        )
        watermark_hash = str(watermark["watermark_hash"])
        reusable = self._reusable(
            listing_set_hash=listing_set_hash,
            market_as_of=market_as_of,
            schedule_hash=schedule_hash,
            watermark_hash=watermark_hash,
            recipe=recipe,
            catalog=catalog,
            catalog_hash=catalog_binding.catalog_hash,
            policies=policies,
            surface=surface,
        )
        if reusable is not None:
            reused_manifest, reused_binding = reusable
            self._publish_progress(
                operation_id=schedule_hash,
                stage_id="causal_execution_outcome",
                completed=1,
                total=1,
                unit_name="snapshot",
                status="REUSED_EXACT",
            )
            return self._finish(
                manifest=reused_manifest,
                method_binding=reused_binding,
                completed_at=completed_at,
                watermark_hash=watermark_hash,
                raw_rows=0,
                action_rows=0,
                started=started,
                action="REUSED_EXACT",
                catalog=catalog,
                catalog_hash=catalog_binding.catalog_hash,
                policies=policies,
                surface=surface,
            )

        source_started = perf_counter()
        profile = cadence_threshold_profile(RebalanceFrequency.DAILY)
        # An overlapping-label method has no admissible sealed cut: the last
        # development label and the first sealed label would describe partly the
        # same holding interval, so "sealed" would not mean unread. Under that
        # scope every formation is development and the snapshot says so in its
        # own contract, rather than a Holdout being written that nobody may use.
        sealed_formations = (
            frozenset()
            if surface.development_only
            else frozenset(anchors[-profile.holdout_observations :])
        )
        # One listing is one unit of derivation (``ListingRowDerivation``):
        # its bars and actions in, its rows and source bindings out, in point
        # order. The store is read here, serially; a large publication hands
        # the units to worker processes and consumes their results in listing
        # order, so the chunk tables and the source-row bindings are the same
        # sequence a single process produces.
        derivation = ListingRowDerivation(
            points=points,
            recipe=recipe,
            ordered_sessions=anchors,
            sealed_formations=sealed_formations,
        )
        tables_by_split_year: dict[tuple[str, int], list[pa.Table]] = {}
        source_row_bindings: list[tuple[str, str, str, str]] = []
        actions_by_listing: dict[str, tuple[CorporateActionEvent, ...]] = {}
        raw_rows_read = 0
        action_rows_read = 0
        self._publish_progress(
            operation_id=schedule_hash,
            stage_id="causal_execution_sources",
            completed=0,
            total=len(listing_ids),
            unit_name="listings",
        )
        workers = _derivation_workers(self.derivation_workers)
        parallel = workers > 1 and len(listings) * len(points) >= PARALLEL_DERIVATION_ROW_THRESHOLD
        pool: ProcessPoolExecutor | None = None
        in_flight: list[tuple[str, Future[ListingExecutionRows]]] = []
        consumed = 0

        def consume(listing_id: str, derived: ListingExecutionRows) -> None:
            nonlocal consumed
            if derived.listing_id != listing_id:
                raise ValueError("causal execution derivation answered for another listing")
            for key, table in derived.tables.items():
                tables_by_split_year.setdefault(key, []).append(table)
            source_row_bindings.extend(derived.source_row_bindings)
            consumed += 1
            self._publish_progress(
                operation_id=schedule_hash,
                stage_id="causal_execution_sources",
                completed=consumed,
                total=len(listing_ids),
                unit_name="listings",
                current_item=listing_id,
                counters={
                    "raw_rows_read": raw_rows_read,
                    "action_rows_read": action_rows_read,
                },
            )

        connection = self.store._connect(read_only=True)
        try:
            if parallel:
                workers = min(workers, len(listings))
                try:
                    pool = ProcessPoolExecutor(
                        max_workers=workers,
                        initializer=_initialize_derivation_worker,
                        initargs=(derivation,),
                    )
                except (OSError, MemoryError) as error:
                    raise ChildStartFailed(error) from error
            window = workers * DERIVATION_WINDOW_PER_WORKER
            for listing in listings:
                bars = self.store.raw_bars(
                    listing.listing_id, through=market_as_of, _connection=connection
                )
                actions = self.store.actions(listing.listing_id, _connection=connection)
                actions = tuple(value for value in actions if value.effective_date <= market_as_of)
                raw_rows_read += len(bars)
                action_rows_read += len(actions)
                actions_by_listing[listing.listing_id] = actions
                if pool is None:
                    consume(
                        listing.listing_id,
                        derivation.derive(
                            listing_id=listing.listing_id,
                            symbol=listing.symbol,
                            bars=bars,
                            actions=actions,
                        ),
                    )
                    continue
                # The oldest unit is consumed (in listing order) before the
                # window admits another; a worker's refusal surfaces here,
                # and nothing is published until every unit has answered.
                if len(in_flight) >= window:
                    oldest_id, oldest = in_flight.pop(0)
                    consume(oldest_id, oldest.result())
                try:
                    future = pool.submit(
                        _derive_in_worker,
                        (listing.listing_id, listing.symbol, tuple(bars), actions),
                    )
                except (OSError, MemoryError) as error:
                    raise ChildStartFailed(error) from error
                in_flight.append((listing.listing_id, future))
            for listing_id, future in in_flight:
                consume(listing_id, future.result())
        finally:
            connection.close()
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
        self._publish_progress(
            operation_id=schedule_hash,
            stage_id="causal_execution_sources",
            completed=len(listing_ids),
            total=len(listing_ids),
            unit_name="listings",
            counters={
                "raw_rows_read": raw_rows_read,
                "action_rows_read": action_rows_read,
            },
            status="SUCCEEDED",
        )
        source_seconds = perf_counter() - source_started
        publish_started = perf_counter()
        chunk_inputs = tuple(sorted(tables_by_split_year.items()))
        self._publish_progress(
            operation_id=schedule_hash,
            stage_id="causal_execution_publication",
            completed=0,
            total=len(chunk_inputs),
            unit_name="chunks",
        )
        published_chunks = []
        for chunk_index, ((split, year), tables) in enumerate(chunk_inputs, start=1):
            published_chunks.append(
                self.artifacts.publish_chunk(split=split, table=pa.concat_tables(tables))
            )
            self._publish_progress(
                operation_id=schedule_hash,
                stage_id="causal_execution_publication",
                completed=chunk_index,
                total=len(chunk_inputs),
                unit_name="chunks",
                current_item=f"{split}:{year}",
            )
        chunks = tuple(published_chunks)
        self._publish_progress(
            operation_id=schedule_hash,
            stage_id="causal_execution_publication",
            completed=len(chunk_inputs),
            total=len(chunk_inputs),
            unit_name="chunks",
            status="SUCCEEDED",
        )
        development = tuple(value for value in chunks if value.split == "DEVELOPMENT")
        sealed = tuple(value for value in chunks if value.split == "SEALED_HOLDOUT")
        development_formations = {
            point.formation_session
            for point in points
            if point.formation_session not in sealed_formations
        }
        sealed_output_formations = {
            point.formation_session
            for point in points
            if point.formation_session in sealed_formations
        }
        shared: dict[str, object] = {
            "research_cadence": RebalanceFrequency.DAILY,
            "market_as_of": market_as_of,
            "listing_ids": listing_ids,
            "listing_set_hash": listing_set_hash,
            "schedule_hash": schedule_hash,
            "ordered_session_triples_hash": canonical_hash(schedule_identity),
            "source_rows_semantic_hash": canonical_hash(source_row_bindings),
            "price_basis": recipe.price_basis,
            # Supplied by the recipe rather than respelled here. On the frozen
            # contract the field is a single-value Literal, so this is also the
            # structural gate that keeps a non-one-session method out of that
            # schema even though publication admission is now a catalog.
            "return_formula_identity": recipe.return_formula_identity,
            "corporate_action_identity": recipe.corporate_action_identity,
            "data_validity_class": "CURRENT_UNIVERSE_RESEARCH_ONLY",
            "development_chunks": development,
            "development_formation_count": len(development_formations),
        }
        outcome: _ManifestLike
        if surface.development_only:
            values = {
                **shared,
                "kind": "DevelopmentOnlyExecutionOutcomeSnapshot",
                "allowed_scope": "DEVELOPMENT_ONLY",
                # Carried on the snapshot because a consumer deriving an embargo
                # needs the span, and reading it back off the method catalog
                # would mean trusting that the catalog still says what it said
                # when these rows were written.
                "maturity_lag_sessions": recipe.maturity_lag_sessions,
                "limitations": (
                    *universe_limits,
                    "YAHOO_CURRENT_SECTOR_BACKFILLED",
                    "NON_POINT_IN_TIME_RESEARCH",
                    "DAILY_BAR_ELIGIBILITY_IS_ASSUMED_NOT_VENUE_VERIFIED",
                    "OVERLAPPING_MULTI_SESSION_LABELS_ARE_NOT_INDEPENDENT",
                    "DEVELOPMENT_ONLY_NO_SEALED_HOLDOUT_EXISTS",
                ),
            }
            identity = DevelopmentOnlyExecutionOutcomeManifest.model_construct(
                **values, snapshot_hash=""
            ).model_dump(mode="json", exclude={"snapshot_hash"})
            outcome = DevelopmentOnlyExecutionOutcomeManifest(
                **values, snapshot_hash=canonical_hash(identity)
            )
        else:
            values = {
                **shared,
                "kind": "CausalExecutionOutcomeSnapshot",
                "sealed_holdout_chunks": sealed,
                "sealed_holdout_formation_count": len(sealed_output_formations),
                "limitations": (
                    *universe_limits,
                    "YAHOO_CURRENT_SECTOR_BACKFILLED",
                    "NON_POINT_IN_TIME_RESEARCH",
                    "DAILY_BAR_ELIGIBILITY_IS_ASSUMED_NOT_VENUE_VERIFIED",
                    "SEALED_HOLDOUT_RELEASE_AUTHORITY_UNAVAILABLE",
                ),
            }
            identity = CausalExecutionOutcomeManifest.model_construct(
                **values, snapshot_hash=""
            ).model_dump(mode="json", exclude={"snapshot_hash"})
            outcome = CausalExecutionOutcomeManifest(
                **values, snapshot_hash=canonical_hash(identity)
            )
        derived_binding = build_execution_outcome_method_binding(
            recipe=recipe,
            catalog_hash=catalog_binding.catalog_hash,
            publication_policy_hash=policy.policy_hash,
            snapshot_hash=outcome.snapshot_hash,
            schedule_hash=outcome.schedule_hash,
            ordered_session_triples_hash=outcome.ordered_session_triples_hash,
            source_watermark_hash=watermark_hash,
            action_lineage=action_lineage_hash(
                actions_by_listing=actions_by_listing,
                intervals=tuple(
                    (point.entry_session, point.holding_end_session) for point in points
                ),
                ordered_sessions=anchors,
            ),
        )
        # A snapshot carries one seal. The watermark is a counter over source
        # revisions, so it moves on a correction that leaves every consumed row
        # as it was (a dividend observed twice and settling on its first
        # payload), and the derivation then lands on a snapshot this publisher
        # already sealed. Minting a second binding for it would leave two
        # well-formed seals that every reader refuses by design, so the seal
        # already on disk is resolved, verified and required to carry the same
        # method and the same action lineage; only the receipt below records
        # this derivation's watermark.
        existing_binding = self._resolve_seal(
            manifest=outcome,
            catalog=catalog,
            catalog_hash=catalog_binding.catalog_hash,
            policies=policies,
            surface=surface,
        )
        if existing_binding is None:
            method_binding = derived_binding
        elif _method_authority(existing_binding) == _method_authority(derived_binding):
            method_binding = existing_binding
        else:
            raise ExecutionOutcomeMethodError(
                "causal_outcomes.existing_snapshot_carries_a_different_method"
            )
        manifest_artifact = self.artifacts.publish_json(
            category=surface.manifest_category,
            payload=outcome.model_dump(mode="json"),
            identity_field="snapshot_hash",
        )
        return self._finish(
            manifest=outcome,
            method_binding=method_binding,
            completed_at=completed_at,
            watermark_hash=watermark_hash,
            raw_rows=raw_rows_read,
            action_rows=action_rows_read,
            started=started,
            action="PUBLISHED",
            catalog=catalog,
            catalog_hash=catalog_binding.catalog_hash,
            policies=policies,
            surface=surface,
            source_seconds=source_seconds,
            publication_seconds=perf_counter() - publish_started,
            manifest_artifact=manifest_artifact,
        )

    def _resolve_seal(
        self,
        *,
        manifest: _ManifestLike,
        catalog: ExecutionOutcomeMethodCatalog,
        catalog_hash: str,
        policies: ExecutionOutcomePublicationPolicyCatalog,
        surface: _PublicationSurface,
    ) -> ExecutionOutcomeMethodBinding | None:
        """Resolve the one Host-authoritative seal for a snapshot, or nothing.

        Duplicates are refused rather than resolved. A dictionary keyed by
        snapshot would have silently taken the last file the directory listing
        produced, which is exactly the shape of an attack: publish a second
        well-formed binding and let ordering decide which method the evidence is
        said to carry. The surviving marker's whole graph is then handed to the
        shared Host verifier with every expected ref derived from content
        identity, so a marker that stores a plausible-but-wrong ref fails the
        whole-model comparison rather than being followed.
        """
        markers = tuple(
            value
            for value in self.artifacts.method_seal_markers()
            if value.snapshot_hash == manifest.snapshot_hash
        )
        if not markers:
            return None
        if len(markers) != 1:
            raise ValueError("causal execution snapshot does not resolve one method seal")
        seal_marker = markers[0]
        outcome_marker = surface.load_marker(self.artifacts, seal_marker.outcome_marker_ref)
        binding = cast(
            ExecutionOutcomeMethodBinding,
            ExecutionOutcomeMethodBinding.model_validate(
                self.artifacts.load_json(
                    category=METHOD_BINDING_CATEGORY, uri=seal_marker.binding_ref
                )
            ),
        )
        verify_execution_outcome_method_seal_marker(
            seal_marker=seal_marker,
            outcome_marker=outcome_marker,
            outcome_marker_ref=self.artifacts._uri(
                surface.marker_category, outcome_marker.marker_hash
            ),
            manifest=manifest,
            manifest_ref=self.artifacts._uri(surface.manifest_category, manifest.snapshot_hash),
            binding=binding,
            binding_ref=self.artifacts._uri(METHOD_BINDING_CATEGORY, binding.binding_hash),
            catalog=catalog,
            expected_catalog_hash=catalog_hash,
            policies=policies,
        )
        return binding

    def _reusable(
        self,
        *,
        listing_set_hash: str,
        market_as_of: date,
        schedule_hash: str,
        watermark_hash: str,
        recipe: ExecutionOutcomeMethodRecipe,
        catalog: ExecutionOutcomeMethodCatalog,
        catalog_hash: str,
        policies: ExecutionOutcomePublicationPolicyCatalog,
        surface: _PublicationSurface,
    ) -> tuple[_ManifestLike, ExecutionOutcomeMethodBinding] | None:
        """Resolve an exactly-reusable snapshot *and the seal that authorizes it*.

        Reuse means "this evidence already carries this method's authority".
        A snapshot published before the seam has no seal and therefore has no
        such authority, so it is not reusable -- and it is not silently sealed
        either. Granting authority to existing evidence is a migration with its
        own output root and its own record, not a side effect of the next
        ordinary run, so this raises and names that migration instead.
        """
        receipt_by_snapshot = {
            value.snapshot_hash: value
            for value in self.artifacts.receipts()
            if value.source_watermark_hash == watermark_hash
        }
        published: tuple[_ManifestLike, ...] = (
            self.artifacts.development_only_manifests()
            if surface.development_only
            else self.artifacts.manifests()
        )
        values = [
            value
            for value in published
            if value.listing_set_hash == listing_set_hash
            and value.market_as_of == market_as_of
            and value.schedule_hash == schedule_hash
            and value.corporate_action_identity == _ACTION_ID
            and value.snapshot_hash in receipt_by_snapshot
        ]
        if not values:
            return None
        manifest = max(values, key=lambda value: value.snapshot_hash)
        binding = self._resolve_seal(
            manifest=manifest,
            catalog=catalog,
            catalog_hash=catalog_hash,
            policies=policies,
            surface=surface,
        )
        if binding is None:
            raise ExecutionOutcomeMethodError(
                "causal_outcomes.legacy_snapshot_requires_explicit_method_migration"
            )
        # The receipt selected above is the attestation that this exact source
        # state derived this snapshot; the seal's own watermark is the state of
        # the derivation that first sealed it and legitimately differs once the
        # same rows have been re-derived under a later revision counter.
        if binding.recipe_hash != recipe.recipe_hash:
            raise ExecutionOutcomeMethodError(
                "causal_outcomes.existing_snapshot_carries_a_different_method"
            )
        return manifest, binding

    def _finish(
        self,
        *,
        manifest: _ManifestLike,
        method_binding: ExecutionOutcomeMethodBinding,
        completed_at: datetime,
        watermark_hash: str,
        raw_rows: int,
        action_rows: int,
        started: float,
        action: Literal["PUBLISHED", "REUSED_EXACT"],
        catalog: ExecutionOutcomeMethodCatalog,
        catalog_hash: str,
        policies: ExecutionOutcomePublicationPolicyCatalog,
        surface: _PublicationSurface,
        source_seconds: float = 0.0,
        publication_seconds: float = 0.0,
        manifest_artifact: ArtifactDescriptor | None = None,
    ) -> PublishedCausalExecutionOutcome:
        if method_binding.snapshot_hash != manifest.snapshot_hash or (
            method_binding.schedule_hash != manifest.schedule_hash
            or method_binding.ordered_session_triples_hash != manifest.ordered_session_triples_hash
        ):
            raise ValueError("causal execution method seal does not describe this snapshot")
        manifest_artifact = manifest_artifact or self.artifacts.publish_json(
            category=surface.manifest_category,
            payload=manifest.model_dump(mode="json"),
            identity_field="snapshot_hash",
        )
        # Manifest, then seal, then marker. The seal is inside the publication
        # transaction rather than appended after it, so marker-last keeps its
        # meaning: a marker on disk implies the seal it authorizes is already
        # there, and a crash between the two leaves no marker to find.
        binding_artifact = self.artifacts.publish_json(
            category=METHOD_BINDING_CATEGORY,
            payload=method_binding.model_dump(mode="json"),
            identity_field="binding_hash",
        )
        marker = surface.build_marker(manifest=manifest, manifest_ref=manifest_artifact.uri)
        marker_artifact = self.artifacts.publish_json(
            category=surface.marker_category,
            payload=marker.model_dump(mode="json"),
            identity_field="marker_hash",
        )
        # Written last, after the marker it names, so that its presence is what
        # makes the snapshot method-bound. A reader never has to scan for a
        # binding that happens to mention the same snapshot.
        seal_marker = build_execution_outcome_method_seal_marker(
            outcome_marker_hash=marker.marker_hash,
            outcome_marker_ref=marker_artifact.uri,
            snapshot_hash=manifest.snapshot_hash,
            manifest_ref=manifest_artifact.uri,
            binding=method_binding,
            binding_ref=binding_artifact.uri,
        )
        seal_marker_artifact = self.artifacts.publish_json(
            category=METHOD_SEAL_MARKER_CATEGORY,
            payload=seal_marker.model_dump(mode="json"),
            identity_field="seal_marker_hash",
        )
        with self.mutation_gate.try_hold(timeout_seconds=30.0):
            # Authority is what is durably on disk, not the objects assembled
            # above: the terminal marker is reloaded, its children are reloaded
            # through the durable marker's *own* refs, the whole graph is
            # re-verified by the shared Host verifier against the descriptors
            # this transaction wrote, and only then is the durable graph
            # required to equal the in-memory one. A tampered or missing child
            # between write and readback fails here, before any receipt exists.
            durable_seal_marker = ExecutionOutcomeMethodSealMarker.model_validate(
                self.artifacts.load_json(
                    category=METHOD_SEAL_MARKER_CATEGORY, uri=seal_marker_artifact.uri
                )
            )
            durable_marker = surface.load_marker(
                self.artifacts, durable_seal_marker.outcome_marker_ref
            )
            durable_manifest = surface.load_manifest(
                self.artifacts, durable_seal_marker.manifest_ref
            )
            durable_binding = ExecutionOutcomeMethodBinding.model_validate(
                self.artifacts.load_json(
                    category=METHOD_BINDING_CATEGORY, uri=durable_seal_marker.binding_ref
                )
            )
            verify_execution_outcome_method_seal_marker(
                seal_marker=durable_seal_marker,
                outcome_marker=durable_marker,
                outcome_marker_ref=marker_artifact.uri,
                manifest=durable_manifest,
                manifest_ref=manifest_artifact.uri,
                binding=durable_binding,
                binding_ref=binding_artifact.uri,
                catalog=catalog,
                expected_catalog_hash=catalog_hash,
                policies=policies,
            )
            if (
                durable_seal_marker != seal_marker
                or durable_marker != marker
                or durable_manifest != manifest
                or durable_binding != method_binding
            ):
                raise ValueError("causal execution marker-last readback differs")
            for chunk in surface.published_chunks(manifest):
                self.artifacts.resolve_chunk(chunk)
        receipt_values = {
            "kind": "CausalExecutionOutcomeReceipt",
            "snapshot_hash": manifest.snapshot_hash,
            "marker_ref": marker_artifact.uri,
            "action": action,
            "source_watermark_hash": watermark_hash,
            "completed_at": completed_at.astimezone(UTC),
            "raw_bar_payload_rows_read": raw_rows,
            "action_payload_rows_read": action_rows,
            "durations_seconds": {
                "source": source_seconds,
                "publication": publication_seconds,
                "total": perf_counter() - started,
            },
        }
        receipt_identity = CausalExecutionOutcomeReceipt.model_construct(
            **receipt_values, receipt_hash=""
        ).model_dump(mode="json", exclude={"receipt_hash"})
        receipt = CausalExecutionOutcomeReceipt(
            **receipt_values, receipt_hash=canonical_hash(receipt_identity)
        )
        receipt_artifact = self.artifacts.publish_json(
            category="receipts",
            payload=receipt.model_dump(mode="json"),
            identity_field="receipt_hash",
        )
        return PublishedCausalExecutionOutcome(
            manifest=manifest,
            manifest_artifact=manifest_artifact,
            marker=marker,
            marker_artifact=marker_artifact,
            receipt=receipt,
            receipt_artifact=receipt_artifact,
            method_binding=method_binding,
            method_binding_artifact=binding_artifact,
            method_seal_marker=seal_marker,
            method_seal_marker_artifact=seal_marker_artifact,
        )


__all__ = [
    "CausalExecutionOutcomePublisher",
    "PublishedCausalExecutionOutcome",
]
