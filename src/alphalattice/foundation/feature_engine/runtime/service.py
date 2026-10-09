"""Deterministic Feature Foundation orchestration, with no agent in the happy path."""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, date, datetime
from typing import Any, NamedTuple, cast

import pandas as pd

from alphalattice.control.observation_runtime.telemetry.process_metrics import worker_processes
from alphalattice.control.observation_runtime.telemetry.progress import WorkProgressUpdate
from alphalattice.control.task_control.child import ChildStartFailed, child_calls
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.control.workspace_runtime.reader_threads import reader_threads
from alphalattice.foundation.feature_engine.catalog.contracts import (
    MARKET_DEPENDENT_FACTOR_IDS,
    FeatureCatalog,
)
from alphalattice.foundation.feature_engine.catalog.layer import (
    FeatureCatalogLayer,
    restricted_catalog,
)
from alphalattice.foundation.feature_engine.contracts import (
    PANEL_ROW_IDENTITY_BY_CROSS_SECTION,
    FeatureBuildOutcome,
    FeatureBuildRequest,
    FeatureBuildStage,
    FeatureBuildStatus,
    FeatureInvalidation,
    FeaturePanelBinding,
    PanelAdmissionSummary,
    PanelMembership,
    PanelMembershipBasisRange,
    PanelMembershipEpoch,
    PanelTemporalRisk,
    TemporalKnowledgeBoundary,
    canonical_hash,
)
from alphalattice.foundation.feature_engine.inputs.contracts import MaterializedFeatureQualification
from alphalattice.foundation.feature_engine.inputs.quality import (
    qualify_materialized_features,
)
from alphalattice.foundation.feature_engine.panels.artifacts import (
    PanelArtifactCompositionOwner,
    PanelColumnExtension,
    PanelCompositionBinding,
    PanelCompositionSession,
    manifest_partition_origins,
)
from alphalattice.foundation.feature_engine.panels.availability import PanelAvailabilityRepository
from alphalattice.foundation.feature_engine.panels.feature_closure_coordinator import (
    feature_row_identities,
)
from alphalattice.foundation.feature_engine.panels.materialization_identity import (
    align_feature_source_sessions,
    feature_source_values_hash,
)
from alphalattice.foundation.feature_engine.producers.base_materializer import (
    _REQUIRED_COLUMNS,
    BaseFeatureMaterializer,
)
from alphalattice.foundation.feature_engine.producers.cross_section import (
    SectorNeutralPanelMaterializer,
    clip_observation_record,
)
from alphalattice.foundation.feature_engine.producers.factors.catalog import (
    default_extension_kernel_registry,
)
from alphalattice.foundation.feature_engine.producers.factors.formula import (
    FORMULA_POINT_IN_TIME_FIELDS,
)
from alphalattice.foundation.feature_engine.producers.factors.registry import FeatureKernelRegistry
from alphalattice.foundation.feature_engine.producers.ineligibility import FeatureIneligibilityStore
from alphalattice.foundation.feature_engine.producers.preprocessing.contracts import (
    PanelClipObservationRecord,
    PanelClippingEvidence,
    build_panel_preprocessing_seal_marker,
)
from alphalattice.foundation.feature_engine.producers.preprocessing.robust_cross_section import (
    PANEL_SESSION_BATCH_SIZE,
    PanelMaterialization,
    allows_missing_source_rows,
)
from alphalattice.foundation.feature_engine.producers.reference_data import (
    MarketReference,
    MarketReferenceMaintainer,
    SectorRefreshOutcome,
    SectorRefreshStager,
    SectorTransportPolicy,
)
from alphalattice.foundation.feature_engine.publication.persistence import (
    FeatureMaterializationPersistence,
    admitted_writes,
)
from alphalattice.foundation.feature_engine.publication.sector_map_activation import (
    SectorRevisionMapActivationCoordinator,
)
from alphalattice.foundation.feature_engine.publication.snapshots import (
    reconcile_feature_panel_lifecycles,
)
from alphalattice.foundation.feature_engine.runtime.factor_invalidation import (
    compile_listing_plan,
    compile_panel_plan,
    restrict_plan_to_sessions,
    targets_by_session,
)
from alphalattice.foundation.feature_engine.storage.contracts import (
    FeatureIneligibilityRun,
    FeatureMaterializationWrite,
    FeatureRowIdentity,
    FeatureSourceWindow,
    SectorReferenceState,
)
from alphalattice.foundation.feature_engine.storage.repositories import (
    FeaturePersistenceTimingSink,
    FeatureSourceInputs,
    FeatureStateRepository,
    PanelStateRepository,
    projected_feature_rows,
)
from alphalattice.foundation.market_data_ops.sources.manifest import (
    ManifestListing,
    UniverseManifest,
)
from alphalattice.foundation.market_data_ops.sources.price_integrity import TradingSessionAuthority
from alphalattice.foundation.market_data_ops.sources.providers import MarketDataProvider
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.kernel.shared_kernel.domain.errors import WorkspaceConflictError
from alphalattice.kernel.shared_kernel.sector_treatment import sector_treatment
from alphalattice.kernel.shared_kernel.spans import span, spanned

_PANEL_CONNECTION_SESSION_BUDGET = 630
_LISTING_ROWS = f"{__name__}:materialize_listing_rows"
"""A listing's computation, run in the Host's kept workers (W10)."""


def catalog_reads_as_traded(catalog: FeatureCatalog) -> bool:
    """Whether a catalog holds a formula reading the as-traded fields (V395).

    Only then does a listing's frame carry them, so a build whose catalog reads none projects
    exactly what it did before and keeps every source window and reuse record it wrote.

    Args:
        catalog: The catalog a build materializes, a workspace's activations included.

    Returns:
        True when any of its factors requires an as-traded field.
    """
    return any(
        set(FORMULA_POINT_IN_TIME_FIELDS) & set(spec.required_fields) for spec in catalog.factors
    )


_WORKER_KERNELS: dict[str, FeatureKernelRegistry] = {}
"""A kept worker's rebuilt shipped kernels, by the capability each sending build installed."""
_WORKER_CUTOFF_SETS: dict[tuple[str, ...], dict[str, tuple[str, str]]] = {}
"""A kept worker's derived cutoff sets, by factor axis: a set belongs to a session, so every
listing offers the same ones and the worker derives each once."""


def feature_source_frame(
    source: FeatureSourceInputs, *, sessions: Sequence[date] | None
) -> pd.DataFrame:
    """A listing's input frame, projected from its stored inputs.

    Args:
        source: What the build's writer read for the listing's window.
        sessions: The calendar's sessions from the window's start, when the Panel policy
            admits missing source rows: the frame is aligned to them.

    Returns:
        One row per bar, or per calendar session when aligned.
    """
    frame = pd.DataFrame(projected_feature_rows(source))
    if sessions is not None:
        frame = align_feature_source_sessions(frame, tuple(sessions))
    return frame


def materialize_listing_rows(
    *,
    catalog: FeatureCatalog,
    kernel_registry: FeatureKernelRegistry | None,
    kernel_capability: str,
    listing_id: str,
    source: FeatureSourceInputs,
    source_sessions: tuple[date, ...] | None,
    market_bars: pd.DataFrame,
    sessions: tuple[date, ...],
    identity_axis: tuple[str, tuple[str, ...]] | None,
    cancelled: Callable[[], bool],
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[FeatureRowIdentity, ...] | None, str]:
    """One listing's Feature rows for `sessions` and its ineligibility facts, in a Host worker.

    The listing's computation, moved off the build's writer, which only reads, writes and
    seals (W10): its input frame is projected here from the stored inputs the writer read
    (V92), and the frame's values hash is answered for the write's source window. Given an
    `identity_axis` -- the catalog hash and factor axis, when the rows are written as computed
    -- the rows' identities are derived here as well, by the derivation the closure
    coordinator runs. The kernels are the shipped ones, rebuilt here and held to the
    capability the build installed, unless the build sends its own.
    """
    projected_bars = feature_source_frame(source, sessions=source_sessions)
    with span("hash", "listing_inputs"):
        stock_input_hash = feature_source_values_hash(projected_bars, _REQUIRED_COLUMNS)
    registry = kernel_registry
    if registry is None:
        registry = _WORKER_KERNELS.get(kernel_capability)
        if registry is None:
            registry = default_extension_kernel_registry()
            if registry.installed_capability_hash != kernel_capability:
                raise ValueError("feature.worker_kernel_capability_mismatch")
            _WORKER_KERNELS[kernel_capability] = registry
    with span("features", "listing_rows"):
        block = BaseFeatureMaterializer(catalog, kernel_registry=registry).materialize_listing(
            listing_id=listing_id,
            projected_bars=projected_bars,
            market_bars=market_bars,
            sessions=sessions,
        )
    rows = block.values
    if identity_axis is None:
        return rows, block.ineligibility, None, stock_input_hash
    catalog_hash, factor_ids = identity_axis
    with span("hash", "row_identities"):
        identities = feature_row_identities(
            rows,
            catalog_hash=catalog_hash,
            factor_ids=factor_ids,
            cutoff_sets=_WORKER_CUTOFF_SETS.setdefault(factor_ids, {}),
        )
    return rows, block.ineligibility, identities, stock_input_hash


_PANEL_CHUNK = f"{__name__}:materialize_panel_chunk"
"""A Panel chunk's cross-section computation, run in the Host's kept workers (W10)."""

_WORKER_PANELS: dict[tuple[str, str], SectorNeutralPanelMaterializer] = {}
"""A kept worker's Panel materializers, by catalog and the implementation each writer runs."""


def materialize_panel_chunk(
    *,
    catalog: FeatureCatalog,
    implementation_binding_hash: str,
    feature_rows: pd.DataFrame,
    active_listing_ids: tuple[str, ...],
    manifest_revision: str,
    sector_revision: str,
    sector_by_listing_id: dict[str, str],
    spy_revision: str,
    factor_ids: tuple[str, ...],
    members_by_session: Mapping[date, Sequence[str]],
    source_exclusions_by_session: Mapping[date, Sequence[str]] | None,
    cancelled: Callable[[], bool],
) -> PanelMaterialization:
    """One chunk of a build's Panel sessions, transformed cross-section by cross-section.

    The computation of a Panel chunk, moved off the build's writer, which reads the chunk's
    Feature rows and then stages, publishes and records the answers in chunk order (W10).
    The materializer is the installed one, rebuilt here from the catalog and held to the
    implementation the writer runs. One chunk is one transformation, so W10's ``cancelled``
    is not consulted within it, as the writer's own loop never consulted it per chunk.
    """
    key = (catalog.binding.catalog_hash, implementation_binding_hash)
    materializer = _WORKER_PANELS.get(key)
    if materializer is None:
        materializer = SectorNeutralPanelMaterializer(catalog)
        if materializer.implementation.implementation_binding_hash != implementation_binding_hash:
            raise ValueError("feature.worker_panel_implementation_mismatch")
        _WORKER_PANELS[key] = materializer
    with span("features", "panel_chunk"):
        return materializer.materialize(
            feature_rows=feature_rows,
            active_listing_ids=active_listing_ids,
            manifest_revision=manifest_revision,
            sector_revision=sector_revision,
            sector_by_listing_id=sector_by_listing_id,
            spy_revision=spy_revision,
            factor_ids=factor_ids,
            members_by_session=members_by_session,
            source_exclusions_by_session=source_exclusions_by_session,
        )


@dataclass
class _PartWork:
    """What one part of a listing's work computes and writes under its own catalog (V92).

    An unlayered catalog is its own only part.
    """

    catalog: FeatureCatalog
    target_factors: dict[date, tuple[str, ...]]
    plan_hash: str
    first_output_session: date
    input_start: date
    """Its source window's first input session; the window's stock input hash is the
    worker's, over the frame it projected from the inputs read here."""
    raw_hash: str
    action_hash: str


@dataclass
class _ListingWork:
    """One listing of a build: read by the writer, computed by a worker, written in its turn.

    A read that failed keeps its failure for the listing's turn, so a build stops at the same
    listing, with the same record, whatever it read ahead. Each part of the catalog's layer the
    listing computes is one worker call, answered in the order the calls were made.
    """

    index: int
    listing: ManifestListing
    target_sessions: tuple[date, ...] = ()
    failure: Exception | None = None
    reports_progress: bool = False
    computed: int = 0
    calendar: tuple[date, ...] = ()
    parts: list[_PartWork] = field(default_factory=list)


class _ListingPlan(NamedTuple):
    """A listing's calendar, its parts' plans with their targets, and its window's input start."""

    calendar: tuple[date, ...]
    planned: tuple[tuple[Any, Any, dict[date, tuple[str, ...]]], ...]
    input_start: date


class _HeldInputs(NamedTuple):
    """A listing's stored inputs and each held part's rows at its targets, from its chunk's read."""

    inputs: FeatureSourceInputs | None
    rows: dict[str, list[dict[str, object]]]


_SET_READ_LISTINGS = 25
"""How many listings' inputs one set read holds: a chunk is read when the read-ahead reaches it,
so a cold or entrant day holds 25 listings' history, not the universe's. Speed only."""


@dataclass
class FeatureFoundationService:
    """Build raw/action-derived base features and one current-sector panel.

    This is intentionally not an Agent tool.  It is a bounded,
    deterministic data operation; a separate task parent can decide when to
    invoke it and route only exhausted failures to the constrained Data
    Engineer remediation surface.
    """

    market_data: MarketDataRepository
    feature_state: FeatureStateRepository
    panel_state: PanelStateRepository
    manifest: UniverseManifest
    provider: MarketDataProvider
    mutation_gate: WorkspaceMutationGate
    panel_artifacts: PanelArtifactCompositionOwner
    feature_persistence: FeatureMaterializationPersistence
    sector_activation: SectorRevisionMapActivationCoordinator
    sector_transport_policy: SectorTransportPolicy = field(
        default_factory=lambda: SectorTransportPolicy(max_workers=2)
    )
    progress_sink: Callable[[WorkProgressUpdate], object] | None = None
    installed_catalog: FeatureCatalog | None = None
    """The catalog revision this service materializes, defaulting to the shipped one.

    Held as composition state rather than loaded per service, so a development
    workspace can install a catalog revision that names an extension factor
    without the shipped catalog -- and therefore every published panel binding
    that quotes its hash -- moving at all.
    """

    kernel_registry: FeatureKernelRegistry | None = None
    persistence_timing_sink: FeaturePersistenceTimingSink | None = None
    """Where the persistence layer reports how long each of its stages took.

    The persistence layer has carried this seam for some time and no caller ever
    supplied one, so the stage timings existed and went nowhere. Attributing a
    slow build then meant re-deriving them from outside, which is measurement
    the process already performs and throws away.
    """
    session_authority_resolver: Callable[..., TradingSessionAuthority] | None = None
    feature_workers: int | None = None
    """How many of the Host's kept workers a build spreads its listings and its Panel chunks
    over (W10).

    An execution parameter: the rows, their order and every identity are the same for any count,
    since a listing's or a chunk's computation reads only its own inputs and the writer takes the
    answers in listing or chunk order. None takes the Task's CPU budget, the operator's
    (`worker_processes` of the cores the Host applied when the Task started, which its execution
    log records); the count a build used is returned beside its result and in its progress.
    """

    def __post_init__(self) -> None:
        """Install the selected catalog and its Feature and Panel materializers."""
        self.catalog = (
            self.installed_catalog if self.installed_catalog is not None else FeatureCatalog.load()
        )
        self.layer = FeatureCatalogLayer.over(self.catalog)
        self.base = BaseFeatureMaterializer(self.catalog, kernel_registry=self.kernel_registry)
        self.panel = SectorNeutralPanelMaterializer(self.catalog)
        self.ineligibility = FeatureIneligibilityStore(self.feature_state)
        self._bounded_warmup_sessions = max(
            item.lookback_sessions + item.formula_skip_sessions + 2
            for item in self.catalog.maintenance_contracts
        )

    def observe_candidate_sectors(
        self, listings: Sequence[ManifestListing]
    ) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]], bool] | None:
        """The current Sector of a recheck's candidates alone, by the Sector refresh's rules.

        Nothing is staged or activated. None when the provider observes no Sector.
        """
        if not hasattr(self.provider, "fetch_current_sector"):
            return None
        return SectorRefreshStager(
            manifest=self.manifest,
            provider=self.provider,
            artifact_root=self.feature_state.workspace / "staging" / "sector-reference",
            staging_id=canonical_hash(
                {
                    "kind": "sector-candidate-observation",
                    "listings": [listing.listing_id for listing in listings],
                }
            ),
            transport_policy=self.sector_transport_policy,
            activation_coordinator=self.sector_activation,
        ).observe(listings)

    def build(
        self,
        request: FeatureBuildRequest,
        *,
        observed_at: datetime | None = None,
        invalidation: FeatureInvalidation | None = None,
        invalidations: Sequence[FeatureInvalidation] = (),
        refresh_sector: bool = False,
        compose_panel: bool = True,
    ) -> FeatureBuildOutcome:
        """Build Feature rows and their Panel under the admitted source identities.

        ``compose_panel=False`` stops once the base values are materialized and the Sector
        revision is active, for a membership its baseline qualification will replace (V311).
        """
        now = observed_at or datetime.now(UTC)
        if now.tzinfo is None:
            raise ValueError("feature build observed_at must be timezone-aware")
        if request.manifest_revision != self.manifest.revision_sha256:
            return FeatureBuildOutcome(
                FeatureBuildStatus.BLOCKED,
                request.request_hash,
                None,
                {},
                failure_code="feature.manifest_revision_mismatch",
            )
        if request.catalog.catalog_hash != self.catalog.binding.catalog_hash:
            return FeatureBuildOutcome(
                FeatureBuildStatus.BLOCKED,
                request.request_hash,
                None,
                {},
                failure_code="feature.catalog_binding_mismatch",
            )
        if invalidation is not None and invalidations:
            raise ValueError("pass one invalidation surface, not both")
        invalidation_items = (
            tuple(invalidations)
            or ((invalidation,) if invalidation else ())
            or request.invalidations
        )
        self.market_data.bootstrap(self.manifest)
        self.feature_state.ensure_current_storage()
        reconcile_feature_panel_lifecycles(
            panel_state=self.panel_state,
            resolver=self.panel_artifacts.resolver,
            mutation_gate=self.mutation_gate,
            observed_at=now,
        )
        reference = self.feature_state.market_reference("SPY")
        if reference is None or str(reference["revision_hash"]) != request.spy_revision:
            return FeatureBuildOutcome(
                FeatureBuildStatus.BLOCKED,
                request.request_hash,
                None,
                {},
                failure_code="feature.spy_reference_not_ready",
            )
        spy = MarketReference.spy(self.manifest)
        try:
            market_payload, _market_raw_hash, _market_action_hash = (
                self.feature_state.projected_feature_frame(
                    spy.manifest, listing_id=spy.listing_id, through=request.as_of_session
                )
            )
        except ValueError:
            return FeatureBuildOutcome(
                FeatureBuildStatus.BLOCKED,
                request.request_hash,
                None,
                {},
                failure_code="feature.spy_projection_unavailable",
            )
        market_frame = pd.DataFrame(market_payload)
        market_input_hashes = {
            request.as_of_session: feature_source_values_hash(
                market_frame, ("session_date", "provider_adjusted_close")
            )
        }
        receipts: list[str] = []
        # A batch is sealed by one part's closure, so each part batches its own writes.
        pending_writes: dict[str, list[FeatureMaterializationWrite]] = {
            catalog_hash: [] for catalog_hash in self.layer.part_hashes
        }
        calendars: dict[str, tuple[date, ...]] = {}

        # Sector metadata is independent of base-feature mathematics. Start
        # its synchronous yfinance transport on one background coordinator so
        # the network wait is hidden behind local feature computation. The
        # maintainer's own bounded provider workers remain unchanged; every
        # DuckDB interaction is serialized by the shared mutation gate.
        sector = self.feature_state.current_sector_state(self.manifest)
        prior_sector_revision = sector.sector_revision if sector is not None else None
        should_refresh_sector = refresh_sector and self.feature_state.sector_revision_refresh_due(
            self.manifest, observed_at=now
        )
        sector_executor: ThreadPoolExecutor | None = None
        sector_future: Future[SectorRefreshOutcome] | None = None
        sector_stager: SectorRefreshStager | None = None
        if (
            (sector is None or should_refresh_sector)
            and refresh_sector
            and hasattr(self.provider, "fetch_current_sector")
        ):
            sector_stager = SectorRefreshStager(
                manifest=self.manifest,
                provider=self.provider,
                artifact_root=self.feature_state.workspace / "staging" / "sector-reference",
                staging_id=canonical_hash(
                    {
                        "kind": "sector-refresh-staging",
                        "request": request.request_hash,
                        "prior_sector_revision": (
                            sector.sector_revision if sector is not None else None
                        ),
                    }
                ),
                transport_policy=self.sector_transport_policy,
                activation_coordinator=self.sector_activation,
            )
            sector_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="sector-refresh-coordinator",
            )
            sector_future = sector_executor.submit(
                sector_stager.acquire,
                observed_at=now,
            )

        def flush_feature_writes(base_connection, catalog_hash: str) -> None:
            writes = pending_writes[catalog_hash]
            if not writes:
                return
            with span("write", "feature_rows"):
                receipts.extend(
                    self.feature_persistence.persist_batch(
                        tuple(writes),
                        connection=base_connection,
                        timing_sink=self.persistence_timing_sink,
                    )
                )
            writes.clear()

        # The calculation axis: every listing some session of the history
        # holds, not only the manifest's current members. Base Formula values
        # are kept for all of them, so a session before a listing's exit
        # still has its member's rows after a catalog change.
        try:
            axis_listings, scope_manifests = self._axis_listings()
        except ValueError as exc:
            return FeatureBuildOutcome(
                FeatureBuildStatus.BLOCKED,
                request.request_hash,
                None,
                {"detail": " ".join(str(exc).split())[:500]},
                failure_code="feature.membership_axis_unresolved",
            )
        base_started = time.perf_counter()
        materialization_failure: dict[str, object] | None = None
        publication_blocked = False
        listing_plans = []
        current_listing_id: str | None = None
        current_target_sessions: tuple[date, ...] = ()
        # Each part of the layer plans from its own state: a part whose closure is complete
        # keeps its rows through a catalog change (V92).
        part_invalidations = self._part_invalidations(invalidation_items, axis_listings, request)
        part_listings = {
            catalog_hash: {
                item.listing_id
                for item in self._listings_for_invalidations(invalidations, axis_listings)
            }
            for catalog_hash, invalidations in part_invalidations.items()
        }
        target_listings = tuple(
            item
            for item in axis_listings
            if any(item.listing_id in listing_ids for listing_ids in part_listings.values())
        )
        base_progress_total = max(1, len(target_listings))
        axis_ids = tuple(item.listing_id for item in axis_listings)
        # The active Panel serves as the base of this catalog's composition when this catalog
        # only adds columns to the one it was built under: the catalog change then reaches the
        # Panel for the added columns alone (V92).
        held_columns = self._held_panel_columns()
        panel_invalidations = (
            tuple(
                replace(
                    item,
                    factor_ids=tuple(
                        sorted(set(self.catalog.factor_ids) - set(held_columns.factor_ids))
                    ),
                )
                if item.kind == "catalog_binding_change"
                else item
                for item in invalidation_items
            )
            if held_columns is not None
            else invalidation_items
        )
        completed_listing_count = 0
        # The Task's cores, as the Host applied its CPU budget when the Task started (B7);
        # outside a Task none is applied and the build takes one worker.
        workers = (
            self.feature_workers
            if self.feature_workers is not None
            else worker_processes(reader_threads() or 1)
        )
        kernel_capability = self.base.kernel_registry.installed_capability_hash
        # The shipped kernels are rebuilt in the worker, since their functions do not
        # pickle; a registry this build was given crosses as itself.
        shipped_kernels = (
            None
            if kernel_capability == default_extension_kernel_registry().installed_capability_hash
            else self.base.kernel_registry
        )
        try:
            self._publish_progress(
                operation_id=request.request_hash,
                stage_id="base_feature_materialization",
                completed=0,
                total=base_progress_total,
                unit_name="listings",
            )
            with (
                self.mutation_gate.hold(),
                self.feature_state.feature_build_connection() as base_connection,
                child_calls(_LISTING_ROWS, workers=workers) as calls,
            ):
                self.feature_persistence.reconcile_pending(
                    self.catalog.binding.catalog_hash,
                    connection=base_connection,
                )
                # A part that holds no row yet (an activation's new column) has no stored row a
                # build could reuse, so its listings skip the read of their verified rows; and a
                # column part that holds none under its genesis head is loaded whole, its rows
                # written as computed with no per-key transition, its head attesting their
                # digest once the stage has written them all (V92).
                parts_holding_rows = frozenset(
                    part.binding.catalog_hash
                    for part in self.layer.parts
                    if part.binding.catalog_hash in part_invalidations
                    and self.feature_state.feature_sessions(
                        listing_ids=axis_ids,
                        catalog_hash=part.binding.catalog_hash,
                        start=request.history_start,
                        end=request.as_of_session,
                        _connection=base_connection,
                    )
                )
                first_loads = self.feature_persistence.begin_first_loads(
                    tuple(
                        part.binding.catalog_hash
                        for part in self.layer.columns
                        if part.binding.catalog_hash in part_invalidations
                        and part.binding.catalog_hash not in parts_holding_rows
                    ),
                    idempotency_key=request.request_hash,
                    observed_at=now,
                    connection=base_connection,
                )

                with span("read", "listing_inputs"):
                    raw_calendars = self.market_data.raw_bar_sessions_by_listing(
                        [listing.listing_id for listing in target_listings],
                        through=request.as_of_session,
                        _connection=base_connection,
                    )

                def plan_listing(
                    listing_index: int, listing: ManifestListing
                ) -> tuple[_ListingWork, _ListingPlan | None]:
                    """One listing's calendar, its parts' targets and its input start."""
                    work = _ListingWork(listing_index, listing)
                    try:
                        calendar = self._base_session_calendar(
                            observed_sessions=raw_calendars[listing.listing_id],
                            calendar_id=scope_manifests[listing.listing_id].profile.calendar_id,
                            history_start=request.history_start,
                            as_of_session=request.as_of_session,
                            observed_at=now,
                            calendars=calendars,
                        )
                        # What of this listing's base work the Panel sees: an
                        # entrant's whole-history values serve its windows, its
                        # cross-sections change from the session it joins.
                        listing_plans.append(
                            (
                                listing.listing_id,
                                compile_listing_plan(
                                    catalog=self.catalog,
                                    request=request,
                                    invalidations=panel_invalidations,
                                    listing_id=listing.listing_id,
                                    sessions=calendar,
                                    reach="panel",
                                ),
                            )
                        )
                        planned = []
                        for part in self.layer.parts:
                            part_hash = part.binding.catalog_hash
                            if listing.listing_id not in part_listings.get(part_hash, ()):
                                continue
                            listing_plan = compile_listing_plan(
                                catalog=part,
                                request=request,
                                invalidations=part_invalidations[part_hash],
                                listing_id=listing.listing_id,
                                sessions=calendar,
                            )
                            target_factors = targets_by_session(listing_plan, calendar)
                            if target_factors:
                                planned.append((part, listing_plan, target_factors))
                        if not planned:
                            work.reports_progress = True
                            return work, None
                        work.target_sessions = tuple(
                            sorted({day for _part, _plan, targets in planned for day in targets})
                        )
                        # Every active formula is evaluated from direct finite
                        # source windows.  Correction and append rebuilds can use
                        # the same bounded prefix without partition-dependent
                        # rolling state.
                        input_start = self._bounded_input_start(
                            calendar=calendar,
                            first_target=work.target_sessions[0],
                            warmup_sessions=self._bounded_warmup_sessions,
                        )
                        return work, _ListingPlan(calendar, tuple(planned), input_start)
                    except Exception as error:
                        work.failure = error
                        return work, None

                planned_listings = [
                    plan_listing(index, listing)
                    for index, listing in enumerate(target_listings, start=1)
                ]

                def planned_in_chunks() -> Iterator[
                    tuple[_ListingWork, _ListingPlan | None, _HeldInputs]
                ]:
                    """Each planned listing with its stored inputs and held rows, read for a
                    chunk of listings when the read-ahead reaches it; a chunk's holdings go
                    with its listings."""
                    for offset in range(0, len(planned_listings), _SET_READ_LISTINGS):
                        chunk = planned_listings[offset : offset + _SET_READ_LISTINGS]
                        with span("read", "listing_inputs"):
                            inputs, rows = self._planned_inputs(
                                chunk,
                                scope_manifests=scope_manifests,
                                parts_holding_rows=parts_holding_rows,
                                through=request.as_of_session,
                                connection=base_connection,
                            )
                        for work, plan in chunk:
                            listing_id = work.listing.listing_id
                            yield (
                                work,
                                plan,
                                _HeldInputs(
                                    inputs.get(listing_id),
                                    {
                                        part: value
                                        for (owner, part), value in rows.items()
                                        if owner == listing_id
                                    },
                                ),
                            )

                @spanned("read", "listing_inputs")
                def read_listing(
                    work: _ListingWork, plan: _ListingPlan, held: _HeldInputs
                ) -> _ListingWork:
                    """Read one listing's inputs and hand each part's computation to a worker."""
                    listing = work.listing
                    calendar, planned, input_start = plan
                    try:
                        # The writer reads each window's stored inputs; the worker computing
                        # the listing projects them. Only a held part's reuse is verified here,
                        # against the window's frame (V92).
                        sources = {
                            input_start: held.inputs
                            or self._base_source_inputs(
                                scope_manifests[listing.listing_id],
                                listing_id=listing.listing_id,
                                through=request.as_of_session,
                                start=input_start,
                                _connection=base_connection,
                            )
                        }
                        verified_frame: pd.DataFrame | None = None
                        remaining = []
                        for part, listing_plan, target_factors in planned:
                            if part.binding.catalog_hash not in parts_holding_rows:
                                remaining.append((part, listing_plan, target_factors))
                                continue
                            prepared = held.rows[part.binding.catalog_hash]
                            reused: set[date] = set()
                            if prepared:
                                if verified_frame is None:
                                    verified_frame = feature_source_frame(
                                        sources[input_start],
                                        sessions=self._source_sessions(calendar, input_start),
                                    )
                                reused = self._verified_source_sessions(
                                    listing_id=listing.listing_id,
                                    catalog_hash=part.binding.catalog_hash,
                                    prepared=prepared,
                                    frame=verified_frame,
                                    market_frame=market_frame,
                                    calendar=calendar,
                                    request=request,
                                    connection=base_connection,
                                    market_input_hashes=market_input_hashes,
                                )
                            target_factors = {
                                day: factors
                                for day, factors in target_factors.items()
                                if day not in reused
                            }
                            if target_factors:
                                remaining.append((part, listing_plan, target_factors))
                        if not remaining:
                            return work
                        work.target_sessions = tuple(
                            sorted({day for _part, _plan, targets in remaining for day in targets})
                        )
                        work.calendar = calendar
                        for part, listing_plan, target_factors in remaining:
                            selected_sessions = tuple(sorted(target_factors))
                            narrowed_start = self._bounded_input_start(
                                calendar=calendar,
                                first_target=selected_sessions[0],
                                warmup_sessions=self._bounded_warmup_sessions,
                            )
                            if narrowed_start not in sources:
                                sources[narrowed_start] = self._base_source_inputs(
                                    scope_manifests[listing.listing_id],
                                    listing_id=listing.listing_id,
                                    through=request.as_of_session,
                                    start=narrowed_start,
                                    _connection=base_connection,
                                )
                            source = sources[narrowed_start]
                            work.parts.append(
                                _PartWork(
                                    catalog=part,
                                    target_factors=target_factors,
                                    plan_hash=listing_plan.plan_hash,
                                    first_output_session=selected_sessions[0],
                                    input_start=narrowed_start,
                                    raw_hash=source.raw_hash,
                                    action_hash=source.action_hash,
                                )
                            )
                            calls.make(
                                {
                                    "kernel_registry": shipped_kernels,
                                    "kernel_capability": kernel_capability,
                                    "listing_id": listing.listing_id,
                                    "source": source,
                                    "source_sessions": self._source_sessions(
                                        calendar, narrowed_start
                                    ),
                                    "sessions": selected_sessions,
                                    # The rows are written as computed unless an unaffected
                                    # factor keeps its stored value; only then are their
                                    # identities the worker's to derive.
                                    "identity_axis": (
                                        (part.binding.catalog_hash, tuple(part.factor_ids))
                                        if self._preserves_nothing(target_factors, part)
                                        else None
                                    ),
                                },
                                # The same for every listing: each worker receives them once.
                                shared={
                                    "catalog": (f"catalog:{part.binding.catalog_hash}", part),
                                    "market_bars": (
                                        f"market:{market_input_hashes[request.as_of_session]}",
                                        market_frame,
                                    ),
                                },
                            )
                            work.computed += 1
                    except Exception as error:
                        work.failure = error
                    return work

                listings = planned_in_chunks()
                queue: deque[_ListingWork] = deque()
                computing = 0
                while True:
                    # Read ahead while a worker is free: listings compute while the
                    # writer writes and seals the ones before them, and are still
                    # written one by one, in listing order.
                    while computing < workers:
                        entry = next(listings, None)
                        if entry is None:
                            break
                        planned_work, plan, held = entry
                        queue.append(
                            planned_work if plan is None else read_listing(planned_work, plan, held)
                        )
                        computing += queue[-1].computed
                    if not queue:
                        break
                    work = queue.popleft()
                    computing -= work.computed
                    listing = work.listing
                    current_listing_id = listing.listing_id
                    current_target_sessions = work.target_sessions
                    if work.failure is not None:
                        raise work.failure
                    for part_work in work.parts:
                        part_hash = part_work.catalog.binding.catalog_hash
                        rows, ineligibility, identities, stock_input_hash = calls.answer()
                        source_window = FeatureSourceWindow(
                            first_output_session=part_work.first_output_session,
                            input_start=part_work.input_start,
                            input_end=request.as_of_session,
                            stock_input_hash=stock_input_hash,
                            market_input_hash=market_input_hashes[request.as_of_session],
                        )
                        with span("read", "preserve_unaffected"):
                            selected_rows = self._preserve_unaffected_feature_values(
                                rows,
                                part_work.target_factors,
                                catalog=part_work.catalog,
                                _connection=base_connection,
                            )
                        identity_by_session = (
                            None
                            if identities is None
                            else {identity.session_date: identity for identity in identities}
                        )
                        for scoped_factors, range_sessions in self._panel_work_groups(
                            work.calendar, part_work.target_factors
                        ):
                            first_session = range_sessions[0]
                            last_session = range_sessions[-1]
                            range_rows = selected_rows.loc[
                                selected_rows["session_date"].isin(range_sessions)
                            ].copy()
                            range_session_set = set(range_sessions)
                            range_ineligibility = ineligibility.loc[
                                ineligibility["session_date"].isin(range_session_set)
                                & ineligibility["factor_id"].isin(scoped_factors)
                            ].copy()
                            pending_writes[part_hash].append(
                                FeatureMaterializationWrite(
                                    listing_id=listing.listing_id,
                                    catalog_hash=part_hash,
                                    rows=range_rows,
                                    ineligibility=range_ineligibility,
                                    factor_ids=scoped_factors,
                                    rows_are_canonical=True,
                                    source_window=source_window,
                                    raw_input_hash=part_work.raw_hash,
                                    action_set_hash_value=part_work.action_hash,
                                    market_reference_revision=request.spy_revision,
                                    idempotency_key=canonical_hash(
                                        [
                                            request.request_hash,
                                            listing.listing_id,
                                            part_work.plan_hash,
                                            first_session.isoformat(),
                                            last_session.isoformat(),
                                            scoped_factors,
                                        ]
                                    ),
                                    revision_reason="factor_session_delta",
                                    observed_at=now,
                                    row_identities=(
                                        None
                                        if identity_by_session is None
                                        else tuple(
                                            identity_by_session[session]
                                            for session in range_rows["session_date"]
                                        )
                                    ),
                                )
                            )
                            if len(pending_writes[part_hash]) == admitted_writes(
                                len(part_work.catalog.factor_ids)
                            ):
                                flush_feature_writes(base_connection, part_hash)
                    if work.computed or work.reports_progress:
                        self._publish_progress(
                            operation_id=request.request_hash,
                            stage_id="base_feature_materialization",
                            completed=work.index,
                            total=len(target_listings),
                            unit_name="listings",
                            current_item=listing.listing_id,
                            counters={"materialization_receipts": len(receipts)},
                        )
                    completed_listing_count = work.index
                for catalog_hash in pending_writes:
                    flush_feature_writes(base_connection, catalog_hash)
                with span("write", "feature_finalize"):
                    self.feature_persistence.complete_first_loads(
                        first_loads, connection=base_connection
                    )
                    self.feature_persistence.assert_ready(self.catalog.binding.catalog_hash)
                    # Every part now holds the axis; rows of a catalog outside the layer (one a
                    # rotation replaced, a deactivated column) answer for nothing (V398, V92).
                    self.feature_state.retire_rows_outside_layer(_connection=base_connection)
                    # The closed years' seals, once each: a year the build closed, or one a
                    # correction reopened, so tomorrow's proofs read only the open year.
                    self.feature_state.seal_closed_feature_years(
                        self.catalog.binding.catalog_hash, _connection=base_connection
                    )
                self._publish_progress(
                    operation_id=request.request_hash,
                    stage_id="base_feature_materialization",
                    completed=base_progress_total,
                    total=base_progress_total,
                    unit_name="listings",
                    counters={"materialization_receipts": len(receipts), "workers": workers},
                    status="SUCCEEDED",
                )
        except (AssertionError, ChildStartFailed):
            if sector_executor is not None:
                sector_executor.shutdown(wait=True, cancel_futures=True)
            raise
        except Exception as exc:
            materialization_failure = {
                "stage": "base_feature_materialization",
                # A worker's failure keeps the name of what it raised there.
                "exception_type": getattr(exc, "type_name", type(exc).__name__),
                "listing_id": current_listing_id,
                "first_target_session": (
                    current_target_sessions[0].isoformat() if current_target_sessions else None
                ),
                "last_target_session": (
                    current_target_sessions[-1].isoformat() if current_target_sessions else None
                ),
                "detail": " ".join(str(exc).split())[:500],
            }
            # A closure file the ledger could not place on disk (a sharing
            # violation that outlasted its retry bound) is the same retryable
            # refusal the sector activation names; the store is unchanged.
            publication_blocked = isinstance(exc, WorkspaceConflictError)

        if materialization_failure is not None:
            detail = str(materialization_failure.get("detail", ""))
            failure_code = (
                detail
                if detail.startswith("feature_closure.")
                else "feature_closure.publication_blocked"
                if publication_blocked
                else "feature.materialization_failed"
            )
            self._publish_progress(
                operation_id=request.request_hash,
                stage_id="base_feature_materialization",
                completed=completed_listing_count,
                total=base_progress_total,
                unit_name="listings",
                current_item=current_listing_id,
                status="FAILED",
                failure_code=failure_code,
                counters={"completed_listings": completed_listing_count},
            )
            if sector_executor is not None:
                # A background worker must never outlive the owning task and
                # continue mutating the workspace after a blocked outcome.
                sector_executor.shutdown(wait=True, cancel_futures=True)
            return FeatureBuildOutcome(
                FeatureBuildStatus.BLOCKED,
                request.request_hash,
                None,
                {
                    "materialized_listings": len(receipts),
                    "materialization_failure": materialization_failure,
                },
                failure_code=failure_code,
                build_stage=self._durable_build_stage(),
            )
        base_elapsed = time.perf_counter() - base_started

        if sector is None or should_refresh_sector:
            if sector_future is None:
                return FeatureBuildOutcome(
                    FeatureBuildStatus.DEFERRED,
                    request.request_hash,
                    None,
                    {"materialized_listings": len(receipts)},
                    failure_code="feature.sector_reference_required",
                )
            try:
                sector_future.result()
                assert sector_stager is not None
                sector_outcome = sector_stager.commit(
                    store=self.feature_state,
                    mutation_gate=self.mutation_gate,
                    observed_at=now,
                )
            finally:
                assert sector_executor is not None
                sector_executor.shutdown(wait=True)
            if sector_outcome.status != "completed":
                # A previously complete taxonomy remains usable while its next
                # full review is merely deferred. Initial initialization has no
                # such safe fallback and must wait for one complete revision.
                if sector is None or sector_outcome.status != "deferred":
                    return FeatureBuildOutcome(
                        FeatureBuildStatus.DEFERRED
                        if sector_outcome.status == "deferred"
                        else FeatureBuildStatus.BLOCKED,
                        request.request_hash,
                        None,
                        {"materialized_listings": len(receipts)},
                        retry_after_at=sector_outcome.retry_after_at,
                        failure_code=sector_outcome.failure_code,
                    )
            else:
                sector = self.feature_state.current_sector_state(self.manifest)
        if sector is None:
            raise AssertionError("completed sector refresh did not activate a revision")
        if not compose_panel:
            return FeatureBuildOutcome(
                FeatureBuildStatus.COMPLETED,
                request.request_hash,
                None,
                {
                    "materialized_listings": len(receipts),
                    "panel": "not_composed",
                    "base_feature_workers": workers,
                },
                build_stage=self._durable_build_stage(),
            )
        sector_revision = sector.sector_revision
        if (
            prior_sector_revision is not None
            and sector_revision != prior_sector_revision
            and not any(item.kind == "sector_revision_change" for item in invalidation_items)
        ):
            invalidation_items = (
                *invalidation_items,
                FeatureInvalidation(
                    "sector_revision_change",
                    source_receipt_hash=sector_revision,
                ),
            )
        active_listing_ids = tuple(item.listing_id for item in self.manifest.listings)
        axis_listing_ids = tuple(item.listing_id for item in axis_listings)
        try:
            # Every axis listing has a classification, or the build is refused by name.
            self._axis_sectors(sector, axis_listings)
        except ValueError as exc:
            return FeatureBuildOutcome(
                FeatureBuildStatus.BLOCKED,
                request.request_hash,
                None,
                {"base_receipts": len(receipts), "detail": str(exc)},
                failure_code="feature.membership_sector_unresolved",
                build_stage=self._durable_build_stage(),
            )
        try:
            source_state_before = self.panel_state.panel_source_state_hash(
                listing_ids=active_listing_ids,
                catalog_hash=self.catalog.binding.catalog_hash,
                as_of_session=request.as_of_session,
                allow_missing_market_observations=allows_missing_source_rows(
                    self.panel.policy_hash
                ),
            )
        except ValueError as exc:
            return FeatureBuildOutcome(
                FeatureBuildStatus.BLOCKED,
                request.request_hash,
                None,
                {"base_receipts": len(receipts)},
                failure_code=(
                    str(exc)
                    if str(exc).startswith("feature.")
                    else "feature.panel_source_state_incomplete"
                ),
                build_stage=self._durable_build_stage(),
            )
        panel_calendar = self.feature_state.feature_sessions(
            listing_ids=axis_listing_ids,
            catalog_hash=self.catalog.binding.catalog_hash,
            start=request.history_start,
            end=request.as_of_session,
        )
        try:
            membership = self._panel_membership(
                calendar=panel_calendar, axis_listing_ids=axis_listing_ids
            )
        except ValueError as exc:
            return FeatureBuildOutcome(
                FeatureBuildStatus.BLOCKED,
                request.request_hash,
                None,
                {"base_receipts": len(receipts), "detail": str(exc)},
                failure_code=(
                    str(exc)
                    if str(exc) == "feature.membership_admission_required"
                    else "feature.membership_manifest_mismatch"
                ),
                build_stage=self._durable_build_stage(),
            )
        # Raw/base preparation may already include a future-effective entrant;
        # the published Panel axis contains only members of its own history.
        axis_listing_ids = membership.listing_ids
        # The Sector each session reads (V346): the backfill before a listing's first
        # reclassification, then each from the session its update observed it.
        sector_history = self.feature_state.sector_history(
            self.manifest, listing_ids=axis_listing_ids
        )
        if sector_history is None:
            raise AssertionError("completed sector refresh did not activate a revision")
        panel_binding = FeaturePanelBinding.create(
            manifest_revision=self.manifest.revision_sha256,
            sector_revision=sector_revision,
            catalog_hash=self.catalog.binding.catalog_hash,
            spy_revision=request.spy_revision,
            policy_hash=self.panel.policy_hash,
        )
        current_snapshot = self.panel_state.feature_panel_snapshot_for_active(
            self.manifest.profile.market_profile_id
        )
        base_manifest = (
            self.panel_artifacts.resolver.load_feature_panel_manifest(
                str(current_snapshot["manifest_uri"])
            )
            if current_snapshot is not None
            else None
        )
        # One Panel build materializes in session batches; the clipping receipt
        # describes the build, so the batches are collected and merged once.
        panel_batches: list[PanelMaterialization] = []
        composition = self.panel_artifacts.begin(
            operation_id=request.request_hash,
            binding=PanelCompositionBinding(
                manifest_revision=self.manifest.revision_sha256,
                sector_revision=sector_revision,
                catalog_hash=self.catalog.binding.catalog_hash,
                policy_hash=panel_binding.policy_hash,
                panel_binding_hash=panel_binding.panel_binding_hash,
                history_start=request.history_start,
                as_of_session=request.as_of_session,
                factor_ids=self.catalog.factor_ids,
                row_identity_basis=PANEL_ROW_IDENTITY_BY_CROSS_SECTION,
            ),
            base_manifest=base_manifest,
            sessions=panel_calendar,
            listing_ids=axis_listing_ids,
            spy_revision=request.spy_revision,
            membership=membership,
            sector_history=sector_history,
            # A Panel built under the catalog this one only adds columns to holds
            # every other column: its partitions are merged with the added ones (V92).
            extension=(
                PanelColumnExtension(
                    catalog_hash=held_columns.binding.catalog_hash,
                    factor_ids=tuple(held_columns.factor_ids),
                )
                if held_columns is not None
                else None
            ),
        )
        if composition.base_extended:
            assert held_columns is not None
            # The base's cells, recorded under this catalog with the batches that
            # computed them; the reuse below judges them as any held cell.
            with (
                self.panel_state.panel_write_connection() as carry_connection,
                self.mutation_gate.hold(),
            ):
                carry_connection.execute("BEGIN TRANSACTION")
                try:
                    PanelAvailabilityRepository().carry(
                        carry_connection,
                        cross_section_by_session=composition.cross_section_by_session(),
                        from_catalog_hash=held_columns.binding.catalog_hash,
                        to_catalog_hash=self.catalog.binding.catalog_hash,
                        policy_hash=panel_binding.policy_hash,
                        factor_ids=held_columns.factor_ids,
                        observed_at=now,
                    )
                    carry_connection.execute("COMMIT")
                except Exception:
                    carry_connection.execute("ROLLBACK")
                    raise
        # A compatible base (same rule, catalog, policy and factor axis; the
        # manifest, sector map and SPY revision may differ) lets the plan
        # stay sparse: only cells a source delta reaches, or sessions whose
        # cross-section is not the one the base computed, are recomputed,
        # and every other partition is the base's. Years the base cannot
        # stand in for -- no partition, a different calendar, cells whose
        # writer no published snapshot names, or a receipt that left no
        # clipping record -- are forced whole. Without a compatible base the
        # planner requests every Panel row, as it always did; underlying
        # Feature values still follow their own incremental plan.
        forced_sessions = self._forced_panel_sessions(
            composition,
            panel_binding=panel_binding,
            base_manifest=base_manifest,
            calendar=panel_calendar,
            request=request,
        )
        # A listing's base delta reaches the Panel on the sessions it is a
        # member of and no other: an entrant's whole-history Formula values
        # change no cross-section before its entry.
        member_sessions = {
            session: frozenset(membership.members(session)) for session in panel_calendar
        }
        panel_plan = compile_panel_plan(
            catalog=self.catalog,
            request=request,
            invalidations=panel_invalidations,
            listing_plans=tuple(
                restrict_plan_to_sessions(
                    plan,
                    calendar=panel_calendar,
                    allowed=lambda session, listing_id=listing_id: (
                        listing_id in member_sessions.get(session, frozenset())
                    ),
                )
                for listing_id, plan in listing_plans
            ),
            sessions=panel_calendar,
            base_reusable=composition.base_compatible,
            forced_sessions=forced_sessions,
            row_identity_basis=composition.binding.row_identity_basis,
        )
        panel_targets = targets_by_session(panel_plan, panel_calendar)
        availability_repository = PanelAvailabilityRepository()
        cross_section_by_session = composition.cross_section_by_session()
        cross_sections = composition.all_cross_sections()
        panel_started = time.perf_counter()
        panel_receipts: list[str] = []
        clip_records: dict[str, PanelClipObservationRecord] = {}
        panel_groups = tuple(self._panel_work_groups(panel_calendar, panel_targets))
        panel_chunk_total = sum(
            (len(grouped_sessions) + PANEL_SESSION_BATCH_SIZE - 1) // PANEL_SESSION_BATCH_SIZE
            for _factor_ids, grouped_sessions in panel_groups
        )
        panel_chunk_total = max(1, panel_chunk_total)
        panel_chunk_completed = 0
        self._publish_progress(
            operation_id=request.request_hash,
            stage_id="panel_compute",
            completed=0,
            total=panel_chunk_total,
            unit_name="chunks",
        )
        implementation = self.panel.implementation.implementation_binding_hash
        # A connection block's chunks compute in the Host's kept workers while the writer
        # reads the next chunk's Feature rows; the writer then stages, publishes and
        # records them in chunk order, as it did one by one (W10). A chunk a Sector
        # reclassification falls inside is one call per run of sessions that read one
        # map (V346).
        with child_calls(_PANEL_CHUNK, workers=workers) as panel_calls:
            for factor_ids, grouped_sessions in panel_groups:
                for connection_start in range(
                    0, len(grouped_sessions), _PANEL_CONNECTION_SESSION_BUDGET
                ):
                    connection_sessions = grouped_sessions[
                        connection_start : connection_start + _PANEL_CONNECTION_SESSION_BUDGET
                    ]
                    block = [
                        (chunk, run_sessions, run_sectors)
                        for chunk in (
                            connection_sessions[start : start + PANEL_SESSION_BATCH_SIZE]
                            for start in range(
                                0, len(connection_sessions), PANEL_SESSION_BATCH_SIZE
                            )
                        )
                        if chunk
                        for run_sessions, run_sectors in sector_history.runs(chunk)
                    ]
                    with self.panel_state.panel_write_connection() as panel_connection:
                        for _chunk, run_sessions, run_sectors in block:
                            panel_calls.make(
                                {
                                    "catalog": self.catalog,
                                    "implementation_binding_hash": implementation,
                                    "feature_rows": self.feature_state.feature_rows(
                                        listing_ids=axis_listing_ids,
                                        catalog_hash=self.catalog.binding.catalog_hash,
                                        start=run_sessions[0],
                                        end=run_sessions[-1],
                                        factor_ids=factor_ids,
                                        include_lineage=False,
                                        as_frame=True,
                                        _connection=panel_connection,
                                    ),
                                    "active_listing_ids": axis_listing_ids,
                                    "manifest_revision": self.manifest.revision_sha256,
                                    "sector_revision": sector_revision,
                                    "sector_by_listing_id": {
                                        listing_id: run_sectors[listing_id]
                                        for listing_id in axis_listing_ids
                                    },
                                    "spy_revision": request.spy_revision,
                                    "factor_ids": factor_ids,
                                    "members_by_session": membership.members_by_session(
                                        run_sessions
                                    ),
                                    "source_exclusions_by_session": (
                                        {
                                            session: membership.excluded_sources(session)
                                            for session in run_sessions
                                        }
                                        if membership.source_exclusions
                                        else None
                                    ),
                                }
                            )
                        for chunk_sessions, run_sessions, _run_sectors in block:
                            panel = cast(PanelMaterialization, panel_calls.answer())
                            panel_batches.append(panel)
                            with span("write", "panel_patch"):
                                composition.stage_patch(
                                    rows=panel.rows,
                                    factor_ids=factor_ids,
                                    materialization_receipt_hash=panel.receipt_hash,
                                )
                            # Measured while transforming, persisted under the
                            # receipt the rows and availability carry, and before
                            # the availability is stamped: a receipt without its
                            # record forces its year to be recomputed next time.
                            record = clip_observation_record(panel)
                            clip_records[record.receipt_hash] = record
                            self.panel_artifacts.resolver.publish_panel_clip_observation(
                                payload=record.model_dump(mode="json"),
                                receipt_hash=record.receipt_hash,
                            )
                            with self.mutation_gate.hold():
                                panel_connection.execute("BEGIN TRANSACTION")
                                try:
                                    availability_repository.upsert(
                                        panel_connection,
                                        cross_section_by_session=cross_section_by_session,
                                        catalog_hash=self.catalog.binding.catalog_hash,
                                        policy_hash=panel.binding.policy_hash,
                                        availability=panel.availability,
                                        materialization_receipt_hash=panel.receipt_hash,
                                        observed_at=now,
                                    )
                                    panel_connection.execute("COMMIT")
                                except Exception:
                                    panel_connection.execute("ROLLBACK")
                                    raise
                            panel_receipts.append(panel.receipt_hash)
                            # A chunk counts once, at its last run.
                            if run_sessions[-1] != chunk_sessions[-1]:
                                continue
                            panel_chunk_completed += 1
                            self._publish_progress(
                                operation_id=request.request_hash,
                                stage_id="panel_compute",
                                completed=panel_chunk_completed,
                                total=panel_chunk_total,
                                unit_name="chunks",
                                current_item=(
                                    f"{chunk_sessions[0].isoformat()}.."
                                    f"{chunk_sessions[-1].isoformat()}"
                                ),
                                counters={"panel_receipts": len(panel_receipts)},
                            )
        self._publish_progress(
            operation_id=request.request_hash,
            stage_id="panel_compute",
            completed=panel_chunk_total,
            total=panel_chunk_total,
            unit_name="chunks",
            counters={"panel_receipts": len(panel_receipts)},
            status="SUCCEEDED",
        )
        as_of_availability = self.panel_state.panel_availability_rows(
            cross_sections=cross_sections,
            catalog_hash=self.catalog.binding.catalog_hash,
            policy_hash=self.panel.policy_hash,
            start=request.as_of_session,
            end=request.as_of_session,
        )
        admission = PanelAdmissionSummary.evaluate(
            as_of_session=request.as_of_session,
            factor_ids=self.catalog.factor_ids,
            availability=as_of_availability,
            sector_distribution=sector.sector_distribution,
        )
        panel_elapsed = time.perf_counter() - panel_started
        risk = PanelTemporalRisk.current_yahoo(
            sector_observed_at=sector.sector_observed_at,
            sector_history_treatment=sector_treatment(
                reclassified=bool(sector_history.reclassifications)
            ),
        )
        temporal_boundary = TemporalKnowledgeBoundary(
            market_as_of_session=request.as_of_session,
            knowledge_cutoff_at=now,
            materialized_at=now,
            universe_source_observed_at=datetime.combine(
                self.manifest.profile.manifest_as_of,
                datetime.min.time(),
                tzinfo=UTC,
            ),
            sector_source_observed_at=sector.sector_observed_at,
            universe_point_in_time_qualified=self.manifest.is_point_in_time_historical,
            sector_point_in_time_qualified=False,
        )
        coverage = {
            "base_receipts": len(receipts),
            "base_feature_elapsed_seconds": base_elapsed,
            "base_feature_workers": workers,
            "panel_elapsed_seconds": panel_elapsed,
            "panel_complete_at_as_of": admission.research_admissible,
            "panel_admission": asdict(admission),
            "panel_invalidation_plan_hash": panel_plan.plan_hash,
            "sector_revision": sector_revision,
            "catalog_hash": self.catalog.binding.catalog_hash,
            "feature_computability": self.qualify_features(
                listing_ids=active_listing_ids,
                factor_ids=self.catalog.factor_ids,
                as_of_session=request.as_of_session,
            ).summary(),
            "row_identity_basis": PANEL_ROW_IDENTITY_BY_CROSS_SECTION,
            "membership": membership.summary_payload(),
            "temporal_risk": asdict(risk),
            "unavailable_factors_at_as_of": tuple(
                item["factor_id"] for item in as_of_availability if item["status"] != "available"
            ),
            "base_ineligibility_runs_at_as_of": len(
                self.ineligibility.find_runs(
                    listing_ids=active_listing_ids,
                    factor_ids=self.catalog.factor_ids,
                    start=request.as_of_session,
                    end=request.as_of_session,
                )
            ),
        }
        receipt_hash = canonical_hash(
            {
                "panel_receipts": panel_receipts,
                "invalidation_plan": panel_plan.plan_hash,
                "source_state": source_state_before,
            }
        )
        if not admission.research_admissible:
            return FeatureBuildOutcome(
                FeatureBuildStatus.BLOCKED,
                request.request_hash,
                receipt_hash,
                coverage,
                failure_code="feature.panel_coverage_not_ready",
            )
        source_state_after = self.panel_state.panel_source_state_hash(
            listing_ids=active_listing_ids,
            catalog_hash=self.catalog.binding.catalog_hash,
            as_of_session=request.as_of_session,
            allow_missing_market_observations=allows_missing_source_rows(self.panel.policy_hash),
        )
        if source_state_after != source_state_before:
            return FeatureBuildOutcome(
                FeatureBuildStatus.BLOCKED,
                request.request_hash,
                receipt_hash,
                coverage,
                failure_code="feature.panel_source_state_changed",
            )
        all_availability = self.panel_state.panel_availability_rows(
            cross_sections=cross_sections,
            catalog_hash=self.catalog.binding.catalog_hash,
            policy_hash=panel_binding.policy_hash,
            start=request.history_start,
            end=request.as_of_session,
        )
        clipping_evidence = (
            self._clipping_evidence(
                all_availability,
                clip_records,
                panel_binding_hash=panel_binding.panel_binding_hash,
                sessions=panel_calendar,
            )
            if panel_batches
            else None
        )
        preprocessing_binding = (
            self.panel.preprocessing_binding(panel_binding_hash=panel_binding.panel_binding_hash)
            if clipping_evidence is not None
            else None
        )
        if clipping_evidence is not None and preprocessing_binding is not None:
            # Children before parents, so nothing ever names a document that does
            # not exist yet: the method binding, then the receipt that cites it,
            # and much later the marker that cites both. Built once and reused
            # here rather than reconstructed at the marker, because two
            # constructions are two chances to disagree.
            self.panel_artifacts.resolver.publish_panel_preprocessing_binding(
                payload=preprocessing_binding.model_dump(mode="json"),
                binding_hash=preprocessing_binding.binding_hash,
            )
            self.panel_artifacts.resolver.publish_panel_clipping_evidence(
                payload=clipping_evidence.model_dump(mode="json"),
                evidence_hash=clipping_evidence.evidence_hash,
            )
        prepared_panel = composition.finalize(
            availability=all_availability,
            progress=lambda completed, total, year: self._publish_progress(
                operation_id=request.request_hash,
                stage_id="panel_year_finalize",
                completed=completed,
                total=total,
                unit_name="years",
                current_item=str(year),
                status="SUCCEEDED" if completed == total else "RUNNING",
            ),
        )
        content = prepared_panel.content
        if clipping_evidence is not None and preprocessing_binding is not None:
            # Marker last, and keyed by the *content* that was actually built.
            # Keyed by the panel binding it would collide: one binding can be
            # rebuilt over a different session coverage, producing different
            # evidence, and a second marker under the same key would let write
            # order decide which method a Panel is said to carry. Content
            # identity makes a rebuild of the same data idempotent and a rebuild
            # of different data a different marker.
            preprocessing_marker = build_panel_preprocessing_seal_marker(
                binding=preprocessing_binding,
                evidence=clipping_evidence,
                panel_content_hash=content.panel_content_hash,
            )
            self.panel_artifacts.resolver.publish_panel_preprocessing_marker(
                payload=preprocessing_marker.model_dump(mode="json"),
                panel_content_hash=content.panel_content_hash,
            )
        materialization_receipt_hash = canonical_hash(
            {
                "request_hash": request.request_hash,
                "panel_binding_hash": panel_binding.panel_binding_hash,
                "panel_content_hash": content.panel_content_hash,
                "panel_invalidation_plan_hash": panel_plan.plan_hash,
                "admission_summary_hash": admission.summary_hash(),
                "temporal_risk_hash": risk.risk_hash(),
                "temporal_identity_hash": temporal_boundary.identity_hash(),
                "source_state_hash": source_state_after,
            }
        )
        self.mutation_gate.run(
            self.panel_state.record_panel_materialization,
            receipt_hash=materialization_receipt_hash,
            market_profile_id=self.manifest.profile.market_profile_id,
            manifest_revision=self.manifest.revision_sha256,
            sector_revision=sector_revision,
            catalog_hash=self.catalog.binding.catalog_hash,
            spy_revision=request.spy_revision,
            policy_hash=panel_binding.policy_hash,
            panel_binding_hash=panel_binding.panel_binding_hash,
            content=content,
            admission_summary=asdict(admission),
            temporal_risk=asdict(risk),
            source_state_hash=source_state_after,
            temporal_identity_hash=temporal_boundary.identity_hash(),
            knowledge_cutoff_at=temporal_boundary.knowledge_cutoff_at,
            observed_at=now,
        )
        self.mutation_gate.run(
            self.panel_state.activate_feature_panel,
            market_profile_id=self.manifest.profile.market_profile_id,
            manifest_revision=self.manifest.revision_sha256,
            sector_revision=sector_revision,
            catalog_hash=self.catalog.binding.catalog_hash,
            spy_revision=request.spy_revision,
            policy_hash=panel_binding.policy_hash,
            panel_binding_hash=panel_binding.panel_binding_hash,
            panel_content_hash=content.panel_content_hash,
            history_start=request.history_start,
            as_of_session=request.as_of_session,
            materialization_receipt_hash=materialization_receipt_hash,
            admission_summary_hash=admission.summary_hash(),
            temporal_risk_hash=risk.risk_hash(),
            temporal_identity_hash=temporal_boundary.identity_hash(),
            source_state_hash=source_state_after,
            knowledge_cutoff_at=temporal_boundary.knowledge_cutoff_at,
            observed_at=now,
        )
        coverage["panel_binding_hash"] = panel_binding.panel_binding_hash
        coverage["panel_content_hash"] = content.panel_content_hash
        coverage["panel_materialization_receipt_hash"] = materialization_receipt_hash
        coverage["panel_partitions_reused"] = prepared_panel.reused_chunk_count
        coverage["panel_partitions_written"] = prepared_panel.written_chunk_count
        coverage["panel_partitions_reused_years"] = list(prepared_panel.reused_years)
        return FeatureBuildOutcome(
            FeatureBuildStatus.COMPLETED,
            request.request_hash,
            materialization_receipt_hash,
            coverage,
            # Derived, not asserted. A completed build has materialised the Panel
            # and recorded its binding; the snapshot, its Gateway admission and
            # its semantic index are the publisher's and the Gateway's to
            # deliver, and stamping "terminal" here claimed all three on the
            # strength of this method having returned.
            build_stage=self._durable_build_stage(),
        )

    def _base_session_calendar(
        self,
        *,
        observed_sessions: tuple[date, ...],
        calendar_id: str,
        history_start: date,
        as_of_session: date,
        observed_at: datetime,
        calendars: dict[str, tuple[date, ...]],
    ) -> tuple[date, ...]:
        if not observed_sessions or not allows_missing_source_rows(self.panel.policy_hash):
            return observed_sessions
        held = calendars.get(calendar_id)
        if held is None or observed_sessions[0] < held[0]:
            if self.session_authority_resolver is None:
                raise ValueError("feature.session_authority_unbound")
            authority = self.session_authority_resolver(
                calendar_ids=tuple(calendar_id.split("_")),
                start=min(history_start, observed_sessions[0]),
                end=as_of_session,
                as_of_timestamp=observed_at,
            )
            if set(authority.calendar_ids) != set(calendar_id.split("_")):
                raise ValueError("feature.source_window_calendar_invalid")
            held = authority.sessions
            calendars[calendar_id] = held
        if set(observed_sessions) - set(held):
            raise ValueError("feature.source_session_outside_calendar")
        # Do not fabricate a stock's pre-coverage history or its absent tail.
        # Interior holes still occupy their real positions in every lookback.
        return tuple(day for day in held if observed_sessions[0] <= day <= observed_sessions[-1])

    def _base_source_frame(
        self,
        manifest: UniverseManifest,
        *,
        listing_id: str,
        through: date,
        start: date,
        calendar: Sequence[date],
        _connection,
    ) -> tuple[pd.DataFrame, str, str]:
        source = self._base_source_inputs(
            manifest, listing_id=listing_id, through=through, start=start, _connection=_connection
        )
        frame = feature_source_frame(source, sessions=self._source_sessions(calendar, start))
        return frame, source.raw_hash, source.action_hash

    def _base_source_inputs(
        self,
        manifest: UniverseManifest,
        *,
        listing_id: str,
        through: date,
        start: date,
        _connection,
    ) -> FeatureSourceInputs:
        """A listing's stored inputs from `start`, as this build's catalog and policy read them."""
        return self.feature_state.feature_source_inputs(
            manifest,
            listing_id=listing_id,
            through=through,
            start=start,
            allow_missing_adjusted=allows_missing_source_rows(self.panel.policy_hash),
            as_traded=catalog_reads_as_traded(self.catalog),
            _connection=_connection,
        )

    def _planned_inputs(
        self,
        planned_listings: Sequence[tuple[_ListingWork, _ListingPlan | None]],
        *,
        scope_manifests: Mapping[str, UniverseManifest],
        parts_holding_rows: frozenset[str],
        through: date,
        connection: Any,
    ) -> tuple[dict[str, FeatureSourceInputs], dict[tuple[str, str], list[dict[str, object]]]]:
        """The planned listings' stored inputs, and each held part's rows at its targets, read
        once for these listings, each from its own input start.

        A listing the set read leaves out (no bars, no provider mapping) reads alone in its turn
        and refuses there as before.
        """
        plans = {work.listing.listing_id: plan for work, plan in planned_listings if plan}
        inputs = self.feature_state.feature_source_inputs_by_listing(
            {
                listing: (scope_manifests[listing], plan.input_start)
                for listing, plan in plans.items()
            },
            through=through,
            allow_missing_adjusted=allows_missing_source_rows(self.panel.policy_hash),
            as_traded=catalog_reads_as_traded(self.catalog),
            _connection=connection,
        )
        rows: dict[tuple[str, str], list[dict[str, object]]] = {}
        for part_hash in parts_holding_rows:
            selected = {
                listing: tuple(sorted(targets))
                for listing, plan in plans.items()
                for part, _plan, targets in plan.planned
                if part.binding.catalog_hash == part_hash
            }
            if not selected:
                continue
            held = self._prepared_rows(selected, catalog_hash=part_hash, connection=connection)
            rows.update({(listing, part_hash): value for listing, value in held.items()})
        return inputs, rows

    def _prepared_rows(
        self, selected: Mapping[str, tuple[date, ...]], *, catalog_hash: str, connection: Any
    ) -> dict[str, list[dict[str, object]]]:
        """Each listing's stored rows at its selected sessions, as a held part's reuse reads."""
        prepared = self.feature_state.feature_rows(
            listing_ids=tuple(selected),
            catalog_hash=catalog_hash,
            start=min(sessions[0] for sessions in selected.values()),
            end=max(sessions[-1] for sessions in selected.values()),
            include_values=False,
            include_lineage=False,
            include_verification=True,
            _connection=connection,
        )
        assert isinstance(prepared, list)
        wanted = {listing: set(sessions) for listing, sessions in selected.items()}
        out: dict[str, list[dict[str, object]]] = {listing: [] for listing in selected}
        for row in prepared:
            listing = str(row["listing_id"])
            if row["session_date"] in wanted[listing]:
                out[listing].append(row)
        return out

    def _source_sessions(self, calendar: Sequence[date], start: date) -> tuple[date, ...] | None:
        """The calendar's sessions a window from `start` is aligned to, when the policy admits
        missing source rows; otherwise a frame holds the bars as stored."""
        if not allows_missing_source_rows(self.panel.policy_hash):
            return None
        return tuple(day for day in calendar if day >= start)

    def _verified_source_sessions(
        self,
        *,
        listing_id: str,
        catalog_hash: str,
        prepared: list[dict[str, object]],
        frame: pd.DataFrame,
        market_frame: pd.DataFrame,
        calendar: Sequence[date],
        request: FeatureBuildRequest,
        connection,
        market_input_hashes: dict[date, str],
    ) -> set[date]:
        """Reuse a computed row only under the same verified numerical inputs."""
        identities = tuple(
            sorted(
                {
                    str(row["source_verification_receipt_hash"])
                    for row in prepared
                    if row.get("source_verification_receipt_hash") is not None
                }
            )
        )
        windows = self.feature_state.materialization_source_windows(
            listing_id=listing_id,
            catalog_hash=catalog_hash,
            receipt_hashes=identities,
            _connection=connection,
        )
        verified: set[str] = set()
        for identity, (window, _start, _end) in windows.items():
            if self._source_window_verifies(
                window,
                frame=frame,
                market_frame=market_frame,
                calendar=calendar,
                as_of_session=request.as_of_session,
                market_input_hashes=market_input_hashes,
            ):
                verified.add(identity)
        result: set[date] = set()
        for row in prepared:
            identity = row.get("source_verification_receipt_hash")
            if identity is None or str(identity) not in verified:
                continue
            day = row["session_date"]
            if (
                not isinstance(day, date)
                or not windows[str(identity)][1] <= day <= windows[str(identity)][2]
            ):
                raise ValueError("feature.source_verification_row_outside_receipt")
            result.add(day)
        return result

    def _source_window_verifies(
        self,
        window: FeatureSourceWindow,
        *,
        frame: pd.DataFrame,
        market_frame: pd.DataFrame,
        calendar: Sequence[date],
        as_of_session: date,
        market_input_hashes: dict[date, str],
    ) -> bool:
        """Whether one materialization's recorded inputs are today's inputs.

        The window's stock inputs over ``[input_start, input_end]`` and the
        market inputs through ``input_end`` are hashed from the current
        frames and compared with what the materialization recorded; the
        window must sit on this listing's calendar as the bounded warm-up
        rule places it, and consume nothing past ``as_of_session``. This is
        the one rule for reusing a computed row (the build) and for judging a
        membership from one (the qualification).
        """
        if window.first_output_session not in calendar or window.input_end > as_of_session:
            return False
        if (
            self._bounded_input_start(
                calendar=tuple(calendar),
                first_target=window.first_output_session,
                warmup_sessions=self._bounded_warmup_sessions,
            )
            != window.input_start
        ):
            return False
        if frame.empty or min(frame["session_date"]) > window.input_start:
            # Do not expand a cheap bounded correction into a historical
            # read just to avoid computing it. Membership backfills already
            # hold the full input prefix and can prove reuse here.
            return False
        bounded = frame.loc[
            (frame.session_date >= window.input_start) & (frame.session_date <= window.input_end)
        ]
        market = market_frame.loc[market_frame.session_date <= window.input_end]
        if window.input_end not in market_input_hashes and not market.empty:
            market_input_hashes[window.input_end] = feature_source_values_hash(
                market, ("session_date", "provider_adjusted_close")
            )
        return (
            not bounded.empty
            and not market.empty
            and feature_source_values_hash(bounded, _REQUIRED_COLUMNS) == window.stock_input_hash
            and market_input_hashes.get(window.input_end) == window.market_input_hash
        )

    def qualify_features(
        self,
        *,
        listing_ids: Sequence[str],
        factor_ids: Sequence[str],
        as_of_session: date,
        require_current_source: bool = False,
        observed_at: datetime | None = None,
    ) -> MaterializedFeatureQualification:
        """Read dated computability from existing materialization, without a second calculation.

        With ``require_current_source`` an exclusion is answered only from a
        row whose materialization still applies to the current inputs: the
        row's source-verification receipt verifies against today's stock and
        market inputs (``_source_window_verifies``, the build's own reuse
        rule) and every unavailability reason at the session was recorded by
        that same materialization. A listing whose exclusion cannot be
        proved that way is reported pending -- not excluded, not admitted --
        so the caller leaves the judgement to the build that recomputes it.
        Admitted listings need no proof here: the build re-verifies every
        row before reusing it and the judgement after the build sees the
        recomputed ones.
        """
        catalog_hash = self.catalog.binding.catalog_hash
        rows = self.feature_state.feature_rows(
            listing_ids=listing_ids,
            factor_ids=factor_ids,
            catalog_hash=catalog_hash,
            start=as_of_session,
            end=as_of_session,
            include_verification=require_current_source,
        )
        assert isinstance(rows, list)
        runs = self.ineligibility.find_runs(
            listing_ids=listing_ids,
            factor_ids=factor_ids,
            start=as_of_session,
            end=as_of_session,
        )

        def judged(admitted_rows: list[dict[str, object]]) -> MaterializedFeatureQualification:
            return qualify_materialized_features(
                session=as_of_session,
                catalog_hash=catalog_hash,
                factor_ids=factor_ids,
                listing_ids=listing_ids,
                rows=admitted_rows,
                ineligibility=runs,
                factor_catalogs=self.layer.factor_parts() if self.layer.layered else None,
            )

        qualification = judged(rows)
        if not require_current_source or not qualification.exclusions:
            return qualification
        rows_by_listing = {str(row["listing_id"]): row for row in rows}
        market_frame: pd.DataFrame | None = None
        unproven: list[str] = []
        for listing_id, reasons in qualification.exclusions:
            if market_frame is None:
                spy = MarketReference.spy(self.manifest)
                payload, _market_raw, _market_actions = self.feature_state.projected_feature_frame(
                    spy.manifest, listing_id=spy.listing_id, through=as_of_session
                )
                market_frame = pd.DataFrame(payload)
            if not self._exclusion_source_current(
                listing_id=listing_id,
                row=rows_by_listing[listing_id],
                missing_factor_ids=tuple(factor for factor, _reason in reasons),
                runs=runs,
                as_of_session=as_of_session,
                observed_at=observed_at,
                market_frame=market_frame,
            ):
                unproven.append(listing_id)
        if not unproven:
            return qualification
        return judged([row for row in rows if str(row["listing_id"]) not in unproven])

    def _exclusion_source_current(
        self,
        *,
        listing_id: str,
        row: Mapping[str, object],
        missing_factor_ids: Sequence[str],
        runs: Sequence[FeatureIneligibilityRun],
        as_of_session: date,
        observed_at: datetime | None,
        market_frame: pd.DataFrame,
    ) -> bool:
        """Whether a listing's unavailability at ``as_of_session`` is today's.

        One bounded read of this listing (its window of source rows), only
        for a listing about to be excluded; the market frame is the caller's,
        read once per judgement. A layered catalog's row was verified part by
        part, so each part holding a missing factor proves its own (V92).
        """
        if not self.layer.layered:
            return self._exclusion_proved(
                listing_id=listing_id,
                catalog_hash=self.catalog.binding.catalog_hash,
                row=row,
                missing_factor_ids=missing_factor_ids,
                runs=runs,
                as_of_session=as_of_session,
                observed_at=observed_at,
                market_frame=market_frame,
            )
        owners = self.layer.factor_parts()
        for part_hash in sorted({owners[factor_id] for factor_id in missing_factor_ids}):
            part_rows = self.feature_state.feature_rows(
                listing_ids=(listing_id,),
                catalog_hash=part_hash,
                start=as_of_session,
                end=as_of_session,
                include_values=False,
                include_lineage=False,
                include_verification=True,
            )
            assert isinstance(part_rows, list)
            if len(part_rows) != 1 or not self._exclusion_proved(
                listing_id=listing_id,
                catalog_hash=part_hash,
                row=part_rows[0],
                missing_factor_ids=tuple(
                    factor_id for factor_id in missing_factor_ids if owners[factor_id] == part_hash
                ),
                runs=runs,
                as_of_session=as_of_session,
                observed_at=observed_at,
                market_frame=market_frame,
            ):
                return False
        return True

    def _exclusion_proved(
        self,
        *,
        listing_id: str,
        catalog_hash: str,
        row: Mapping[str, object],
        missing_factor_ids: Sequence[str],
        runs: Sequence[FeatureIneligibilityRun],
        as_of_session: date,
        observed_at: datetime | None,
        market_frame: pd.DataFrame,
    ) -> bool:
        """Whether the materialization that computed one catalog's row still applies."""
        receipt = row.get("source_verification_receipt_hash")
        if receipt is None:
            return False
        receipt_hash = str(receipt)
        if any(
            run.materialization_receipt_hash != receipt_hash
            for run in runs
            if run.listing_id == listing_id
            and run.factor_id in missing_factor_ids
            and run.first_session <= as_of_session <= run.last_session
        ):
            return False
        try:
            windows = self.feature_state.materialization_source_windows(
                listing_id=listing_id,
                catalog_hash=catalog_hash,
                receipt_hashes=(receipt_hash,),
            )
        except ValueError as error:
            if str(error).startswith("feature.source_verification_receipt"):
                return False
            raise
        window, start, end = windows[receipt_hash]
        if not start <= as_of_session <= end:
            return False
        observed = self.market_data.raw_bar_sessions(listing_id, through=as_of_session)
        if not observed:
            return False
        if observed_at is None and allows_missing_source_rows(self.panel.policy_hash):
            # A partial-source calendar is resolved at an instant; without one
            # the judgement waits for the build.
            return False
        calendar = self._base_session_calendar(
            observed_sessions=observed,
            calendar_id=self.manifest.profile.calendar_id,
            history_start=min(window.input_start, observed[0]),
            as_of_session=as_of_session,
            observed_at=observed_at if observed_at is not None else datetime.now(UTC),
            calendars={},
        )
        frame, _raw_hash, _action_hash = self._base_source_frame(
            self.manifest,
            listing_id=listing_id,
            through=as_of_session,
            start=window.input_start,
            calendar=calendar,
            _connection=None,
        )
        return self._source_window_verifies(
            window,
            frame=frame,
            market_frame=market_frame,
            calendar=calendar,
            as_of_session=as_of_session,
            market_input_hashes={},
        )

    def _forced_panel_sessions(
        self,
        composition: PanelCompositionSession,
        *,
        panel_binding: FeaturePanelBinding,
        base_manifest: dict[str, object] | None,
        calendar: Sequence[date],
        request: FeatureBuildRequest,
    ) -> tuple[date, ...]:
        """Sessions whose Panel rows must be recomputed although no delta reaches them.

        Reuse is decided per calendar year from durable facts only. The base
        must hold the year under the current calendar (the composition's own
        rule), and every cell the year already holds must be attributable:
        its availability row names a binding some published snapshot records
        as an origin (or this build's) and a receipt whose clipping record
        exists, and the year holds every cell. Otherwise the year is
        recomputed whole, which restamps its cells with this build. A build
        before partition reuse left no records, so the first build over such
        a base recomputes once and every later build reuses.
        """
        if not composition.base_compatible:
            return ()
        # Under a column extension (V92) the added columns are computed on every
        # session; the cells a year must already hold are the base's columns'.
        held_factors = frozenset(
            composition.extension.factor_ids
            if composition.base_extended and composition.extension is not None
            else self.catalog.factor_ids
        )
        forced_years = set(composition.unreusable_years())
        # Sessions the composition computes regardless (a whole unreusable
        # year, or the sessions beyond a prefix-held year's partition) hold no
        # cells yet, or cells about to be replaced; they are not judged here.
        pending = set(composition.unreusable_sessions())
        known_bindings = {panel_binding.panel_binding_hash}
        if base_manifest is not None:
            known_bindings.update(manifest_partition_origins(base_manifest))
        rows = self.panel_state.panel_availability_rows(
            cross_sections=composition.all_cross_sections(),
            catalog_hash=self.catalog.binding.catalog_hash,
            policy_hash=panel_binding.policy_hash,
            start=request.history_start,
            end=request.as_of_session,
        )
        cells_by_year: dict[int, int] = {}
        receipts_by_year: dict[int, set[str]] = {}
        for item in rows:
            session = date.fromisoformat(str(item["session_date"]))
            if session in pending or str(item["factor_id"]) not in held_factors:
                continue
            year = session.year
            cells_by_year[year] = cells_by_year.get(year, 0) + 1
            binding_hash = item.get("panel_binding_hash")
            receipt = item.get("materialization_receipt_hash")
            if not binding_hash or not receipt or str(binding_hash) not in known_bindings:
                forced_years.add(year)
                continue
            receipts_by_year.setdefault(year, set()).add(str(receipt))
        sessions_by_year: dict[int, int] = {}
        for session in calendar:
            if session <= request.as_of_session and session not in pending:
                sessions_by_year[session.year] = sessions_by_year.get(session.year, 0) + 1
        resolver = self.panel_artifacts.resolver
        recorded: dict[str, bool] = {}
        for year, session_count in sessions_by_year.items():
            if year in forced_years:
                continue
            if cells_by_year.get(year, 0) != session_count * len(held_factors):
                forced_years.add(year)
                continue
            for receipt in sorted(receipts_by_year.get(year, ())):
                if receipt not in recorded:
                    recorded[receipt] = resolver.load_panel_clip_observation(receipt) is not None
                if not recorded[receipt]:
                    forced_years.add(year)
                    break
        return tuple(
            session for session in calendar if session.year in forced_years or session in pending
        )

    def _clipping_evidence(
        self,
        availability: Sequence[dict[str, object]],
        records: dict[str, PanelClipObservationRecord],
        *,
        panel_binding_hash: str,
        sessions: Sequence[date],
    ) -> PanelClippingEvidence:
        """Fold the Panel's evidence from every batch its cells came from.

        The availability rows say which receipt owns each (session, factor)
        cell; this build's records are in hand and every other receipt's was
        persisted by the build that produced it (a year whose receipts lack
        one was forced into this build's plan). The fold owner joins each
        cell of the Panel's calendar to the record that observed it.
        """
        owned: dict[tuple[str, str], set[str]] = {}
        for item in availability:
            receipt = item.get("materialization_receipt_hash")
            if not receipt:
                raise ValueError("feature.panel_clip_observation_missing")
            owned.setdefault((str(receipt), str(item["factor_id"])), set()).add(
                date.fromisoformat(str(item["session_date"])).isoformat()
            )
        resolver = self.panel_artifacts.resolver
        for receipt, _factor_id in owned:
            if receipt in records:
                continue
            payload = resolver.load_panel_clip_observation(receipt)
            if payload is None:
                raise ValueError("feature.panel_clip_observation_missing")
            records[receipt] = PanelClipObservationRecord.model_validate(payload)
        return self.panel.clipping_evidence_from_records(
            tuple(records.values()),
            panel_binding_hash=panel_binding_hash,
            owned={key: frozenset(value) for key, value in owned.items()},
            sessions=tuple(session.isoformat() for session in sessions),
        )

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
        status: str = "RUNNING",
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

    def refresh_spy(self, *, as_of_session: date, observed_at: datetime) -> FeatureBuildOutcome:
        """Refresh the bounded SPY market reference and report its revision."""
        result = MarketReferenceMaintainer(
            market_data=self.market_data,
            feature_state=self.feature_state,
            parent_manifest=self.manifest,
            provider=self.provider,
        ).refresh(as_of_session=as_of_session, observed_at=observed_at)
        return FeatureBuildOutcome(
            FeatureBuildStatus(result.status),
            canonical_hash(["SPY", as_of_session.isoformat()]),
            result.revision_hash,
            {"reference": "SPY"},
            retry_after_at=result.retry_after_at,
            failure_code=result.failure_code,
        )

    def _durable_build_stage(self) -> FeatureBuildStage:
        """Report which stage the durable state has actually reached.

        Recovery depends on this and nothing else: a pending closure transition
        must be reconciled, an incomplete closure has listings left to compute,
        and a complete closure with no published Panel needs composition alone --
        never a recomputation of every listing, and never a scratch reset.

        Two things this deliberately does not do. It does not read closure state
        from the exception ``assert_ready`` happened to raise: the closure owner
        reports its own disposition, so damaged authority is not announced as a
        reconcilable transition. And it does not accept a positive row count as a
        finished closure -- one materialised listing out of the manifest's is a
        positive count and nothing else.

        Terminal means the whole terminal contract, because that is the only
        claim worth making: an active snapshot this catalog produced, inputs the
        Feature Input Gateway admitted, and a semantic index research can
        actually resolve to that snapshot. A Panel missing the last of those is
        published and unusable, which is a real state with a cheap repair, and
        naming it terminal would hide exactly that repair.
        """
        catalog_hash = self.catalog.binding.catalog_hash
        try:
            axis_listings, _scope = self._axis_listings()
        except ValueError:
            return FeatureBuildStage.CLOSURE_AUTHORITY_UNAVAILABLE
        disposition = self.feature_persistence.closure_disposition(
            catalog_hash,
            expected_listing_ids=tuple(item.listing_id for item in axis_listings),
        )
        if disposition == "TRANSITION_RECOVERY_PENDING":
            return FeatureBuildStage.TRANSITION_RECOVERY_PENDING
        if disposition == "CLOSURE_AUTHORITY_UNAVAILABLE":
            return FeatureBuildStage.CLOSURE_AUTHORITY_UNAVAILABLE
        if disposition != "MEMBERSHIP_COMPLETE":
            return FeatureBuildStage.BASE_CLOSURE_INCOMPLETE
        snapshot = self.panel_state.feature_panel_snapshot_for_active(
            self.manifest.profile.market_profile_id
        )
        if snapshot is None or str(snapshot.get("catalog_hash", "")) != catalog_hash:
            return FeatureBuildStage.BASE_CLOSURE_COMPLETE_PANEL_PENDING
        disclosure = self.panel_state.feature_input_quality_disclosure(
            result_manifest_revision=self.manifest.revision_sha256
        )
        if not bool(disclosure.get("gateway_qualified")):
            return FeatureBuildStage.PANEL_PUBLISHED_AWAITING_GATEWAY_ADMISSION
        try:
            resolved = self.panel_artifacts.resolver.find_feature_panel_semantic_index(
                panel_snapshot_hash=str(snapshot["snapshot_hash"])
            )
        except (OSError, ValueError):
            # An index that will not resolve is not one research can read, so the
            # Panel is not terminal either way. The handoff owner refuses loudly
            # if what is there cannot be repaired by completing it.
            return FeatureBuildStage.PANEL_SEMANTIC_INDEX_PENDING
        if resolved is None:
            return FeatureBuildStage.PANEL_SEMANTIC_INDEX_PENDING
        return FeatureBuildStage.TERMINAL_PANEL_COMPLETE

    def _listings_for_invalidations(
        self,
        invalidations: tuple[FeatureInvalidation, ...],
        axis_listings: tuple[ManifestListing, ...],
    ) -> tuple[ManifestListing, ...]:
        """Which listings the base stage computes for these invalidations.

        Whole-history work (a backfill, a catalog or SPY change) covers the
        calculation axis: every listing some session of the history holds,
        so a member that has since left keeps its base rows under the new
        catalog for the sessions it was a member of. New sessions reach current
        members and explicitly changed historical names: a future-effective
        EXIT must not discard the source tail still owed before its cutoff.
        """
        if invalidations and all(
            item.kind
            in {
                "action_correction",
                "action_evidence_change",
                "manifest_removal",
                "panel_binding_change",
                "sector_revision_change",
            }
            for item in invalidations
        ):
            return ()
        if not invalidations or any(
            item.kind
            in {
                "initial_backfill",
                "spy_correction",
                "catalog_binding_change",
            }
            for item in invalidations
        ):
            return axis_listings
        requested = {item.listing_id for item in invalidations if item.listing_id}
        if any(item.kind == "normal_new_session" for item in invalidations):
            requested.update(item.listing_id for item in self.manifest.listings)
        target = tuple(item for item in axis_listings if item.listing_id in requested)
        if {item.listing_id for item in target} != requested:
            raise ValueError("feature invalidation listing is outside the calculation axis")
        return target

    def _axis_listings(
        self,
    ) -> tuple[tuple[ManifestListing, ...], dict[str, UniverseManifest]]:
        """Every listing the Panel's history holds, with the manifest that admits each.

        The manifest's current members, plus every listing the Universe
        journal records as a member of some earlier session. The latter are
        read from the manifest revision that admitted them -- the bootstrap
        cohort's or their entry's -- so a listing that has left keeps its
        symbol, provider mapping and data scope without a synthetic manifest.
        The axis is in calculation order: by symbol, as the manifest lists
        its members, so a listing joining or leaving never reorders the
        others' floating-point sums.
        """
        listings = {item.listing_id: item for item in self.manifest.listings}
        scope = {item.listing_id: self.manifest for item in self.manifest.listings}
        manifests = {self.manifest.revision_sha256: self.manifest}
        for listing_id, revision in self.market_data.research_listing_sources(
            self.manifest
        ).items():
            if listing_id in listings:
                continue
            manifest = manifests.get(revision)
            if manifest is None:
                manifest = self.market_data.load_universe_manifest_revision(revision)
                manifests[revision] = manifest
            item = next(
                (entry for entry in manifest.listings if entry.listing_id == listing_id), None
            )
            if item is None:
                raise ValueError(
                    f"membership journal listing {listing_id} is absent from the "
                    "manifest revision that admitted it"
                )
            listings[listing_id] = item
            scope[listing_id] = manifest
        ordered = tuple(sorted(listings.values(), key=lambda item: (item.symbol, item.listing_id)))
        return ordered, scope

    def _axis_sectors(
        self, sector: SectorReferenceState, axis_listings: tuple[ManifestListing, ...]
    ) -> dict[str, str]:
        """Return each axis listing's sector from manifest state or storage."""
        sectors = dict(sector.sector_by_listing_id)
        missing = [item.listing_id for item in axis_listings if item.listing_id not in sectors]
        if missing:
            sectors.update(self.feature_state.sector_classifications(missing))
        unresolved = [item.listing_id for item in axis_listings if item.listing_id not in sectors]
        if unresolved:
            raise ValueError(
                "calculation axis listings have no sector classification: "
                + ", ".join(unresolved[:10])
            )
        return {item.listing_id: sectors[item.listing_id] for item in axis_listings}

    def _panel_membership(
        self, *, calendar: Sequence[date], axis_listing_ids: tuple[str, ...]
    ) -> PanelMembership:
        """Return each calendar session's members over the calculation axis.

        Resolved from the Universe journal by its owner; before a bootstrap
        is recorded every session holds the manifest under construction, as
        the disclosed initial-cohort backfill. The manifest must match the
        journal's admitted roster; some decisions may take effect after this
        Panel's as-of session. They can prepare base history without entering
        this Panel's historical calculation axis or changing its members.
        """
        schedule = self.market_data.membership_schedule(
            self.manifest.profile.market_profile_id,
            sessions=calendar,
            fallback_listing_ids=tuple(item.listing_id for item in self.manifest.listings),
        )
        latest = set(schedule.admitted_listing_ids)
        current = {item.listing_id for item in self.manifest.listings}
        if current - latest:
            raise ValueError("feature.membership_admission_required")
        if latest != current:
            raise ValueError(
                "the Universe journal's admitted members differ from the manifest: "
                f"{len(latest - current)} not in the manifest, {len(current - latest)} not in "
                "the journal"
            )
        in_history = set(schedule.union)
        axis_listing_ids = tuple(value for value in axis_listing_ids if value in in_history)
        positions = {listing_id: index for index, listing_id in enumerate(axis_listing_ids)}
        epochs = []
        for epoch in schedule.epochs:
            try:
                members = tuple(sorted(epoch.listing_ids, key=positions.__getitem__))
            except KeyError as exc:
                raise ValueError(
                    f"journal member {exc.args[0]} is outside the calculation axis"
                ) from exc
            epochs.append(PanelMembershipEpoch(epoch.first_session, epoch.last_session, members))
        return PanelMembership(
            listing_ids=axis_listing_ids,
            epochs=tuple(epochs),
            basis_ranges=tuple(
                PanelMembershipBasisRange(item.first_session, item.last_session, item.basis)
                for item in schedule.basis_ranges
            ),
            journal_sequence=schedule.journal_sequence,
            bootstrap_record_hash=(
                schedule.bootstrap.record_hash if schedule.bootstrap is not None else None
            ),
            bootstrap_t0_session=(
                schedule.bootstrap.t0_session if schedule.bootstrap is not None else None
            ),
            source_exclusions=self.panel_state.panel_source_exclusions(
                market_profile_id=self.manifest.profile.market_profile_id,
                sessions=calendar,
                listing_ids=axis_listing_ids,
            ),
        )

    def _held_panel_columns(self) -> FeatureCatalog | None:
        """The catalog the active Panel was built under, when this one only adds columns to it.

        It is this catalog restricted to the Panel's factors, and its hash must be the one the
        Panel's lineage records: then every value the Panel holds is the one this catalog
        computes for that column, and its partitions serve as the base of this catalog's
        composition, merged with the added columns (V92).
        """
        snapshot = self.panel_state.feature_panel_snapshot_for_active(
            self.manifest.profile.market_profile_id
        )
        if snapshot is None or str(snapshot.get("catalog_hash")) == (
            self.catalog.binding.catalog_hash
        ):
            return None
        manifest = self.panel_artifacts.resolver.load_feature_panel_manifest(
            str(snapshot["manifest_uri"])
        )
        summary = manifest.get("safe_summary")
        factors = summary.get("factor_catalog_summary") if isinstance(summary, Mapping) else None
        if not isinstance(factors, Mapping):
            return None
        held = {str(value) for value in factors}
        if not held or not held < set(self.catalog.factor_ids):
            return None
        catalog = restricted_catalog(self.catalog, held)
        if catalog.binding.catalog_hash != str(snapshot.get("catalog_hash")):
            return None
        return catalog

    @staticmethod
    def _preserves_nothing(
        target_factors: Mapping[date, Sequence[str]], catalog: FeatureCatalog
    ) -> bool:
        """Whether every targeted session recomputes the whole catalog, keeping no stored value."""
        catalog_scope = set(catalog.factor_ids)
        return all(set(factors) == catalog_scope for factors in target_factors.values())

    def _part_invalidations(
        self,
        invalidations: tuple[FeatureInvalidation, ...],
        axis_listings: tuple[ManifestListing, ...],
        request: FeatureBuildRequest,
    ) -> dict[str, tuple[FeatureInvalidation, ...]]:
        """What each part of the layer recomputes (V92).

        An unlayered catalog is its own only part and plans from the build's invalidations as
        they are. A layered catalog's part plans from its own closure: a part complete for the
        axis keeps its rows through a catalog change, which then reaches only the Panel; any
        other part recomputes its whole history. Each invalidation reaches a part's own factors
        only, and a part none reaches computes nothing.
        """
        if not self.layer.layered:
            return {self.catalog.binding.catalog_hash: invalidations}
        expected = tuple(item.listing_id for item in axis_listings)
        planned: dict[str, tuple[FeatureInvalidation, ...]] = {}
        for part in self.layer.parts:
            part_hash = part.binding.catalog_hash
            factors = set(part.factor_ids)
            complete = (
                self.feature_persistence.closure_disposition(
                    part_hash, expected_listing_ids=expected
                )
                == "MEMBERSHIP_COMPLETE"
            )
            scoped: list[FeatureInvalidation] = []
            for item in invalidations:
                if item.kind == "catalog_binding_change" and complete:
                    continue
                named = item.factor_ids or (
                    tuple(sorted(MARKET_DEPENDENT_FACTOR_IDS))
                    if item.kind == "spy_correction"
                    else ()
                )
                if named:
                    kept = tuple(factor_id for factor_id in named if factor_id in factors)
                    if not kept:
                        continue
                    item = replace(item, factor_ids=kept)
                scoped.append(item)
            if not complete and not any(
                item.kind in {"catalog_binding_change", "initial_backfill"} for item in scoped
            ):
                scoped.append(
                    FeatureInvalidation(
                        "catalog_binding_change", earliest_session=request.history_start
                    )
                )
            if scoped or not invalidations:
                planned[part_hash] = tuple(scoped)
        return planned

    def _preserve_unaffected_feature_values(
        self,
        rows: pd.DataFrame,
        target_factors: dict[date, tuple[str, ...]],
        *,
        catalog: FeatureCatalog,
        _connection=None,
    ) -> pd.DataFrame:
        if rows.empty or self._preserves_nothing(target_factors, catalog):
            return rows
        listing_id = str(rows.iloc[0]["listing_id"])
        start = min(rows["session_date"])
        end = max(rows["session_date"])
        existing_by_session = {
            str(item["session_date"]): item
            for item in self.feature_state.feature_rows(
                listing_ids=(listing_id,),
                catalog_hash=catalog.binding.catalog_hash,
                start=start,
                end=end,
                _connection=_connection,
            )
        }
        for row_index in rows.index:
            session = rows.at[row_index, "session_date"]
            existing = existing_by_session.get(str(session))
            if existing is None:
                continue
            selected = set(target_factors[session])
            for factor_id in catalog.factor_ids:
                if factor_id not in selected:
                    rows.at[row_index, factor_id] = existing[factor_id]
            current_cutoffs = json.loads(str(rows.at[row_index, "input_cutoffs_json"]))
            prior_cutoffs = json.loads(str(existing["input_cutoffs_json"]))
            for factor_id in catalog.factor_ids:
                if factor_id not in selected:
                    current_cutoffs[factor_id] = prior_cutoffs[factor_id]
            rows.at[row_index, "input_cutoffs_json"] = json.dumps(
                current_cutoffs, sort_keys=True, separators=(",", ":")
            )
        return rows

    @staticmethod
    def _bounded_input_start(
        *,
        calendar: tuple[date, ...],
        first_target: date,
        warmup_sessions: int,
    ) -> date:
        """Choose a finite prefix that also retains the M-13 month anchor."""
        if warmup_sessions < 1:
            raise ValueError("bounded feature warm-up must be positive")
        try:
            first_target_position = calendar.index(first_target)
        except ValueError as exc:
            raise ValueError("feature target is outside the listing calendar") from exc
        bounded_position = max(0, first_target_position - warmup_sessions)
        anchor_month = first_target.year * 12 + first_target.month - 13
        calendar_anchor_position = next(
            index
            for index, session in enumerate(calendar)
            if session.year * 12 + session.month >= anchor_month
        )
        return calendar[min(bounded_position, calendar_anchor_position)]

    @staticmethod
    def _contiguous_target_ranges(
        calendar: tuple[date, ...], selected: set[date]
    ) -> tuple[tuple[date, date], ...]:
        positions = {session: index for index, session in enumerate(calendar)}
        ordered = [session for session in calendar if session in selected]
        if not ordered:
            return ()
        ranges: list[tuple[date, date]] = []
        first = previous = ordered[0]
        for session in ordered[1:]:
            if positions[session] != positions[previous] + 1:
                ranges.append((first, previous))
                first = session
            previous = session
        ranges.append((first, previous))
        return tuple(ranges)

    @staticmethod
    def _panel_work_groups(
        calendar: Sequence[date], targets: dict[date, tuple[str, ...]]
    ) -> tuple[tuple[tuple[str, ...], tuple[date, ...]], ...]:
        """Group adjacent sessions with the same sparse factor scope."""
        groups: list[tuple[tuple[str, ...], list[date]]] = []
        positions = {session: index for index, session in enumerate(calendar)}
        previous: date | None = None
        for session in calendar:
            factors = targets.get(session)
            if not factors:
                continue
            if (
                groups
                and groups[-1][0] == factors
                and previous is not None
                and positions[session] == positions[previous] + 1
            ):
                groups[-1][1].append(session)
            else:
                groups.append((factors, [session]))
            previous = session
        return tuple((factors, tuple(sessions)) for factors, sessions in groups)
