"""Read admitted workspace inputs for component inference, never future labels.

Data owns bar/action status and returns, Feature owns observation values,
Sector owns its causal states and Alpha owns Context assembly. This composition
only resolves their axes and delegates; it publishes no Foundation or book.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import date, datetime
from itertools import groupby
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from alphalattice.control.data_platform.data_decision_rule import DataDecisionRule
from alphalattice.control.data_platform.maintenance.contracts import (
    WorkspaceDataUpdateBinding,
    WorkspaceInputStatus,
)
from alphalattice.control.product_host.composition.research_workspace import (
    RESEARCH_WORKSPACE_MANIFEST_NAME,
    read_research_workspace_manifest,
)
from alphalattice.control.product_host.maintenance.data_update import (
    installed_data_update_binding,
    read_workspace_inputs,
)
from alphalattice.control.workspace_runtime.verified_facts import file_sha256
from alphalattice.foundation.causal_outcomes.execution.compile import (
    _action_dividends,
    causal_execution_simple_return,
)
from alphalattice.foundation.causal_outcomes.execution.contracts import CausalExecutionSchedulePoint
from alphalattice.foundation.causal_outcomes.execution.methods import (
    ONE_SESSION_RECIPE_ID,
    ExecutionOutcomeSessionAxis,
    build_installed_execution_outcome_method_catalog,
    period_dividend_for_point,
    resolve_schedule_points,
)
from alphalattice.foundation.feature_engine.catalog.contracts import desktop_core_feature_bundle
from alphalattice.foundation.feature_engine.producers.factors.catalog import (
    default_extension_kernel_registry,
)
from alphalattice.foundation.feature_engine.producers.factors.interactions import (
    interaction_market_state_children,
)
from alphalattice.foundation.feature_engine.producers.factors.session_liquidity import (
    session_liquidity_factor_specs,
)
from alphalattice.foundation.feature_engine.producers.factors.session_observation import (
    append_session_observation_values,
    session_observation_factor_specs,
)
from alphalattice.foundation.feature_engine.storage.listing_proofs import (
    feature_source_listing_proofs,
)
from alphalattice.foundation.feature_engine.storage.repositories import (
    FeatureStateRepository,
    PanelStateRepository,
)
from alphalattice.foundation.market_data_ops.sources.contracts import (
    CorporateActionEvent,
    RawDailyBar,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.foundation.market_data_ops.storage.rebase import recorded as recorded_rebases
from alphalattice.foundation.market_data_ops.storage.source_prefix import (
    market_data_source_prefix_proofs,
)
from alphalattice.investment.alpha_research.inputs.frozen_price_volume import (
    BoolArray,
    FloatArray,
    FrozenPriceVolumeInputs,
)
from alphalattice.investment.alpha_research.inputs.panel_feature_views import (
    IntArray,
    PanelFeatureSourceArrays,
    assemble_panel_context_arrays,
)
from alphalattice.investment.alpha_research.publication.contracts import (
    StoredRebase,
    WorkspaceObservationHistoryHead,
)
from alphalattice.investment.alpha_research.publication.workspace_history import (
    RebaseKeep,
    rebase_keep,
    rebased,
)
from alphalattice.investment.alpha_research.targets.component_training import (
    compile_frozen_component_training_targets,
)
from alphalattice.investment.sector_research.inputs.surface import compile_sector_context_arrays
from alphalattice.kernel.data.calendar import materialize_calendar_schedule
from alphalattice.kernel.quant.factor_contracts import FactorSpec
from alphalattice.kernel.quant.sector_history import sector_ids, sector_subset
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.spans import span


@dataclass(frozen=True)
class PreparedWorkspaceComponentInputs:
    """One Task's independently held, verified raw and Feature observation history.

    Array storage is immutable bytes. A formation receives its own arrays and
    its own membership and causal Context calculation; no later maturity is
    visible through this held history.
    """

    workspace: Path
    through: date
    observed_at: datetime
    source_hash: str
    source_proof: tuple[tuple[str, str | None], ...]
    formation_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]
    ohlcv: Mapping[str, FloatArray]
    observations: Mapping[str, FloatArray]
    formula_values: Mapping[str, FloatArray]
    raw_simple: FloatArray
    actions: tuple[tuple[CorporateActionEvent, ...], ...]
    dependency_prefix_hash: str | None = None
    stable_session_count: int = 0
    reused_history_hash: str | None = None
    axes_proof: str | None = None
    listing_digests: tuple[str, ...] = ()
    listing_closed: tuple[str, ...] = ()
    restated_listing_ids: tuple[str, ...] = ()
    rebases: tuple[StoredRebase, ...] = ()
    """Every rebase of the store's bars its kept rows know, with the rows each still needs."""
    kept_rows: RebaseKeep = MappingProxyType({})
    """A rebased listing's rows kept as stored, `rebase_keep`'s, for its publishers."""
    rebase_agreement: tuple[float, float] = (0.0, 0.0)
    """The data rule's agreement those rows were kept within."""


def _market_source_proof(market: MarketDataRepository) -> tuple[tuple[str, str | None], ...]:
    # A revision journal is not a content proof. Include the WAL's presence and
    # exact bytes: a live writable instance can commit without checkpointing.
    # A file is hashed whole unless its verified fact still holds (any write
    # moves its size or mtime).
    proof = []
    for path in (market.path, Path(str(market.path) + ".wal")):
        try:
            digest = file_sha256(path)
        except FileNotFoundError:
            if path == market.path:
                raise
            digest = None
        proof.append((str(path.resolve()), digest))
    return tuple(proof)


def _immutable_arrays(values: Mapping[str, FloatArray]) -> Mapping[str, FloatArray]:
    return MappingProxyType(
        {
            name: np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)
            for name, array in values.items()
        }
    )


class _PrefixProof(NamedTuple):
    """A prefix's proof whole, by its shared axes alone, and each listing's market part, read
    only when asked for (a whole match never asks)."""

    whole: str
    axes: str
    market: Callable[[], Mapping[str, str]]

    @classmethod
    def of(
        cls, axes: str, data: str, features: str | None, market: Callable[[], Mapping[str, str]]
    ) -> _PrefixProof:
        """The proof from its axes, the scope's market and Feature proofs and each listing's."""
        parts = {"kind": "WorkspaceObservationPrefixProof.v3", "axes": axes, "data": data}
        return cls(str(canonical_hash({**parts, "features": features})), axes, market)

    def listings(self, feature: Mapping[str, str]) -> dict[str, str]:
        """Each listing's proof: its market part and its own Feature values."""
        return {
            name: str(canonical_hash((value, feature.get(name))))
            for name, value in self.market().items()
        }


_KERNEL_IDS = sorted(
    spec.factor_id
    for spec in (*session_observation_factor_specs(), *session_liquidity_factor_specs())
)
PER_LISTING_REVIEWED = "27d51f63ffd56539840ed4f2a67620a82a7cd1eccfa2aa2435ffebaf0459a701"
"""The digest of the extension kernel ids reviewed to compute one listing at a time (each gives a
listing subset its full frame's bytes; the test names them)."""
PER_LISTING_EXTENSIONS = (
    frozenset(_KERNEL_IDS) if canonical_hash(_KERNEL_IDS) == PER_LISTING_REVIEWED else frozenset()
)
"""The reviewed per-listing kernels; none while a kernel is added or changed unreviewed. Only
with every kept extension among them may a restated listing rebuild alone; any other kernel, a
cross-sectional one, keeps the whole capture cold."""


def _workspace_history_prefix_proof(
    market: MarketDataRepository,
    feature: FeatureStateRepository,
    *,
    sessions: tuple[date, ...],
    listings: tuple[str, ...],
    observed_at: datetime,
    catalog_hash: str,
    factor_ids: tuple[str, ...],
    extension_factor_ids: tuple[str, ...],
    feature_prefix_proof: str | None = None,
) -> _PrefixProof:
    """Bind actual source values and the dependency axes of a completed prefix, whole, by its
    axes alone, and for each listing (its raw bars, actions and stored Feature values).

    Daily source identities and the observation instant are provenance. The
    old completed clocks and actual membership, exclusions, Sector and source
    values must still agree before an immutable preparation can be reused. The
    market reference's adjusted closes are not among them: no captured value reads
    them (training reads them live), and a provider restating them daily kept every
    real day cold.
    """
    readiness = market.readiness.load("us-current-index-research")
    if readiness is None or readiness.active_manifest_id is None:
        raise ValueError("strategy_score.workspace_not_qualified")
    manifest = market.load_universe_manifest(readiness.active_manifest_id)
    with market.database.read_transaction() as connection:
        data, market_listing = market_data_source_prefix_proofs(
            market,
            listing_ids=listings,
            start=sessions[0],
            through=sessions[-1],
            _connection=connection,
        )
        features = (
            feature.feature_source_prefix_proof(
                listing_ids=listings,
                catalog_hash=catalog_hash,
                start=sessions[0],
                end=sessions[-1],
                factor_ids=factor_ids,
                _connection=connection,
            )
            if feature_prefix_proof is None
            else feature_prefix_proof
        )
    calendar = materialize_calendar_schedule(
        ("XNAS", "XNYS"),
        start=sessions[0],
        end=sessions[-1],
        as_of_timestamp=observed_at,
    )
    schedule = market.membership_schedule(
        manifest.profile.market_profile_id,
        sessions=sessions,
        fallback_listing_ids=tuple(item.listing_id for item in manifest.listings),
    )
    membership = []
    for (members, basis), days in groupby(
        sessions, key=lambda day: (schedule.members(day), schedule.basis(day))
    ):
        run = tuple(days)
        membership.append((run[0], run[-1], members, basis))
    history = feature.sector_history(manifest, listing_ids=listings)
    if history is None or set(listings) - set(history):
        raise ValueError("strategy_score.sector_coverage_incomplete")
    exclusions = PanelStateRepository(market.database, market_data=market).panel_source_exclusions(
        market_profile_id=manifest.profile.market_profile_id,
        sessions=sessions,
        listing_ids=listings,
    )
    registry = default_extension_kernel_registry()
    bundle = desktop_core_feature_bundle()
    axes = str(
        canonical_hash(
            {
                "kind": "WorkspaceObservationPrefixAxes.v1",
                "sessions": sessions,
                "listings": listings,
                "calendar": calendar.to_pylist(),
                "membership": membership,
                "admitted_listings": schedule.admitted_listing_ids,
                "exclusions": tuple(
                    (
                        e.listing_id,
                        max(sessions[0], e.first_session),
                        min(sessions[-1], e.last_session),
                        e.evidence_hash,
                        e.reason_codes,
                    )
                    for e in exclusions
                ),
                "sector_map": dict(history),
                "sector_history": history.lineage_payload(),
                "execution_method": build_installed_execution_outcome_method_catalog()
                .resolve(ONE_SESSION_RECIPE_ID)
                .recipe_hash,
                "extension_rules": tuple(
                    (s.factor_id, registry.implementation_hash(s, core_bundle=bundle))
                    for s in (
                        *session_observation_factor_specs(),
                        *session_liquidity_factor_specs(),
                    )
                    if s.factor_id in extension_factor_ids
                ),
            }
        )
    )
    return _PrefixProof.of(axes, data, features, market_listing)


def workspace_observation_history_columns(
    value: PreparedWorkspaceComponentInputs,
) -> Mapping[str, FloatArray]:
    """Expose the held immutable numerical families to Alpha's preparation store."""
    return MappingProxyType(
        {
            **{"ohlcv::" + k: v for k, v in value.ohlcv.items()},
            **{"observations::" + k: v for k, v in value.observations.items()},
            **{"formula::" + k: v for k, v in value.formula_values.items()},
            "raw_simple": value.raw_simple,
        }
    )


def prepare_workspace_component_inputs(
    workspace: Path,
    *,
    through: date,
    observed_at: datetime,
    expected_source_hash: str,
    ordered_feature_ids: tuple[str, ...] = (),
    training_factor_ids: tuple[str, ...] = (),
    history: tuple[WorkspaceObservationHistoryHead, Mapping[str, FloatArray]] | None = None,
) -> PreparedWorkspaceComponentInputs:
    """Verify and hold one maximum-cutoff history for projections in one Task.

    Capture validates the source, raw and Feature history and Sector mapping.
    Every formation verifies the current database bytes and performs its own
    ordinary Context and optional training-target checks.
    The caller owns the value's lifetime and discards it when its Task stage ends.

    Args:
        workspace: Admitted workspace; its retained writer lease and mutation gate
            must cover this read boundary, as the scoring application arranges.
        through: Maximum completed formation in the Task's pending requests.
        observed_at: Shared explicit calendar observation timestamp.
        expected_source_hash: Required admitted workspace score source.
        ordered_feature_ids: Exact inference Feature selections to prepare.
        training_factor_ids: Optional sorted unique mature training factors.
        history: A verified durable candidate; its current source prefix is
            independently proved here before any of its values are used.

    Returns:
        Independently held immutable raw and Feature history.

    Raises:
        ValueError: Ordinary cutoff admission or exact source proof fails.
    """
    _, _, prepared = _read_workspace_component_inputs(
        workspace,
        formation=through,
        observed_at=observed_at,
        expected_source_hash=expected_source_hash,
        ordered_feature_ids=ordered_feature_ids,
        training_factor_ids=training_factor_ids,
        capture=True,
        history=history,
    )
    assert prepared is not None
    with MarketDataRepository(workspace).database.read_transaction() as connection:
        known = recorded_rebases(connection, prepared.ordered_listing_ids)
    if not known and (history is None or not history[0].rebases):
        return prepared
    held = (workspace / RESEARCH_WORKSPACE_MANIFEST_NAME).exists()  # a data-only one has none
    rule = (read_research_workspace_manifest(workspace).data_decision_rule if held else None) or (
        DataDecisionRule()
    )
    columns = workspace_observation_history_columns(prepared)
    agreement = rule.rebase_agreement
    listings = prepared.ordered_listing_ids
    entries, keep = rebase_keep(history, listings, columns, known, agreement, rule.rule_hash)
    return replace(prepared, rebases=entries, kept_rows=keep, rebase_agreement=agreement)


def workspace_score_source_identity(workspace: Path) -> str:
    """Read the current qualified workspace score-source identity.

    Args:
        workspace: Caller-owned admitted workspace root.

    Returns:
        Deterministic identity of installed score inputs.
    """
    status = read_workspace_inputs(workspace, installed_data_update_binding(workspace))
    return score_source_identity(status)


def score_source_identity(status: WorkspaceInputStatus) -> str:
    """Identity of a verified input-status value, including captured Task readback."""
    if status.readiness_status != "RESEARCH_READY":
        raise ValueError("strategy_score.workspace_inputs_not_ready")
    return str(
        canonical_hash(
            {
                "manifest": status.manifest_revision,
                "data": status.data_revision_hash,
                "panel": status.panel_hash,
            }
        )
    )


def build_workspace_score_inputs(
    workspace: Path,
    *,
    formation: date,
    observed_at: datetime,
    expected_source_hash: str,
    ordered_feature_ids: tuple[str, ...] = (),
    prepared: PreparedWorkspaceComponentInputs | None = None,
) -> FrozenPriceVolumeInputs:
    """Build exact qualified price-volume inference inputs for one completed formation.

    Args:
        workspace: Caller-owned admitted workspace root.
        formation: Explicit completed formation session.
        observed_at: Explicit calendar observation timestamp.
        expected_source_hash: Required exact workspace score source.
        ordered_feature_ids: Explicit inference feature selections.
        prepared: Optional Task-local maximum-cutoff history to project.

    Returns:
        Frozen source arrays assembled by the existing Data and Feature owners.
    """
    source, _ = read_workspace_component_inputs(
        workspace,
        formation=formation,
        observed_at=observed_at,
        expected_source_hash=expected_source_hash,
        ordered_feature_ids=ordered_feature_ids,
        prepared=prepared,
    )
    assert source is not None
    if prepared is None or not prepared.rebases:
        return source
    # The frozen observation lists the rebases its own rows and listings still need.
    rows, listings = len(source.formation_sessions), set(source.ordered_listing_ids)
    return replace(
        source,
        rebases=tuple(
            item.model_copy(
                update={
                    "rows": tuple((a, min(b, rows)) for a, b in item.rows if a < rows)
                    if item.listing_id in listings
                    else ()
                }
            )
            for item in prepared.rebases
        ),
        kept_rows=prepared.kept_rows,
        rebase_agreement=prepared.rebase_agreement,
    )


def read_workspace_component_inputs(
    workspace: Path,
    *,
    formation: date,
    observed_at: datetime,
    expected_source_hash: str,
    ordered_feature_ids: tuple[str, ...] = (),
    training_factor_ids: tuple[str, ...] = (),
    prepared: PreparedWorkspaceComponentInputs | None = None,
) -> tuple[FrozenPriceVolumeInputs, PanelFeatureSourceArrays | None]:
    """Read exact qualified inference inputs and optional mature training support.

    Source identity is checked before and after reads. Membership, exclusions, calendar, causal
    outcomes and context arrays are supplied by their existing deterministic owners; unavailable
    future outcomes remain absent.

    Args:
        workspace: Caller-owned admitted workspace root.
        formation: Explicit completed formation session.
        observed_at: Explicit calendar observation timestamp.
        expected_source_hash: Required exact workspace score source.
        ordered_feature_ids: Explicit inference feature selections.
        training_factor_ids: Optional sorted unique factors selecting a mature training view.
        prepared: Optional Task-local maximum-cutoff history to project.

    Returns:
        Frozen inference inputs and optional training arrays with known one-session maturities.

    Raises:
        ValueError: Source changes, qualified support/calendar/sector coverage is absent or selected
            training support is invalid.
    """
    source, training, _ = _read_workspace_component_inputs(
        workspace,
        formation=formation,
        observed_at=observed_at,
        expected_source_hash=expected_source_hash,
        ordered_feature_ids=ordered_feature_ids,
        training_factor_ids=training_factor_ids,
        prepared=prepared,
    )
    assert source is not None
    return source, training


def _restated(
    head: WorkspaceObservationHistoryHead,
    proof: _PrefixProof,
    listings: tuple[str, ...],
    feature: Mapping[str, str],
) -> tuple[int, ...]:
    """The listings whose own proof moved while the shared axes held: only they rebuild (a
    corrected action, a restated history)."""
    if head.axes_proof != proof.axes or len(head.listing_digests) != len(listings):
        return ()
    digests, current = head.listing_digests, proof.listings(feature)
    return tuple(c for c, name in enumerate(listings) if digests[c] != current[name])


def _reads(
    listings: tuple[str, ...], first: date, rebuilt: list[str], since: date
) -> tuple[tuple[tuple[str, ...], date], ...]:
    """Each read's listings and first session, as the cold path reads them: the kept listings
    from ``first``, the rebuilt ones from ``since``; each listing once, no empty read."""
    kept = tuple(name for name in listings if name not in set(rebuilt))
    return tuple((ids, start) for ids, start in ((kept, first), (tuple(rebuilt), since)) if ids)


def _bars(
    market: MarketDataRepository,
    reads: tuple[tuple[tuple[str, ...], date], ...],
    through: date,
    connection: Any,
) -> dict[str, tuple[RawDailyBar, ...]]:
    """Each read's raw bars (`_reads`), every listing once."""
    return {
        listing: bars
        for ids, start in reads
        for listing, bars in market.raw_bars_by_listing(
            ids, start=start, through=through, _connection=connection
        ).items()
    }


def _rows_since(
    feature: FeatureStateRepository,
    catalog: str,
    reads: tuple[tuple[tuple[str, ...], date], ...],
    end: date,
    factor_ids: tuple[str, ...],
    connection: object = None,
) -> pd.DataFrame:
    """Stored Feature rows of each read (`_reads`): their keys and values, one frame."""
    extra = {} if connection is None else {"_connection": connection}
    frames = [
        feature.feature_rows(
            listing_ids=ids,
            catalog_hash=catalog,
            start=start,
            end=end,
            factor_ids=factor_ids,
            include_lineage=False,
            as_frame=True,
            **extra,
        )
        for ids, start in reads
        if ids
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _fill(
    values: Mapping[str, FloatArray],
    rows: pd.DataFrame,
    positions: Mapping[date, int],
    listings: Mapping[str, int],
) -> None:
    """Write each row's values at its session and listing, a missing value as NaN."""
    if rows.empty:
        return
    at = rows["session_date"].map(positions)
    kept = at.notna().to_numpy()
    row: IntArray = at.to_numpy()[kept].astype(np.int64)
    column: IntArray = rows["listing_id"].map(listings).to_numpy()[kept].astype(np.int64)
    for name, target in values.items():
        target[row, column] = rows[name].to_numpy(dtype=np.float64, na_value=np.nan)[kept]


def _rebuild_listings(
    values: FloatArray, frame: pd.DataFrame, spec: FactorSpec, rebuilt: list[str]
) -> FloatArray:
    """The values with the rebuilt listings' rows computed from their first session, as a cold
    capture computes them; the kernel groups its rows by listing (`PER_LISTING_EXTENSIONS`)."""
    if not rebuilt:
        return values
    mask = frame["listing_id"].isin(rebuilt).to_numpy()
    flat = values.reshape(-1).copy()
    flat[mask] = append_session_observation_values(
        frame[mask], spec, previous=None, compute=default_extension_kernel_registry().compute
    ).to_numpy(dtype=np.float64)
    return flat.reshape(values.shape)


def _kept_columns(columns: Mapping[str, FloatArray], kept: set[str]) -> Mapping[str, FloatArray]:
    """The kept history's columns, refused unless they are exactly the capture's."""
    expected = {
        *("ohlcv::" + name for name in ("open", "high", "low", "close", "volume")),
        "observations::dist_52w_high",
        "observations::dist_52w_low",
        "observations::close_to_close",
        "raw_simple",
        *("formula::" + name for name in kept),
    }
    if set(columns) != expected:
        raise ValueError("strategy_score.prepared_input_axis_invalid")
    return columns


class _Reuse(NamedTuple):
    """A capture's reuse of its kept history: the kept columns (whole, or all but the restated
    listings), the restated listing columns, and the current prefix's proof."""

    restored: Mapping[str, FloatArray] | None
    restated: tuple[int, ...]
    proved: _PrefixProof
    digests: tuple[str, ...]
    closed: tuple[str, ...]

    def parts(self) -> dict[str, Any]:
        """The capture's fields the next head seals: the current prefix's proof parts."""
        return {
            "dependency_prefix_hash": self.proved.whole,
            "axes_proof": self.proved.axes,
            "listing_digests": self.digests,
            "listing_closed": self.closed,
        }


def _kept_candidate(
    history: tuple[WorkspaceObservationHistoryHead, Mapping[str, FloatArray]] | None,
    listings: tuple[str, ...],
    sessions: tuple[date, ...],
    points: tuple[CausalExecutionSchedulePoint, ...],
) -> tuple[date, ...] | None:
    """The kept history's sessions when it is a prefix of today's axes, else None."""
    if history is None:
        return None
    head = history[0]
    kept: tuple[date, ...] = head.formation_sessions
    if (
        head.ordered_listing_ids == listings
        and len(kept) <= len(sessions)
        and kept == sessions[: len(kept)]
        and head.stable_session_count
        == sum(point.holding_end_session <= kept[-1] for point in points)
    ):
        return kept
    return None


def _carried_closed(
    head: WorkspaceObservationHistoryHead, listings: tuple[str, ...], end: date
) -> tuple[str, ...]:
    """The head's closed-year digests in sorted listing order, when they still answer for
    today's cutoff year (its prefix proved whole); else none, and they are read."""
    if len(head.listing_closed) != len(listings) or head.formation_sessions[-1].year != end.year:
        return ()
    closed_of = dict(zip(listings, head.listing_closed, strict=True))
    return tuple(closed_of[name] for name in sorted(listings))


def _kept_history_reuse(
    market: MarketDataRepository,
    feature: FeatureStateRepository,
    binding: WorkspaceDataUpdateBinding,
    history: tuple[WorkspaceObservationHistoryHead, Mapping[str, FloatArray]] | None,
    *,
    sessions: tuple[date, ...],
    listings: tuple[str, ...],
    points: tuple[CausalExecutionSchedulePoint, ...],
    observed_at: datetime,
    kept: set[str],
    selections: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]],
) -> _Reuse:
    """Prove the kept history's prefix against the current sources and decide its reuse.

    The immutable child proves its own bytes. Reuse also needs the source owner to prove the
    *current* old values, not a revision or last date: whole when the prefix proof holds,
    listing by listing when only listings moved beside the same axes.
    """
    kept_stored, selected_stored, kept_extensions, selected_extensions = selections
    restored: Mapping[str, FloatArray] | None = None
    restated: tuple[int, ...] = ()
    candidate_sessions = _kept_candidate(history, listings, sessions, points)
    same_columns = kept_stored == selected_stored and kept_extensions == selected_extensions
    ends = (
        (candidate_sessions[-1], sessions[-1])
        if candidate_sessions is not None and same_columns
        else (sessions[-1],)
    )
    with (
        market.database.read_transaction() as connection,
        span("verify", "feature_prefix_proofs"),
    ):
        catalog, start = binding.feature_catalog_hash, sessions[0]
        feature_proofs = feature.feature_source_prefix_proofs(
            listing_ids=listings,
            catalog_hash=catalog,
            start=start,
            ends=ends,
            factor_ids=selected_stored,
            _connection=connection,
        )
        # A wider kept history's prefix is proved over its own Features, apart.
        kept_proof = (
            feature_proofs[0]
            if candidate_sessions is None or same_columns
            else feature.feature_source_prefix_proofs(
                listing_ids=listings,
                catalog_hash=catalog,
                start=start,
                ends=(candidate_sessions[-1],),
                factor_ids=kept_stored,
                _connection=connection,
            )[0]
        )
    kept_prefix, whole = None, False
    listing_ends: tuple[date, ...] = (sessions[-1],)
    carried: tuple[str, ...] = ()
    if history is not None and candidate_sessions is not None:
        head, columns = history
        kept_prefix = _workspace_history_prefix_proof(
            market,
            feature,
            sessions=candidate_sessions,
            listings=listings,
            observed_at=observed_at,
            catalog_hash=catalog,
            factor_ids=kept_stored,
            extension_factor_ids=kept_extensions,
            feature_prefix_proof=kept_proof,
        )
        whole = kept_prefix.whole == head.dependency_prefix_hash
        if whole and same_columns:
            # The prefix held, its closed years with it: carry their digests, read the open year.
            carried = _carried_closed(head, listings, sessions[-1])
        elif not whole and same_columns and set(kept_extensions) <= PER_LISTING_EXTENSIONS:
            # Each listing's own values at the kept cutoff, so a restated listing rebuilds
            # alone; a cross-sectional kernel keeps the capture cold (no kept proofs read).
            listing_ends = (candidate_sessions[-1], sessions[-1])
    with span("verify", "feature_listing_proofs"):
        listing_proofs, closed = feature_source_listing_proofs(
            feature,
            listing_ids=listings,
            catalog_hash=catalog,
            start=start,
            ends=listing_ends,
            factor_ids=selected_stored,
            closed=carried,
        )
    if history is not None and kept_prefix is not None:
        head, columns = history
        restated = (
            _restated(head, kept_prefix, listings, listing_proofs[0]) if listing_ends[1:] else ()
        )
        if whole or restated:
            restored = rebased(_kept_columns(columns, kept), listings, head.rebases)
    with span("verify", "history_prefix_proof"):
        proved = _workspace_history_prefix_proof(
            market,
            feature,
            sessions=sessions,
            listings=listings,
            observed_at=observed_at,
            catalog_hash=catalog,
            factor_ids=selected_stored,
            extension_factor_ids=selected_extensions,
            feature_prefix_proof=feature_proofs[-1],
        )
    current = proved.listings(listing_proofs[-1])
    closed_of = dict(zip(sorted(listings), closed, strict=True))
    return _Reuse(
        restored,
        restated,
        proved,
        tuple(current[name] for name in listings),
        tuple(closed_of[name] for name in listings),
    )


def _read_workspace_component_inputs(
    workspace: Path,
    *,
    formation: date,
    observed_at: datetime,
    expected_source_hash: str,
    ordered_feature_ids: tuple[str, ...] = (),
    training_factor_ids: tuple[str, ...] = (),
    prepared: PreparedWorkspaceComponentInputs | None = None,
    capture: bool = False,
    history: tuple[WorkspaceObservationHistoryHead, Mapping[str, FloatArray]] | None = None,
) -> tuple[
    FrozenPriceVolumeInputs | None,
    PanelFeatureSourceArrays | None,
    PreparedWorkspaceComponentInputs | None,
]:
    if workspace_score_source_identity(workspace) != expected_source_hash:
        raise ValueError("strategy_score.source_revision_changed")
    market = MarketDataRepository(workspace)
    proof = _market_source_proof(market) if capture or prepared is not None else None
    if prepared is not None:
        if (
            prepared.workspace != workspace.resolve()
            or prepared.source_hash != expected_source_hash
            or prepared.observed_at != observed_at
            or formation > prepared.through
        ):
            raise ValueError("strategy_score.prepared_input_binding_invalid")
        if proof != prepared.source_proof:
            raise ValueError("strategy_score.source_revision_changed")
    readiness = market.readiness.load("us-current-index-research")
    if readiness is None or readiness.active_manifest_id is None:
        raise ValueError("strategy_score.workspace_not_qualified")
    manifest = market.load_universe_manifest(readiness.active_manifest_id)
    feature = FeatureStateRepository(market.database, market_data=market)
    sector = feature.current_sector_state(manifest)
    covered = market.manifest_raw_range(manifest)
    if sector is None or covered is None or not covered[0] <= formation <= covered[1]:
        raise ValueError("strategy_score.local_input_support_unavailable")
    # Only completed sessions exist here. Missing future maturities stay absent.
    calendar = materialize_calendar_schedule(
        ("XNAS", "XNYS"),
        start=covered[0],
        end=formation,
        as_of_timestamp=observed_at,
    )
    clocks = {}
    counts: dict[date, int] = {}
    for row in calendar.to_pylist():
        day = row["session_date"]
        clocks[day] = row
        counts[day] = counts.get(day, 0) + 1
    axis = tuple(day for day in sorted(clocks) if counts[day] == 2)
    sessions = tuple(day for day in axis if day <= formation)
    if not sessions or sessions[-1] != formation:
        raise ValueError("strategy_score.formation_not_a_completed_session")
    schedule = market.membership_schedule(
        manifest.profile.market_profile_id,
        sessions=sessions,
        fallback_listing_ids=tuple(item.listing_id for item in manifest.listings),
    )
    # Historical leavers remain observations; already admitted future entrants
    # may warm up their own history without entering earlier references.
    listings = tuple(sorted(set(schedule.union) | set(schedule.admitted_listing_ids)))
    shape = (len(sessions), len(listings))
    positions = {value: index for index, value in enumerate(sessions)}
    ohlcv: dict[str, FloatArray] = {
        name: np.full(shape, np.nan, dtype=np.float64)
        for name in ("open", "high", "low", "close", "volume")
    }
    raw_simple: FloatArray = np.full(shape, np.nan, dtype=np.float64)
    method = build_installed_execution_outcome_method_catalog().resolve(ONE_SESSION_RECIPE_ID)
    points = resolve_schedule_points(recipe=method, ordered_sessions=axis, session_clocks=clocks)
    by_formation = {point.formation_session: point for point in points}
    dividend_axis = ExecutionOutcomeSessionAxis(axis)
    binding = installed_data_update_binding(workspace)
    extension = {
        s.factor_id: s
        for s in (*session_observation_factor_specs(), *session_liquidity_factor_specs())
    }
    formula_ids = tuple(
        sorted(
            {
                name.split("::")[1]
                for name in ordered_feature_ids
                if name.startswith(
                    ("RELATIVE_STOCK_CROSS_SECTION::", "NON_NEUTRAL_STOCK_CROSS_SECTION::")
                )
            }
            | set(training_factor_ids)
        )
    )
    stored = (
        tuple(sorted((set(formula_ids) | {"mom_252_21"}) - extension.keys())) if formula_ids else ()
    )
    selected_stored = tuple(sorted({"dist_52w_high", "dist_52w_low", *stored}))
    selected_extensions = tuple(
        sorted({"close_to_close", *(name for name in formula_ids if name in extension)})
    )
    # A kept history serves when it holds every column this capture needs; its prefix is proved
    # over the Features it was kept for, so a renewal day's wider history serves the next day.
    kept_stored, kept_extensions = selected_stored, selected_extensions
    needed = {*stored, *(name for name in formula_ids if name in extension)}
    kept = needed
    if capture and history is not None:
        kept = {name.split("::", 1)[1] for name in history[1] if name.startswith("formula::")}
        if (
            needed
            <= kept
            <= set(feature._feature_factor_ids(binding.feature_catalog_hash)) | set(extension)
        ):
            kept_stored = tuple(
                sorted({"dist_52w_high", "dist_52w_low", *(kept - extension.keys())})
            )
            kept_extensions = tuple(sorted({"close_to_close", *(kept & extension.keys())}))
        else:
            history = None
    proof_parts: dict[str, Any] = {}
    restored: Mapping[str, FloatArray] | None = None
    restated: tuple[int, ...] = ()
    reused_count = return_start = 0
    if capture:
        reuse = _kept_history_reuse(
            market,
            feature,
            binding,
            history,
            sessions=sessions,
            listings=listings,
            points=points,
            observed_at=observed_at,
            kept=kept,
            selections=(kept_stored, selected_stored, kept_extensions, selected_extensions),
        )
        restored, restated = reuse.restored, reuse.restated
        if restored is not None and history is not None:
            reused_count = len(history[0].formation_sessions)
            return_start = history[0].stable_session_count
            for name in ohlcv:
                ohlcv[name][:reused_count] = restored["ohlcv::" + name]
            raw_simple[:reused_count] = restored["raw_simple"]
        proof_parts = reuse.parts()
    rebuilt = [listings[column] for column in restated]
    actions: list[tuple[CorporateActionEvent, ...]] = []
    if prepared is None:
        with market._connect(read_only=True) as connection, connection.snapshot():
            with span("materialize", "ohlcv_returns"):
                # Every listing's bars and actions in one read each, as each listing's own
                # reads answer them (two queries a listing and component were most of this).
                bars_by_listing = _bars(
                    market,
                    _reads(listings, sessions[return_start], rebuilt, sessions[0]),
                    formation,
                    connection,
                )
                actions_by_listing = market.actions_by_listing(listings, _connection=connection)
                for values in (*ohlcv.values(), raw_simple):
                    values[:, restated] = np.nan
                for column, listing in enumerate(listings):
                    bars = bars_by_listing[listing]
                    by_session = {bar.session_date: bar for bar in bars}
                    listing_actions = actions_by_listing[listing]
                    actions.append(listing_actions)
                    try:
                        dividends = _action_dividends(listing_actions, through=formation)
                    except ValueError as error:
                        if proof is not None and _market_source_proof(market) != proof:
                            raise ValueError("strategy_score.source_revision_changed") from error
                        raise
                    for bar in bars:
                        if bar.session_date in positions:
                            for name in ohlcv:
                                value = getattr(bar, name)
                                ohlcv[name][positions[bar.session_date], column] = (
                                    np.nan if value is None else float(value)
                                )
                    for row in range(0 if column in restated else return_start, len(sessions)):
                        session = sessions[row]
                        point = by_formation.get(session)
                        if point is None:
                            continue
                        value = causal_execution_simple_return(
                            entry_bar=by_session.get(point.entry_session),
                            holding_bar=by_session.get(point.holding_end_session),
                            period_dividend_split_adjusted=period_dividend_for_point(
                                recipe=method,
                                dividends=dividends,
                                ordered_sessions=dividend_axis,
                                point=point,
                            ),
                        )
                        raw_simple[row, column] = np.nan if value is None else value
            with span("read", "feature_rows"):
                first = sessions[reused_count] if reused_count < len(sessions) else formation
                observations = _rows_since(
                    feature,
                    binding.feature_catalog_hash,
                    _reads(listings, first, rebuilt, sessions[0]),
                    formation,
                    ("dist_52w_high", "dist_52w_low"),
                    connection,
                )
        observed: dict[str, FloatArray] = {
            name: np.full(shape, np.nan, dtype=np.float64)
            for name in ("dist_52w_high", "dist_52w_low")
        }
        if restored is not None:
            for name in observed:
                observed[name][:reused_count] = restored["observations::" + name]
                observed[name][:, restated] = np.nan
    else:
        if sessions != prepared.formation_sessions[: len(sessions)] or not set(listings) <= set(
            prepared.ordered_listing_ids
        ):
            raise ValueError("strategy_score.prepared_input_axis_invalid")
        column_positions = [prepared.ordered_listing_ids.index(listing) for listing in listings]

        def bounded(values: FloatArray) -> FloatArray:
            return values[: len(sessions), column_positions].copy()

        ohlcv = {name: bounded(values) for name, values in prepared.ohlcv.items()}
        observed = {name: bounded(values) for name, values in prepared.observations.items()}
        raw_simple = bounded(prepared.raw_simple)
        # Only maturities visible at this cutoff are admitted, including the
        # trailing observation rows which were mature at the later capture.
        raw_simple[len(points) :] = np.nan
        actions = [prepared.actions[column] for column in column_positions]
        for listing_actions in actions:
            _action_dividends(listing_actions, through=formation)
    listing_positions = {listing: index for index, listing in enumerate(listings)}
    reference_eligible: BoolArray | None = None
    nominal: IntArray | None = None
    if schedule.bootstrap is not None:
        reference_eligible = np.zeros(shape, dtype=np.bool_)
        for row, day in enumerate(sessions):
            reference_eligible[row, [listing_positions[name] for name in schedule.members(day)]] = (
                True
            )
        nominal = reference_eligible.sum(axis=1, dtype=np.int64)
        for exclusion in PanelStateRepository(
            market.database, market_data=market
        ).panel_source_exclusions(
            market_profile_id=manifest.profile.market_profile_id,
            sessions=sessions,
            listing_ids=listings,
        ):
            rows = [
                i
                for i, day in enumerate(sessions)
                if exclusion.first_session <= day <= exclusion.last_session
            ]
            reference_eligible[rows, listing_positions[exclusion.listing_id]] = False
        # A missing close is not a usable observation. Zero volume is retained;
        # each selected feature still decides its own mathematical availability.
        reference_eligible &= np.isfinite(ohlcv["close"]) & (ohlcv["close"] > 0.0)
        reference_eligible.setflags(write=False)
        nominal.setflags(write=False)
    if prepared is None:
        _fill(observed, observations, positions, listing_positions)
        del observations
    # This formula's kernel is installed, although the ordinary stored catalog
    # need not materialize its column. Invoke its owner; do not invent an alias
    # or change the workspace's global Feature catalog.
    if prepared is None:
        observation_frame = pd.DataFrame(
            {
                "session_date": np.repeat(np.asarray(sessions, dtype=object), len(listings)),
                "listing_id": np.tile(np.asarray(listings, dtype=object), len(sessions)),
                "close_split_adjusted": ohlcv["close"].reshape(-1),
            }
        )
        observation_spec = next(
            value
            for value in session_observation_factor_specs()
            if value.factor_id == "close_to_close"
        )
        with span("features", "close_to_close"):
            observed["close_to_close"] = _rebuild_listings(
                append_session_observation_values(
                    observation_frame,
                    observation_spec,
                    previous=(
                        pd.Series(restored["observations::close_to_close"].reshape(-1))
                        if restored is not None
                        else None
                    ),
                    compute=default_extension_kernel_registry().compute,
                )
                .to_numpy(dtype=np.float64)
                .reshape(shape),
                observation_frame,
                observation_spec,
                rebuilt,
            )
    formula_values: dict[str, FloatArray] = {}
    if formula_ids:
        required = set(stored) | (set(formula_ids) & extension.keys())
        if prepared is not None and not required <= prepared.formula_values.keys():
            # A renewed authority can ask for another Feature set. The ordinary
            # full reader retains the new authority's exact declaration.
            result = _read_workspace_component_inputs(
                workspace,
                formation=formation,
                observed_at=observed_at,
                expected_source_hash=expected_source_hash,
                ordered_feature_ids=ordered_feature_ids,
                training_factor_ids=training_factor_ids,
            )
            if _market_source_proof(market) != proof:
                raise ValueError("strategy_score.source_revision_changed")
            return result
        if prepared is not None:
            selected = (*stored, *(name for name in formula_ids if name in extension))
            formula_values = {name: bounded(prepared.formula_values[name]) for name in selected}
        else:
            with span("read", "formula_rows"):
                first = sessions[reused_count] if reused_count < len(sessions) else formation
                stored_rows = _rows_since(
                    feature,
                    binding.feature_catalog_hash,
                    _reads(listings, first, rebuilt, sessions[0]),
                    formation,
                    stored,
                )
            formula_values = {name: np.full(shape, np.nan, dtype=np.float64) for name in stored}
            if restored is not None:
                for name in stored:
                    formula_values[name][:reused_count] = restored["formula::" + name]
                    formula_values[name][:, restated] = np.nan
            _fill(formula_values, stored_rows, positions, listing_positions)
            del stored_rows
            frame = pd.DataFrame(
                {
                    "session_date": np.repeat(np.asarray(sessions, dtype=object), len(listings)),
                    "listing_id": np.tile(np.asarray(listings, dtype=object), len(sessions)),
                    **{
                        f"{name}_split_adjusted": ohlcv[name].reshape(-1)
                        for name in ("open", "high", "low", "close")
                    },
                    "volume": ohlcv["volume"].reshape(-1),
                    "close_raw": ohlcv["close"].reshape(-1),
                    "volume_raw": ohlcv["volume"].reshape(-1),
                }
            )
            with span("features", "extension_formulas"):
                for name in formula_ids:
                    if name in extension:
                        formula_values[name] = _rebuild_listings(
                            append_session_observation_values(
                                frame,
                                extension[name],
                                previous=(
                                    pd.Series(restored["formula::" + name].reshape(-1))
                                    if restored is not None
                                    else None
                                ),
                                compute=default_extension_kernel_registry().compute,
                            )
                            .to_numpy(dtype=np.float64)
                            .reshape(shape),
                            frame,
                            extension[name],
                            rebuilt,
                        )
    # The Sector each session reads, the store's history over these names (V346).
    sector_history = feature.sector_history(manifest, listing_ids=listings)
    if sector_history is None or set(listings) - set(sector_history):
        raise ValueError("strategy_score.sector_coverage_incomplete")
    mapping = sector_subset(sector_history, listings)
    sectors = sector_ids(mapping)
    if capture:
        if training_factor_ids and (
            not points or tuple(sorted(set(training_factor_ids))) != training_factor_ids
        ):
            raise ValueError("strategy_score.training_source_axis_invalid")
        assert proof is not None
        held_history = PreparedWorkspaceComponentInputs(
            workspace=workspace.resolve(),
            through=formation,
            observed_at=observed_at,
            source_hash=expected_source_hash,
            source_proof=proof,
            formation_sessions=sessions,
            ordered_listing_ids=listings,
            ohlcv=_immutable_arrays(ohlcv),
            observations=_immutable_arrays(observed),
            formula_values=_immutable_arrays(formula_values),
            raw_simple=_immutable_arrays({"returns": raw_simple})["returns"],
            actions=tuple(actions),
            stable_session_count=len(points),
            reused_history_hash=(
                history[0].head_hash if restored is not None and history and not restated else None
            ),
            restated_listing_ids=tuple(rebuilt),
            **proof_parts,
        )
        if (
            workspace_score_source_identity(workspace) != expected_source_hash
            or _market_source_proof(market) != proof
        ):
            raise ValueError("strategy_score.source_revision_changed")
        return None, None, held_history
    raw_log = np.log1p(raw_simple)
    holdings = tuple(
        by_formation[session].holding_end_session if session in by_formation else None
        for session in sessions
    )
    known = tuple(point.formation_session for point in points)
    states, _identity = compile_sector_context_arrays(
        sessions=known,
        holding_end_sessions=tuple(point.holding_end_session for point in points),
        listing_ids=listings,
        raw_log_returns=raw_log[: len(known)],
        target_evidence_hash=expected_source_hash,
        sector_by_listing_id=mapping,
        sector_revision=sector.sector_revision,
        output_sessions=sessions,
        reference_eligible=(
            reference_eligible[: len(known)] if reference_eligible is not None else None
        ),
    )
    interactions: FloatArray = np.full((len(sessions), 2), np.nan, dtype=np.float64)
    if training_factor_ids:
        reference = feature.market_reference("SPY")
        if reference is None:
            raise ValueError("strategy_score.training_market_reference_absent")
        points_by_session = {
            p.session_date: float(p.adjusted_close)
            for p in market.provider_adjusted_closes(
                str(reference["listing_id"]),
                start=sessions[0],
                through=formation,
            )
        }
        if set(sessions) - points_by_session.keys():
            raise ValueError("strategy_score.training_market_reference_incomplete")
        interactions = interaction_market_state_children(
            pd.DataFrame(
                {
                    "session_date": sessions,
                    "listing_id": [str(reference["listing_id"])] * len(sessions),
                    "market_provider_adjusted_close": [points_by_session[d] for d in sessions],
                }
            )
        ).to_numpy(dtype=np.float64)
    sector_context, market_context = assemble_panel_context_arrays(
        formation_sessions=sessions,
        holding_end_sessions=holdings,
        ordered_listing_ids=listings,
        ordered_sector_ids=sectors,
        sector_by_listing_id=mapping,
        raw_log_execution_returns=raw_log,
        raw_simple_execution_returns=raw_simple,
        sector_state_values=states,
        # Inference does not need these columns; the frozen training-support
        # view does. Its values come from the existing Feature owner, not zeros.
        market_interaction_state_values=interactions,
        observation_returns=observed["close_to_close"],
        observation_high_distance=observed["dist_52w_high"],
        observation_low_distance=observed["dist_52w_low"],
        observation_volume_state=formula_values.get("volume_zscore_21"),
        observation_dollar_volume=formula_values.get("session_dollar_volume"),
        reference_eligible=reference_eligible,
        selected_sector_source_ids=None if training_factor_ids else ("sector_trend_20",),
        selected_market_source_ids=(
            None
            if training_factor_ids
            else (
                "market_drawdown_252",
                "observed_breadth_positive_share",
                "observed_new_high_low_share",
            )
        ),
    )
    if workspace_score_source_identity(workspace) != expected_source_hash:
        raise ValueError("strategy_score.source_revision_changed")
    source = FrozenPriceVolumeInputs(
        formation_sessions=sessions,
        ordered_listing_ids=listings,
        sector_by_listing_id=mapping,
        **ohlcv,
        market_context_values=(
            market_context[:, [3, 8, 15]]
            if training_factor_ids
            else np.asfortranarray(market_context)
        ),
        sector_trend_values=(
            sector_context[:, :, 1] if training_factor_ids else sector_context.squeeze(axis=-1)
        ),
        source_binding_hash=expected_source_hash,
        formula_values=formula_values,
        reference_eligible=reference_eligible,
        nominal_member_count=nominal,
    )
    training = None
    if training_factor_ids:
        # A training view contains only rows with a known h1 maturity. Daily
        # inference retains the later observations without any target columns.
        size = len(points)
        if not size or tuple(sorted(set(training_factor_ids))) != training_factor_ids:
            raise ValueError("strategy_score.training_source_axis_invalid")
        target, _ = compile_frozen_component_training_targets(
            source=source,
            formation_sessions=sessions[:size],
            holding_end_sessions=tuple(p.holding_end_session for p in points),
            raw_log_returns=raw_log[:size],
            simple_returns=raw_simple[:size],
            target_method_id="EXACT_FROZEN_G0_H1_WHOLE_UNIVERSE_TARGET",
            reference_eligible=(
                reference_eligible[:size] if reference_eligible is not None else None
            ),
            nominal_member_count=(nominal[:size] if nominal is not None else None),
        )
        formula = np.stack([formula_values[name][:size] for name in training_factor_ids], axis=-1)
        arrays = [
            formula,
            target,
            raw_log[:size],
            raw_simple[:size],
            sector_context[:size],
            market_context[:size],
        ]
        for values in arrays:
            values.setflags(write=False)
        training = PanelFeatureSourceArrays(
            formation_sessions=sessions[:size],
            holding_end_sessions=tuple(p.holding_end_session for p in points),
            ordered_listing_ids=listings,
            ordered_factor_ids=training_factor_ids,
            absolute_state_factor_ids=training_factor_ids,
            ordered_sector_ids=sectors,
            sector_by_listing_id=mapping,
            raw_formula_values=formula,
            total_return_target_z=target,
            raw_log_execution_returns=arrays[2],
            raw_simple_execution_returns=arrays[3],
            sector_context_values=arrays[4],
            market_context_values=arrays[5],
            source_identity_hashes=MappingProxyType({"workspace": expected_source_hash}),
            reference_eligible=(
                reference_eligible[:size] if reference_eligible is not None else None
            ),
        )
    if proof is not None and _market_source_proof(market) != proof:
        raise ValueError("strategy_score.source_revision_changed")
    return source, training, prepared
