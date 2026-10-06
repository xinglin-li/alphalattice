"""Read admitted workspace inputs for component inference, never future labels.

Data owns bar/action status and returns, Feature owns observation values,
Sector owns its causal states and Alpha owns Context assembly. This composition
only resolves their axes and delegates; it publishes no Foundation or book.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from alphalattice.control.data_platform.maintenance.contracts import WorkspaceInputStatus
from alphalattice.control.product_host.maintenance.data_update import (
    installed_data_update_binding,
    read_workspace_inputs,
)
from alphalattice.foundation.causal_outcomes.execution.compile import (
    _action_dividends,
    derive_causal_execution_row,
)
from alphalattice.foundation.causal_outcomes.execution.methods import (
    ONE_SESSION_RECIPE_ID,
    ExecutionOutcomeSessionAxis,
    build_installed_execution_outcome_method_catalog,
    period_dividend_for_point,
    resolve_schedule_points,
)
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
    session_observation_factor_specs,
)
from alphalattice.foundation.feature_engine.storage.repositories import (
    FeatureStateRepository,
    PanelStateRepository,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
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
from alphalattice.investment.alpha_research.targets.component_training import (
    compile_frozen_component_training_targets,
)
from alphalattice.investment.sector_research.inputs.surface import compile_sector_context_arrays
from alphalattice.kernel.data.calendar import materialize_calendar_schedule
from alphalattice.kernel.quant.sector_history import sector_ids, sector_subset
from alphalattice.kernel.shared_kernel.identity import canonical_hash


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
) -> FrozenPriceVolumeInputs:
    """Build exact qualified price-volume inference inputs for one completed formation.

    Args:
        workspace: Caller-owned admitted workspace root.
        formation: Explicit completed formation session.
        observed_at: Explicit calendar observation timestamp.
        expected_source_hash: Required exact workspace score source.
        ordered_feature_ids: Explicit inference feature selections.

    Returns:
        Frozen source arrays assembled by the existing Data and Feature owners.
    """
    return read_workspace_component_inputs(
        workspace,
        formation=formation,
        observed_at=observed_at,
        expected_source_hash=expected_source_hash,
        ordered_feature_ids=ordered_feature_ids,
    )[0]


def read_workspace_component_inputs(
    workspace: Path,
    *,
    formation: date,
    observed_at: datetime,
    expected_source_hash: str,
    ordered_feature_ids: tuple[str, ...] = (),
    training_factor_ids: tuple[str, ...] = (),
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

    Returns:
        Frozen inference inputs and optional training arrays with known one-session maturities.

    Raises:
        ValueError: Source changes, qualified support/calendar/sector coverage is absent or selected
            training support is invalid.
    """
    if workspace_score_source_identity(workspace) != expected_source_hash:
        raise ValueError("strategy_score.source_revision_changed")
    market = MarketDataRepository(workspace)
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
    # One read connection shared with both existing Data readers.
    with market._connect(read_only=True) as connection:
        for column, listing in enumerate(listings):
            bars = market.raw_bars(
                listing, start=sessions[0], through=formation, _connection=connection
            )
            by_session = {bar.session_date: bar for bar in bars}
            dividends = _action_dividends(
                market.actions(listing, _connection=connection), through=formation
            )
            for bar in bars:
                if bar.session_date in positions:
                    for name in ohlcv:
                        value = getattr(bar, name)
                        ohlcv[name][positions[bar.session_date], column] = (
                            np.nan if value is None else float(value)
                        )
            for row, session in enumerate(sessions):
                point = by_formation.get(session)
                if point is None:
                    continue
                result = derive_causal_execution_row(
                    listing_id=listing,
                    symbol=listing,
                    point=point,
                    entry_bar=by_session.get(point.entry_session),
                    holding_bar=by_session.get(point.holding_end_session),
                    period_dividend_split_adjusted=period_dividend_for_point(
                        recipe=method,
                        dividends=dividends,
                        ordered_sessions=dividend_axis,
                        point=point,
                    ),
                )
                value = result["simple_return"]
                raw_simple[row, column] = np.nan if value is None else value
        observations = feature.feature_rows(
            listing_ids=listings,
            catalog_hash=installed_data_update_binding(workspace).feature_catalog_hash,
            start=sessions[0],
            end=formation,
            factor_ids=("dist_52w_high", "dist_52w_low"),
            _connection=connection,
        )
    observed: dict[str, FloatArray] = {
        name: np.full(shape, np.nan, dtype=np.float64) for name in ("dist_52w_high", "dist_52w_low")
    }
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
    for item in observations:
        if item["session_date"] not in positions:
            continue
        row, column = positions[item["session_date"]], listing_positions[item["listing_id"]]
        for name in observed:
            observed[name][row, column] = np.nan if item[name] is None else item[name]
    raw_log = np.log1p(raw_simple)
    # This formula's kernel is installed, although the ordinary stored catalog
    # need not materialize its column. Invoke its owner; do not invent an alias
    # or change the workspace's global Feature catalog.
    observation_frame = pd.DataFrame(
        {
            "session_date": np.repeat(np.asarray(sessions, dtype=object), len(listings)),
            "listing_id": np.tile(np.asarray(listings, dtype=object), len(sessions)),
            "close_split_adjusted": ohlcv["close"].reshape(-1),
        }
    )
    observation_spec = next(
        value for value in session_observation_factor_specs() if value.factor_id == "close_to_close"
    )
    observed["close_to_close"] = (
        default_extension_kernel_registry()
        .compute(observation_frame, observation_spec)
        .to_numpy(dtype=np.float64)
        .reshape(shape)
    )
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
    formula_values: dict[str, FloatArray] = {}
    if formula_ids:
        extension = {
            s.factor_id: s
            for s in (*session_observation_factor_specs(), *session_liquidity_factor_specs())
        }
        stored = tuple(sorted((set(formula_ids) | {"mom_252_21"}) - extension.keys()))
        rows = feature.feature_rows(
            listing_ids=listings,
            catalog_hash=installed_data_update_binding(workspace).feature_catalog_hash,
            start=sessions[0],
            end=formation,
            factor_ids=stored,
        )
        formula_values = {name: np.full(shape, np.nan, dtype=np.float64) for name in stored}
        for item in rows:
            if item["session_date"] in positions:
                row, column = positions[item["session_date"]], listing_positions[item["listing_id"]]
                for name in stored:
                    formula_values[name][row, column] = np.nan if item[name] is None else item[name]
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
        for name in formula_ids:
            if name in extension:
                formula_values[name] = (
                    default_extension_kernel_registry()
                    .compute(frame, extension[name])
                    .to_numpy(dtype=np.float64)
                    .reshape(shape)
                )
    holdings = tuple(
        by_formation[session].holding_end_session if session in by_formation else None
        for session in sessions
    )
    # The Sector each session reads, the store's history over these names (V346).
    history = feature.sector_history(manifest, listing_ids=listings)
    if history is None or set(listings) - set(history):
        raise ValueError("strategy_score.sector_coverage_incomplete")
    mapping = sector_subset(history, listings)
    sectors = sector_ids(mapping)
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
    )
    if workspace_score_source_identity(workspace) != expected_source_hash:
        raise ValueError("strategy_score.source_revision_changed")
    source = FrozenPriceVolumeInputs(
        formation_sessions=sessions,
        ordered_listing_ids=listings,
        sector_by_listing_id=mapping,
        **ohlcv,
        market_context_values=market_context[:, [3, 8, 15]],
        sector_trend_values=sector_context[:, :, 1],
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
    return source, training
