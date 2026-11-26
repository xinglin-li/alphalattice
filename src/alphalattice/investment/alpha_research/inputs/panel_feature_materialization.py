"""Fold materialization and causal scaling for installed Panel feature views.

Catalog/recipe contracts and source-lane assembly remain in
``panel_feature_views``; this module owns common-axis preflight, temporal
projection, fold fitting and immutable model matrices.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from types import MappingProxyType
from typing import Literal, Protocol, cast

import numpy as np
import numpy.typing as npt

from alphalattice.foundation.feature_engine.producers.preprocessing.joint_primary import (
    formula_pretransform,
    formula_pretransform_descriptor,
)
from alphalattice.investment.alpha_research.inputs.panel_feature_views import (
    _NEAR_CONSTANT_THRESHOLD,
    BASE_MARKET_CONTEXT_SOURCE_IDS,
    OBSERVED_MARKET_SOURCE_IDS,
    SECTOR_CONTEXT_SOURCE_IDS,
    SPARSE_SESSION_AMPLITUDE_REQUIRED_FORMULA_IDS,
    SPARSE_SESSION_AMPLITUDE_VIEW_IDS,
    BoolArray,
    FeatureRoleId,
    FeatureRoleSelection,
    FeatureTransformId,
    FloatArray,
    FoldScaleReceipt,
    InstalledPanelViewId,
    IntArray,
    PanelFeatureBoundaryError,
    PanelFeaturePreflight,
    PanelFeatureSourceArrays,
    PanelFeatureViewCatalog,
    PanelFeatureViewRecipe,
    ScaleState,
    ViewRetentionEvidence,
    _transform_lookback,
    build_installed_panel_feature_view_catalog,
    panel_array_hash,
)
from alphalattice.investment.alpha_research.inputs.preprocessing.sector_context import (
    SectorListingSessionScaleResult,
    scale_mapped_sector_context_by_listing_session,
)
from alphalattice.kernel.quant.cross_section import (
    MAD_SCALE,
    median_mad_winsor,
)
from alphalattice.kernel.quant.sector_history import sector_codes, sector_slices
from alphalattice.kernel.shared_kernel.identity import canonical_hash


@dataclass(frozen=True, slots=True)
class RoleSurface:
    role: FeatureRoleSelection
    ordered_feature_ids: tuple[str, ...]
    raw_values: FloatArray
    native_entity_axis_hash: str


@dataclass(frozen=True, slots=True)
class RoleSurfaceDescriptor:
    role: FeatureRoleSelection
    ordered_feature_ids: tuple[str, ...]
    native_entity_axis_hash: str


@dataclass(frozen=True, slots=True)
class PanelFeatureProjection:
    """Retain one role-normalized feature-view projection and its paired target/economic lanes.

    Training and transformation arrays retain dated row/listing axes, exact common-row commitments,
    aggregation readiness and final scaling receipts.
    """

    method: PanelFeatureViewRecipe
    training_sessions: tuple[date, ...]
    transform_sessions: tuple[date, ...]
    row_sessions: tuple[date, ...]
    row_listing_ids: tuple[str, ...]
    ordered_feature_ids: tuple[str, ...]
    training_features: FloatArray
    transformed_features: FloatArray
    training_targets: FloatArray
    transformed_targets: FloatArray
    training_raw_simple_returns: FloatArray
    transformed_raw_simple_returns: FloatArray
    aggregation_ready: BoolArray
    common_training_row_axis_hash: str
    common_transform_row_axis_hash: str
    scale_receipts: tuple[FoldScaleReceipt, ...]


def _readonly(values: npt.NDArray[np.generic], *, dtype: np.dtype[np.generic]) -> npt.NDArray:
    output = np.ascontiguousarray(values, dtype=dtype)
    output.setflags(write=False)
    return output


def _causal_transform(
    values: FloatArray,
    transform_id: FeatureTransformId,
    *,
    output_positions: IntArray | None = None,
) -> FloatArray:
    """Transform along accepted sessions; missing data breaks, never fills, state."""

    source = np.asarray(values, dtype=np.float64)
    if source.ndim < 1:
        raise PanelFeatureBoundaryError("alpha_research.feature_transform_axis_invalid")
    positions = (
        np.arange(source.shape[0], dtype=np.int64)
        if output_positions is None
        else np.asarray(output_positions, dtype=np.int64)
    )
    if (
        positions.ndim != 1
        or positions.size == 0
        or np.any(positions < 0)
        or np.any(positions >= source.shape[0])
        or np.any(positions[1:] <= positions[:-1])
    ):
        raise PanelFeatureBoundaryError("alpha_research.feature_transform_positions_invalid")
    result = np.full((len(positions), *source.shape[1:]), np.nan, dtype=np.float64)
    if transform_id == "current":
        result[:] = source[positions]
    else:
        lookback = _transform_lookback(transform_id)
        if transform_id.startswith("lag"):
            admitted = positions >= lookback
            result[admitted] = source[positions[admitted] - lookback]
        elif transform_id.startswith("delta"):
            admitted = positions >= lookback
            current = positions[admitted]
            result[admitted] = source[current] - source[current - lookback]
        elif transform_id.startswith(("mean", "max", "min", "std")):
            # One window shape, four statistics. Each requires the whole window
            # finite, so a partially observed window is absent rather than
            # computed over whatever happens to be there.
            kind = transform_id.rstrip("0123456789")
            statistics: dict[str, Callable[[FloatArray], FloatArray]] = {
                "mean": lambda window: cast(FloatArray, np.mean(window, axis=0)),
                "max": lambda window: cast(FloatArray, np.max(window, axis=0)),
                "min": lambda window: cast(FloatArray, np.min(window, axis=0)),
                "std": lambda window: cast(FloatArray, np.std(window, axis=0, ddof=1)),
            }
            statistic = statistics[kind]
            if kind == "std" and lookback < 2:
                raise PanelFeatureBoundaryError("alpha_research.feature_transform_not_installed")
            for output_position, position in enumerate(positions):
                if position < lookback - 1:
                    continue
                window = source[position - lookback + 1 : position + 1]
                finite = np.isfinite(window).all(axis=0)
                value = statistic(np.where(np.isfinite(window), window, 0.0))
                result[output_position] = np.where(finite, value, np.nan)
        elif transform_id.startswith("EWMA"):
            alpha = 2.0 / (lookback + 1.0)
            state = np.full(source.shape[1:], np.nan, dtype=np.float64)
            consecutive = np.zeros(source.shape[1:], dtype=np.int64)
            output_by_position = {int(value): index for index, value in enumerate(positions)}
            for position in range(int(positions[-1]) + 1):
                current = source[position]
                finite = np.isfinite(current)
                state = np.where(
                    finite,
                    np.where(np.isfinite(state), alpha * current + (1.0 - alpha) * state, current),
                    np.nan,
                )
                consecutive = np.where(finite, consecutive + 1, 0)
                selected_output_position = output_by_position.get(position)
                if selected_output_position is not None:
                    result[selected_output_position] = np.where(
                        consecutive >= lookback, state, np.nan
                    )
        else:
            raise PanelFeatureBoundaryError("alpha_research.feature_transform_not_installed")
    return cast(FloatArray, _readonly(result, dtype=np.dtype(np.float64)))


def _causal_availability(finite_source: BoolArray, transform_id: FeatureTransformId) -> BoolArray:
    """Derive exact transform availability without materializing numerical values."""

    if finite_source.ndim != 2:
        raise PanelFeatureBoundaryError("alpha_research.feature_availability_axis_invalid")
    sessions = finite_source.shape[0]
    result = np.zeros(finite_source.shape, dtype=np.bool_)
    if transform_id == "current":
        result[:] = finite_source
    else:
        lookback = _transform_lookback(transform_id)
        if transform_id.startswith("lag"):
            result[lookback:] = finite_source[:-lookback]
        elif transform_id.startswith("delta"):
            result[lookback:] = finite_source[lookback:] & finite_source[:-lookback]
        elif transform_id.startswith(("mean", "max", "min", "std")):
            # Availability is the window shape, which the four statistics share.
            cumulative = np.vstack(
                (
                    np.zeros((1, finite_source.shape[1]), dtype=np.int64),
                    np.cumsum(finite_source, axis=0, dtype=np.int64),
                )
            )
            counts = cumulative[lookback:] - cumulative[:-lookback]
            result[lookback - 1 :] = counts == lookback
        elif transform_id.startswith("EWMA"):
            consecutive = np.zeros(finite_source.shape[1], dtype=np.int64)
            for position in range(sessions):
                consecutive = np.where(finite_source[position], consecutive + 1, 0)
                result[position] = consecutive >= lookback
        else:
            raise PanelFeatureBoundaryError("alpha_research.feature_transform_not_installed")
    return cast(BoolArray, _readonly(result, dtype=np.dtype(np.bool_)))


def _formula_surface(source: PanelFeatureSourceArrays) -> FloatArray:
    result = np.full(source.raw_formula_values.shape, np.nan, dtype=np.float64)
    for factor_position, factor_id in enumerate(source.ordered_factor_ids):
        descriptor = formula_pretransform_descriptor(factor_id)
        if descriptor["owner"] != "feature_engine.formula":
            raise PanelFeatureBoundaryError("alpha_research.formula_pretransform_owner_invalid")
        result[:, :, factor_position] = formula_pretransform(
            factor_id, source.raw_formula_values[:, :, factor_position]
        )
    return cast(FloatArray, _readonly(result, dtype=np.dtype(np.float64)))


def _stock_role_formula_values(
    *, role: FeatureRoleSelection, source: PanelFeatureSourceArrays, formula: FloatArray
) -> FloatArray:
    """Project Formula-owned pretransforms onto one stock role's declared axis."""

    factor_position = {value: index for index, value in enumerate(source.ordered_factor_ids)}
    result = np.empty((*formula.shape[:2], len(role.source_ids)), dtype=np.float64)
    for output_position, factor_id in enumerate(role.source_ids):
        result[:, :, output_position] = formula[:, :, factor_position[factor_id]]
    return cast(FloatArray, _readonly(result, dtype=np.dtype(np.float64)))


def _latest_matured_positions(source: PanelFeatureSourceArrays) -> IntArray:
    result: IntArray = np.full(len(source.formation_sessions), -1, dtype=np.int64)
    holding = np.asarray(source.holding_end_sessions, dtype=object)
    for position, formation in enumerate(source.formation_sessions):
        eligible = np.flatnonzero(holding < formation)
        if eligible.size:
            result[position] = int(eligible[-1])
    return cast(IntArray, _readonly(result, dtype=np.dtype(np.int64)))


def _matured_lane(values: FloatArray, latest: IntArray) -> FloatArray:
    result = np.full(values.shape, np.nan, dtype=np.float64)
    eligible = latest >= 0
    result[eligible] = values[latest[eligible]]
    return cast(FloatArray, _readonly(result, dtype=np.dtype(np.float64)))


class ListingSectorAxis(Protocol):
    """The scale consumes an axis, not training targets or future outcomes."""

    @property
    def ordered_listing_ids(self) -> tuple[str, ...]: ...

    @property
    def ordered_sector_ids(self) -> tuple[str, ...]: ...

    @property
    def sector_by_listing_id(self) -> Mapping[str, str]: ...


def _sector_positions(source: ListingSectorAxis) -> tuple[IntArray, ...]:
    return tuple(
        cast(
            IntArray,
            _readonly(
                np.asarray(
                    [
                        position
                        for position, listing in enumerate(source.ordered_listing_ids)
                        if source.sector_by_listing_id[listing] == sector
                    ],
                    dtype=np.int64,
                ),
                dtype=np.dtype(np.int64),
            ),
        )
        for sector in source.ordered_sector_ids
    )


def _sector_residual_log_return(source: PanelFeatureSourceArrays) -> FloatArray:
    """Derive the raw Sector-residual lane without taking final demean authority.

    The Sector context owner publishes equal-weight raw simple returns.  This
    converts that declared lane to log-return units and subtracts it from each
    listing's raw execution log return.  The resulting raw residual is still
    bounded and Sector-demeaned exactly once *after* its causal temporal
    transform by the role's final preprocessing owner.
    """

    sector_simple = source.sector_context_values[:, :, 0]
    sector_log = np.full(sector_simple.shape, np.nan, dtype=np.float64)
    admitted = np.isfinite(sector_simple) & (sector_simple > -1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sector_log[admitted] = np.log1p(sector_simple[admitted])
    mapped = _map_sector_to_listings(
        cast(FloatArray, _readonly(sector_log[:, :, None], dtype=np.dtype(np.float64))),
        source,
        source.formation_sessions,
    )[:, :, 0]
    result = source.raw_log_execution_returns - mapped
    return cast(FloatArray, _readonly(result, dtype=np.dtype(np.float64)))


def _transformed_columns(
    *,
    role_id: FeatureRoleId,
    source_ids: tuple[str, ...],
    transform_ids_by_source: tuple[tuple[FeatureTransformId, ...], ...],
    values: FloatArray,
    output_positions: IntArray | None = None,
) -> tuple[tuple[str, ...], FloatArray]:
    if values.shape[-1] != len(source_ids):
        raise PanelFeatureBoundaryError("alpha_research.feature_source_column_axis_invalid")
    output_session_count = values.shape[0] if output_positions is None else len(output_positions)
    column_count = sum(len(value) for value in transform_ids_by_source)
    feature_ids: list[str] = []
    grouped_positions: dict[FeatureTransformId, list[tuple[int, int]]] = {}
    output_column = 0
    for source_position, (source_id, transforms) in enumerate(
        zip(source_ids, transform_ids_by_source, strict=True)
    ):
        for transform_id in transforms:
            feature_ids.append(f"{role_id}::{source_id}::{transform_id}")
            grouped_positions.setdefault(transform_id, []).append((source_position, output_column))
            output_column += 1
    if output_column != column_count:
        raise PanelFeatureBoundaryError("alpha_research.feature_transform_axis_invalid")

    result = np.empty(
        (output_session_count, *values.shape[1:-1], column_count),
        dtype=np.float64,
    )
    # A role commonly applies the same transform to tens of source columns.
    # Evaluate each transform once over its complete source block, then scatter
    # into the stable source-major feature axis. Arithmetic remains owned by
    # the same causal transform implementation.
    for transform_id, positions in grouped_positions.items():
        source_positions = [value[0] for value in positions]
        output_columns = [value[1] for value in positions]
        transformed = _causal_transform(
            values[..., source_positions],
            transform_id,
            output_positions=output_positions,
        )
        result[..., output_columns] = transformed
    return tuple(feature_ids), cast(FloatArray, _readonly(result, dtype=np.dtype(np.float64)))


def scale_alpha_stock_cross_section(
    values: FloatArray,
    *,
    source: ListingSectorAxis,
    sector_neutral: bool,
    reference_eligible: npt.NDArray[np.bool_] | None = None,
    sessions: Sequence[date] | None = None,
) -> FloatArray:
    """MAD bound then exactly one declared center/demean and sample-std Z.

    A Sector-neutral scale of rows whose `sessions` a reclassification splits scales each run
    of sessions by the Sectors in force there; a scale of one run is the one it was.

    Args:
        values: The rows to scale, one a session.
        source: The listing and Sector axes, with the map or the history the rows read.
        sector_neutral: Whether the center is each Sector's, or the universe's.
        reference_eligible: Each row's members the center and scale are taken over.
        sessions: The rows' sessions, when a Sector history may split them.

    Returns:
        The scaled rows, read-only.
    """
    runs = (
        sector_slices(source.sector_by_listing_id, sessions)
        if sector_neutral and sessions is not None
        else ()
    )
    if len(runs) > 1:
        return cast(
            FloatArray,
            _readonly(
                np.concatenate(
                    [
                        _scale_cross_section(
                            values[rows],
                            source=_RunSectorAxis(source, mapping),
                            sector_neutral=True,
                            reference_eligible=(
                                None if reference_eligible is None else reference_eligible[rows]
                            ),
                        )
                        for rows, mapping in runs
                    ],
                    axis=0,
                ),
                dtype=np.dtype(np.float64),
            ),
        )
    return _scale_cross_section(
        values,
        source=source,
        sector_neutral=sector_neutral,
        reference_eligible=reference_eligible,
    )


@dataclass(frozen=True, slots=True)
class _RunSectorAxis:
    """One run's axis: the source's listings and Sectors, the run's map."""

    axis: ListingSectorAxis
    mapping: Mapping[str, str]

    @property
    def ordered_listing_ids(self) -> tuple[str, ...]:
        return self.axis.ordered_listing_ids

    @property
    def ordered_sector_ids(self) -> tuple[str, ...]:
        return self.axis.ordered_sector_ids

    @property
    def sector_by_listing_id(self) -> Mapping[str, str]:
        return self.mapping


def _scale_cross_section(
    values: FloatArray,
    *,
    source: ListingSectorAxis,
    sector_neutral: bool,
    reference_eligible: npt.NDArray[np.bool_] | None,
) -> FloatArray:
    if values.ndim != 3:
        raise PanelFeatureBoundaryError("alpha_research.cross_section_scale_axis_invalid")
    sessions, listings, columns = values.shape
    if reference_eligible is not None and (
        reference_eligible.shape != (sessions, listings) or reference_eligible.dtype != np.bool_
    ):
        raise PanelFeatureBoundaryError("alpha_research.cross_section_reference_axis_invalid")
    output = np.full(values.shape, np.nan, dtype=np.float64)
    sector_axes = _sector_positions(source) if sector_neutral else ()
    # Fit compact reference populations, never zero-pad them with query names.
    # C-ordered reference reductions make future membership/group boundaries
    # irrelevant to earlier results. None retains the frozen legacy layout.
    groups = None
    if reference_eligible is not None:
        masks, inverse = np.unique(reference_eligible, axis=0, return_inverse=True)
        groups = tuple(
            (np.flatnonzero(inverse == i), np.flatnonzero(mask)) for i, mask in enumerate(masks)
        )
    for start in range(0, columns, 16):
        stop = min(columns, start + 16)
        if groups is None:
            rows = values[:, :, start:stop].transpose(0, 2, 1).reshape(-1, listings)
            scaled = _scale_stock_rows(rows, sector_axes=sector_axes)
            output[:, :, start:stop] = scaled.reshape(sessions, stop - start, listings).transpose(
                0, 2, 1
            )
            continue
        for days, members in groups:
            if len(members) < 2:
                continue
            covered = values[days, :, start:stop].transpose(0, 2, 1).reshape(-1, listings)
            reference = np.ascontiguousarray(covered[:, members])
            reference_sectors = tuple(np.flatnonzero(np.isin(members, p)) for p in sector_axes)
            scaled = _scale_stock_rows(
                reference,
                sector_axes=reference_sectors,
                coverage=covered,
                coverage_sectors=sector_axes,
                stable_reduction=True,
            )
            output[days, :, start:stop] = scaled.reshape(
                len(days), stop - start, listings
            ).transpose(0, 2, 1)
    return cast(FloatArray, _readonly(output, dtype=np.dtype(np.float64)))


def _scale_stock_rows(
    rows: FloatArray,
    *,
    sector_axes: tuple[IntArray, ...],
    coverage: FloatArray | None = None,
    coverage_sectors: tuple[IntArray, ...] = (),
    stable_reduction: bool = False,
) -> FloatArray:
    """One estimator and transform; query rows never enter its fitted moments."""
    bounded, _median, _mad, lower, upper = median_mad_winsor(
        rows, multiplier=3.5, mad_scale=MAD_SCALE
    )
    covered = bounded if coverage is None else np.clip(coverage, lower[:, None], upper[:, None])
    reference_centered = np.full_like(bounded, np.nan)
    centered = reference_centered if coverage is None else np.full_like(covered, np.nan)
    axes = (
        tuple(zip(sector_axes, coverage_sectors or sector_axes, strict=True))
        if sector_axes
        else ((None, None),)
    )
    for positions, query_positions in axes:
        block = bounded if positions is None else bounded[:, positions]
        if stable_reduction:
            block = np.ascontiguousarray(block)
        finite = np.isfinite(block)
        count = np.sum(finite, axis=1)
        mean = np.divide(
            np.sum(np.where(finite, block, 0.0), axis=1),
            count,
            out=np.full(block.shape[0], np.nan, dtype=np.float64),
            where=count > 0,
        )
        if positions is None:
            reference_centered = bounded - mean[:, None]
            centered = reference_centered if coverage is None else covered - mean[:, None]
        else:
            reference_centered[:, positions] = block - mean[:, None]
            if coverage is not None:
                centered[:, query_positions] = covered[:, query_positions] - mean[:, None]
    if stable_reduction:
        reference_centered = np.ascontiguousarray(reference_centered)
    finite_centered = np.isfinite(reference_centered)
    count = np.sum(finite_centered, axis=1)
    variance = np.divide(
        np.sum(np.where(finite_centered, np.square(reference_centered), 0.0), axis=1),
        count - 1,
        out=np.full(reference_centered.shape[0], np.nan, dtype=np.float64),
        where=count >= 2,
    )
    dispersion = np.sqrt(variance)
    scaled = np.divide(
        centered,
        dispersion[:, None],
        out=np.full_like(centered, np.nan),
        where=np.isfinite(centered) & (dispersion[:, None] > 0.0),
    )
    constant = (count >= 2) & np.isfinite(dispersion) & (dispersion == 0.0)
    scaled[constant] = np.where(np.isfinite(centered[constant]), 0.0, np.nan)
    scaled[~np.isfinite(dispersion) & ~constant] = np.nan
    return cast(FloatArray, scaled)


def _map_sector_to_listings(
    values: FloatArray, source: PanelFeatureSourceArrays, sessions: Sequence[date]
) -> FloatArray:
    # Each row takes its session's Sector of each listing.
    codes = sector_codes(
        source.sector_by_listing_id,
        sessions,
        source.ordered_listing_ids,
        source.ordered_sector_ids,
    )
    rows: IntArray = np.arange(len(sessions), dtype=np.int64)[:, None]
    return cast(FloatArray, _readonly(values[rows, codes, :], dtype=np.dtype(np.float64)))


def _broadcast_market(values: FloatArray, source: PanelFeatureSourceArrays) -> FloatArray:
    return cast(
        FloatArray,
        _readonly(
            np.repeat(values[:, None, :], len(source.ordered_listing_ids), axis=1),
            dtype=np.dtype(np.float64),
        ),
    )


def _selected_context_values(
    *, role: FeatureRoleSelection, source: PanelFeatureSourceArrays
) -> FloatArray:
    """Project a context role onto its declared source axis before transforms."""

    if role.role_id == "SECTOR_CONTEXT":
        available = SECTOR_CONTEXT_SOURCE_IDS
        values = source.sector_context_values
    elif role.role_id == "MARKET_CONTEXT":
        available = (
            BASE_MARKET_CONTEXT_SOURCE_IDS
            if source.market_context_values.shape[1] == len(BASE_MARKET_CONTEXT_SOURCE_IDS)
            else (*BASE_MARKET_CONTEXT_SOURCE_IDS, *OBSERVED_MARKET_SOURCE_IDS)
        )
        values = source.market_context_values
    else:
        raise PanelFeatureBoundaryError("alpha_research.feature_context_role_invalid")
    positions = {value: index for index, value in enumerate(available)}
    if not set(role.source_ids).issubset(positions):
        raise PanelFeatureBoundaryError("alpha_research.feature_context_source_not_installed")
    selected = values[..., [positions[value] for value in role.source_ids]]
    return cast(FloatArray, _readonly(selected, dtype=np.dtype(np.float64)))


def _output_sessions(
    source: PanelFeatureSourceArrays, output_positions: IntArray | None
) -> tuple[date, ...]:
    """The sessions of a role surface's rows: every formation, or the selected ones."""
    sessions: tuple[date, ...] = (
        source.formation_sessions
        if output_positions is None
        else tuple(source.formation_sessions[int(index)] for index in output_positions)
    )
    return sessions


def _category_values(
    source: PanelFeatureSourceArrays, *, output_positions: IntArray | None = None
) -> FloatArray:
    # Each session's Sector of each listing.
    sessions = (
        source.formation_sessions
        if output_positions is None
        else tuple(source.formation_sessions[int(index)] for index in output_positions)
    )
    codes = sector_codes(
        source.sector_by_listing_id,
        sessions,
        source.ordered_listing_ids,
        source.ordered_sector_ids,
    )
    one_hot: FloatArray = np.zeros(
        (len(sessions), len(source.ordered_listing_ids), len(source.ordered_sector_ids)),
        dtype=np.float64,
    )
    one_hot[
        np.arange(len(sessions), dtype=np.int64)[:, None],
        np.arange(len(source.ordered_listing_ids), dtype=np.int64)[None, :],
        codes,
    ] = 1.0
    return cast(FloatArray, _readonly(one_hot, dtype=np.dtype(np.float64)))


def _build_role_surface(
    *,
    role: FeatureRoleSelection,
    source: PanelFeatureSourceArrays,
    formula: FloatArray,
    latest_matured: IntArray,
    output_positions: IntArray | None = None,
) -> RoleSurface:
    reference = source.reference_eligible
    if reference is not None and output_positions is not None:
        reference = reference[output_positions]
    if role.role_id in {
        "RELATIVE_STOCK_CROSS_SECTION",
        "NON_NEUTRAL_STOCK_CROSS_SECTION",
    }:
        selected = _stock_role_formula_values(role=role, source=source, formula=formula)
        feature_ids, values = _transformed_columns(
            role_id=role.role_id,
            source_ids=role.source_ids,
            transform_ids_by_source=role.source_transform_ids,
            values=selected,
            output_positions=output_positions,
        )
        final = scale_alpha_stock_cross_section(
            values,
            source=source,
            sector_neutral=role.role_id == "RELATIVE_STOCK_CROSS_SECTION",
            reference_eligible=reference,
            sessions=_output_sessions(source, output_positions),
        )
        return RoleSurface(
            role=role,
            ordered_feature_ids=feature_ids,
            raw_values=final,
            native_entity_axis_hash=str(canonical_hash(list(source.ordered_listing_ids))),
        )
    if role.role_id == "ABSOLUTE_STOCK_TIME_SERIES_STATE":
        selected = _stock_role_formula_values(role=role, source=source, formula=formula)
        feature_ids, values = _transformed_columns(
            role_id=role.role_id,
            source_ids=role.source_ids,
            transform_ids_by_source=role.source_transform_ids,
            values=selected,
            output_positions=output_positions,
        )
        return RoleSurface(
            role=role,
            ordered_feature_ids=feature_ids,
            raw_values=values,
            native_entity_axis_hash=str(canonical_hash(list(source.ordered_listing_ids))),
        )
    if role.role_id == "MATURED_STOCK_OUTCOME_HISTORY":
        # The third lane is a raw stock-minus-Sector log return.  Its only
        # cross-sectional Sector demean happens in final preprocessing after
        # temporal transformation; no residual-before-bound/re-demean remains.
        base = np.stack(
            (
                _matured_lane(source.total_return_target_z, latest_matured),
                _matured_lane(source.raw_log_execution_returns, latest_matured),
                _matured_lane(_sector_residual_log_return(source), latest_matured),
            ),
            axis=2,
        )
        feature_ids, transformed = _transformed_columns(
            role_id=role.role_id,
            source_ids=role.source_ids,
            transform_ids_by_source=role.source_transform_ids,
            values=base,
            output_positions=output_positions,
        )
        per_source = len(feature_ids) // len(role.source_ids)
        first = scale_alpha_stock_cross_section(
            transformed[:, :, : 2 * per_source],
            source=source,
            sector_neutral=False,
            reference_eligible=reference,
        )
        residual = scale_alpha_stock_cross_section(
            transformed[:, :, 2 * per_source :],
            source=source,
            sector_neutral=True,
            reference_eligible=reference,
            sessions=_output_sessions(source, output_positions),
        )
        values = np.concatenate((first, residual), axis=2)
        return RoleSurface(
            role=role,
            ordered_feature_ids=feature_ids,
            raw_values=cast(FloatArray, _readonly(values, dtype=np.dtype(np.float64))),
            native_entity_axis_hash=str(canonical_hash(list(source.ordered_listing_ids))),
        )
    if role.role_id == "SECTOR_CONTEXT":
        selected = _selected_context_values(role=role, source=source)
        feature_ids, values = _transformed_columns(
            role_id=role.role_id,
            source_ids=role.source_ids,
            transform_ids_by_source=role.source_transform_ids,
            values=selected,
            output_positions=output_positions,
        )
        return RoleSurface(
            role=role,
            ordered_feature_ids=feature_ids,
            raw_values=values,
            native_entity_axis_hash=str(canonical_hash(list(source.ordered_sector_ids))),
        )
    if role.role_id == "MARKET_CONTEXT":
        selected = _selected_context_values(role=role, source=source)
        feature_ids, values = _transformed_columns(
            role_id=role.role_id,
            source_ids=role.source_ids,
            transform_ids_by_source=role.source_transform_ids,
            values=selected,
            output_positions=output_positions,
        )
        return RoleSurface(
            role=role,
            ordered_feature_ids=feature_ids,
            raw_values=values,
            native_entity_axis_hash=str(canonical_hash(["MARKET"])),
        )
    values = _category_values(source, output_positions=output_positions)
    return RoleSurface(
        role=role,
        ordered_feature_ids=tuple(f"{role.role_id}::{value}::current" for value in role.source_ids),
        raw_values=values,
        native_entity_axis_hash=str(canonical_hash(list(source.ordered_listing_ids))),
    )


@dataclass(frozen=True, slots=True)
class PanelFeaturePlan:
    """Retain admitted feature views, causal source arrays and common-row preflight evidence.

    The plan holds the formula surface, latest matured positions, role descriptors and
    common/aggregation masks used by deterministic materialization.
    """

    source: PanelFeatureSourceArrays
    catalog: PanelFeatureViewCatalog
    selected_methods: tuple[PanelFeatureViewRecipe, ...]
    formula_surface: FloatArray
    latest_matured_positions: IntArray
    role_descriptors_by_selection_hash: MappingProxyType[str, RoleSurfaceDescriptor]
    preflight: PanelFeaturePreflight
    common_row_mask: BoolArray
    aggregation_row_mask: BoolArray


def _role_descriptor(
    *, role: FeatureRoleSelection, source: PanelFeatureSourceArrays
) -> RoleSurfaceDescriptor:
    feature_ids = tuple(
        f"{role.role_id}::{source_id}::{transform_id}"
        for source_id, transforms in zip(role.source_ids, role.source_transform_ids, strict=True)
        for transform_id in transforms
    )
    native_ids = (
        source.ordered_sector_ids
        if role.role_id == "SECTOR_CONTEXT"
        else (("MARKET",) if role.role_id == "MARKET_CONTEXT" else source.ordered_listing_ids)
    )
    return RoleSurfaceDescriptor(
        role=role,
        ordered_feature_ids=feature_ids,
        native_entity_axis_hash=str(canonical_hash(list(native_ids))),
    )


def _role_availability(
    *,
    role: FeatureRoleSelection,
    source: PanelFeatureSourceArrays,
    formula: FloatArray,
    latest_matured: IntArray,
) -> BoolArray:
    """Preflight exact finite-row availability without a full transformed cube."""

    session_count = len(source.formation_sessions)
    listing_count = len(source.ordered_listing_ids)
    # Each session's Sector of each listing.
    listing_sector_codes = sector_codes(
        source.sector_by_listing_id,
        source.formation_sessions,
        source.ordered_listing_ids,
        source.ordered_sector_ids,
    )
    session_rows: IntArray = np.arange(session_count, dtype=np.int64)[:, None]
    complete: BoolArray = np.ones((session_count, listing_count), dtype=np.bool_)
    if role.role_id == "SECTOR_CATEGORY":
        return cast(BoolArray, _readonly(complete, dtype=np.dtype(np.bool_)))

    if role.role_id in {
        "RELATIVE_STOCK_CROSS_SECTION",
        "NON_NEUTRAL_STOCK_CROSS_SECTION",
        "ABSOLUTE_STOCK_TIME_SERIES_STATE",
    }:
        selected_formula = _stock_role_formula_values(role=role, source=source, formula=formula)
        source_finite = tuple(
            np.isfinite(selected_formula[:, :, position])
            for position in range(len(role.source_ids))
        )
    elif role.role_id == "MATURED_STOCK_OUTCOME_HISTORY":
        matured_target: BoolArray = np.zeros((session_count, listing_count), dtype=np.bool_)
        matured_raw = np.zeros_like(matured_target)
        matured_residual = np.zeros_like(matured_target)
        eligible = latest_matured >= 0
        matured_target[eligible] = np.isfinite(
            source.total_return_target_z[latest_matured[eligible]]
        )
        matured_raw[eligible] = np.isfinite(
            source.raw_log_execution_returns[latest_matured[eligible]]
        )
        sector_simple = source.sector_context_values[:, :, 0]
        sector_log_finite = np.isfinite(sector_simple) & (sector_simple > -1.0)
        mapped_sector_finite = sector_log_finite[session_rows, listing_sector_codes]
        matured_residual[eligible] = (
            np.isfinite(source.raw_log_execution_returns[latest_matured[eligible]])
            & mapped_sector_finite[latest_matured[eligible]]
        )
        source_finite = (matured_target, matured_raw, matured_residual)
    elif role.role_id == "SECTOR_CONTEXT":
        selected_context = _selected_context_values(role=role, source=source)
        source_finite = tuple(
            np.isfinite(selected_context[:, :, position])
            for position in range(len(role.source_ids))
        )
    elif role.role_id == "MARKET_CONTEXT":
        selected_context = _selected_context_values(role=role, source=source)
        source_finite = tuple(
            np.isfinite(selected_context[:, position])[:, None]
            for position in range(len(role.source_ids))
        )
    else:
        raise PanelFeatureBoundaryError("alpha_research.feature_role_not_installed")

    for finite_lane, transforms in zip(source_finite, role.source_transform_ids, strict=True):
        for transform_id in transforms:
            transformed = _causal_availability(cast(BoolArray, finite_lane), transform_id)
            if role.role_id == "SECTOR_CONTEXT":
                mapped = transformed[session_rows, listing_sector_codes]
            elif role.role_id == "MARKET_CONTEXT":
                mapped = np.repeat(transformed, listing_count, axis=1)
            else:
                mapped = transformed
            if role.role_id in {
                "RELATIVE_STOCK_CROSS_SECTION",
                "NON_NEUTRAL_STOCK_CROSS_SECTION",
                "MATURED_STOCK_OUTCOME_HISTORY",
            }:
                reference = (
                    mapped
                    if source.reference_eligible is None
                    else mapped & source.reference_eligible
                )
                mapped = mapped & (np.sum(reference, axis=1) >= 2)[:, None]
            complete &= mapped
    return cast(BoolArray, _readonly(complete, dtype=np.dtype(np.bool_)))


def _listing_row_axis(
    source: PanelFeatureSourceArrays, mask: BoolArray
) -> tuple[tuple[date, ...], tuple[str, ...], str]:
    if mask.shape != (
        len(source.formation_sessions),
        len(source.ordered_listing_ids),
    ):
        raise PanelFeatureBoundaryError("alpha_research.feature_row_axis_shape_invalid")
    session_positions, listing_positions = np.nonzero(mask)
    selected_sessions = tuple(
        source.formation_sessions[int(position)] for position in session_positions
    )
    selected_listings = tuple(
        source.ordered_listing_ids[int(position)] for position in listing_positions
    )
    return (
        selected_sessions,
        selected_listings,
        str(
            canonical_hash(
                [
                    (session.isoformat(), listing)
                    for session, listing in zip(selected_sessions, selected_listings, strict=True)
                ]
            )
        ),
    )


def preflight_panel_feature_plan(
    *,
    source: PanelFeatureSourceArrays,
    selected_method_ids: tuple[InstalledPanelViewId, ...],
    maximum_aggregation_span: int,
) -> PanelFeaturePlan:
    """Construct every common axis and retention count without fitting a scaler."""
    if selected_method_ids != tuple(dict.fromkeys(selected_method_ids)):
        raise PanelFeatureBoundaryError("alpha_research.feature_view_selection_duplicated")
    if maximum_aggregation_span not in {1, 21, 42, 63}:
        raise PanelFeatureBoundaryError("alpha_research.aggregation_span_not_installed")
    if set(selected_method_ids).intersection(SPARSE_SESSION_AMPLITUDE_VIEW_IDS) and not set(
        SPARSE_SESSION_AMPLITUDE_REQUIRED_FORMULA_IDS
    ).issubset(source.ordered_factor_ids):
        raise PanelFeatureBoundaryError(
            "alpha_research.sparse_session_amplitude_required_formula_missing"
        )
    catalog = build_installed_panel_feature_view_catalog(
        factor_ids=source.ordered_factor_ids,
        absolute_state_factor_ids=source.absolute_state_factor_ids,
        sector_ids=source.ordered_sector_ids,
    )
    methods = tuple(catalog.resolve(value) for value in selected_method_ids)
    formula = _formula_surface(source)
    latest = _latest_matured_positions(source)
    descriptors: dict[str, RoleSurfaceDescriptor] = {}
    completeness_by_selection_hash: dict[str, BoolArray] = {}
    for method in methods:
        for role in method.roles:
            if role.selection_hash in descriptors:
                continue
            descriptor = _role_descriptor(role=role, source=source)
            complete = _role_availability(
                role=role,
                source=source,
                formula=formula,
                latest_matured=latest,
            )
            completeness_by_selection_hash[role.selection_hash] = cast(
                BoolArray, _readonly(complete, dtype=np.dtype(np.bool_))
            )
            descriptors[role.selection_hash] = descriptor
    total_rows = len(source.formation_sessions) * len(source.ordered_listing_ids)
    target_complete = np.isfinite(source.total_return_target_z) & np.isfinite(
        source.raw_simple_execution_returns
    )
    if source.reference_eligible is not None:
        target_complete &= source.reference_eligible
    method_masks: dict[str, BoolArray] = {}
    retention: list[ViewRetentionEvidence] = []
    maximum_lookback = 0
    for method in methods:
        ordered_method_feature_ids = tuple(
            feature_id
            for role in method.roles
            for feature_id in descriptors[role.selection_hash].ordered_feature_ids
        )
        if ordered_method_feature_ids != tuple(dict.fromkeys(ordered_method_feature_ids)):
            raise PanelFeatureBoundaryError("alpha_research.feature_view_axis_duplicate")
        mask = np.array(target_complete, copy=True, dtype=np.bool_)
        excluded: dict[str, int] = {}
        excluded_sessions: dict[str, int] = {}
        if source.reference_eligible is not None:
            excluded["REFERENCE_ELIGIBILITY"] = int(np.sum(~source.reference_eligible))
            excluded_sessions["REFERENCE_ELIGIBILITY"] = int(
                np.sum(~source.reference_eligible.any(axis=1))
            )
        for role in method.roles:
            complete = completeness_by_selection_hash[role.selection_hash]
            excluded[role.role_id] = int(np.sum(~complete))
            excluded_sessions[role.role_id] = int(np.sum(~np.any(complete, axis=1)))
            mask &= complete
            maximum_lookback = max(
                maximum_lookback,
                *(
                    _transform_lookback(transform)
                    for values_by_source in role.source_transform_ids
                    for transform in values_by_source
                ),
            )
        method_masks[method.method_id] = mask
        excluded["SCORE_FILTER_ROW_AXIS"] = 0
        excluded_sessions["SCORE_FILTER_ROW_AXIS"] = 0
        axis = _listing_row_axis(source, mask)
        retained = int(np.sum(mask))
        retention.append(
            ViewRetentionEvidence(
                method_id=method.method_id,
                total_row_count=total_rows,
                retained_row_count=retained,
                retained_fraction=retained / total_rows,
                excluded_by_role=excluded,
                excluded_session_count_by_role=excluded_sessions,
                row_axis_hash=axis[2],
            )
        )
    common = np.array(target_complete, copy=True, dtype=np.bool_)
    for mask in method_masks.values():
        common &= mask
    common.setflags(write=False)
    aggregation_common = cast(BoolArray, common)
    common_axis = _listing_row_axis(source, aggregation_common)
    common_sessions = tuple(sorted(set(common_axis[0])))
    preflight = PanelFeaturePreflight.create(
        selected_method_ids=selected_method_ids,
        feature_count_by_method={
            method.method_id: len(
                tuple(
                    feature_id
                    for role in method.roles
                    for feature_id in descriptors[role.selection_hash].ordered_feature_ids
                )
            )
            for method in methods
        },
        ordered_feature_axis_hash_by_method={
            method.method_id: str(
                canonical_hash(
                    [
                        feature_id
                        for role in method.roles
                        for feature_id in descriptors[role.selection_hash].ordered_feature_ids
                    ]
                )
            )
            for method in methods
        },
        common_row_retention_by_method=tuple(retention),
        common_row_axis_hash=common_axis[2],
        common_session_axis_hash=str(
            canonical_hash([value.isoformat() for value in common_sessions])
        ),
        maximum_lookback_sessions=maximum_lookback,
        maximum_aggregation_span=maximum_aggregation_span,
    )
    return PanelFeaturePlan(
        source=source,
        catalog=catalog,
        selected_methods=methods,
        formula_surface=formula,
        latest_matured_positions=latest,
        role_descriptors_by_selection_hash=MappingProxyType(descriptors),
        preflight=preflight,
        common_row_mask=common,
        aggregation_row_mask=aggregation_common,
    )


def _session_positions(source: PanelFeatureSourceArrays, sessions: tuple[date, ...]) -> IntArray:
    indexed = {value: position for position, value in enumerate(source.formation_sessions)}
    try:
        values: IntArray = np.asarray([indexed[value] for value in sessions], dtype=np.int64)
    except KeyError as error:
        raise PanelFeatureBoundaryError(
            "alpha_research.feature_fold_session_unavailable"
        ) from error
    if sessions != tuple(sorted(set(sessions))):
        raise PanelFeatureBoundaryError("alpha_research.feature_fold_sessions_unordered")
    return cast(IntArray, _readonly(values, dtype=np.dtype(np.int64)))


def _fit_temporal_scale(
    values: FloatArray,
    training_positions: IntArray,
    *,
    training_mask: BoolArray | None = None,
) -> tuple[FloatArray, dict[str, tuple[float | None, ...]], tuple[ScaleState, ...]]:
    """Fit per transformed column on its native, equally weighted training rows."""

    columns = values.shape[-1]
    center = np.full(columns, np.nan, dtype=np.float64)
    scale = np.full(columns, np.nan, dtype=np.float64)
    states: list[ScaleState] = []
    if training_mask is not None and training_mask.shape != values[training_positions].shape[:-1]:
        raise PanelFeatureBoundaryError("alpha_research.feature_scale_training_mask_invalid")
    for column in range(columns):
        training = values[training_positions, ..., column].reshape(-1)
        if training_mask is not None:
            training = training[training_mask.reshape(-1)]
        finite = training[np.isfinite(training)]
        if finite.size == 0:
            states.append("INVALID_NONFINITE_SCALE")
            continue
        median = float(np.median(finite))
        normalized_mad = float(np.median(np.abs(finite - median)) * MAD_SCALE)
        if not np.isfinite(median) or not np.isfinite(normalized_mad):
            states.append("INVALID_NONFINITE_SCALE")
            continue
        center[column] = median
        if normalized_mad == 0.0:
            scale[column] = 0.0
            states.append("CONSTANT_TRAINING_FEATURE")
        elif normalized_mad > 0.0:
            scale[column] = normalized_mad
            states.append("VALID_SCALE")
        else:
            states.append("INVALID_NONFINITE_SCALE")
    if "INVALID_NONFINITE_SCALE" in states:
        raise PanelFeatureBoundaryError("alpha_research.feature_scale_invalid_nonfinite")
    output = apply_frozen_temporal_scale(values, center=center, scale=scale)
    parameters: dict[str, tuple[float | None, ...]] = {
        "center": tuple(float(value) for value in center),
        "scale": tuple(float(value) for value in scale),
        "near_constant_threshold": (_NEAR_CONSTANT_THRESHOLD,),
    }
    return (
        cast(FloatArray, _readonly(output, dtype=np.dtype(np.float64))),
        parameters,
        tuple(states),
    )


def apply_frozen_temporal_scale(
    values: FloatArray, *, center: npt.ArrayLike, scale: npt.ArrayLike
) -> FloatArray:
    """Apply admitted training statistics without fitting on prediction rows."""
    center, scale = np.asarray(center, dtype=np.float64), np.asarray(scale, dtype=np.float64)
    if (
        center.shape != (values.shape[-1],)
        or scale.shape != center.shape
        or not np.isfinite(center).all()
        or not np.isfinite(scale).all()
        or np.any(scale < 0.0)
    ):
        raise PanelFeatureBoundaryError("alpha_research.frozen_scale_invalid")
    output = np.full(values.shape, np.nan, dtype=np.float64)
    for column in range(len(center)):
        finite = np.isfinite(values[..., column])
        if scale[column] == 0.0:
            output[..., column] = np.where(finite, 0.0, np.nan)
            continue
        left = center[column] - 3.5 * scale[column]
        right = center[column] + 3.5 * scale[column]
        bounded = np.clip(values[..., column], left, right)
        output[..., column] = (bounded - center[column]) / scale[column]
    return cast(FloatArray, _readonly(output, dtype=np.dtype(np.float64)))


def _seal_fold_constant_columns(
    *,
    values: FloatArray,
    training_positions: IntArray,
    training_mask: BoolArray,
    continuous: bool,
) -> tuple[FloatArray, tuple[ScaleState, ...], int]:
    """Let the fold owner, never an adapter, project legal constants to zero."""

    if not continuous:
        return values, ("VALID_SCALE",) * values.shape[2], 0
    flattened = values[training_positions].reshape(-1, values.shape[2])
    selected = flattened[training_mask.reshape(-1)]
    if selected.size == 0 or not np.isfinite(selected).all():
        raise PanelFeatureBoundaryError("alpha_research.feature_fold_matrix_nonfinite")
    standard_deviation = np.std(selected, axis=0, ddof=1)
    if not np.isfinite(standard_deviation).all():
        raise PanelFeatureBoundaryError("alpha_research.feature_scale_invalid_nonfinite")
    constant_columns = np.flatnonzero(standard_deviation == 0.0)
    states: tuple[ScaleState, ...] = tuple(
        "CONSTANT_TRAINING_FEATURE" if value == 0.0 else "VALID_SCALE"
        for value in standard_deviation
    )
    near = int(np.sum((standard_deviation > 0.0) & (standard_deviation < _NEAR_CONSTANT_THRESHOLD)))
    if constant_columns.size == 0:
        return values, states, near
    output = np.array(values, copy=True, dtype=np.float64)
    for column in constant_columns:
        output[..., column] = np.where(np.isfinite(output[..., column]), 0.0, np.nan)
    return (
        cast(FloatArray, _readonly(output, dtype=np.dtype(np.float64))),
        states,
        near,
    )


def _receipt(
    *,
    program_hash: str,
    fold_index: int,
    boundary_id: Literal["INNER", "OUTER"],
    method: PanelFeatureViewRecipe,
    surface: RoleSurfaceDescriptor,
    fit_sessions: tuple[date, ...],
    transform_sessions: tuple[date, ...],
    parameters: dict[str, tuple[float | None, ...]],
    source_array_hash: str,
    transformed_array_hash: str,
    states: tuple[ScaleState, ...],
    near: int,
    receipt_positions: IntArray,
    sector_listing_scale: SectorListingSessionScaleResult | None,
    final_scale_entity_axis_hash: str | None,
) -> FoldScaleReceipt:
    parameter_hash = str(canonical_hash(parameters))
    final_scale_identity = str(
        canonical_hash(
            {
                "role_selection_hash": surface.role.selection_hash,
                "final_scale_method_id": surface.role.final_scale_method_id,
                "formula_pretransform_descriptors": [
                    formula_pretransform_descriptor(value)
                    for value in surface.role.source_ids
                    if surface.role.role_id
                    in {
                        "RELATIVE_STOCK_CROSS_SECTION",
                        "NON_NEUTRAL_STOCK_CROSS_SECTION",
                        "ABSOLUTE_STOCK_TIME_SERIES_STATE",
                    }
                ],
                "parameter_hash": parameter_hash,
            }
        )
    )
    successor_evidence: dict[str, object] = {}
    if sector_listing_scale is not None:
        missing = sector_listing_scale.source_missing_session_feature[receipt_positions]
        constant = sector_listing_scale.constant_session_feature[receipt_positions]
        nondegenerate = sector_listing_scale.nondegenerate_session_feature[receipt_positions]
        scaled = sector_listing_scale.values[receipt_positions]
        scaled_means = np.mean(scaled, axis=1)
        scaled_stds = np.std(scaled, axis=1, ddof=0)
        mean_error = (
            float(np.max(np.abs(scaled_means[nondegenerate])))
            if bool(np.any(nondegenerate))
            else 0.0
        )
        std_error = (
            float(np.max(np.abs(scaled_stds[nondegenerate] - 1.0)))
            if bool(np.any(nondegenerate))
            else 0.0
        )
        successor_evidence = {
            "final_scale_entity_axis_hash": final_scale_entity_axis_hash,
            "source_missing_session_feature_count": int(np.sum(missing)),
            "source_missing_session_feature_axis_hash": str(canonical_hash(missing.tolist())),
            "constant_session_feature_count": int(np.sum(constant)),
            "constant_session_feature_axis_hash": str(canonical_hash(constant.tolist())),
            "nondegenerate_session_feature_count": int(np.sum(nondegenerate)),
            "listing_axis_mean_max_abs": mean_error,
            "listing_axis_population_std_max_abs_error": std_error,
        }
    return FoldScaleReceipt.create(
        program_hash=program_hash,
        fold_index=fold_index,
        boundary_id=boundary_id,
        role_id=surface.role.role_id,
        method_id=method.method_id,
        final_scale_method_id=surface.role.final_scale_method_id,
        ordered_feature_ids=surface.ordered_feature_ids,
        fit_sessions=fit_sessions,
        transform_sessions=transform_sessions,
        native_entity_axis_hash=surface.native_entity_axis_hash,
        source_array_hash=source_array_hash,
        parameter_hash=parameter_hash,
        fitted_center_scale_hash=str(canonical_hash(parameters)),
        transformed_array_hash=transformed_array_hash,
        final_scale_identity=final_scale_identity,
        scale_states=states,
        constant_feature_count=sum(value == "CONSTANT_TRAINING_FEATURE" for value in states),
        near_constant_feature_count=near,
        future_fit_violation_count=0,
        **successor_evidence,
    )


@dataclass(frozen=True, slots=True)
class _FoldRoleMaterialization:
    values: FloatArray
    parameters: dict[str, tuple[float | None, ...]]
    states: tuple[ScaleState, ...]
    near_constant_feature_count: int
    source_array_hash: str
    transformed_array_hash: str
    sector_listing_scale: SectorListingSessionScaleResult | None = None


def fold_role_materialization_key(
    plan: PanelFeaturePlan,
    role: FeatureRoleSelection,
    training_sessions: tuple[date, ...],
    transform_sessions: tuple[date, ...],
) -> str:
    """Bind a cached role matrix to its source, recipe and consuming windows.

    Program, fold and boundary labels belong to the receipt made for each call;
    they do not change the matrix. Operator controls likewise do not enter it.

    Args:
        plan: Admitted immutable source and common-row preflight.
        role: Complete declared role recipe.
        training_sessions: Sessions used to fit the role's scaling state.
        transform_sessions: Sessions receiving that fitted state.

    Returns:
        Canonical identity of the role matrix's complete executed scope.
    """
    return str(
        canonical_hash(
            {
                "source_identities": dict(plan.source.source_identity_hashes),
                "preflight_hash": plan.preflight.preflight_hash,
                "role_selection_hash": role.selection_hash,
                "training_sessions": [value.isoformat() for value in training_sessions],
                "transform_sessions": [value.isoformat() for value in transform_sessions],
            }
        )
    )


def materialize_panel_feature_projection(
    *,
    plan: PanelFeaturePlan,
    method_id: InstalledPanelViewId,
    program_hash: str,
    fold_index: int,
    boundary_id: Literal["INNER", "OUTER"],
    training_sessions: tuple[date, ...],
    transform_sessions: tuple[date, ...],
    role_materialization_cache: dict[str, _FoldRoleMaterialization] | None = None,
) -> PanelFeatureProjection:
    """Fit exactly at one consuming boundary and transform its validation rows."""
    if len(program_hash) != 64:
        raise PanelFeatureBoundaryError("alpha_research.feature_program_identity_invalid")
    if (
        not training_sessions
        or not transform_sessions
        or max(training_sessions) >= min(transform_sessions)
    ):
        raise PanelFeatureBoundaryError("alpha_research.feature_fold_future_fit_violation")
    source = plan.source
    method = plan.catalog.resolve(method_id)
    if method not in plan.selected_methods:
        raise PanelFeatureBoundaryError("alpha_research.feature_view_not_in_program")
    train_positions = _session_positions(source, training_sessions)
    transform_positions = _session_positions(source, transform_sessions)
    selected_positions = cast(
        IntArray,
        _readonly(
            np.concatenate((train_positions, transform_positions), axis=0),
            dtype=np.dtype(np.int64),
        ),
    )
    if np.any(selected_positions[1:] <= selected_positions[:-1]):
        raise PanelFeatureBoundaryError("alpha_research.feature_boundary_positions_invalid")
    training_selected_positions: IntArray = np.arange(len(train_positions), dtype=np.int64)
    transform_selected_positions: IntArray = np.arange(
        len(train_positions), len(selected_positions), dtype=np.int64
    )
    training_mask = plan.common_row_mask[train_positions]
    transform_mask = plan.common_row_mask[transform_positions]
    role_arrays: list[FloatArray] = []
    receipts: list[FoldScaleReceipt] = []
    cache = role_materialization_cache if role_materialization_cache is not None else {}
    receipt_positions: IntArray = np.arange(len(selected_positions), dtype=np.int64)
    for role in method.roles:
        descriptor = plan.role_descriptors_by_selection_hash[role.selection_hash]
        key = fold_role_materialization_key(plan, role, training_sessions, transform_sessions)
        materialized = cache.get(key)
        if materialized is None:
            surface = _build_role_surface(
                role=role,
                source=source,
                formula=plan.formula_surface,
                latest_matured=plan.latest_matured_positions,
                output_positions=selected_positions,
            )
            if (
                surface.ordered_feature_ids != descriptor.ordered_feature_ids
                or surface.native_entity_axis_hash != descriptor.native_entity_axis_hash
            ):
                raise PanelFeatureBoundaryError(
                    "alpha_research.feature_boundary_descriptor_mismatch"
                )
            values = surface.raw_values
            sector_listing_scale: SectorListingSessionScaleResult | None = None
            parameters: dict[str, tuple[float | None, ...]] = {
                "near_constant_threshold": (_NEAR_CONSTANT_THRESHOLD,)
            }
            if role.role_id == "SECTOR_CONTEXT":
                mapped = _map_sector_to_listings(
                    values,
                    source,
                    tuple(source.formation_sessions[int(index)] for index in selected_positions),
                )
                sector_listing_scale = scale_mapped_sector_context_by_listing_session(
                    mapped,
                    reference_eligible=(
                        source.reference_eligible[selected_positions]
                        if source.reference_eligible is not None
                        else None
                    ),
                )
                values = sector_listing_scale.values
                parameters = {"population_standard_deviation_ddof": (0.0,)}
                temporal_states = ("VALID_SCALE",) * len(surface.ordered_feature_ids)
            elif role.role_id in {
                "ABSOLUTE_STOCK_TIME_SERIES_STATE",
                "MARKET_CONTEXT",
            }:
                scaled_native, parameters, temporal_states = _fit_temporal_scale(
                    values,
                    training_selected_positions,
                    training_mask=(
                        training_mask
                        if source.reference_eligible is not None
                        and role.role_id == "ABSOLUTE_STOCK_TIME_SERIES_STATE"
                        else None
                    ),
                )
                if role.role_id == "MARKET_CONTEXT":
                    values = _broadcast_market(scaled_native, source)
                else:
                    values = scaled_native
            else:
                valid_scale: ScaleState = "VALID_SCALE"
                temporal_states = (valid_scale,) * len(surface.ordered_feature_ids)
            values, constant_states, near = _seal_fold_constant_columns(
                values=values,
                training_positions=training_selected_positions,
                training_mask=training_mask,
                continuous=role.role_id != "SECTOR_CATEGORY",
            )
            states: tuple[ScaleState, ...] = tuple(
                "INVALID_NONFINITE_SCALE"
                if left == "INVALID_NONFINITE_SCALE" or right == "INVALID_NONFINITE_SCALE"
                else (
                    "CONSTANT_TRAINING_FEATURE"
                    if left == "CONSTANT_TRAINING_FEATURE" or right == "CONSTANT_TRAINING_FEATURE"
                    else "VALID_SCALE"
                )
                for left, right in zip(temporal_states, constant_states, strict=True)
            )
            if "INVALID_NONFINITE_SCALE" in states:
                raise PanelFeatureBoundaryError("alpha_research.feature_scale_invalid_nonfinite")
            materialized = _FoldRoleMaterialization(
                values=values,
                parameters=parameters,
                states=states,
                near_constant_feature_count=near,
                source_array_hash=panel_array_hash(surface.raw_values[training_selected_positions]),
                transformed_array_hash=panel_array_hash(values),
                sector_listing_scale=sector_listing_scale,
            )
            cache[key] = materialized
        receipt = _receipt(
            program_hash=program_hash,
            fold_index=fold_index,
            boundary_id=boundary_id,
            method=method,
            surface=descriptor,
            fit_sessions=training_sessions,
            transform_sessions=transform_sessions,
            parameters=materialized.parameters,
            source_array_hash=materialized.source_array_hash,
            transformed_array_hash=materialized.transformed_array_hash,
            states=materialized.states,
            near=materialized.near_constant_feature_count,
            receipt_positions=receipt_positions,
            sector_listing_scale=materialized.sector_listing_scale,
            final_scale_entity_axis_hash=(
                str(canonical_hash(list(source.ordered_listing_ids)))
                if materialized.sector_listing_scale is not None
                else None
            ),
        )
        receipts.append(receipt)
        role_arrays.append(materialized.values)
    ordered_feature_ids = tuple(
        feature_id
        for role in method.roles
        for feature_id in plan.role_descriptors_by_selection_hash[
            role.selection_hash
        ].ordered_feature_ids
    )
    if ordered_feature_ids != tuple(dict.fromkeys(ordered_feature_ids)):
        raise PanelFeatureBoundaryError("alpha_research.feature_view_axis_duplicate")
    # Project only the two consuming row sets.  Concatenating every role across
    # the complete source cube would create a second full joint matrix even
    # though a fold consumes only its training and validation rows.
    training = np.concatenate(
        tuple(
            values[training_selected_positions].reshape(-1, values.shape[2])[
                training_mask.reshape(-1)
            ]
            for values in role_arrays
        ),
        axis=1,
    )
    transformed = np.concatenate(
        tuple(
            values[transform_selected_positions].reshape(-1, values.shape[2])[
                transform_mask.reshape(-1)
            ]
            for values in role_arrays
        ),
        axis=1,
    )
    if not np.isfinite(training).all() or not np.isfinite(transformed).all():
        raise PanelFeatureBoundaryError("alpha_research.feature_fold_matrix_nonfinite")
    training_axis_mask = np.zeros(plan.common_row_mask.shape, dtype=np.bool_)
    training_axis_mask[train_positions] = plan.common_row_mask[train_positions]
    transform_axis_mask = np.zeros(plan.common_row_mask.shape, dtype=np.bool_)
    transform_axis_mask[transform_positions] = plan.common_row_mask[transform_positions]
    training_axis = _listing_row_axis(source, training_axis_mask)
    transform_axis = _listing_row_axis(source, transform_axis_mask)
    train_target = source.total_return_target_z[train_positions].reshape(-1)[
        training_mask.reshape(-1)
    ]
    transformed_target = source.total_return_target_z[transform_positions].reshape(-1)[
        transform_mask.reshape(-1)
    ]
    train_returns = source.raw_simple_execution_returns[train_positions].reshape(-1)[
        training_mask.reshape(-1)
    ]
    transformed_returns = source.raw_simple_execution_returns[transform_positions].reshape(-1)[
        transform_mask.reshape(-1)
    ]
    aggregation_ready: BoolArray = np.ones(int(np.sum(transform_mask)), dtype=np.bool_)
    arrays = (
        training,
        transformed,
        train_target,
        transformed_target,
        train_returns,
        transformed_returns,
    )
    readonly_arrays = tuple(
        cast(FloatArray, _readonly(value, dtype=np.dtype(np.float64))) for value in arrays
    )
    return PanelFeatureProjection(
        method=method,
        training_sessions=training_sessions,
        transform_sessions=transform_sessions,
        row_sessions=transform_axis[0],
        row_listing_ids=transform_axis[1],
        ordered_feature_ids=ordered_feature_ids,
        training_features=readonly_arrays[0],
        transformed_features=readonly_arrays[1],
        training_targets=readonly_arrays[2],
        transformed_targets=readonly_arrays[3],
        training_raw_simple_returns=readonly_arrays[4],
        transformed_raw_simple_returns=readonly_arrays[5],
        aggregation_ready=cast(
            BoolArray,
            _readonly(aggregation_ready, dtype=np.dtype(np.bool_)),
        ),
        common_training_row_axis_hash=training_axis[2],
        common_transform_row_axis_hash=transform_axis[2],
        scale_receipts=tuple(receipts),
    )


__all__ = [
    "PanelFeaturePlan",
    "PanelFeatureProjection",
    "materialize_panel_feature_projection",
    "preflight_panel_feature_plan",
]
