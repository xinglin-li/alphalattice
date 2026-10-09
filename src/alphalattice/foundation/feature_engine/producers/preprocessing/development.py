"""Development-only absolute-state and state-interaction Panel transforms."""

from __future__ import annotations

from datetime import date
from typing import cast

import numpy as np
import pandas as pd

from alphalattice.foundation.feature_engine.contracts import (
    FeaturePanelBinding,
    PanelAdmissionSummary,
)
from alphalattice.kernel.quant.cross_section import (
    MAD_SCALE,
    WINSOR_MULTIPLIER,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

from .robust_cross_section import (
    PanelCrossSectionKernel,
    PanelFactorClipObservation,
    PanelMaterialization,
)

ABSOLUTE_LOOKBACK = 252
ABSOLUTE_MINIMUM_FINITE = 126
ABSOLUTE_VECTOR_CHUNK = 16
INTERACTION_CLIP = 5.0


def _numeric_identity(values: np.ndarray) -> str:
    array = np.asarray(values, dtype=np.float64)
    return str(
        canonical_hash(
            {
                "shape": array.shape,
                "values": tuple(
                    None if not np.isfinite(value) else float(value).hex()
                    for value in array.reshape(-1)
                ),
            }
        )
    )


def _axes(
    feature_rows: pd.DataFrame,
    active_listing_ids: tuple[str, ...],
) -> tuple[pd.DataFrame, tuple[date, ...], pd.MultiIndex]:
    frame = feature_rows.copy()
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.date
    frame = frame.loc[frame["listing_id"].isin(active_listing_ids)]
    if frame.duplicated(["session_date", "listing_id"]).any():
        raise ValueError("development preprocessing rows contain duplicate listing sessions")
    sessions = tuple(cast(date, item) for item in sorted(frame["session_date"].unique()))
    grid = pd.MultiIndex.from_product(
        (sessions, active_listing_ids), names=("session_date", "listing_id")
    )
    return frame, sessions, grid


def _rolling_absolute_state(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized rolling median/MAD over bounded listing chunks."""
    transformed = np.full_like(matrix, np.nan)
    clipped = np.zeros_like(matrix, dtype=bool)
    lower = np.full_like(matrix, np.nan)
    upper = np.full_like(matrix, np.nan)
    early_stop = min(len(matrix), ABSOLUTE_LOOKBACK - 1)
    for position in range(ABSOLUTE_MINIMUM_FINITE - 1, early_stop):
        history = matrix[: position + 1]
        finite_count = np.isfinite(history).sum(axis=0)
        median = np.nanmedian(history, axis=0)
        scale = MAD_SCALE * np.nanmedian(np.abs(history - median), axis=0)
        current = matrix[position]
        valid = (
            (finite_count >= ABSOLUTE_MINIMUM_FINITE)
            & np.isfinite(current)
            & np.isfinite(scale)
            & (scale > 0.0)
        )
        low = median - WINSOR_MULTIPLIER * scale
        high = median + WINSOR_MULTIPLIER * scale
        bounded = np.clip(current, low, high)
        lower[position] = np.where(valid, low, np.nan)
        upper[position] = np.where(valid, high, np.nan)
        clipped[position] = valid & (bounded != current)
        transformed[position] = np.where(valid, (bounded - median) / scale, np.nan)
    if len(matrix) >= ABSOLUTE_LOOKBACK:
        rows = slice(ABSOLUTE_LOOKBACK - 1, len(matrix))
        for start in range(0, matrix.shape[1], ABSOLUTE_VECTOR_CHUNK):
            stop = min(matrix.shape[1], start + ABSOLUTE_VECTOR_CHUNK)
            windows = np.lib.stride_tricks.sliding_window_view(
                matrix[:, start:stop],
                window_shape=ABSOLUTE_LOOKBACK,
                axis=0,
            )
            finite_count = np.isfinite(windows).sum(axis=-1)
            median = np.nanmedian(windows, axis=-1)
            scale = MAD_SCALE * np.nanmedian(np.abs(windows - median[..., None]), axis=-1)
            current = matrix[rows, start:stop]
            valid = (
                (finite_count >= ABSOLUTE_MINIMUM_FINITE)
                & np.isfinite(current)
                & np.isfinite(scale)
                & (scale > 0.0)
            )
            low = median - WINSOR_MULTIPLIER * scale
            high = median + WINSOR_MULTIPLIER * scale
            bounded = np.clip(current, low, high)
            lower[rows, start:stop] = np.where(valid, low, np.nan)
            upper[rows, start:stop] = np.where(valid, high, np.nan)
            clipped[rows, start:stop] = valid & (bounded != current)
            transformed[rows, start:stop] = np.where(valid, (bounded - median) / scale, np.nan)
    return transformed, clipped, lower, upper


def materialize_absolute_state(
    *,
    feature_rows: list[dict[str, object]] | pd.DataFrame,
    active_listing_ids: tuple[str, ...],
    factor_ids: tuple[str, ...],
    binding: FeaturePanelBinding,
) -> PanelMaterialization:
    """Scale each listing against up to 252 trailing sessions, requiring 126 finite observations.

    Args:
        feature_rows: Listing-session source feature values.
        active_listing_ids: Declared calculation axis in execution order.
        factor_ids: Ordered factor scope to transform.
        binding: Panel lineage and recorded numerical policy authority.

    Returns:
        Listing median/MAD scores with winsor observations and identities; missing or
            zero-scale histories remain unavailable rather than being filled.

    Raises:
        KeyError: A required listing, session, or factor source column is absent.
        ValueError: Listing-session rows are duplicated.
    """
    frame = feature_rows if isinstance(feature_rows, pd.DataFrame) else pd.DataFrame(feature_rows)
    frame, sessions, grid = _axes(frame, active_listing_ids)
    aligned = frame.set_index(["session_date", "listing_id"]).reindex(grid)
    rows = pd.DataFrame(
        {
            "session_date": [item[0].isoformat() for item in grid],
            "listing_id": [item[1] for item in grid],
        }
    )
    availability: list[dict[str, object]] = []
    observations: list[PanelFactorClipObservation] = []
    raw_identities: list[tuple[str, str]] = []
    output_identities: list[tuple[str, str]] = []
    for factor_id in factor_ids:
        matrix = (
            pd.to_numeric(aligned[factor_id], errors="coerce")
            .to_numpy(float)
            .reshape(len(sessions), len(active_listing_ids))
        )
        # A non-finite raw value is missing, as NaN is, in the trailing median and MAD too.
        matrix = np.where(np.isfinite(matrix), matrix, np.nan)
        transformed, clipped, lower, upper = _rolling_absolute_state(matrix)
        finite_input = np.isfinite(matrix)
        observations.append(
            PanelFactorClipObservation(
                factor_id=factor_id,
                finite_input_count=int(finite_input.sum()),
                per_session_finite_counts=tuple(int(value) for value in finite_input.sum(axis=1)),
                per_session_clipped_counts=tuple(int(value) for value in clipped.sum(axis=1)),
                boundary_identity=cast(
                    str,
                    canonical_hash(
                        {
                            "factor_id": factor_id,
                            "lower_identity": _numeric_identity(lower),
                            "upper_identity": _numeric_identity(upper),
                        }
                    ),
                ),
            )
        )
        rows[factor_id] = transformed.reshape(-1)
        for index, session in enumerate(sessions):
            count = int(np.isfinite(transformed[index]).sum())
            availability.append(
                {
                    "session_date": session.isoformat(),
                    "factor_id": factor_id,
                    "universe_size": len(active_listing_ids),
                    "computed_count": count,
                    "coverage": count / len(active_listing_ids),
                    "sector_counts": {},
                    "status": "available" if count else "unavailable",
                    "reason": None if count else "absolute_state_history_or_scale_unavailable",
                    "small_sector_warning": False,
                    "small_sector_names": (),
                    "panel_binding_hash": binding.panel_binding_hash,
                }
            )
        raw_identities.append((factor_id, _numeric_identity(matrix)))
        output_identities.append((factor_id, _numeric_identity(transformed)))
    admission = PanelAdmissionSummary.evaluate(
        as_of_session=pd.Timestamp(sessions[-1]).date(),
        factor_ids=factor_ids,
        availability=availability,
        sector_distribution={},
    )
    return PanelMaterialization(
        rows=rows,
        availability=availability,
        binding=binding,
        receipt_hash=str(
            canonical_hash({"binding": binding.panel_binding_hash, "rows": len(rows)})
        ),
        complete=bool(
            np.isfinite(rows.loc[:, factor_ids].to_numpy(float)[-len(active_listing_ids) :]).any()
        ),
        admission=admission,
        clip_observations=tuple(observations),
        raw_input_identity=str(canonical_hash(raw_identities)),
        transformed_identity=str(canonical_hash(output_identities)),
        ordered_sessions_hash=str(canonical_hash([item.isoformat() for item in sessions])),
        ordered_listing_ids_hash=str(canonical_hash(list(active_listing_ids))),
    )


def materialize_state_interactions(
    *,
    feature_rows: list[dict[str, object]] | pd.DataFrame,
    active_listing_ids: tuple[str, ...],
    sector_by_listing_id: dict[str, str],
    factor_ids: tuple[str, ...],
    binding: FeaturePanelBinding,
) -> PanelMaterialization:
    """Multiply robust sector-neutral stock values by their state children and clip to [-5, 5].

    Args:
        feature_rows: Listing-session source feature values.
        active_listing_ids: Declared calculation axis in execution order.
        sector_by_listing_id: Sector authority for every axis listing.
        factor_ids: Ordered factor scope to transform.
        binding: Panel lineage and recorded numerical policy authority.

    Returns:
        Transformed rows, availability and admission, measured clipping observations,
            and ordered input/output identities bound to the supplied Panel lineage.

    Raises:
        KeyError: A required listing, session, or factor source column is absent.
        ValueError: The cross-section is invalid, rows are duplicated, or a factor
            lacks its state child column.
    """
    frame = feature_rows if isinstance(feature_rows, pd.DataFrame) else pd.DataFrame(feature_rows)
    child = PanelCrossSectionKernel().materialize(
        feature_rows=frame,
        active_listing_ids=active_listing_ids,
        sector_by_listing_id=sector_by_listing_id,
        factor_ids=factor_ids,
        binding=binding,
    )
    frame, sessions, grid = _axes(frame, active_listing_ids)
    aligned = frame.set_index(["session_date", "listing_id"]).reindex(grid)
    rows = child.rows.copy()
    observations: list[PanelFactorClipObservation] = []
    availability: list[dict[str, object]] = []
    raw_pairs: list[tuple[str, str]] = []
    transformed_pairs: list[tuple[str, str]] = []
    for factor_id in factor_ids:
        state_column = f"{factor_id}__state"
        if state_column not in aligned:
            raise ValueError("STATE_INTERACTION_MARKET_CHILD_MISSING")
        stock = pd.to_numeric(rows[factor_id], errors="coerce").to_numpy(float)
        state = pd.to_numeric(aligned[state_column], errors="coerce").to_numpy(float)
        product = stock * state
        bounded = np.clip(product, -INTERACTION_CLIP, INTERACTION_CLIP)
        bounded[~np.isfinite(product)] = np.nan
        clipped = np.isfinite(product) & (bounded != product)
        rows[factor_id] = bounded
        matrix = bounded.reshape(len(sessions), len(active_listing_ids))
        for session_position, session in enumerate(sessions):
            finite = np.isfinite(matrix[session_position])
            sector_counts = {
                sector: int(
                    sum(
                        finite[position]
                        for position, listing in enumerate(active_listing_ids)
                        if sector_by_listing_id[listing] == sector
                    )
                )
                for sector in sorted(set(sector_by_listing_id.values()))
            }
            coverage = float(finite.mean())
            available = coverage >= 0.98 and all(value >= 5 for value in sector_counts.values())
            availability.append(
                {
                    "session_date": session.isoformat(),
                    "factor_id": factor_id,
                    "universe_size": len(active_listing_ids),
                    "computed_count": int(finite.sum()),
                    "coverage": coverage,
                    "sector_counts": sector_counts,
                    "status": "available" if available else "unavailable",
                    "reason": None if available else "interaction_child_or_state_unavailable",
                    "small_sector_warning": False,
                    "small_sector_names": (),
                    "panel_binding_hash": binding.panel_binding_hash,
                }
            )
        observations.append(
            PanelFactorClipObservation(
                factor_id=factor_id,
                finite_input_count=int(np.isfinite(product).sum()),
                per_session_finite_counts=tuple(
                    int(value)
                    for value in np.isfinite(product).reshape(len(sessions), -1).sum(axis=1)
                ),
                per_session_clipped_counts=tuple(
                    int(value) for value in clipped.reshape(len(sessions), -1).sum(axis=1)
                ),
                boundary_identity=str(canonical_hash({"lower": -5.0, "upper": 5.0})),
            )
        )
        raw_pairs.append((factor_id, _numeric_identity(product)))
        transformed_pairs.append((factor_id, _numeric_identity(bounded)))
    sector_distribution = {
        sector: sum(value == sector for value in sector_by_listing_id.values())
        for sector in sorted(set(sector_by_listing_id.values()))
    }
    admission = PanelAdmissionSummary.evaluate(
        as_of_session=pd.Timestamp(sessions[-1]).date(),
        factor_ids=factor_ids,
        availability=availability,
        sector_distribution=sector_distribution,
    )
    return PanelMaterialization(
        rows=rows,
        availability=availability,
        binding=binding,
        receipt_hash=str(canonical_hash({"child": child.receipt_hash, "clip": 5.0})),
        complete=admission.research_admissible,
        admission=admission,
        clip_observations=tuple(observations),
        raw_input_identity=str(canonical_hash(raw_pairs)),
        transformed_identity=str(canonical_hash(transformed_pairs)),
        ordered_sessions_hash=child.ordered_sessions_hash,
        ordered_listing_ids_hash=child.ordered_listing_ids_hash,
    )


__all__ = [
    "ABSOLUTE_LOOKBACK",
    "ABSOLUTE_MINIMUM_FINITE",
    "ABSOLUTE_VECTOR_CHUNK",
    "INTERACTION_CLIP",
    "materialize_absolute_state",
    "materialize_state_interactions",
]
