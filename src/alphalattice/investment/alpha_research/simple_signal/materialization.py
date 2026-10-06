"""Read one Panel feature over a named axis and standardize it. Nothing else.

The read goes through ``FeaturePanelReader``, which is the installed, governed,
target-free way to get per-(session, listing) feature values: it refuses a
snapshot that is not ACTIVE and physically available, and it streams rather than
materializing the panel. No second reader is introduced here.

Scattering into a preallocated array by index rather than concatenating batches
is what makes the result independent of batch order, so the reader's own
threading cannot move a number.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.feature_engine.panels.reader import (
    FeaturePanelReader,
    FeaturePanelReadRequest,
)

from .standardize import (
    MINIMUM_FINITE_LISTINGS,
    FloatArray,
    SimpleSignalError,
    finite_listing_counts,
    score_values_identity,
    standardize_cross_section,
)


@dataclass(frozen=True, slots=True)
class MaterializedSimpleScore:
    """One standardized score surface and the axis it was measured on."""

    feature_id: str
    ordered_formation_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]
    raw_values: FloatArray
    standardized_values: FloatArray
    values_identity: str
    resolved_cell_count: int
    minimum_finite_listings_observed: int


def materialize_simple_score(
    *,
    resolver: ArtifactResolver,
    panel_manifest_ref: str,
    feature_id: str,
    ordered_formation_sessions: tuple[date, ...],
    ordered_listing_ids: tuple[str, ...],
) -> MaterializedSimpleScore:
    """Read ``feature_id`` on the named axis and standardize each formation.

    The axis is stated by the caller and honoured exactly: a session or listing
    the panel does not carry stays unresolved rather than shrinking the axis,
    because the cross-section a z was measured over is part of what it means.
    """
    if ordered_formation_sessions != tuple(sorted(set(ordered_formation_sessions))):
        raise SimpleSignalError("alpha_research.simple_signal_session_axis_unordered")
    if len(set(ordered_listing_ids)) != len(ordered_listing_ids):
        raise SimpleSignalError("alpha_research.simple_signal_listing_duplicated")
    if len(ordered_listing_ids) < MINIMUM_FINITE_LISTINGS:
        raise SimpleSignalError("alpha_research.simple_signal_axis_invalid")

    row_at = {value: index for index, value in enumerate(ordered_formation_sessions)}
    column_at = {value: index for index, value in enumerate(ordered_listing_ids)}
    raw: FloatArray = np.full(
        (len(ordered_formation_sessions), len(ordered_listing_ids)), np.nan, dtype=np.float64
    )
    request = FeaturePanelReadRequest(
        manifest_ref=panel_manifest_ref,
        start_session=ordered_formation_sessions[0],
        end_session=ordered_formation_sessions[-1],
        factor_columns=(feature_id,),
        exact_sessions=ordered_formation_sessions,
    )
    for batch in FeaturePanelReader(resolver).batches(request):
        sessions = batch.column("session_date").to_pylist()
        listings = batch.column("listing_id").to_pylist()
        values = batch.column(feature_id).to_pylist()
        for session, listing, value in zip(sessions, listings, values, strict=True):
            row = row_at.get(session)
            column = column_at.get(listing)
            if row is None or column is None or value is None:
                continue
            raw[row, column] = float(value)

    standardized = standardize_cross_section(raw)
    counts = finite_listing_counts(standardized)
    resolved = int(counts.sum())
    if resolved < 1:
        raise SimpleSignalError("alpha_research.simple_signal_entirely_unresolved")
    standardized.setflags(write=False)
    raw.setflags(write=False)
    return MaterializedSimpleScore(
        feature_id=feature_id,
        ordered_formation_sessions=ordered_formation_sessions,
        ordered_listing_ids=ordered_listing_ids,
        raw_values=raw,
        standardized_values=standardized,
        values_identity=score_values_identity(standardized),
        resolved_cell_count=resolved,
        minimum_finite_listings_observed=int(counts.min()),
    )


__all__ = ["MaterializedSimpleScore", "materialize_simple_score"]
