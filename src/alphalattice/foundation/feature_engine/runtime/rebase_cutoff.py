"""A rebased listing's early cutoff: the Feature cells its rebased bars move past the rule.

The feature engine owns it; the maintenance coordinator consumes it, planning only the moved
cells' rebuild after a split's rebase of the stored bars.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from datetime import date, datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from alphalattice.foundation.feature_engine.producers.reference_data import MarketReference
from alphalattice.foundation.feature_engine.runtime.service import feature_source_frame

if TYPE_CHECKING:
    from alphalattice.foundation.feature_engine.runtime.service import FeatureFoundationService

Moved = tuple[tuple[str, tuple[date, ...]], ...]
"""Each moved Feature with its sessions, by Feature."""
PanelSpans = tuple[tuple[date, date], ...]
"""The active Panel's immutable first/last-session spans, shared by one reach generation."""


def rebase_moved(
    service: FeatureFoundationService,
    listing_id: str,
    *,
    as_of_session: date,
    observed_at: datetime,
    agreement: tuple[float, float],
    spans: PanelSpans | None = None,
) -> Moved:
    """Each stored Feature of a rebased listing that its rebased bars move past `agreement`.

    An early cutoff, as build systems name it: a value recomputed from the rebased bars that
    agrees with its stored one within the data rule's agreement (relative, then absolute near
    zero; a missing value agrees only with a missing one) keeps its row, so only the cells past
    it are rebuilt and reach the Panel. A row it keeps is the cold rebuild's value within the
    agreement, so it stands for it. Every stored cell is compared, so whichever kernel a rescale
    moves (a tie its inexact prices flip, a volume a whole-share rounding moves) is found. The
    listing is computed once over its stored history, as a cold build computes it. In each
    existing Panel span these cells reach, all its Features are rebuilt: standardisation
    must not amplify a kept cell's gap in a part the Panel already rewrites.

    Returns:
        Each moved Feature with its sessions; empty when none moved.
    """
    observed = service.market_data.raw_bar_sessions_by_listing([listing_id], through=as_of_session)
    if not observed.get(listing_id):
        return ()
    calendar = service._base_session_calendar(
        observed_sessions=tuple(observed[listing_id]),
        calendar_id=service.manifest.profile.calendar_id,
        history_start=observed[listing_id][0],
        as_of_session=as_of_session,
        observed_at=observed_at,
        calendars={},
    )
    source = service._base_source_inputs(
        service.manifest, listing_id=listing_id, through=as_of_session, start=calendar[0],
        _connection=None,
    )  # fmt: skip
    spy = MarketReference.spy(service.manifest)
    market, _raw, _actions = service.feature_state.projected_feature_frame(
        spy.manifest, listing_id=spy.listing_id, through=as_of_session
    )
    values = service.base.materialize_listing(
        listing_id=listing_id,
        projected_bars=feature_source_frame(
            source, sessions=service._source_sessions(calendar, calendar[0])
        ),
        market_bars=pd.DataFrame(market),
        sessions=calendar,
    ).values
    fresh = values.set_index(pd.to_datetime(values["session_date"]).dt.date)
    moved: dict[str, tuple[date, ...]] = {}
    for part in service.layer.parts:
        stored = service.feature_state.feature_rows(
            listing_ids=(listing_id,), catalog_hash=part.binding.catalog_hash,
            start=calendar[0], end=as_of_session, include_lineage=False, as_frame=True,
        )  # fmt: skip
        assert isinstance(stored, pd.DataFrame)
        days = pd.to_datetime(stored["session_date"]).dt.date
        held = fresh.reindex(days)
        for factor in part.factor_ids:
            past = ~np.isclose(
                stored[factor].to_numpy(float),
                held[factor].to_numpy(float),
                rtol=agreement[0],
                atol=agreement[1],
                equal_nan=True,
            )
            if past.any():
                moved[factor] = tuple(days[past])
    if moved:
        sessions = _panel_sessions(
            panel_spans(service) if spans is None else spans, moved, calendar
        )
        if sessions:
            for part in service.layer.parts:
                for factor in part.factor_ids:
                    moved[factor] = tuple(sorted(set(moved.get(factor, ())) | sessions))
    return tuple(sorted(moved.items()))


def panel_spans(service: FeatureFoundationService) -> PanelSpans:
    """Resolve the active Panel descriptor once for all listings in a reach generation.

    The caller holds this immutable value only for that generation; standalone cutoff calls
    resolve their own descriptor. No process cache may outlive a new active Panel.
    """
    snapshot = service.panel_state.feature_panel_snapshot_for_active(
        service.manifest.profile.market_profile_id
    )
    if snapshot is None:
        return ()
    manifest = service.panel_artifacts.resolver.load_feature_panel_manifest(
        str(snapshot["manifest_uri"])
    )
    return tuple(
        (date.fromisoformat(chunk["first_session"]), date.fromisoformat(chunk["last_session"]))
        for chunk in manifest["chunks"]
    )


def _panel_sessions(
    spans: PanelSpans, moved: dict[str, tuple[date, ...]], calendar: tuple[date, ...]
) -> set[date]:
    """Sessions of existing Panel parts reached by a moved cell, without adding a part."""
    changed = sorted({day for days in moved.values() for day in days})
    selected: set[date] = set()
    for first, last in spans:
        index = bisect_left(changed, first)
        if index < len(changed) and changed[index] <= last:
            selected.update(calendar[bisect_left(calendar, first) : bisect_right(calendar, last)])
    return selected
