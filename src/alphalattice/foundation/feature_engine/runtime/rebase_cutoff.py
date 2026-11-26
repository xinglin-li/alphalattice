"""Feature-owned rebase cutoffs; maintenance consumes their exact rebuild cells."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Callable, Sequence
from datetime import date, datetime
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from alphalattice.foundation.feature_engine.panels.reader import (
    FeaturePanelReader,
    FeaturePanelReadRequest,
)
from alphalattice.foundation.feature_engine.producers.reference_data import MarketReference
from alphalattice.foundation.feature_engine.runtime.service import (
    PANEL_SESSION_BATCH_SIZE,
    feature_source_frame,
)

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
    panel_values: pd.DataFrame | None = None,
    panel_agreement: tuple[float, float] | None = None,
    kept: Callable[[str, int], None] | None = None,
) -> Moved:
    """Compute the listing cold once, returning Feature work under both agreements.

    Feature disagreement rebuilds all Features in reached Panel spans; outside those spans,
    Panel disagreement escalates its exact Feature cells. Missing values agree only together.
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
    sessions: set[date] = set()
    if moved:
        sessions = _panel_sessions(
            panel_inputs(service)[0] if spans is None else spans, moved, calendar
        )
        if sessions:
            for part in service.layer.parts:
                for factor in part.factor_ids:
                    moved[factor] = tuple(sorted(set(moved.get(factor, ())) | sessions))
    if panel_values is not None and panel_agreement is not None:
        extra, count = _panel_moved(
            service, listing_id, fresh, panel_values, sessions, panel_agreement
        )
        for factor, moved_days in extra:
            moved[factor] = tuple(sorted(set(moved.get(factor, ())) | set(moved_days)))
        if kept is not None:
            kept(listing_id, count)
    return tuple(sorted(moved.items()))


def panel_inputs(
    service: FeatureFoundationService, listing_ids: Sequence[str] = ()
) -> tuple[PanelSpans, pd.DataFrame | None]:
    """Admit/read the stored Panel once per reach generation, keeping only rebased listings."""
    snapshot = service.panel_state.feature_panel_snapshot_for_active(
        service.manifest.profile.market_profile_id
    )
    if snapshot is None:
        return (), None
    ref = str(snapshot["manifest_uri"])
    reader = FeaturePanelReader(service.panel_artifacts.resolver)
    manifest = (
        reader.admitted_manifest(ref, consumer="Feature maintenance")
        if listing_ids
        else service.panel_artifacts.resolver.load_feature_panel_manifest(ref)
    )
    spans = tuple(
        (date.fromisoformat(c["first_session"]), date.fromisoformat(c["last_session"]))
        for c in cast(list[dict[str, str]], manifest["chunks"])
    )
    if not listing_ids:
        return spans, None
    request = FeaturePanelReadRequest(
        ref, spans[0][0], spans[-1][1], tuple(sorted(service.catalog.factor_ids))
    )
    frames = []
    for batch in reader._admitted_batches(manifest, request):
        frame = batch.to_pandas()
        frames.append(frame.loc[frame.listing_id.isin(listing_ids)])
    return spans, pd.concat(frames, ignore_index=True)


def _panel_moved(
    service: FeatureFoundationService,
    listing_id: str,
    fresh: pd.DataFrame,
    stored: pd.DataFrame,
    rewritten: set[date],
    agreement: tuple[float, float],
) -> tuple[Moved, int]:
    """Compare against fresh listing Features through the installed Panel preprocessing."""
    stored = stored.loc[
        (stored.listing_id == listing_id)
        & ~pd.to_datetime(stored.session_date).dt.date.isin(rewritten)
    ].copy()
    if stored.empty:
        return (), 0
    stored["session_date"] = pd.to_datetime(stored.session_date).dt.date
    days = tuple(sorted(stored.session_date.unique()))
    axis, _scope = service._axis_listings()
    membership = service._panel_membership(
        calendar=days, axis_listing_ids=tuple(x.listing_id for x in axis)
    )
    axis_ids, factors = membership.listing_ids, tuple(sorted(service.catalog.factor_ids))
    rows = service.feature_state.feature_rows(
        listing_ids=axis_ids, catalog_hash=service.catalog.binding.catalog_hash, start=days[0],
        end=days[-1], factor_ids=factors, include_lineage=False, as_frame=True,
    )  # fmt: skip
    assert isinstance(rows, pd.DataFrame)
    rows["session_date"] = pd.to_datetime(rows.session_date).dt.date
    rows = rows.loc[rows.session_date.isin(days)].copy()
    selected = rows.listing_id == listing_id
    rows.loc[selected, list(factors)] = fresh.reindex(rows.loc[selected, "session_date"])[
        list(factors)
    ].to_numpy(float)
    sectors = service.feature_state.sector_history(service.manifest, listing_ids=axis_ids)
    assert sectors is not None
    reference = service.feature_state.market_reference()
    assert reference is not None
    cold = pd.concat([
        service.panel.materialize(
            feature_rows=rows.loc[rows.session_date.isin(chunk)], active_listing_ids=axis_ids,
            manifest_revision=service.manifest.revision_sha256,
            sector_revision=sectors.current_revision, sector_by_listing_id=mapping,
            spy_revision=str(reference["revision_hash"]), factor_ids=factors,
            members_by_session=membership.members_by_session(chunk),
            source_exclusions_by_session={d: membership.excluded_sources(d) for d in chunk},
        ).rows.loc[lambda frame: frame.listing_id == listing_id]
        for run, mapping in sectors.runs(days)
        for chunk in (run[i:i + PANEL_SESSION_BATCH_SIZE]
                      for i in range(0, len(run), PANEL_SESSION_BATCH_SIZE))
    ], ignore_index=True)  # fmt: skip
    cold = cold.set_index("session_date")
    cold.index = pd.to_datetime(cold.index).date
    held = cold.reindex(stored.session_date)
    close = np.isclose(
        stored[list(factors)].to_numpy(float),
        held[list(factors)].to_numpy(float),
        rtol=agreement[0],
        atol=agreement[1],
        equal_nan=True,
    )
    return tuple(
        (f, tuple(stored.loc[~close[:, i], "session_date"]))
        for i, f in enumerate(factors)
        if not close[:, i].all()
    ), int(close.sum())


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
