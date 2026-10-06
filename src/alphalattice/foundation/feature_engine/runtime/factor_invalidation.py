"""Compile source deltas into sparse, bounded factor/session work."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import date
from typing import Literal

from alphalattice.foundation.feature_engine.catalog.contracts import (
    MARKET_DEPENDENT_FACTOR_IDS,
    FeatureCatalog,
)
from alphalattice.foundation.feature_engine.contracts import (
    PANEL_ROW_IDENTITY_BY_BINDING,
    PANEL_ROW_IDENTITY_BY_CROSS_SECTION,
    FactorSessionInvalidation,
    FactorSessionInvalidationPlan,
    FeatureBuildRequest,
    FeatureInvalidation,
    PanelInvalidationRange,
)

_FULL_BASE_KINDS = frozenset({"initial_backfill", "catalog_binding_change", "manifest_addition"})
_FULL_PANEL_KINDS = frozenset(
    {
        "initial_backfill",
        "catalog_binding_change",
        "sector_revision_change",
    }
)
# A membership change reaches the Panel from the session it takes effect:
# every session's cross-section from then on is a different computation,
# and every session before it is exactly what it was. A manifest that
# changed for governance alone (``panel_binding_change``) reaches no Panel
# row at all; what the base cannot serve, the composition forces.
_MEMBERSHIP_PANEL_KINDS = frozenset({"manifest_addition", "manifest_removal"})
_EVIDENCE_ONLY_KINDS = frozenset(
    {"action_correction", "action_evidence_change", "panel_binding_change"}
)


def compile_listing_plan(
    *,
    catalog: FeatureCatalog,
    request: FeatureBuildRequest,
    invalidations: Sequence[FeatureInvalidation],
    listing_id: str,
    sessions: Sequence[date],
    reach: Literal["base", "panel"] = "base",
) -> FactorSessionInvalidationPlan:
    """Return exact output cells for one listing.

    Source observations are expanded through each feature's finite skip/window
    contract.  The durable result remains sparse ranges; no dense matrix is
    written to task state.

    ``reach`` names which owner the plan feeds. The base Formula values of a
    listing that joins (or rejoins) are computed over its whole history for
    the windows -- the ``base`` reach; the Panel's cross-sections change from
    the session the membership takes effect and nothing earlier moves, so the
    ``panel`` reach of a ``manifest_addition`` starts at its effective session
    whatever the listing's earlier membership was.
    """
    calendar = tuple(session for session in sessions if session <= request.as_of_session)
    if not calendar:
        return FactorSessionInvalidationPlan.create(items=())
    positions = {session: index for index, session in enumerate(calendar)}
    targets: dict[str, set[date]] = defaultdict(set)
    receipts: list[str] = []
    relevant = tuple(item for item in invalidations if item.listing_id in {None, listing_id})
    if not relevant:
        if invalidations:
            return FactorSessionInvalidationPlan.create(items=())
        relevant = (FeatureInvalidation("initial_backfill"),)

    # A catalog-wide base rebuild has one exact durable shape: every factor
    # owns the complete listing calendar.  Construct that shape directly
    # instead of allocating one full session set per factor and coalescing it
    # back into the same single range.  Keep the shortcut deliberately narrow
    # so mixed and factor-scoped invalidations retain the sparse compiler's
    # validation and union semantics.
    if len(relevant) == 1 and _whole_calendar(relevant[0], reach=reach):
        full_item = relevant[0]
        factor_ids = _factor_scope(catalog, full_item)
        if set(factor_ids) == set(catalog.factor_ids):
            full_range = (PanelInvalidationRange(calendar[0], calendar[-1]),)
            return FactorSessionInvalidationPlan.create(
                items=tuple(
                    FactorSessionInvalidation(factor_id, full_range)
                    for factor_id in catalog.factor_ids
                ),
                source_receipt_hashes=(
                    (full_item.source_receipt_hash,)
                    if full_item.source_receipt_hash is not None
                    else ()
                ),
            )

    for item in relevant:
        if item.source_receipt_hash is not None:
            receipts.append(item.source_receipt_hash)
        if item.kind in _EVIDENCE_ONLY_KINDS or item.kind in {
            "sector_revision_change",
            "manifest_removal",
        }:
            continue
        factor_ids = _factor_scope(catalog, item)
        if _whole_calendar(item, reach=reach):
            for factor_id in factor_ids:
                targets[factor_id].update(calendar)
            continue
        source_sessions = _source_sessions(item, request=request, calendar=calendar)
        if item.kind == "manifest_addition":
            for factor_id in factor_ids:
                targets[factor_id].update(source_sessions)
            continue
        if item.kind == "normal_new_session":
            for factor_id in factor_ids:
                targets[factor_id].update(source_sessions)
            continue
        for factor_id in factor_ids:
            contract = catalog.contracts_by_factor[factor_id]
            for source_session in source_sessions:
                try:
                    source_position = positions[source_session]
                except KeyError as exc:
                    raise ValueError(
                        "feature source delta is outside the listing calendar"
                    ) from exc
                first = source_position + contract.formula_skip_sessions
                # Source observations can feed a one-session derived edge
                # (diff/shift/pairwise range) before the catalog's declared
                # rolling window.  Corrections therefore need one conservative
                # successor beyond the direct skip/window span.  A Formula with
                # no economic skip is invalidated from the corrected session
                # itself, which is what an observation-session clock implies.
                # New sessions are handled above and have no observed successor.
                last = first + contract.maximum_invalidation_sessions
                targets[factor_id].update(calendar[first : min(last + 1, len(calendar))])

    return FactorSessionInvalidationPlan.create(
        items=tuple(
            FactorSessionInvalidation(factor_id, _ranges(calendar, factor_sessions))
            for factor_id, factor_sessions in sorted(targets.items())
            if factor_sessions
        ),
        source_receipt_hashes=tuple(receipts),
    )


def compile_panel_plan(
    *,
    catalog: FeatureCatalog,
    request: FeatureBuildRequest,
    invalidations: Sequence[FeatureInvalidation],
    listing_plans: Sequence[FactorSessionInvalidationPlan],
    sessions: Sequence[date],
    base_reusable: bool = True,
    forced_sessions: Sequence[date] = (),
    row_identity_basis: str = PANEL_ROW_IDENTITY_BY_BINDING,
) -> FactorSessionInvalidationPlan:
    """Union listing work into full-cross-section factor/session patches.

    ``forced_sessions`` are sessions whose Panel rows cannot come from the base
    snapshot even though no source delta reaches them -- a year the base holds
    no compatible partition for, or one whose recorded provenance cannot be
    resolved. They are requested for every factor, exactly as a Panel without
    any base is; the listing plans still carry only what the source changed.
    A catalog change scoped to the columns a catalog only adds (V92) requests
    those columns on every session over a reusable base, and every factor
    without one.
    """
    calendar = tuple(session for session in sessions if session <= request.as_of_session)
    targets: dict[str, set[date]] = defaultdict(set)
    receipts: list[str] = []
    if row_identity_basis not in {
        PANEL_ROW_IDENTITY_BY_BINDING,
        PANEL_ROW_IDENTITY_BY_CROSS_SECTION,
    }:
        raise ValueError("feature.panel_plan_identity_basis_invalid")
    if not invalidations:
        invalidations = (FeatureInvalidation("initial_backfill"),)
    full_rebuild_kinds = _FULL_PANEL_KINDS
    if row_identity_basis == PANEL_ROW_IDENTITY_BY_CROSS_SECTION:
        # Composition already identifies each changed member/Sector footprint.
        # Adding a future member to a Sector map does not change past groups.
        full_rebuild_kinds = full_rebuild_kinds - {"sector_revision_change"}
    whole = tuple(item for item in invalidations if item.kind in full_rebuild_kinds)
    only_added_columns = bool(whole) and all(
        item.kind == "catalog_binding_change" and item.factor_ids for item in whole
    )
    if not base_reusable or (whole and not only_added_columns):
        for factor_id in catalog.factor_ids:
            targets[factor_id].update(calendar)
    else:
        for item in whole:
            for factor_id in item.factor_ids:
                targets[factor_id].update(calendar)
        for plan in listing_plans:
            receipts.extend(plan.source_receipt_hashes)
            for item in plan.items:
                targets[item.factor_id].update(expand_ranges(item.ranges, calendar))
        forced = set(forced_sessions).intersection(calendar)
        for item in invalidations:
            if item.kind in _MEMBERSHIP_PANEL_KINDS:
                forced.update(_source_sessions(item, request=request, calendar=calendar))
        if forced:
            for factor_id in catalog.factor_ids:
                targets[factor_id].update(forced)
    receipts.extend(
        item.source_receipt_hash for item in invalidations if item.source_receipt_hash is not None
    )
    return FactorSessionInvalidationPlan.create(
        items=tuple(
            FactorSessionInvalidation(factor_id, _ranges(calendar, factor_sessions))
            for factor_id, factor_sessions in sorted(targets.items())
            if factor_sessions
        ),
        source_receipt_hashes=tuple(receipts),
    )


def restrict_plan_to_sessions(
    plan: FactorSessionInvalidationPlan,
    *,
    calendar: Sequence[date],
    allowed: Callable[[date], bool],
) -> FactorSessionInvalidationPlan:
    """Restrict a listing's invalidation plan to its Panel member sessions.

    A listing's base Formula values may be recomputed over its whole history
    (an entrant's, for its windows) while its Panel rows exist only on the
    sessions it is a member of; a base delta on any other session changes no
    cross-section and must not force one to be recomputed.
    """
    items = []
    # Every factor of a plan usually carries the same ranges (one new session
    # a day); expand and restrict each distinct range set once, not per factor.
    restricted: dict[tuple[PanelInvalidationRange, ...], tuple[PanelInvalidationRange, ...]] = {}
    for item in plan.items:
        key = tuple(item.ranges)
        ranges = restricted.get(key)
        if ranges is None:
            sessions = {session for session in expand_ranges(key, calendar) if allowed(session)}
            ranges = _ranges(calendar, sessions)
            restricted[key] = ranges
        if ranges:
            items.append(FactorSessionInvalidation(item.factor_id, ranges))
    return FactorSessionInvalidationPlan.create(
        items=tuple(items),
        source_receipt_hashes=plan.source_receipt_hashes,
        expected_input_hashes=plan.expected_input_hashes,
    )


def expand_ranges(
    ranges: Sequence[PanelInvalidationRange], sessions: Sequence[date]
) -> tuple[date, ...]:
    """Return sessions inside any range, preserving their given order."""
    if not ranges or not sessions:
        return ()
    if all(sessions[index] <= sessions[index + 1] for index in range(len(sessions) - 1)):
        # A calendar: each range is one contiguous slice found by bisection,
        # instead of testing every session against every range.
        chosen: set[int] = set()
        for item in ranges:
            low = bisect_left(sessions, item.first_session)
            high = bisect_right(sessions, item.last_session)
            chosen.update(range(low, high))
        return tuple(sessions[index] for index in sorted(chosen))
    return tuple(
        session
        for session in sessions
        if any(item.first_session <= session <= item.last_session for item in ranges)
    )


def targets_by_session(
    plan: FactorSessionInvalidationPlan, sessions: Sequence[date]
) -> dict[date, tuple[str, ...]]:
    """Map affected sessions to the factors that require recomputation."""
    if plan.items:
        shared_ranges = plan.items[0].ranges
        if len(shared_ranges) == 1 and all(item.ranges == shared_ranges for item in plan.items[1:]):
            factors = plan.factor_ids
            return {
                session: factors for session in sorted(set(expand_ranges(shared_ranges, sessions)))
            }
    result: dict[date, list[str]] = defaultdict(list)
    for item in plan.items:
        for session in expand_ranges(item.ranges, sessions):
            result[session].append(item.factor_id)
    return {session: tuple(sorted(factors)) for session, factors in sorted(result.items())}


def _factor_scope(catalog: FeatureCatalog, item: FeatureInvalidation) -> tuple[str, ...]:
    if item.factor_ids:
        if not set(item.factor_ids).issubset(catalog.factor_ids):
            raise ValueError("feature invalidation references an unknown factor")
        return item.factor_ids
    if item.kind == "spy_correction":
        return tuple(sorted(MARKET_DEPENDENT_FACTOR_IDS))
    if item.source_fields:
        fields = set(item.source_fields)
        return tuple(
            factor.factor_id
            for factor in catalog.factors
            if fields.intersection(factor.required_fields)
        )
    return catalog.factor_ids


def _whole_calendar(item: FeatureInvalidation, *, reach: Literal["base", "panel"]) -> bool:
    """Whether ``item`` owns the listing's entire calendar for this reach."""
    if item.kind not in _FULL_BASE_KINDS:
        return False
    if item.kind != "manifest_addition" or reach == "base":
        return True
    # An addition without an effective session is the older, whole-history
    # spelling and keeps its whole-history Panel reach.
    return item.earliest_session is None and not item.affected_sessions


def _source_sessions(
    item: FeatureInvalidation,
    *,
    request: FeatureBuildRequest,
    calendar: tuple[date, ...],
) -> tuple[date, ...]:
    if item.affected_sessions:
        calendar_scope = set(calendar)
        return tuple(session for session in item.affected_sessions if session in calendar_scope)
    if item.earliest_session is not None:
        return tuple(session for session in calendar if session >= item.earliest_session)
    if item.kind == "normal_new_session":
        return (request.as_of_session,)
    raise ValueError("bounded source invalidation requires exact sessions or an earliest bound")


def _ranges(calendar: Sequence[date], selected: set[date]) -> tuple[PanelInvalidationRange, ...]:
    ordered = [session for session in calendar if session in selected]
    if not ordered:
        return ()
    positions = {session: index for index, session in enumerate(calendar)}
    result: list[PanelInvalidationRange] = []
    first = previous = ordered[0]
    for session in ordered[1:]:
        if positions[session] != positions[previous] + 1:
            result.append(PanelInvalidationRange(first, previous))
            first = session
        previous = session
    result.append(PanelInvalidationRange(first, previous))
    return tuple(result)
