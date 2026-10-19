"""A split's reach into the stored Features: the cells its rebase moves past the data rule.

The data platform owns it; the maintenance coordinator consumes it as it assembles a cycle's
change set, so the Feature build rebuilds exactly those cells and the Panel sees only them.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from alphalattice.foundation.feature_engine.runtime.rebase_cutoff import panel_spans, rebase_moved
from alphalattice.foundation.market_data_ops.storage.rebase import ex_date_deltas

from .contracts import (
    ListingMarketDataChange,
    MaintenancePhase,
    MaintenanceStatus,
    MarketDataChangeSet,
    gathered_cells,
)


def rebase_reach(
    coordinator: Any,
    cycle: Any,
    changes: MarketDataChangeSet,
    active_panel: Any,
    observed_at: datetime,
) -> tuple[ListingMarketDataChange, ...]:
    """The cycle's listing changes, each listing a split rebased naming its moved Features.

    A split rebased since the Panel's activation names the Features its rebased bars move past
    the data rule's agreement, as the Feature engine's early cutoff finds them. The agreement is
    the rule's while its Goal holds. Without one, any cell that changed at all moved, so no
    rebased listing keeps a value a cold rebuild would not give.

    The cells are an obligation until the Panel holds them. The cycle records them before any
    Feature write, so a cycle restarted after the base rows were rebuilt still owes the Panel
    those cells. The restart's cutoff finds none, but the cells it reads back stand, and its
    next cycle starts without them once the Panel is active.

    A later request reads the recorded obligations since that same Panel prefix, including
    cancelled and blocked cycles. A repaired base cannot rediscover cells its Panel still owes.
    """
    activated = (active_panel or {}).get("activated_at")
    if not isinstance(activated, datetime):
        return changes.listing_changes
    if activated.tzinfo is not None:  # as the store keeps a rebase's observation
        activated = activated.astimezone(UTC).replace(tzinfo=None)
    through = cycle.request.target_market_session
    agreement = getattr(coordinator.diagnose, "agreement", lambda: (0.0, 0.0))()
    with coordinator.market_data.database.read_transaction() as connection:
        rebased = {
            row[0] for row in ex_date_deltas(connection, coordinator.manifest, activated, through)
        }
    by_listing = {item.listing_id: item for item in changes.listing_changes}
    owed = (
        coordinator.registry.rebase_moved_since(
            coordinator.manifest.profile.market_profile_id, after=activated, through=through
        )
        if rebased
        else {}
    )
    spans = panel_spans(coordinator.feature_foundation) if rebased else ()
    for listing_id in sorted(rebased):
        moved = rebase_moved(
            coordinator.feature_foundation, listing_id, as_of_session=through,
            observed_at=observed_at, agreement=agreement, spans=spans,
        )  # fmt: skip
        moved = gathered_cells(owed.get(listing_id, ()), moved)
        if moved:
            change = by_listing.get(listing_id) or ListingMarketDataChange(listing_id)
            # A resumed cycle's recorded cells stand: their rows may be rebuilt, not yet the Panel.
            by_listing[listing_id] = replace(
                change, rebase_moved=gathered_cells(change.rebase_moved, moved)
            )
    listing_changes = tuple(by_listing[key] for key in sorted(by_listing))
    if listing_changes != changes.listing_changes:  # owed from now on: recorded before the build
        coordinator.registry.update(
            cycle.cycle_id, phase=MaintenancePhase.FEATURE, status=MaintenanceStatus.RUNNING,
            observed_at=observed_at, change_set=replace(
                changes, listing_changes=listing_changes, change_set_hash=""
            ),
        )  # fmt: skip
    return listing_changes
