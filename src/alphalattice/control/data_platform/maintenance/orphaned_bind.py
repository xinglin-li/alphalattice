"""A working membership a cancelled Task left active, returned to its Panel's own.

Cancelling a deferred Task runs none of its stages, so its cycle keeps the membership it bound
(`record_working_manifest`) and every next plan meets one its Panel never published. A saga's
compensating action for the uncommitted step: only for the cancelled Task's own last bind, while
the Panel still holds what the cycle was planned on, once, recorded and disclosed in the answer.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

from alphalattice.foundation.feature_engine.storage.repositories import PanelStateRepository
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.kernel.shared_kernel.project_layout import runtime_json

from .contracts import MaintenanceStatus
from .registry import DuckDbWorkspaceMaintenanceRegistry

RESTORED = DuckDbWorkspaceMaintenanceRegistry.MEMBERSHIP_RESTORED_EVENT
UNPROVED = "workspace_maintenance.membership_owner_unproved"


class _Tasks(Protocol):
    def task(self, task_id: UUID) -> Any: ...


def restore_orphaned_membership(
    registry: DuckDbWorkspaceMaintenanceRegistry,
    tasks: _Tasks,
    observed_at: datetime,
    derived: Callable[[str], str | None] | None = None,
) -> dict[str, object]:
    """Return a cancelled Task's orphaned working membership to its Panel's own.

    The answer carries it as `restored_membership`; nothing when no restore is owed. `derived`
    names the Task whose own plan derives a cycle, for binds recorded before they named their
    Task (0.2.3 and earlier): a plan passes it, and a bind it cannot prove is refused by name. A
    cancel passes none: it restores only a proved owner's bind, and never refuses.
    """
    binds = registry.working_binds()
    last = binds[-1] if binds else None
    if last is None or (last.owner_task is None and derived is None):
        return {}
    cycle = registry.cycle(last.cycle_id)
    profile, before = cycle.request.market_profile_id, cycle.request.membership_revision
    market = MarketDataRepository(registry.database_path.parent)
    record = market.readiness.load(profile)
    state = PanelStateRepository(market.database, market_data=market)
    panel = state.feature_panel_snapshot_for_active(profile)
    if (
        record is None
        or panel is None
        or record.active_manifest_revision != last.manifest_revision
        or panel["manifest_revision"] != before
        or before == last.manifest_revision
        or cycle.status in {MaintenanceStatus.COMPLETED, MaintenanceStatus.NOOP}
        or any(
            e.manifest_revision == last.manifest_revision for e in market.membership_events(profile)
        )
    ):
        return {}
    owners = {b.owner_task for b in binds if b.cycle_id == last.cycle_id}
    if derived is not None and None in owners:
        owners = (owners - {None}) | {derived(last.cycle_id)}
    if len(owners) != 1 or None in owners:
        if derived is None:
            return {}
        raise ValueError(f"{UNPROVED}:{last.cycle_id}")
    if not _cancelled(tasks, owner := str(owners.pop())):
        return {}
    details = {
        "reason": RESTORED,
        "owner_task": owner,
        "from_revision": last.manifest_revision,
        "restored_revision": before,
    }
    registry.record_membership_restored(last.cycle_id, details=details, observed_at=observed_at)
    market.readiness.save(
        market_profile_id=profile,
        status="RESEARCH_READY",
        active_manifest_id=market.load_universe_manifest_revision(before).manifest_id,
        active_manifest_revision=before,
        active_membership_fingerprint=record.active_membership_fingerprint,
        active_candidate_manifest_document=record.active_candidate_manifest_document,
        pending_membership_fingerprint=record.pending_membership_fingerprint,
        pending_candidate_manifest_document=record.pending_candidate_manifest_document,
        last_checked_at=record.last_checked_at,
        last_changed_at=record.last_changed_at,
        failure_code=None,
        observed_at=observed_at,
    )
    words = runtime_json("config/registries/messages.json")["entries"][RESTORED]["words"]
    return {"restored_membership": {"cycle_id": last.cycle_id, **details, "words": words}}


def _cancelled(tasks: _Tasks, owner: str) -> bool:
    try:
        return str(tasks.task(UUID(owner)).lifecycle.value) == "CANCELLED"
    except (KeyError, ValueError):
        return False
