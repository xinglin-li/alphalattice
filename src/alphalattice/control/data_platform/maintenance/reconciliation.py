"""Typed bridge from verified workspace maintenance to Factor reconciliation.

The bridge owns no market, feature, or Factor computation.  It binds an
authoritative maintenance prefix to the existing Factor Research reuse-only
admission contract so a background refresh cannot silently acquire screening
authority.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from alphalattice.control.data_platform.maintenance.contracts import (
    MaintenancePhase,
    MaintenanceStatus,
    WorkspaceMaintenanceCycle,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash


class MaintenanceReconciliationError(ValueError):
    """A stable fail-closed admission failure with a user-safe code."""

    def __init__(self, message: str, *, failure_code: str) -> None:
        """Retain a reconciliation failure with its stable cause.

        Args:
            message: Human-readable error detail.
            failure_code: Stable failure code retained for diagnosis.
        """
        super().__init__(message)
        self.failure_code = failure_code


class VerifiedMaintenanceCompletion(BaseModel):  # type: ignore[misc]
    """Content-addressed verified prefix emitted by the maintenance owner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["VerifiedMaintenanceCompletion"] = "VerifiedMaintenanceCompletion"
    market_profile_id: str
    cycle_id: str
    maintenance_request_hash: str
    maintenance_status: Literal["completed", "noop"]
    target_market_session: date
    membership_revision: str
    change_set_hash: str | None = None
    panel_snapshot_hash: str
    panel_manifest_ref: str
    panel_as_of_session: date
    catalog_hash: str
    completion_hash: str

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> VerifiedMaintenanceCompletion:
        """Require the reconciled panel to reach its target and verify completion identity.

        Returns:
            This validated contract.

        Raises:
            ValueError: Panel session is behind the target or canonical completion identity differs.
        """
        if self.panel_as_of_session < self.target_market_session:
            raise ValueError("maintenance completion panel is behind its target")
        identity = self.model_dump(mode="json", exclude={"completion_hash"})
        if self.completion_hash != canonical_hash(identity):
            raise ValueError("maintenance completion hash is invalid")
        return self


def capture_verified_maintenance_completion(
    *,
    cycle: WorkspaceMaintenanceCycle,
    panel_snapshot: Mapping[str, object],
) -> VerifiedMaintenanceCompletion:
    """Capture only a terminal cycle backed by an active immutable panel."""
    if cycle.status not in {MaintenanceStatus.COMPLETED, MaintenanceStatus.NOOP}:
        raise MaintenanceReconciliationError(
            "workspace maintenance has not reached a reusable terminal state",
            failure_code="MAINTENANCE_PREFIX_NOT_VERIFIED",
        )
    if cycle.phase is not MaintenancePhase.COMPLETED:
        raise MaintenanceReconciliationError(
            "workspace maintenance terminal state lacks a completed phase",
            failure_code="MAINTENANCE_PHASE_NOT_VERIFIED",
        )
    required = {
        "snapshot_hash",
        "manifest_uri",
        "as_of_session",
        "catalog_hash",
    }
    if not required.issubset(panel_snapshot):
        raise MaintenanceReconciliationError(
            "active Feature Panel snapshot metadata is incomplete",
            failure_code="ACTIVE_PANEL_SNAPSHOT_INCOMPLETE",
        )
    change_set_hash = (
        cycle.market_data_change_set.change_set_hash
        if cycle.market_data_change_set is not None
        else None
    )
    values = {
        "market_profile_id": cycle.request.market_profile_id,
        "cycle_id": cycle.cycle_id,
        "maintenance_request_hash": cycle.request.request_hash,
        "maintenance_status": cycle.status.value,
        "target_market_session": cycle.request.target_market_session,
        "membership_revision": cycle.request.membership_revision,
        "change_set_hash": change_set_hash,
        "panel_snapshot_hash": str(panel_snapshot["snapshot_hash"]),
        "panel_manifest_ref": str(panel_snapshot["manifest_uri"]),
        "panel_as_of_session": panel_snapshot["as_of_session"],
        "catalog_hash": str(panel_snapshot["catalog_hash"]),
    }
    identity = VerifiedMaintenanceCompletion.model_construct(
        **values, completion_hash=""
    ).model_dump(mode="json", exclude={"completion_hash"})
    return VerifiedMaintenanceCompletion(**values, completion_hash=canonical_hash(identity))


__all__ = [
    "MaintenanceReconciliationError",
    "VerifiedMaintenanceCompletion",
    "capture_verified_maintenance_completion",
]
