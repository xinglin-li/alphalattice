"""Typed contracts for current-state desktop storage governance."""

from __future__ import annotations

import math
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash

GIB = 1024**3


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class StorageRootKind(StrEnum):
    """Name the active, rollback, explicitly pinned and in-flight retention root kinds."""

    CURRENT_ACTIVE = "CURRENT_ACTIVE"
    PREVIOUS_ROLLBACK = "PREVIOUS_ROLLBACK"
    USER_PINNED = "USER_PINNED"
    IN_FLIGHT_RECOVERY = "IN_FLIGHT_RECOVERY"


class PhysicalAvailability(StrEnum):
    """Distinguish available bytes, recorded retention eviction and missing/tampered storage."""

    AVAILABLE = "AVAILABLE"
    EVICTED_BY_RETENTION = "EVICTED_BY_RETENTION"
    MISSING_OR_TAMPERED = "MISSING_OR_TAMPERED"


class StorageBudgetResolution(_Contract):
    """Verify historical listing/session budget receipts, including their original cap."""

    active_listing_count: int = Field(gt=0)
    research_session_count: int = Field(gt=0)
    retained_generations: int = Field(default=2, ge=1)
    calibrated_bytes_per_pair: int = Field(default=1024, gt=0)
    listing_session_pairs: int = Field(gt=0)
    estimated_core_bytes: int = Field(gt=0)
    managed_cap_bytes: int = Field(ge=10 * GIB)
    cleanup_target_bytes: int = Field(gt=0)
    high_water_bytes: int = Field(gt=0)
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_resolution(self) -> StorageBudgetResolution:
        """Require exact pair/core/cap derivation, cleanup thresholds and budget policy identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Derived pair/core/cap, 80% cleanup/90% high-water threshold or policy_hash
                differs.
        """
        expected_pairs = self.active_listing_count * self.research_session_count
        expected_core = expected_pairs * self.calibrated_bytes_per_pair * self.retained_generations
        expected_cap = max(10 * GIB, math.ceil(expected_core * 1.25 / GIB) * GIB)
        if self.listing_session_pairs != expected_pairs:
            raise ValueError("storage listing-session pair count is invalid")
        if self.estimated_core_bytes != expected_core or self.managed_cap_bytes != expected_cap:
            raise ValueError("storage managed-cap derivation is invalid")
        if self.cleanup_target_bytes != int(expected_cap * 0.8):
            raise ValueError("storage cleanup target is invalid")
        if self.high_water_bytes != int(expected_cap * 0.9):
            raise ValueError("storage high-water mark is invalid")
        if self.policy_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"policy_hash"})
        ):
            raise ValueError("storage budget policy hash is invalid")
        return self


class StorageRetentionRoot(_Contract):
    """Declare one explicit retention root and any byte-bound human pin approval."""

    root_kind: StorageRootKind
    root_id: str
    head_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    panel_snapshot_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    approved_bytes: int | None = Field(default=None, ge=0)
    approval_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_pin(self) -> StorageRetentionRoot:
        """Require byte-bound approval only for explicitly user-pinned roots.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: A user pin lacks bytes/approval or another root carries a pin override.
        """
        if self.root_kind == StorageRootKind.USER_PINNED:
            if self.approved_bytes is None or self.approval_hash is None:
                raise ValueError("user-pinned storage requires a byte-bound approval")
        elif self.approved_bytes is not None or self.approval_hash is not None:
            raise ValueError("only user-pinned storage may carry an override")
        return self


class CurrentStateStorageInventory(_Contract):
    """Seal captured storage counts, current/previous heads and downstream identity references."""

    kind: Literal["CurrentStateStorageInventory"] = "CurrentStateStorageInventory"
    operation_id: str
    captured_at: datetime
    database_bytes: int = Field(ge=0)
    artifact_bytes: int = Field(ge=0)
    staging_bytes: int = Field(ge=0)
    available_disk_bytes: int = Field(ge=0)
    active_catalog_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    active_factor_ids: tuple[str, ...]
    stored_feature_columns: tuple[str, ...]
    orphan_feature_columns: tuple[str, ...]
    feature_generation_rows: tuple[tuple[str, int], ...]
    active_feature_rows: int = Field(ge=0)
    cutoff_set_count: int = Field(ge=0)
    active_listing_count: int = Field(gt=0)
    research_session_count: int = Field(gt=0)
    current_head_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    previous_head_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    panel_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    downstream_identities: tuple[tuple[str, str], ...]
    inventory_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_inventory(self) -> CurrentStateStorageInventory:
        """Require aware capture, unique Feature axis and exact inventory binding.

        Require aware capture time, ordered unique Feature axis and exact orphan/inventory identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Clock, active axis, derived orphan columns or inventory_hash differs.
        """
        if self.captured_at.tzinfo is None or self.captured_at.utcoffset() is None:
            raise ValueError("storage inventory clock must be timezone-aware")
        if self.active_factor_ids != tuple(dict.fromkeys(self.active_factor_ids)):
            raise ValueError("active Feature axis must be ordered and unique")
        expected_orphans = tuple(
            sorted(set(self.stored_feature_columns) - set(self.active_factor_ids))
        )
        if self.orphan_feature_columns != expected_orphans:
            raise ValueError("storage inventory orphan Feature columns are invalid")
        if self.inventory_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"inventory_hash"})
        ):
            raise ValueError("storage inventory hash is invalid")
        return self


class StorageEvictionPlan(_Contract):
    """Seal explicit retained roots, sorted eviction paths and byte estimates for cleanup."""

    kind: Literal["StorageEvictionPlan"] = "StorageEvictionPlan"
    operation_id: str
    inventory_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    roots: tuple[StorageRetentionRoot, ...]
    evict_paths: tuple[str, ...]
    evict_bytes: int = Field(ge=0)
    retained_bytes: int = Field(ge=0)
    managed_cap_bytes: int | None = Field(default=None, gt=0)
    """Historical plans' recorded cap; new plans leave execution settings out."""
    expected_final_bytes: int = Field(ge=0)
    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_plan(self) -> StorageEvictionPlan:
        """Require sorted unique eviction paths, an active root and exact plan identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Eviction paths, required current root or canonical plan_hash differs.
        """
        if self.evict_paths != tuple(sorted(set(self.evict_paths))):
            raise ValueError("storage eviction paths must be sorted and unique")
        if not any(root.root_kind == StorageRootKind.CURRENT_ACTIVE for root in self.roots):
            raise ValueError("storage eviction plan has no current root")
        payload = self.model_dump(mode="json", exclude={"plan_hash"})
        if self.managed_cap_bytes is None:
            payload.pop("managed_cap_bytes")
        if self.plan_hash != canonical_hash(payload):
            raise ValueError("storage eviction plan hash is invalid")
        return self


def storage_input_estimate(
    *, active_listing_count: int, research_session_count: int
) -> dict[str, int]:
    """Estimate two generations of core inputs only, excluding future panels/models/studies.

    This sizing estimate admits preparation's initial writes. It grants no workspace capacity
    and carries no operator setting or policy identity.
    """
    if active_listing_count <= 0 or research_session_count <= 0:
        raise ValueError("storage input axes must be positive")
    pairs = active_listing_count * research_session_count
    return {
        "active_listing_count": active_listing_count,
        "research_session_count": research_session_count,
        "listing_session_pairs": pairs,
        "estimated_core_bytes": pairs * 2 * 1024,
    }


def resolve_storage_budget(
    *,
    active_listing_count: int,
    research_session_count: int,
    retained_generations: int = 2,
    calibrated_bytes_per_pair: int = 1024,
) -> StorageBudgetResolution:
    """Derive an explicit managed storage cap and cleanup thresholds from source axes.

    Args:
        active_listing_count: Positive admitted listing count.
        research_session_count: Positive admitted research session count.
        retained_generations: Positive retained generation count.
        calibrated_bytes_per_pair: Positive declared byte estimate per listing/session pair.

    Returns:
        Validated budget with at least 10 GiB cap, 25% rounded headroom, 80% cleanup target and 90%
        high-water mark.
    """
    pairs = active_listing_count * research_session_count
    estimated = pairs * retained_generations * calibrated_bytes_per_pair
    cap = max(10 * GIB, math.ceil(estimated * 1.25 / GIB) * GIB)
    values = {
        "active_listing_count": active_listing_count,
        "research_session_count": research_session_count,
        "retained_generations": retained_generations,
        "calibrated_bytes_per_pair": calibrated_bytes_per_pair,
        "listing_session_pairs": pairs,
        "estimated_core_bytes": estimated,
        "managed_cap_bytes": cap,
        "cleanup_target_bytes": int(cap * 0.8),
        "high_water_bytes": int(cap * 0.9),
    }
    return StorageBudgetResolution(**values, policy_hash=canonical_hash(values))


__all__ = [
    "GIB",
    "CurrentStateStorageInventory",
    "PhysicalAvailability",
    "StorageBudgetResolution",
    "StorageEvictionPlan",
    "StorageRetentionRoot",
    "StorageRootKind",
    "resolve_storage_budget",
]
