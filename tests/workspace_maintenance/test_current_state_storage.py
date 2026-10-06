from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from alphalattice.control.product_host.storage.contracts import (
    GIB,
    StorageBudgetResolution,
    StorageEvictionPlan,
    resolve_storage_budget,
)
from alphalattice.control.product_host.storage.inventory import (
    StorageInventoryError,
    require_storage_capacity,
    workspace_storage_capacity,
)
from alphalattice.control.workspace_runtime.storage.capacity import StorageCapStore
from alphalattice.kernel.shared_kernel.identity import canonical_hash


def test_historical_budget_receipts_keep_their_exact_derivation() -> None:
    desktop = resolve_storage_budget(active_listing_count=466, research_session_count=2515)
    assert desktop.managed_cap_bytes == 10 * GIB
    assert desktop.cleanup_target_bytes == 8 * GIB
    assert desktop.high_water_bytes == 9 * GIB
    large = resolve_storage_budget(active_listing_count=3000, research_session_count=2515)
    assert large.managed_cap_bytes == 18 * GIB
    assert large.managed_cap_bytes > desktop.managed_cap_bytes
    assert StorageBudgetResolution.model_validate_json(desktop.model_dump_json()) == desktop


def test_cleanup_plan_checks_old_cap_receipts_and_new_plans_exclude_execution_capacity(
    tmp_path: Path,
) -> None:
    """V680: legacy content integrity survives without carrying capacity into a new plan."""
    fields = dict(
        kind="StorageEvictionPlan",
        operation_id="cleanup",
        inventory_hash="a" * 64,
        roots=[
            dict(
                root_kind="CURRENT_ACTIVE",
                root_id="current",
                head_hash=None,
                panel_snapshot_hash=None,
                approved_bytes=None,
                approval_hash=None,
            )
        ],
        evict_paths=[],
        evict_bytes=0,
        retained_bytes=12,
        expected_final_bytes=12,
    )
    current = StorageEvictionPlan.model_validate({**fields, "plan_hash": canonical_hash(fields)})
    assert current.managed_cap_bytes is None
    old_fields = {**fields, "managed_cap_bytes": 10 * GIB}
    old = StorageEvictionPlan.model_validate(
        {**old_fields, "plan_hash": canonical_hash(old_fields)}
    )
    recorded = old.model_dump_json()
    StorageCapStore(tmp_path).write(20 * GIB, chosen_by="HUMAN", chosen_at=datetime.now(UTC))
    assert StorageEvictionPlan.model_validate_json(recorded) == old
    with pytest.raises(ValueError, match="plan hash is invalid"):
        StorageEvictionPlan.model_validate(
            {**old.model_dump(mode="json"), "managed_cap_bytes": 20 * GIB}
        )


def test_storage_cap_counts_managed_models_panels_and_artifacts_and_follows_the_operator(
    tmp_path: Path,
) -> None:
    """V680: live execution capacity covers physical files, including linked inputs once."""
    contents = {
        "market-data.duckdb": b"data",
        "artifacts/feature-panel/panel.parquet": b"feature panel",
        "research-experiments/model.bin": b"model",
        "runtime/artifacts/product-host/result.json": b"artifact",
    }
    for name, body in contents.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
    alias = tmp_path / "research-inputs/linked-panel.parquet"
    alias.parent.mkdir()
    os.link(tmp_path / "artifacts/feature-panel/panel.parquet", alias)
    used = sum(map(len, contents.values()))
    store = StorageCapStore(tmp_path)
    automatic = workspace_storage_capacity(tmp_path)
    assert automatic.measured_data_bytes == used
    assert automatic.setting.cap_bytes == "auto"
    assert automatic.cap_bytes == used + min(
        max(used // 4, automatic.free_disk_bytes // 100), automatic.free_disk_bytes // 4
    )
    assert "cannot predict future" in automatic.estimate_limit
    retained = {name: (tmp_path / name).read_bytes() for name in contents}
    store.write(used, chosen_by="HUMAN", chosen_at=datetime.now(UTC))
    with pytest.raises(StorageInventoryError) as refusal:
        require_storage_capacity(tmp_path, additional_bytes=1)
    assert refusal.value.failure_code == "storage.managed_capacity_exceeded"
    assert "Raise the cap in Settings or plan a cleanup" in str(refusal.value)
    store.write(str(used + 1), chosen_by="HUMAN", chosen_at=datetime.now(UTC))
    require_storage_capacity(tmp_path, additional_bytes=1)
    assert {name: (tmp_path / name).read_bytes() for name in contents} == retained
    store.write("auto", chosen_by="HUMAN", chosen_at=datetime.now(UTC))
    assert store.read().cap_bytes == "auto"


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "0", "-1", "1.5", "unknown"])
def test_storage_cap_rejects_values_that_are_not_positive_whole_bytes(
    tmp_path: Path, value: object
) -> None:
    with pytest.raises(ValueError, match=r"storage\.cap_setting_invalid"):
        StorageCapStore(tmp_path).write(value, chosen_by="HUMAN", chosen_at=datetime.now(UTC))
    assert not StorageCapStore(tmp_path).path.exists()
