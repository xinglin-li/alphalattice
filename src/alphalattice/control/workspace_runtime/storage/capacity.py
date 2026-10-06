"""Operator storage configuration, independent of sealed computation identities."""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class StorageCapSetting(BaseModel):  # type: ignore[misc]
    """Keep the selected byte cap and its operator provenance, never a method binding."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    cap_bytes: Annotated[int, Field(strict=True, gt=0)] | Literal["auto"] = "auto"
    chosen_by: str | None = None
    chosen_at: datetime | None = None


class StorageCapacity(BaseModel):  # type: ignore[misc]
    """Read the current setting against measured managed data and available disk."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    cap_bytes: int = Field(gt=0)
    automatic_cap_bytes: int = Field(gt=0)
    measured_data_bytes: int = Field(ge=0)
    free_disk_bytes: int = Field(ge=0)
    setting: StorageCapSetting
    estimate_limit: str = (
        "The estimate counts managed files already held, including panels, models and artifacts; "
        "it cannot predict future studies or models."
    )
    automatic_basis: str = (
        "Automatic capacity adds the larger of a quarter of managed data and one percent of "
        "free disk, limited to a quarter of free disk."
    )


class StorageCapStore:
    """Read and atomically set one workspace's operator storage cap."""

    def __init__(self, workspace: Path) -> None:
        """Locate configuration under the caller's explicit workspace."""
        self.workspace = workspace
        self.path = workspace / "runtime/execution/storage-cap.json"

    def read(self) -> StorageCapSetting:
        """Read a typed setting, defaulting to automatic capacity on this machine."""
        if not self.path.is_file():
            return StorageCapSetting()
        try:
            setting = StorageCapSetting.model_validate_json(self.path.read_bytes())
            self.parse(setting.cap_bytes)
            return setting
        except (OSError, ValueError) as error:
            raise ValueError("storage.cap_setting_unreadable") from error

    @staticmethod
    def parse(value: object) -> int | Literal["auto"]:
        """Accept automatic capacity or a strictly positive whole byte count."""
        if value == "auto":
            return "auto"
        if type(value) is int and value > 0:
            return value
        if isinstance(value, str) and value.isdecimal() and int(value) > 0:
            return int(value)
        raise ValueError("storage.cap_setting_invalid")

    def write(self, value: object, *, chosen_by: str, chosen_at: datetime) -> StorageCapSetting:
        """Atomically retain the operator's setting; nothing is evicted or resealed."""
        setting = StorageCapSetting(
            cap_bytes=self.parse(value), chosen_by=chosen_by, chosen_at=chosen_at
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        staged = self.path.with_suffix(".partial")
        staged.write_bytes(setting.model_dump_json().encode("utf-8"))
        os.replace(staged, self.path)
        return setting

    def capacity(self, *, measured_data_bytes: int) -> StorageCapacity:
        """Resolve the same live setting for every storage reader and write admission."""
        setting = self.read()
        free = shutil.disk_usage(self.workspace).free
        allowance = min(max(measured_data_bytes // 4, free // 100), free // 4)
        automatic = max(1, measured_data_bytes + allowance)
        return StorageCapacity(
            cap_bytes=automatic if setting.cap_bytes == "auto" else setting.cap_bytes,
            automatic_cap_bytes=automatic,
            measured_data_bytes=measured_data_bytes,
            free_disk_bytes=free,
            setting=setting,
        )
