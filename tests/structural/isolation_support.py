"""Exercise the owned-root checks on a loaded test support package."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import ModuleType

import pytest


def check_support_roots(
    module: ModuleType, root: str, named: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep the other root owned, then reload with this root set, unset or at the default."""
    base = tmp_path / "person"
    monkeypatch.setenv("LOCALAPPDATA", str(base))
    monkeypatch.setenv("XDG_DATA_HOME", str(base))
    default = base / ("AlphaLattice" if os.name == "nt" else "alphalattice")
    variable = f"ALPHALATTICE_{root}_ROOT"
    monkeypatch.setenv("ALPHALATTICE_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("ALPHALATTICE_BACKUP_ROOT", str(tmp_path / "backups"))
    if named == "unset":
        monkeypatch.delenv(variable, raising=False)
    else:
        monkeypatch.setenv(variable, str(default if named == "default" else tmp_path / "own"))
    if named == "owned":
        importlib.reload(module)
    else:
        with pytest.raises(RuntimeError, match=rf"tests\.support_{root.lower()}_root_unisolated"):
            importlib.reload(module)
