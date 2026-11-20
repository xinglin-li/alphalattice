"""Shared test support requires data and backup roots outside the person's default root."""

import os
from pathlib import Path


def require_isolated_roots() -> None:
    """Refuse unset and default data or backup roots before loading a support package."""
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    default = Path(os.path.realpath(base / ("AlphaLattice" if os.name == "nt" else "alphalattice")))
    for kind in ("DATA", "BACKUP"):
        variable = f"ALPHALATTICE_{kind}_ROOT"
        named = os.environ.get(variable, "").strip()
        reason = "is unset" if not named else "names the person's default root"
        if not named or Path(os.path.realpath(named)).is_relative_to(default):
            raise RuntimeError(
                f"tests.support_{kind.lower()}_root_unisolated: {variable} {reason}; "
                "name a run-owned root before importing test support"
            )


require_isolated_roots()
