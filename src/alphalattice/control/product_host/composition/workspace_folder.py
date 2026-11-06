"""Whether a folder can become a research workspace, answerable before a Host's libraries load."""

from __future__ import annotations

from pathlib import Path
from typing import Final

RESEARCH_WORKSPACE_MANIFEST_NAME: Final = "research-workspace.json"


def holds_foreign_files(workspace: Path) -> bool:
    """Whether a folder with no manifest holds anything that bars making it a workspace."""
    root = workspace.resolve()
    if (root / RESEARCH_WORKSPACE_MANIFEST_NAME).exists() or not root.is_dir():
        return False
    pending = [root]
    while pending:
        for p in pending.pop().iterdir():
            if p.name == ".alphalattice-writer.lock" and p.parent == root:
                continue
            if p.is_symlink() or p.is_junction() or not p.is_dir():
                return True
            pending.append(p)
    return False
