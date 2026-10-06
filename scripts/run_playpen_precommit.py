"""Run pre-commit with sandbox-safe caches and the locked root interpreter."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import cast

from pre_commit.main import main as pre_commit_main  # type: ignore[import-untyped]

PLAYPEN_ROOT = Path(__file__).resolve().parents[1]


def _prepare_environment() -> None:
    cache_root = Path(tempfile.gettempdir()) / "alphalattice-playpen-hooks"
    pre_commit_home = cache_root / "pre-commit"
    uv_cache = cache_root / "uv"
    pre_commit_home.mkdir(parents=True, exist_ok=True)
    uv_cache.mkdir(parents=True, exist_ok=True)
    os.environ["PRE_COMMIT_HOME"] = str(pre_commit_home)
    os.environ["UV_CACHE_DIR"] = str(uv_cache)

    path_entries = [str(Path(sys.executable).resolve().parent)]
    git = shutil.which("git")
    if git is None and os.name == "nt":
        program_files = Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
        candidate = program_files / "Git" / "cmd" / "git.exe"
        if candidate.is_file():
            path_entries.append(str(candidate.parent))
    path_entries.append(os.environ.get("PATH", ""))
    os.environ["PATH"] = os.pathsep.join(path_entries)


def main() -> int:
    _prepare_environment()
    return cast(
        int,
        pre_commit_main(
            (
                "run",
                "--config",
                str(PLAYPEN_ROOT / ".pre-commit-config.yaml"),
                "--hook-stage",
                "pre-commit",
            )
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
