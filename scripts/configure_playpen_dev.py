"""Configure the nested Playpen repository's deterministic local hook path."""

from __future__ import annotations

import sys
from pathlib import Path

PLAYPEN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLAYPEN_ROOT / "src"))
from alphalattice.kernel.shared_kernel import spawn  # noqa: E402


def main() -> int:
    hook = PLAYPEN_ROOT / ".githooks" / "pre-commit"
    if not hook.is_file():
        raise FileNotFoundError(hook)
    spawn.run(
        ("git", "config", "--local", "core.hooksPath", ".githooks"),
        cwd=PLAYPEN_ROOT,
        check=True,
    )
    configured = spawn.run(
        ("git", "config", "--local", "--get", "core.hooksPath"),
        cwd=PLAYPEN_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if configured != ".githooks":
        raise RuntimeError("Playpen hook path configuration did not persist")
    print(f"Configured Playpen hooks with {sys.executable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
