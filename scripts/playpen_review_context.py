"""Emit an identity-bound context for reviewers of the nested Playpen repository."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

PLAYPEN_ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", "-C", str(PLAYPEN_ROOT), *args),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", default="HEAD")
    args = parser.parse_args()
    repository_root = Path(_git("rev-parse", "--show-toplevel")).resolve()
    if repository_root != PLAYPEN_ROOT:
        raise RuntimeError("review context resolved the outer repository instead of Playpen")
    base_sha = _git("rev-parse", f"{args.base}^{{commit}}")
    head_sha = _git("rev-parse", f"{args.head}^{{commit}}")
    if _git("merge-base", "--is-ancestor", base_sha, head_sha) not in {"", base_sha}:
        raise RuntimeError("review base is not an ancestor of the frozen head")
    status = _git("status", "--short")
    if status:
        raise RuntimeError("review snapshot requires a clean nested Playpen worktree")
    payload = {
        "repository_kind": "nested-local-git",
        "repository_root": str(repository_root),
        "base_sha": base_sha,
        "head_sha": head_sha,
        "required_git_prefix": ["git", "-C", str(repository_root)],
        "diff_command": [
            "git",
            "-C",
            str(repository_root),
            "diff",
            "--find-renames",
            f"{base_sha}..{head_sha}",
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
