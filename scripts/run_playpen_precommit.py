"""Run pre-commit with sandbox-safe caches and the locked root interpreter."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
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


def _format_staged() -> None:
    """Format staged Python in a private mirror, never a partially staged file."""
    sys.path.insert(0, str(PLAYPEN_ROOT / "src"))
    from check_playpen import _batched  # type: ignore[import-not-found]
    from devtools.architecture.git_inputs import read_blobs, staged_changes

    pairs = staged_changes(PLAYPEN_ROOT)
    staged = [
        (meta[1], meta[3], path)
        for meta, path in pairs
        if meta[4] in {"A", "M", "T"} and path.endswith(".py")
    ]
    if not staged:
        return
    if unsafe := [path for mode, _, path in staged if mode not in {"100644", "100755"}]:
        raise ValueError(
            f"playpen.format_unsafe: {unsafe[0]}; stage a regular file before formatting"
        )
    blobs = read_blobs([oid for _, oid, _ in staged] + [":pyproject.toml"], PLAYPEN_ROOT)
    if any(oid not in blobs for _, oid, _ in staged) or ":pyproject.toml" not in blobs:
        raise ValueError(
            "playpen.format_unreadable: staged source or pyproject.toml; "
            "restore the index and retry"
        )
    original = {
        path: blobs[oid].encode("utf-8", errors="surrogateescape") for _, oid, path in staged
    }
    with tempfile.TemporaryDirectory(
        prefix="staged-format-", dir=os.environ["PRE_COMMIT_HOME"]
    ) as folder:
        mirror = Path(folder)
        before, after = mirror / "before", mirror / "after"
        before.mkdir()
        after.mkdir()
        for root in (before, after):
            (root / "pyproject.toml").write_bytes(
                blobs[":pyproject.toml"].encode("utf-8", errors="surrogateescape")
            )
        for _, _, path in staged:
            for root in (before, after):
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(original[path])
        prefix = (
            sys.executable,
            "-m",
            "ruff",
            "format",
            "--no-cache",
            "--config",
            str(after / "pyproject.toml"),
            "--",
        )
        for batch in _batched(tuple(path for _, _, path in staged), prefix=prefix):
            subprocess.run((*prefix, *batch), cwd=after, check=True)
        changes, partial = [], []
        for mode, oid, path in staged:
            formatted = (after / path).read_bytes()
            if formatted == original[path]:
                continue
            target = PLAYPEN_ROOT / path
            if target.is_symlink() or target.resolve() != target:
                raise ValueError(
                    f"playpen.format_unsafe: {path}; restore a regular file inside the checkout"
                )
            try:
                same = target.read_bytes() == original[path]
            except OSError:
                same = False
            if not same:
                partial.append(path)
                (after / path).write_bytes(original[path])
            else:
                changes.append((mode, oid, path, formatted))
        if partial:
            command = " ".join(shlex.quote(path) for path in partial)
            print(
                f"playpen.format_partial: {', '.join(partial)}; review unstaged changes, "
                f"then run python -m ruff format -- {command}",
                file=sys.stderr,
            )
        if changes:
            current = {path: (meta[1], meta[3]) for meta, path in staged_changes(PLAYPEN_ROOT)}
            if any(current.get(path) != (mode, oid) for mode, oid, path, _ in changes):
                raise ValueError(
                    "playpen.format_index_changed: staged files changed during formatting; "
                    "retry the commit"
                )
            encoded = b"".join(
                path.encode("utf-8", errors="surrogateescape") + b"\0" for _, _, path, _ in changes
            )
            # Git apply reads working-tree attributes; staged attributes alone are insufficient.
            for cached in (True, False):
                attrs = subprocess.run(
                    (
                        "git",
                        "check-attr",
                        *(["--cached"] if cached else []),
                        "-z",
                        "--stdin",
                        "filter",
                        "working-tree-encoding",
                    ),
                    input=encoded,
                    cwd=PLAYPEN_ROOT,
                    capture_output=True,
                    check=True,
                ).stdout.split(b"\0")
                if attrs[-1:] != [b""] or len(attrs) != 6 * len(changes) + 1:
                    raise ValueError(
                        "playpen.format_attributes_unreadable: restore Git attributes and retry"
                    )
                expected = [
                    (path.encode("utf-8", errors="surrogateescape"), name)
                    for _, _, path, _ in changes
                    for name in (b"filter", b"working-tree-encoding")
                ]
                for (encoded_path, name), (seen, attribute, value) in zip(
                    expected, zip(attrs[::3], attrs[1::3], attrs[2::3], strict=False), strict=True
                ):
                    if (encoded_path, name) != (seen, attribute):
                        raise ValueError(
                            "playpen.format_attributes_unreadable: restore Git attributes and retry"
                        )
                    if value not in {b"unspecified", b"unset"}:
                        raise ValueError(
                            f"playpen.format_unsafe: {os.fsdecode(encoded_path)} "
                            f"has {os.fsdecode(name)}; "
                            "format and stage it manually"
                        )
            patch = subprocess.run(
                ("git", "diff", "--no-index", "--no-prefix", "--binary", "--", "before", "after"),
                cwd=mirror,
                capture_output=True,
            )
            if patch.returncode != 1:
                raise ValueError(
                    "playpen.format_patch_unreadable: inspect the staged files and retry"
                )
            subprocess.run(
                ("git", "apply", "--index", "-p1", "--whitespace=nowarn", "-"),
                input=patch.stdout,
                cwd=PLAYPEN_ROOT,
                check=True,
            )


def main() -> int:
    _prepare_environment()
    try:
        _format_staged()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"playpen.format_refused: {error}; inspect the named file and retry", file=sys.stderr)
        return 1
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
