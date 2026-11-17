"""Run pre-commit with sandbox-safe caches and the locked root interpreter."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter
from typing import cast

HOOK_STARTED = perf_counter()
PLAYPEN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLAYPEN_ROOT / "src"))
from alphalattice.kernel.shared_kernel import spawn  # noqa: E402

GITLEAKS_SHA256 = {
    "darwin-arm64": "ba52fb1bfabbcde42f032afad3d6e0b19dff8ed105229a16e7caa338bbc0e84f",
    "darwin-x64": "cee01fea7173f1b779dff188e1c26ecbcb4027d394acc573b23aaf0be260e291",
    "linux-arm64": "00e91bbe655bd7c47753e8cfe61cb76ea1a5d7e7702fe161ee40102b46b3823b",
    "linux-x64": "88f91962aa2f93ac6ab281d553b9e125f5197bbbce38f9f2437f7299c32e5509",
    "win32-arm64": "200df852fdecbedb19a33960657333cba5e231740bc8968972b507b50f93b194",
    "win32-x64": "17157e2ee8b76fc8b1d8bee607a250e34b8a8023c8bc81822d4b5ee4d78fcb7c",
}


def verified_gitleaks(root: Path, platform_key: str) -> Path:
    """Return the exact official release executable the hook will run."""
    expected = GITLEAKS_SHA256.get(platform_key)
    if expected is None:
        raise ValueError(f"playpen.gitleaks_unavailable: unsupported platform {platform_key}")
    workshop = next(
        (parent for parent in root.parents if (parent / "tools/container-roots.json").is_file()),
        None,
    )
    if workshop is not None:
        name = "gitleaks.exe" if platform_key.startswith("win32-") else "gitleaks"
        binary = workshop / "tools/.state/gitleaks/8.30.1" / name
    else:
        found = shutil.which("gitleaks")
        if found is None:
            raise ValueError(
                "playpen.gitleaks_unavailable: install official gitleaks 8.30.1 on PATH"
            )
        binary = Path(found)
    try:
        with binary.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
    except FileNotFoundError as error:
        raise ValueError(
            f"playpen.gitleaks_unavailable: {binary}; restore official gitleaks 8.30.1"
        ) from error
    except OSError as error:
        raise ValueError(f"playpen.gitleaks_unreadable: {binary}; restore access") from error
    if digest != expected:
        raise ValueError(f"playpen.gitleaks_untrusted: {binary}; restore official gitleaks 8.30.1")
    return binary.absolute()


def run_gitleaks(arguments: list[str]) -> int:
    architecture = {"amd64": "x64", "x86_64": "x64", "aarch64": "arm64", "arm64": "arm64"}.get(
        platform.machine().lower(), platform.machine().lower()
    )
    try:
        binary = verified_gitleaks(PLAYPEN_ROOT, f"{sys.platform}-{architecture}")
        return spawn.run(
            [str(binary), *arguments],
            check=False,
            stdin=subprocess.DEVNULL,
            timeout=120,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).returncode
    except subprocess.TimeoutExpired:
        print("playpen.gitleaks_timeout: secret scan exceeded 120 seconds", file=sys.stderr)
        return 1
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1


def run_secrets(filenames: list[str]) -> int:
    """Run the existing private-key and gitleaks checks on the same staged files."""
    private_keys = importlib.import_module("pre_commit_hooks.detect_private_key").main

    try:
        keys = private_keys(["--", *filenames])
    finally:
        leaks = run_gitleaks(
            ["git", "--staged", "--no-banner", "--redact", "--config", ".github/gitleaks.toml"]
        )
    return int(bool(keys or leaks))


def _prepare_environment() -> None:
    cache_root = Path(tempfile.gettempdir()) / "alphalattice-playpen-hooks"
    pre_commit_home = cache_root / "pre-commit"
    uv_cache = cache_root / "uv"
    pre_commit_home.mkdir(parents=True, exist_ok=True)
    uv_cache.mkdir(parents=True, exist_ok=True)
    os.environ["PRE_COMMIT_HOME"] = str(pre_commit_home)
    os.environ["UV_CACHE_DIR"] = str(uv_cache)

    path_entries = [str(Path(sys.executable).absolute().parent)]
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
            str(Path(sys.executable).with_name("ruff.exe" if os.name == "nt" else "ruff")),
            "format",
            "--no-cache",
            "--config",
            str(after / "pyproject.toml"),
            "--",
        )
        for batch in _batched(tuple(path for _, _, path in staged), prefix=prefix):
            spawn.run(
                (*prefix, *batch),
                cwd=after,
                check=True,
                stdin=subprocess.DEVNULL,
                timeout=60,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
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
                attrs = spawn.run(
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
                    timeout=60,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
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
            patch = spawn.run(
                ("git", "diff", "--no-index", "--no-prefix", "--binary", "--", "before", "after"),
                cwd=mirror,
                stdin=subprocess.DEVNULL,
                timeout=60,
                capture_output=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if patch.returncode != 1:
                raise ValueError(
                    "playpen.format_patch_unreadable: inspect the staged files and retry"
                )
            spawn.run(
                ("git", "apply", "--index", "-p1", "--whitespace=nowarn", "-"),
                input=patch.stdout,
                cwd=PLAYPEN_ROOT,
                check=True,
                timeout=60,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )


def run_hooks() -> int:
    """Run every configured hook, regardless of a caller's inherited skip list."""
    from pre_commit.main import main as pre_commit_main  # type: ignore[import-untyped]

    skipped = os.environ.pop("SKIP", None)
    try:
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
    finally:
        if skipped is not None:
            os.environ["SKIP"] = skipped


def _keep_bytecode() -> None:
    """Let the hook's Python children keep their bytecode outside the checkout.

    A caller that forbids bytecode (PYTHONDONTWRITEBYTECODE, set to keep the tree clean) made
    every check compile the product's modules again on every commit, seconds each. The cache
    lives beside the pre-commit cache, never in the tree; the commit stage keeps its own setting.
    """
    os.environ.pop("PYTHONDONTWRITEBYTECODE", None)
    cache = Path(tempfile.gettempdir()) / "alphalattice-playpen-hooks" / "pycache"
    os.environ["PYTHONPYCACHEPREFIX"] = str(cache)


def main() -> int:
    try:
        _keep_bytecode()
        _prepare_environment()
        _format_staged()
    except subprocess.TimeoutExpired as error:
        print(
            f"playpen.format_timeout: {error}; inspect the named command and retry", file=sys.stderr
        )
        return 1
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"playpen.format_refused: {error}; inspect the named file and retry", file=sys.stderr)
        return 1
    result = run_hooks()
    elapsed = perf_counter() - HOOK_STARTED
    system = platform.system()
    over_budget = int(elapsed >= 5)
    try:
        with (Path(os.environ["PRE_COMMIT_HOME"]) / "hook-timing.jsonl").open(
            "a", encoding="utf-8", newline="\n"
        ) as record:
            record.write(
                json.dumps({"os": system, "seconds": elapsed, "over_budget": over_budget}) + "\n"
            )
    except (KeyError, OSError) as error:
        print(f"playpen.hook_timing_unrecorded: {error}", file=sys.stderr)
    print(f"PLAYPEN_HOOK_TIMING os={system} seconds={elapsed:.3f} over_budget={over_budget}")
    if elapsed >= 5:
        print(
            f"playpen.hook_budget: {elapsed:.3f}s exceeds the 5s target; "
            "the commit is judged by the checks, not by elapsed time",
            file=sys.stderr,
        )
    return result


if __name__ == "__main__":
    if sys.argv[1:2] == ["--gitleaks"]:
        raise SystemExit(run_gitleaks(sys.argv[2:]))
    raise SystemExit(run_secrets(sys.argv[2:]) if sys.argv[1:2] == ["--secrets"] else main())
