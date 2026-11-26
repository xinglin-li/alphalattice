"""The package import law, checked by import-linter and Ruff from the one registry.

``config/package-architecture.json`` owns the law: each package's dependency level and declared
peers, and the owners a cutover retired. ``build`` turns it into
an import-linter configuration (one ``forbidden`` contract per package, direct imports only, as the
home-built check had it) and a Ruff configuration that bans every retired name with TID251, which
bans an import by name whether or not its module still exists (grimp drops such an import). Both
are built at run time, never committed; ``check`` writes them into the caller's run as evidence and
runs both tools over the tracked tree.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from devtools.architecture.structural import (
    _PHYSICAL_NAMESPACE_GROUPS,
    PackageArchitectureRegistry,
)

ROOTS = ("alphalattice", "devtools")
"""The source roots grimp builds its graph over."""

COMMAND_LINE = 32_000
"""UTF-16 units one Ruff launch may take: Windows' CreateProcess refuses 32,767 and more."""


def _physical(registry: PackageArchitectureRegistry, logical: str) -> tuple[str, ...]:
    """Every physical name a logical module name stands for, as the old check normalized imports:
    a registered package through its target, and any name under each namespace group and bare."""
    head, _, rest = logical.partition(".")
    names = {logical, *(f"alphalattice.{group}.{logical}" for group in _PHYSICAL_NAMESPACE_GROUPS)}
    if package := registry.by_package.get(head):
        names.add(package.physical_namespace_target + (f".{rest}" if rest else ""))
    return tuple(sorted(names))


def build(registry: PackageArchitectureRegistry) -> tuple[str, str]:
    """The import-linter INI and the Ruff TOML for this registry."""
    packages = registry.by_package
    lines = ["[importlinter]", "root_packages =", *(f"    {root}" for root in ROOTS), ""]
    for package_id, package in sorted(packages.items()):
        forbidden = sorted(
            other.physical_namespace_target
            for other in packages.values()
            if other.package_id != package_id
            and (
                other.dependency_level > package.dependency_level
                or (
                    other.dependency_level == package.dependency_level
                    and other.package_id not in package.allowed_peer_dependencies
                )
            )
        )
        if not forbidden:
            continue
        lines += [
            f"[importlinter:contract:{package_id}]",
            f"name = {package_id} imports no higher level and no undeclared peer",
            "type = forbidden",
            f"source_modules =\n    {package.physical_namespace_target}",
            "forbidden_modules =",
            *(f"    {name}" for name in forbidden),
            "allow_indirect_imports = true",
            "",
        ]
    banned = {
        name: f"retired: {reason}"
        for prefix, reason in sorted(registry.retired_imports.items())
        for name in _physical(registry, prefix)
    }
    ruff = [
        'output-format = "concise"',
        "[lint]",
        'select = ["TID251"]',
        "[lint.flake8-tidy-imports.banned-api]",
    ]
    ruff += [
        f"{json.dumps(name)} = {{ msg = {json.dumps(msg)} }}"
        for name, msg in sorted(banned.items())
    ]
    return "\n".join(lines) + "\n", "\n".join(ruff) + "\n"


def run(argv: list[object], root: Path, **env: str) -> subprocess.CompletedProcess[str]:
    """One tool over the checkout, as text; a finding is its output, and a launch that fails is
    exit 127 with its error, never an exception."""
    words = [str(word) for word in argv]
    try:
        return subprocess.run(
            words,
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, **env},
        )
    except OSError as error:
        return subprocess.CompletedProcess(words, 127, "", f"cannot launch: {error}")


def _python_files(root: Path) -> tuple[list[str], str]:
    """The old law's population, relative to ``root``: every tracked file (ignored ones too) and
    every untracked one Git does not ignore, plus every file under src on disk; or, when Git's
    listing fails, the reason the population is unknown."""
    listed = run(
        [
            "git",
            "-c",
            "core.quotepath=off",
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "*.py",
        ],
        root,
    )
    if listed.returncode:  # a partial listing is no population
        return [], f"git ls-files exited {listed.returncode}: {listed.stderr.strip()[-300:]}"
    files = {path for path in listed.stdout.split("\0") if path and (root / path).is_file()}
    files |= {path.relative_to(root).as_posix() for path in (root / "src").rglob("*.py")}
    return sorted(files), ""


def _units(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def launch_batches(command: list[object], files: list[str]) -> tuple[list[list[str]], list[str]]:
    """``files`` split so each launch's Windows command line stays under COMMAND_LINE, counted as
    CreateProcess counts it, in UTF-16 units (a supplementary character is two), and the files too
    long to launch even alone."""
    base = _units(subprocess.list2cmdline([str(word) for word in command]))
    batches: list[list[str]] = []
    current: list[str] = []
    alone: list[str] = []
    size = base
    for path in files:
        cost = _units(subprocess.list2cmdline([path])) + 1
        if base + cost > COMMAND_LINE:
            alone.append(path)
            continue
        if size + cost > COMMAND_LINE:
            batches.append(current)
            current, size = [], base
        current.append(path)
        size += cost
    return batches + [current] * bool(current), alone


def check(root: Path, out: Path, registry: PackageArchitectureRegistry) -> tuple[str, ...]:
    """Each violation of the import law over the checkout, from both tools; the configs and logs
    that produced them stay in ``out``. A tool that cannot run is a violation named by its output,
    never a pass."""
    out.mkdir(parents=True, exist_ok=True)
    contracts, bans = build(registry)
    (out / "importlinter.ini").write_text(contracts, encoding="utf-8", newline="\n")
    (out / "ruff-retired.toml").write_text(bans, encoding="utf-8", newline="\n")
    lint = [Path(sys.executable).parent / "lint-imports", "--config", out / "importlinter.ini"]
    ruff = [sys.executable, "-m", "ruff", "check", "--config", out / "ruff-retired.toml"]
    ruff += ["--ignore-noqa", "--no-cache"]  # no noqa silences a ban
    files, unknown = _python_files(root)
    batches, alone = launch_batches(ruff, files)
    found = [f"ruff: the checked files are unknown: {unknown}"] if unknown else []
    found += [f"ruff: {path} cannot be launched: its command line is too long" for path in alone]
    found += [] if files or unknown else ["ruff: no Python file found to check"]
    tools = {
        "lint-imports": [run([*lint, "--no-cache"], root, PYTHONPATH=str(root / "src"))],
        "ruff": [run([*ruff, *batch], root) for batch in batches],
    }
    for tool, runs in tools.items():
        marker = " -> " if tool == "lint-imports" else " TID251 "
        (out / f"{tool}.log").write_text(
            "".join(r.stdout + r.stderr for r in runs), encoding="utf-8"
        )
        for done in runs:
            hits = [
                f"{tool}: {line.strip()}" for line in done.stdout.splitlines() if marker in line
            ]
            tail = (done.stdout + done.stderr).strip()[-300:]
            found += hits or (
                [f"{tool} exited {done.returncode}: {tail}"] if done.returncode else []
            )
    return tuple(found)
