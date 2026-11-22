"""The package import law, checked by import-linter and Ruff from the one registry.

``config/package-architecture.json`` owns the law: each package's dependency level and declared
peers, owned external imports and retired owners. The generator builds the package-level and
kernel contracts for Import Linter and the scoped and retired import bans for Ruff. Ruff also
bans an import by name whether or not its module still exists (grimp drops such an import). Both
are built at run time, never committed; ``check`` writes them into the caller's run as evidence and
runs both tools over the tracked tree. Each Python file is scanned once with its applicable bans.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from alphalattice.kernel.shared_kernel import spawn
from devtools.architecture.structural import (
    _PHYSICAL_NAMESPACE_GROUPS,
    PackageArchitectureRegistry,
    _import_sites,
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


def build(
    registry: PackageArchitectureRegistry, *, deferred_backend_edges: tuple[str, ...] = ()
) -> tuple[str, str]:
    """The import-linter INI and the Ruff TOML for this registry."""
    packages = registry.by_package
    kernels = sorted(
        package.physical_namespace_target
        for package in packages.values()
        if package.dependency_level < 0
    )
    lines = ["[importlinter]", "root_packages =", *(f"    {root}" for root in ROOTS)]
    if kernels:
        lines.append("include_external_packages = true")
    lines.append("")
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
    if kernels:
        lines.extend(_kernel_contracts(kernels, deferred_backend_edges))
    return "\n".join(lines) + "\n", _ruff_bans(_retired_bans(registry))


def _retired_bans(registry: PackageArchitectureRegistry) -> dict[str, str]:
    return {
        name: f"retired: {reason}"
        for prefix, reason in sorted(registry.retired_imports.items())
        for name in _physical(registry, prefix)
    }


def _ruff_bans(banned: dict[str, str]) -> str:
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
    return "\n".join(ruff) + "\n"


def _kernel_contracts(packages: list[str], deferred_backend_edges: tuple[str, ...]) -> list[str]:
    """Allow only standard-library roots and protect every module below the public package port."""
    lines = [
        "[importlinter:contract:kernel_stdlib]",
        "name = Kernel ports import only stdlib and declared first-call backends",
        "type = forbidden",
        "source_modules =",
        *(f"    {name}" for package in packages for name in (package, package + ".**")),
        "forbidden_modules = *",  # Grimp squashes external modules to their root.
        "as_packages = false",  # Match roots only, preserving the separate internal layer rules.
        "allow_indirect_imports = true",
        "unmatched_ignore_imports_alerting = none",
        "ignore_imports =",
        *(f"    alphalattice.kernel.** -> {name}" for name in sorted(sys.stdlib_module_names)),
        *(f"    {edge}" for edge in deferred_backend_edges),
        "",
    ]
    for package in packages:
        lines += [
            f"[importlinter:contract:{package.rsplit('.', 1)[-1]}_internals]",
            f"name = Only {package} imports its internal modules",
            "type = protected",
            f"protected_modules = {package}.**",
            f"allowed_importers =\n    {package}\n    {package}.**",
            "as_packages = false",
            "",
        ]
    return lines


def _deferred_backend_edges(root: Path, registry: PackageArchitectureRegistry) -> tuple[str, ...]:
    """Grimp groups sites by edge. Exempt an edge only if all of its sites are deferred."""
    permitted: list[str] = []
    for module, backends in registry.kernel_deferred_backends.items():
        path = (root / "src").joinpath(*module.split("."))
        package = path / "__init__.py"
        if package.is_file():
            path, qualified = package, module + ".__init__"
        else:
            path, qualified = path.with_suffix(".py"), module
        if not path.is_file():
            continue  # The registry can declare a backend before its owning module lands.
        source = path.read_bytes()
        eager: set[str] = set()
        deferred: set[str] = set()
        for _, _, _, in_function, _, _, _, targets in _import_sites(source, qualified):
            (deferred if in_function else eager).update(t.partition(".")[0] for t in targets)
        permitted.extend(f"{module} -> {backend}" for backend in backends & (deferred - eager))
    return tuple(sorted(permitted))


def run(argv: list[object], root: Path, **env: str) -> subprocess.CompletedProcess[str]:
    """One tool over the checkout, as text; a finding is its output, and a launch that fails is
    exit 127 with its error, never an exception."""
    words = [str(word) for word in argv]
    try:
        return spawn.run(
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


def launch_batches(
    command: Sequence[object], files: list[str]
) -> tuple[list[list[str]], list[str]]:
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


_POOL_CONSTRUCTORS = ("ThreadPoolExecutor", "ProcessPoolExecutor")


def _pool_names(tree: ast.AST) -> set[str]:
    """The spellings bound to the standard executors by imports in this file."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "concurrent.futures":
            names.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name in _POOL_CONSTRUCTORS
            )
        elif isinstance(node, ast.ImportFrom) and node.module == "concurrent":
            names.update(
                f"{alias.asname or alias.name}.{kind}"
                for alias in node.names
                if alias.name == "futures"
                for kind in _POOL_CONSTRUCTORS
            )
        elif isinstance(node, ast.Import):
            names.update(
                f"{alias.asname or alias.name}.{kind}"
                for alias in node.names
                if alias.name == "concurrent.futures"
                for kind in _POOL_CONSTRUCTORS
            )
    return names


def _worker_arguments(tree: ast.AST) -> list[ast.expr]:
    positional = _pool_names(tree)
    arguments: list[ast.expr] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "max_workers":
            arguments.append(node.value)
        elif (
            isinstance(node, ast.Call)
            and node.args
            and positional
            and (
                isinstance(node.func, (ast.Name, ast.Attribute))
                and ast.unparse(node.func) in positional
            )
        ):
            arguments.append(node.args[0])
    return arguments


def _literal_worker_counts(root: Path, files: list[str]) -> list[str]:
    """Find literal worker counts in product code outside the resource owner."""
    findings: list[str] = []
    for file in files:
        source = (root / file).read_bytes()
        if b"max_workers" not in source and b"concurrent" not in source:
            continue
        try:
            tree = ast.parse(source, filename=file)
        except SyntaxError as error:
            findings.append(f"{file}:{error.lineno}: resources.source_invalid: {error.msg}")
            continue
        for argument in _worker_arguments(tree):
            value = argument.operand if isinstance(argument, ast.UnaryOp) else argument
            if isinstance(value, ast.Constant):
                findings.append(
                    f"{file}:{argument.lineno}: resources.literal_max_workers: use a worker grant"
                )
    return findings


def _ruff_partitions(
    registry: PackageArchitectureRegistry, files: list[str], out: Path
) -> list[tuple[Path, list[str]]]:
    scoped: dict[str, dict[str, str]] = defaultdict(dict)
    for owner, names in sorted(registry.owned_imports.items()):
        prefix = f"src/{owner.replace('.', '/')}"
        scoped[prefix].update((name, f"Use {owner} for this import") for name in names)
    if registry.wall_clock_reads:
        scoped["src/alphalattice/kernel/clock"].update(registry.wall_clock_reads)

    if registry.worker_count_reads:
        scoped["src/alphalattice/kernel/resources"].update(registry.worker_count_reads)

    groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for path in files:
        applicable = (
            tuple(
                owner
                for owner in scoped
                if path != owner + ".py" and not path.startswith(owner + "/")
            )
            if path.startswith("src/alphalattice/")
            else ()
        )
        groups[applicable].append(path)

    retired = _retired_bans(registry)
    partitions = [(out / "ruff-retired.toml", groups.pop((), []))]
    for index, (owners, paths) in enumerate(groups.items(), 1):
        config = out / f"ruff-scope-{index}.toml"
        scoped_bans = {name: reason for owner in owners for name, reason in scoped[owner].items()}
        config.write_text(_ruff_bans(retired | scoped_bans), encoding="utf-8", newline="\n")
        partitions.append((config, paths))
    return partitions


def check(root: Path, out: Path, registry: PackageArchitectureRegistry) -> tuple[str, ...]:
    """Each violation of the import law over the checkout, from both tools; the configs and logs
    that produced them stay in ``out``. A tool that cannot run is a violation named by its output,
    never a pass."""
    out.mkdir(parents=True, exist_ok=True)
    contracts, bans = build(
        registry, deferred_backend_edges=_deferred_backend_edges(root, registry)
    )
    (out / "importlinter.ini").write_text(contracts, encoding="utf-8", newline="\n")
    (out / "ruff-retired.toml").write_text(bans, encoding="utf-8", newline="\n")
    lint = [Path(sys.executable).parent / "lint-imports", "--config", out / "importlinter.ini"]
    ruff = [sys.executable, "-m", "ruff", "check", "--ignore-noqa", "--no-cache"]
    files, unknown = _python_files(root)
    found = [f"ruff: the checked files are unknown: {unknown}"] if unknown else []
    found += [] if files or unknown else ["ruff: no Python file found to check"]
    tools = {"lint-imports": [run([*lint, "--no-cache"], root, PYTHONPATH=str(root / "src"))]}
    if registry.worker_count_reads:
        product = [
            path
            for path in files
            if path.startswith("src/alphalattice/")
            and not path.startswith("src/alphalattice/kernel/resources/")
        ]
        found.extend(_literal_worker_counts(root, product))
    tools["ruff"] = []
    for config, selected in _ruff_partitions(registry, files, out):
        command = [*ruff, "--config", config]
        batches, alone = launch_batches(command, selected)
        found += [
            f"ruff: {path} cannot be launched: its command line is too long" for path in alone
        ]
        tools["ruff"].extend(run([*command, *batch], root) for batch in batches)
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
