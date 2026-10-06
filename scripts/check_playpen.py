"""Run the bounded local playpen gate with this workspace's own locked environment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
STRICT_MODULES = ROOT / "config" / "mypy-strict-modules.txt"
STRUCTURAL_TESTS = (
    ROOT / "tests" / "structural" / "test_structural_guards.py",
    ROOT / "tests" / "structural" / "test_playpen_gate_routing.py",
    ROOT / "tests" / "structural" / "test_entry_point_startup.py",
    ROOT / "tests" / "structural" / "test_sealed_hash_admissions.py",
    ROOT / "tests" / "structural" / "test_retired_spellings.py",
    ROOT / "tests" / "structural" / "test_book_walkers_refuse_short_pools.py",
    ROOT / "tests" / "structural" / "test_product_scripts_answer_their_writes.py",
)
STRUCTURAL_TEST_PATHS = frozenset(path.relative_to(ROOT).as_posix() for path in STRUCTURAL_TESTS)
"""The same declaration, in the form the impact runner needs to skip them.

`check_alpha_verification_impact` imports this instead of keeping a second
list. The two were hand-kept copies and drifted once: a file named in one and
not the other runs twice per commit, once in the gate's parallel batch and
again, serially, under `--structural-tests-already-passed`.
"""

_ROOT_ENV_SENTINEL = "ALPHALATTICE_PLAYPEN_GATE_ROOT_ENV"

# `CreateProcess` accepts a command line of at most 32,767 UTF-16 code units
# including its terminator, so 32,766 reach the OS. The gate never spawns the
# OS directly: it re-executes into `<root>/.venv`, whose `python.exe` is a
# trampoline that rebuilds the line with the *base* interpreter's path. That
# substitution is longer here by three units, and its size is a property of two
# absolute paths on the machine, so it is not portable. Measured on this one:
# through the trampoline the largest line whose child actually ran was 32,763,
# while a direct base-interpreter call reached 32,766. `_LAUNCHER_RESERVE`
# covers any realistic difference at about 1.6% of capacity -- cheap, and it
# leaves the real-target list at three batches either way.
#
# `--all-python` names about 1,100 files and serializes to roughly 72,000
# units, which is where the documented `WinError 206` came from. Splitting that
# list across invocations keeps both the target set and the explicit-file
# semantics: ruff applies `extend-exclude` to a directory walk, not to files
# handed to it by name, so passing the roots instead would silently check fewer
# files. Strict mypy is deliberately not batched -- it is one command line well
# inside the bound, and with `follow_imports = "skip"` a subset is a different
# check, not a smaller one.
_LAUNCHER_RESERVE = 512
_COMMAND_LINE_UNITS = 32_766 - _LAUNCHER_RESERVE


def _ensure_workspace_environment() -> None:
    """Re-execute under `<root>/.venv`, the environment `uv.lock` describes.

    There is deliberately no other candidate: a parent checkout's interpreter
    would satisfy imports with source nobody reviewed here, so an absent
    workspace environment is a stop with the bootstrap command, not a fallback.
    """

    candidates = (
        ROOT / ".venv" / "Scripts" / "python.exe",
        ROOT / ".venv" / "bin" / "python",
    )
    target = next((candidate for candidate in candidates if candidate.is_file()), None)
    if target is None:
        raise RuntimeError(
            "playpen gate requires the workspace .venv; run "
            f"'uv sync --locked --all-extras' in {ROOT} (see README.md)"
        )
    if Path(sys.executable).resolve() == target.resolve():
        return
    if os.environ.get(_ROOT_ENV_SENTINEL) == "1":
        raise RuntimeError("playpen gate failed to enter the workspace .venv")
    os.environ[_ROOT_ENV_SENTINEL] = "1"
    result = subprocess.run(
        (str(target), str(Path(__file__).resolve()), *sys.argv[1:]),
        check=False,
    )
    raise SystemExit(result.returncode)


def _run(label: str, *args: str) -> None:
    started = perf_counter()
    try:
        subprocess.run(args, cwd=ROOT, check=True)
    finally:
        elapsed = perf_counter() - started
        print(f"PLAYPEN_GATE_TIMING {label} {elapsed:.3f}s", flush=True)


def _command_line_units(args: tuple[str, ...]) -> int:
    """UTF-16 code units Windows sees for one argument list.

    `len()` is the wrong ruler twice over. A non-BMP character is one Python
    character but two UTF-16 units, so eighty of them cost 160; and `subprocess`
    does not pass a list, it serializes one string with `list2cmdline`, which
    adds quotes around anything containing a space and escapes embedded quotes
    and trailing backslashes. Both are measured here rather than estimated.
    """

    return len(subprocess.list2cmdline(args).encode("utf-16-le")) // 2


def _batched(targets: tuple[str, ...], *, prefix: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    """Split targets into command lines Windows accepts, preserving order.

    Every batch is planned before the caller spawns anything, so an argument no
    process could ever receive is refused here instead of surfacing as a
    `WinError 206` after earlier batches have already run. Each argument costs
    its own serialization plus the separator in front of it, and the assembled
    batch is measured once more before it is returned: an accounting mistake
    fails as a stop with a number in it, not as an opaque OS error.
    """

    budget = _COMMAND_LINE_UNITS - _command_line_units(prefix)
    batches: list[tuple[str, ...]] = []
    batch: list[str] = []
    used = 0
    for target in targets:
        cost = _command_line_units((target,)) + 1
        if cost > budget:
            raise ValueError(
                "playpen gate cannot pass this argument to any process: "
                f"{cost} of {budget} available UTF-16 units: {target!r}"
            )
        if batch and used + cost > budget:
            batches.append(tuple(batch))
            batch, used = [], 0
        batch.append(target)
        used += cost
    if batch:
        batches.append(tuple(batch))
    for planned in batches:
        measured = _command_line_units((*prefix, *planned))
        if measured > _COMMAND_LINE_UNITS:
            raise ValueError(f"playpen gate planned an oversized command line: {measured} units")
    return tuple(batches)


def _run_over_targets(label: str, *prefix: str, targets: tuple[str, ...]) -> None:
    """Run one check over every target, in as few processes as the OS allows.

    Every batch runs before the step fails, so one batch's diagnostics never
    hide another's, and the first failure is re-raised with the same
    `CalledProcessError` a single invocation produced.
    """

    batches = _batched(targets, prefix=prefix)
    started = perf_counter()
    first_failure: subprocess.CompletedProcess[bytes] | None = None
    try:
        for batch in batches:
            completed = subprocess.run((*prefix, *batch), cwd=ROOT, check=False)
            if completed.returncode != 0 and first_failure is None:
                first_failure = completed
    finally:
        elapsed = perf_counter() - started
        print(f"PLAYPEN_GATE_TIMING {label} {elapsed:.3f}s", flush=True)
    if first_failure is not None:
        first_failure.check_returncode()


def _git_lines(*args: str) -> set[str]:
    result = subprocess.run(("git", *args), cwd=ROOT, check=True, capture_output=True, text=True)
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _python_targets(*, staged: bool, all_python: bool) -> tuple[str, ...]:
    if all_python:
        roots = ("src", "scripts", "tests", "benchmark", "probe", "case-study")
        return tuple(
            sorted(
                str(path.relative_to(ROOT)).replace("\\", "/")
                for root in roots
                for path in (ROOT / root).rglob("*.py")
                if not any(part in {"experiments", "tmp", "workspaces"} for part in path.parts)
            )
        )
    if staged:
        changed = _git_lines("diff", "--cached", "--name-only", "--diff-filter=ACMR")
    else:
        changed = _git_lines("diff", "HEAD", "--name-only", "--diff-filter=ACMR")
        changed |= _git_lines("ls-files", "--others", "--exclude-standard")
    return tuple(
        sorted(path for path in changed if path.endswith(".py") and (ROOT / path).is_file())
    )


def _report_identity_closures(*, staged: bool) -> None:
    """Report, never refuse: which source identities the changed files rotate.

    The binding plan's closure map (B5, B14). A commit that edits a file a closure
    tracks moves that identity; saying so at the commit is what lets its author
    record the move, or keep the edit out of the closure, before anyone meets it.
    """

    changed = _git_lines(
        "diff", "--cached" if staged else "HEAD", "--name-only", "--diff-filter=ACMRD"
    )
    if not changed:
        return
    started = perf_counter()
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from devtools.architecture.identity_closures import report
    finally:
        sys.path.pop(0)
    for line in report(ROOT, sorted(changed)):
        print(f"IDENTITY_CLOSURE {line}", flush=True)
    print(f"PLAYPEN_GATE_TIMING identity-closures {perf_counter() - started:.3f}s", flush=True)


def _imports_of(source: str | None) -> frozenset[tuple[int, str, str]]:
    if source is None:
        return frozenset()
    import ast

    found: set[tuple[int, str, str]] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update((0, alias.name, "") for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.update((node.level, node.module or "", alias.name) for alias in node.names)
    return frozenset(found)


def _reach_may_grow(*, staged: bool) -> list[str]:
    """The changed source modules that can widen an area's reach (GB).

    A rule closure follows imports, so only a module added, deleted or re-imported can
    change what a role hashes, and only where a closure walks: an area the reach table
    lists, or a number-deciding package a closure could newly import from.
    """

    table = json.loads((ROOT / "config/registries/identity-reach.json").read_text(encoding="utf-8"))
    areas = set(table["areas"])
    deciding = set(
        json.loads((ROOT / "config/identity-roles.json").read_text(encoding="utf-8"))[
            "number_deciding_packages"
        ]
    )
    changed = _git_lines(
        "diff", "--cached" if staged else "HEAD", "--name-only", "--diff-filter=ACMRD"
    )
    found = []
    for path in sorted(changed):
        if not (path.startswith("src/alphalattice/") and path.endswith(".py")):
            continue
        parts = path.removesuffix(".py").split("/")[1:]
        module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        area = module if parts[-1] == "__init__" else module.rpartition(".")[0]
        if area not in areas and not (len(parts) > 2 and parts[2] in deciding):
            continue
        before = subprocess.run(
            ["git", "show", f"HEAD:{path}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        after = subprocess.run(
            ["git", "show", f":{path}"] if staged else ["git", "show", f"HEAD:{path}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        after_text = after.stdout if staged else None
        if not staged and (ROOT / path).is_file():
            after_text = (ROOT / path).read_text(encoding="utf-8")
        if staged and after.returncode != 0:
            after_text = None
        before_text = before.stdout if before.returncode == 0 else None
        if _imports_of(before_text) != _imports_of(after_text):
            found.append(path)
    return found


def _check_identity_reach(*, staged: bool) -> bool:
    """Refuse a change that makes an area reach an identity role outside its row (GB, ID8).

    Measured by the readout's own functions (about 45 s), and only when a changed module
    can widen the reach; the table only shrinks.
    """

    started = perf_counter()
    candidates = _reach_may_grow(staged=staged)
    if not candidates:
        return True
    print(f"identity reach: {len(candidates)} changed modules can widen it; measuring", flush=True)
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "identity_readout.py"), "--reach-check"],
        cwd=ROOT,
        check=False,
    )
    print(f"PLAYPEN_GATE_TIMING identity-reach {perf_counter() - started:.3f}s", flush=True)
    return result.returncode == 0


def _check_registries(*, staged: bool) -> bool:
    """Refuse a change that adds to a registry without registering it (binding plan G0).

    Only the changed files are read: a parameter, a failure code, a format or a private test
    import can only be added where a file changed, so the check costs what the change costs.
    """

    changed = sorted(
        path
        for path in _git_lines(
            "diff", "--cached" if staged else "HEAD", "--name-only", "--diff-filter=ACMR"
        )
        if path.endswith(".py")
    )
    started = perf_counter()
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from devtools.architecture.registries import private_test_growth, problems
    finally:
        sys.path.pop(0)
    found = problems(ROOT, changed)
    # The tests' ratchet against the registry this change started from: registering a new
    # private import by hand passes the scan above, never this (LAWS.md TE5, V16).
    ratchet = "config/registries/test-private.json"

    def shown(revision: str) -> str | None:
        result = subprocess.run(
            ["git", "show", f"{revision}:{ratchet}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return result.stdout if result.returncode == 0 else None

    base = shown("HEAD")
    current = shown("") if staged else (ROOT / ratchet).read_text(encoding="utf-8")
    if base is not None and current is not None:
        found += private_test_growth(json.loads(base)["entries"], json.loads(current)["entries"])
    for line in found:
        print(f"REGISTRY {line}", flush=True)
    print(f"PLAYPEN_GATE_TIMING registries {perf_counter() - started:.3f}s", flush=True)
    return not found


def _check_docstrings() -> bool:
    """Refuse a source file whose public docstring findings grew (C2's ratchet, lower only).

    Ruff's Google-convention docstring rules over the public API of ``src/alphalattice``,
    counted per file against ``config/docstring-baseline.json``; a new public file starts at
    zero. ``scripts/check_docstrings.py --update`` only lowers the baseline.
    """

    started = perf_counter()
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_docstrings.py")], cwd=ROOT, check=False
    )
    print(f"PLAYPEN_GATE_TIMING docstrings {perf_counter() - started:.3f}s", flush=True)
    return result.returncode == 0


def _check_evidence_bindings() -> bool:
    """Refuse a tree whose Evidence binding tuple moved without its predecessor listed."""

    started = perf_counter()
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from devtools.architecture.evidence_bindings import problems

        found = problems(ROOT)
    finally:
        sys.path.pop(0)
    for line in found:
        print(f"EVIDENCE_BINDINGS {line}", flush=True)
    print(f"PLAYPEN_GATE_TIMING evidence-bindings {perf_counter() - started:.3f}s", flush=True)
    return not found


def _check_operation_registry() -> bool:
    """Refuse a tree whose operations disagree with the registry, or whose CLI answers any
    command outside its contract without a Host (C1 rule 3 and the CLI walk)."""

    started = perf_counter()
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from devtools.architecture.operation_registry import problems, walk

        found = problems(ROOT) + walk(ROOT)
    finally:
        sys.path.pop(0)
    for line in found:
        print(f"OPERATION_REGISTRY {line}", flush=True)
    print(f"PLAYPEN_GATE_TIMING operation-registry {perf_counter() - started:.3f}s", flush=True)
    return not found


def _strict_targets() -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in STRICT_MODULES.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def _require_private_structural_inputs() -> bool:
    missing = [path.relative_to(ROOT).as_posix() for path in STRUCTURAL_TESTS if not path.is_file()]
    if not missing:
        return True
    print(
        "playpen.private_structural_inputs_unavailable: "
        + ", ".join(missing)
        + "; these guards read maintainers' private records and history. "
        "Run scripts/check_playpen.py --all-python --fast for the public gate.",
        file=sys.stderr,
    )
    return False


def main(*, ensure_environment: Callable[[], None] = _ensure_workspace_environment) -> int:
    """Run the gate; `ensure_environment` enters the workspace .venv (a test passes its own)."""
    ensure_environment()
    gate_started = perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--all-python", action="store_true")
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Per-commit gate: lint and strict-type only what changed. Skips the "
            "whole-repo structural guards and the impact-selected suites, which "
            "the full gate runs, at the user's word (LAWS TE7)."
        ),
    )
    parser.add_argument(
        "--evidence",
        action="store_true",
        help=(
            "Run only the explicit real-evidence lane over the same selection: the "
            "tests marked real_evidence, reported as REQUIRED / RAN / SKIPPED. Lint, "
            "types and the structural batch belong to the routed gate and are not "
            "repeated here."
        ),
    )
    args = parser.parse_args()
    if args.evidence and args.fast:
        parser.error("--evidence and --fast are mutually exclusive")
    if args.evidence:
        sys.path.insert(0, str(ROOT / "src"))
        try:
            from devtools.architecture.evidence_roots import EvidenceRootError, EvidenceRoots

            try:
                EvidenceRoots.load(worktree=ROOT)
            except EvidenceRootError as error:
                print(str(error), file=sys.stderr)
                return 1
        finally:
            sys.path.pop(0)
        # The lane the routed gate deselects. Same selection, inverted marker,
        # every skip named: a SHA whose REQUIRED set is non-empty is handed to
        # review with this output, not with the routed gate's alone.
        evidence_args: tuple[str, ...] = (
            sys.executable,
            str(ROOT / "scripts" / "check_alpha_verification_impact.py"),
            "--run-tests",
            "--evidence",
        )
        if args.staged:
            evidence_args = (*evidence_args, "--staged")
        _run("evidence-tests", *evidence_args)
        print(
            f"PLAYPEN_GATE_TIMING total {perf_counter() - gate_started:.3f}s",
            flush=True,
        )
        return 0

    if not args.fast and not _require_private_structural_inputs():
        return 1
    python_targets = _python_targets(staged=args.staged, all_python=args.all_python)
    if python_targets:
        _run_over_targets(
            "ruff-check",
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--config",
            "pyproject.toml",
            targets=python_targets,
        )
        _run_over_targets(
            "ruff-format",
            sys.executable,
            "-m",
            "ruff",
            "format",
            "--check",
            "--config",
            "pyproject.toml",
            targets=python_targets,
        )
    # Always the full strict set, never a subset. With follow_imports = "skip",
    # a module absent from the command line types as Any, so narrowing the list
    # both invents errors and hides real ones -- it is not the same check on
    # fewer files. It also buys nothing: warm-cache mypy over all of them is
    # about a second, and was never what made this gate slow.
    _run(
        "strict-mypy",
        sys.executable,
        "-m",
        "mypy",
        "--config-file",
        "pyproject.toml",
        *_strict_targets(),
    )
    _report_identity_closures(staged=args.staged)
    if not _check_identity_reach(staged=args.staged):
        return 1
    if not _check_evidence_bindings():
        return 1
    if not _check_operation_registry():
        return 1
    if not _check_registries(staged=args.staged):
        return 1
    if not _check_docstrings():
        return 1
    if args.fast:
        # What actually cost minutes per commit: the whole-repo structural guard
        # suite and the impact runner, which selects and executes entire test
        # suites. Both answer questions about the repository rather than about
        # this edit, so they belong in the full gate that runs before review.
        print(
            f"PLAYPEN_GATE_TIMING total {perf_counter() - gate_started:.3f}s",
            flush=True,
        )
        return 0
    # These are short independent checks over the same immutable tree. Every
    # worker still collects both structural modules, so bound the pool at the
    # measured Windows optimum instead of repeating imports under `auto`.
    # Impact-selected suites below retain their separate batching and
    # process-isolation policy.
    _run(
        "structural-guards",
        sys.executable,
        "-m",
        "pytest",
        "--no-cov",
        *(str(path) for path in STRUCTURAL_TESTS),
        "-q",
        "-n",
        "8",
    )
    impact_args: tuple[str, ...] = (
        sys.executable,
        str(ROOT / "scripts" / "check_alpha_verification_impact.py"),
        "--run-tests",
        "--structural-tests-already-passed",
    )
    if args.staged:
        impact_args = (*impact_args, "--staged")
    _run("impact-tests", *impact_args)
    print(
        f"PLAYPEN_GATE_TIMING total {perf_counter() - gate_started:.3f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
