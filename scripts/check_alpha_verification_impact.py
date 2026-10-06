"""Derive and optionally execute the bounded Alpha verification plan."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import NamedTuple
from xml.etree import ElementTree

SCRIPTS_ROOT = Path(__file__).resolve().parent
PLAYPEN_ROOT = SCRIPTS_ROOT.parent
SRC = PLAYPEN_ROOT / "src"
for _entry in (str(SRC), str(SCRIPTS_ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from alphalattice.investment.alpha_research.verification.identity import (  # noqa: E402
    alpha_execution_domain_hashes,
)
from alphalattice.investment.alpha_research.verification.impact import (  # noqa: E402
    alpha_verification_impact,
)

# The gate owns the structural set; this runner only needs to know which paths
# it has already executed. `check_playpen` imports no product runtime, so
# consuming it here costs nothing and removes the second hand-kept list. The
# ignore is because `scripts/` is not on `mypy_path`, and putting it there means
# editing `pyproject.toml`, which sits inside four byte-exact closures.
from check_playpen import STRUCTURAL_TEST_PATHS  # type: ignore[import-not-found]  # noqa: E402

# The build-time owner of import resolution. Reused rather than reimplemented:
# it already resolves relative imports and package initializers, and a second
# parser here is what produced a wrong answer for `__init__.py`.
from devtools.architecture.neutral_changes import (  # noqa: E402
    git_previous,
    moved_files,
    neutral_verdicts,
)
from devtools.architecture.structural import (  # noqa: E402
    discover_module_dependencies,
    discover_python_imports,
    discover_python_modules,
)

_PARENT_REVISION = "HEAD"
"""The commit a change is measured against: `--base`, else HEAD. `main` sets it."""

# This case study is deterministic and imports no Agent/model harness. Running
# its selected files in one pytest process lets its session-scoped real-workspace
# fixtures be built once. Agent-bearing cases keep the one-file/one-process
# boundary below.
_BATCH_SAFE_TEST_DIRS = frozenset(
    {
        "tests/researcher_methodology_extension",
        "tests/researcher_methodology_surface",
    }
)
"""Packages that run as one process because their session workspaces make it pay.

Batching is not a default: `feature_engine`, `workspace_readiness` and
`workspace_task_runner` were audited batch-safe and measured CPU-neutral in one
process (feature_engine about 147 s either way), which only turns seventeen short,
parallelizable jobs into one long one.
"""

# Two lanes over one selection. The routed lane is what every gate runs; it
# deselects the tests that read a dogfood evidence root. The evidence lane runs
# exactly those, explicitly, and reports what it could and could not reach.
# The marker is registered in `pyproject.toml`; `--strict-markers` refuses any
# other spelling, so a typo cannot smuggle a test out of the routed lane.
_REAL_EVIDENCE_MARKER = "real_evidence"
_ROUTED_LANE_MARKER_EXPRESSION = f"not {_REAL_EVIDENCE_MARKER}"
_EVIDENCE_LANE_COMMAND = "scripts/check_playpen.py --staged --evidence"
_NOTHING_COLLECTED = 5
"""pytest's exit code when a selection deselects every test in a file."""

# Wall-clock hints from earlier runs, written after every routed lane. Ordering
# is all they decide: the pool below is fixed at four workers, so the wall clock
# is set by whichever job starts last and runs longest. Path order put the
# longest job in the middle of the list and measured 1.5-2.0x the ideal
# makespan; longest-first is the standard answer and needs a duration to sort
# by. A group never timed is estimated from its size, which is only a proxy but
# beats sorting it by name.
_TIMING_HINTS = PLAYPEN_ROOT / "tmp" / "playpen-impact-timings.json"
_SIZE_ESTIMATE_BYTES_PER_SECOND = 5_000.0


_BATCH_MAX_FILES: dict[str, int] = {
    "tests/researcher_methodology_surface": 7,
}
"""Target file occupancy when dividing audited directories into workload groups.

One process for the whole methodology package set the routed lane's makespan
(about 420 s against a 270 s runner-up) because its twenty-seven files shared
one eighty-second workspace build. The build now comes from the golden cache
(`tests/researcher_methodology_surface/real_workspace.py`) and a copy costs
seconds, so the package runs as cost-balanced chunks that the four-worker pool
overlaps; each chunk still shares its session fixtures within one process.
"""


def _balanced_chunks(files: list[str], count: int) -> tuple[tuple[str, ...], ...]:
    """Greedy cost balancing; file size is only the unmeasured fallback.

    Membership can change as measurements arrive. Group hints therefore bind
    the full member set, never only the human-readable first-file label.
    """

    hints = _timing_hints()
    sizes = {path: _estimated_seconds((path,), hints) for path in files}
    chunks: list[list[str]] = [[] for _ in range(count)]
    loads = [0.0] * count
    for test_path in sorted(files, key=lambda path: (-sizes[path], path)):
        index = min(range(count), key=lambda position: (loads[position], position))
        chunks[index].append(test_path)
        loads[index] += sizes[test_path]
    return tuple(tuple(sorted(chunk)) for chunk in chunks if chunk)


def _execution_groups(test_paths: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    """Batch only explicitly audited deterministic cases; isolate everything else."""

    batched: dict[str, list[str]] = {}
    isolated: list[tuple[str, ...]] = []
    for test_path in test_paths:
        parent = Path(test_path).parent.as_posix()
        if parent in _BATCH_SAFE_TEST_DIRS:
            batched.setdefault(parent, []).append(test_path)
        else:
            isolated.append((test_path,))
    groups: list[tuple[str, ...]] = list(isolated)
    for parent, values in batched.items():
        files = sorted(values)
        limit = _BATCH_MAX_FILES.get(parent)
        if limit is not None and len(files) > limit:
            groups.extend(_balanced_chunks(files, -(-len(files) // limit)))
        else:
            groups.append(tuple(files))
    return tuple(groups)


def _group_label(group: tuple[str, ...]) -> str:
    if len(group) == 1:
        return group[0]
    # The first member names the chunk: members are disjoint across chunks, so
    # two chunks of one directory never share a label or a timing hint.
    return f"{Path(group[0]).parent.as_posix()} ({len(group)}) {Path(group[0]).stem}"


def _files_with_real_evidence(test_paths: tuple[str, ...]) -> frozenset[str]:
    """The files that carry the marker, read from their source, not from pytest.

    A syntactic fact answered syntactically: `pytest.mark.real_evidence` as a
    decorator, a `pytestmark` assignment, or a member of a `pytestmark` list all
    put an attribute named `real_evidence` on `mark`. A file that cannot be
    parsed is reported as evidence-bearing so its breakage is looked at in the
    explicit lane rather than vanishing from both.
    """

    found: set[str] = set()
    for test_path in test_paths:
        path = PLAYPEN_ROOT / test_path
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError):
            found.add(test_path)
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == _REAL_EVIDENCE_MARKER
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "mark"
            ):
                found.add(test_path)
                break
    return frozenset(found)


def _timing_hints() -> dict[str, float]:
    try:
        payload = json.loads(_TIMING_HINTS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(label): float(seconds)
        for label, seconds in payload.items()
        if isinstance(seconds, (int, float))
    }


def _record_timings(
    results: list[tuple[tuple[str, ...], int, float]],
    *,
    file_seconds: dict[str, float] | None = None,
    lane: str = "routed",
) -> None:
    hints = _timing_hints()
    hints.update({_timing_key(group, lane): round(elapsed, 3) for group, _, elapsed in results})
    hints.update(
        {f"{lane}:{path}": round(seconds, 3) for path, seconds in (file_seconds or {}).items()}
    )
    try:
        _TIMING_HINTS.parent.mkdir(parents=True, exist_ok=True)
        _TIMING_HINTS.write_text(
            json.dumps(dict(sorted(hints.items())), indent=2) + "\n", encoding="utf-8"
        )
    except OSError:
        # A hint. Losing it costs ordering quality on the next run, nothing else.
        return


def _timing_key(group: tuple[str, ...], lane: str) -> str:
    return f"{lane}:" + (group[0] if len(group) == 1 else "group:" + "|".join(group))


def _estimated_seconds(
    group: tuple[str, ...], hints: dict[str, float], lane: str = "routed"
) -> float:
    known = hints.get(_timing_key(group, lane))
    if known is None and len(group) == 1 and lane == "routed":
        known = hints.get(group[0])  # Old singleton hints have exact membership too.
    if known is not None:
        return known
    if len(group) > 1:
        return sum(_estimated_seconds((path,), hints, lane) for path in group)
    size = 0
    for test_path in group:
        try:
            size += (PLAYPEN_ROOT / test_path).stat().st_size
        except OSError:
            continue
    return size / _SIZE_ESTIMATE_BYTES_PER_SECOND


def _ordered_groups(
    groups: tuple[tuple[str, ...], ...], *, lane: str = "routed"
) -> tuple[tuple[str, ...], ...]:
    """Longest first, by the last measured wall clock or a size estimate."""

    hints = _timing_hints()
    return tuple(
        sorted(
            groups, key=lambda group: (-_estimated_seconds(group, hints, lane), _group_label(group))
        )
    )


class TestGroupRun(NamedTuple):
    group: tuple[str, ...]
    code: int
    elapsed: float
    file_seconds: dict[str, float]
    stdout: str
    stderr: str


def _junit_file_seconds(path: Path, group: tuple[str, ...]) -> dict[str, float]:
    """Use pytest's built-in total setup/call/teardown timing, not file length."""
    if not path.is_file():
        return {}
    try:
        document = ElementTree.parse(path)
    except (OSError, ElementTree.ParseError):
        return {}  # Missing hints never change the process exit or selected tests.
    prefixes = {file: file.removesuffix(".py").replace("/", ".") for file in group}
    result: dict[str, float] = {}
    for case in document.iter("testcase"):
        owner = case.get("classname", "")
        matches = [
            file
            for file, prefix in prefixes.items()
            if owner == prefix or owner.startswith(prefix + ".")
        ]
        if not matches:
            matches = [file for file in group if owner.split(".")[0] == Path(file).stem]
        if len(matches) == 1:
            try:
                seconds = max(0.0, float(case.get("time", "0")))
            except ValueError:
                continue
            result[matches[0]] = result.get(matches[0], 0.0) + seconds
    return result


def _run_group(
    group: tuple[str, ...], *marker_arguments: str, capture: bool = False
) -> TestGroupRun:
    started = perf_counter()
    print(f"PLAYPEN_TEST_STARTED {json.dumps(group)}", flush=True)
    with TemporaryDirectory(prefix="playpen-test-report-") as temporary:
        report = Path(temporary) / "junit.xml"
        completed = subprocess.run(
            (
                sys.executable,
                "-m",
                "pytest",
                "--no-cov",
                "-q",
                "--durations=10",
                "--durations-min=0.5",
                f"--junitxml={report}",
                "-o",
                "junit_duration_report=total",
                *marker_arguments,
                *group,
            ),
            cwd=PLAYPEN_ROOT,
            check=False,
            capture_output=capture,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        costs = _junit_file_seconds(report, group)
    return TestGroupRun(
        group,
        completed.returncode,
        perf_counter() - started,
        costs,
        completed.stdout if capture else "",
        completed.stderr if capture else "",
    )


def _run_admitted_tests(test_paths: tuple[str, ...]) -> None:
    """Run Agent-bearing cases in isolation and batch audited deterministic cases.

    The old runner applied Agent process isolation to every test file. That made
    deterministic case studies rebuild the same session fixture once per file,
    even though those files contain no Agent harness at all. Isolation remains
    the conservative default; batching is an explicit, reviewed allow-list.

    This is the routed lane: every selected file runs with the real-evidence
    tests deselected. A file the marker scan found evidence-bearing may
    therefore collect nothing here, which pytest reports as exit 5; that is the
    lane working, not a failure, and it is accepted for those files only. The
    same files are then named as REQUIRED for the explicit evidence lane.
    """

    # One worker per four cores, at most eight (eight on the 32-core machine this
    # was measured on). A lane's wall clock is bounded by its longest group and by
    # its summed time over the pool. Four workers saturated the Evidence/CRO lane
    # (C3's gate: 1,178 s), so its two longest consumer modules were split by
    # lifecycle and the pool doubled: 712 s for the same selection. On a small
    # lane one end-to-end build sets the wall clock and extra workers only add
    # contention between numpy and DuckDB processes (97 s -> 102 s measured with
    # one worker per case), hence a bound by cores, not by cases.
    groups = _ordered_groups(_execution_groups(test_paths))
    evidence_files = _files_with_real_evidence(test_paths)

    def _one(group: tuple[str, ...]) -> TestGroupRun:
        return _run_group(group, "-m", _ROUTED_LANE_MARKER_EXPRESSION)

    results = []
    costs: dict[str, float] = {}
    workers = min(8, len(groups), max(1, (os.cpu_count() or 1) // 4))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(_one, group) for group in groups]):
            result = future.result()
            results.append((result.group, result.code, result.elapsed))
            costs.update(result.file_seconds)
            print(
                f"PLAYPEN_IMPACT_TIMING {_group_label(result.group)} {result.elapsed:.3f}s",
                flush=True,
            )
            if result.code:
                print(f"PLAYPEN_TEST_EXIT {_group_label(result.group)} {result.code}", flush=True)
    _record_timings(results, file_seconds=costs)
    for path in sorted(evidence_files):
        print(f"PLAYPEN_EVIDENCE REQUIRED {path}", flush=True)
    if evidence_files:
        print(f"PLAYPEN_EVIDENCE_COMMAND {_EVIDENCE_LANE_COMMAND}", flush=True)
    failed = [
        path
        for group, code, _ in results
        if code != 0 and not (code == _NOTHING_COLLECTED and set(group) <= evidence_files)
        for path in group
    ]
    if failed:
        # Named, because a bare non-zero exit out of a pool would not say which
        # case failed -- and that is the only part worth reading.
        raise SystemExit(f"admitted case tests failed: {failed}")


_SUMMARY_COUNT = re.compile(
    r"(\d+) (passed|failed|skipped|error|errors|deselected|xfailed|xpassed)"
)
_SKIP_REASON = re.compile(r"^SKIPPED \[\d+\] (.+?): (.*)$", re.MULTILINE)


def _run_evidence_tests(test_paths: tuple[str, ...]) -> None:
    """The explicit lane: only the marked tests, with every skip named.

    Three states, all printed, because the whole point of an explicit lane is
    that nobody has to infer it from a passing gate: REQUIRED is what the
    selection reaches, RAN is what this machine could prove, SKIPPED carries the
    reason the declaration gave. A handoff quotes these lines; a SHA whose
    REQUIRED set is non-empty is not reviewable without them.
    """

    required = _files_with_real_evidence(test_paths)
    for path in sorted(required):
        print(f"PLAYPEN_EVIDENCE REQUIRED {path}", flush=True)
    if not required:
        print("PLAYPEN_EVIDENCE_SUMMARY required=0 ran=0 skipped=0 failed=0", flush=True)
        return
    groups = _ordered_groups(_execution_groups(tuple(sorted(required))), lane="evidence")

    def _one(group: tuple[str, ...]) -> TestGroupRun:
        return _run_group(group, "-rs", "-m", _REAL_EVIDENCE_MARKER, capture=True)

    ran = skipped = failed = 0
    failed_groups: list[str] = []
    results = []
    costs: dict[str, float] = {}
    # Consume futures on completion: one slow model test must not hide completed
    # files, skip reasons or an earlier failure for half an hour.
    with ThreadPoolExecutor(max_workers=min(4, len(groups), max(1, os.cpu_count() or 1))) as pool:
        for future in as_completed([pool.submit(_one, group) for group in groups]):
            completed = future.result()
            group, elapsed = completed.group, completed.elapsed
            results.append((group, completed.code, elapsed))
            costs.update(completed.file_seconds)
            label = _group_label(group)
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            counts = {kind: int(count) for count, kind in _SUMMARY_COUNT.findall(completed.stdout)}
            passed = counts.get("passed", 0)
            skips = counts.get("skipped", 0)
            failures = counts.get("failed", 0) + counts.get("error", 0) + counts.get("errors", 0)
            if completed.code not in (0, _NOTHING_COLLECTED) and failures == 0:
                failures = 1
            ran += passed
            skipped += skips
            failed += failures
            print(
                f"PLAYPEN_EVIDENCE RAN {label} passed={passed} failed={failures} "
                f"skipped={skips} {elapsed:.3f}s",
                flush=True,
            )
            for reason in sorted(
                {reason for _location, reason in _SKIP_REASON.findall(completed.stdout)}
            ):
                print(f"PLAYPEN_EVIDENCE SKIPPED {label} {reason}", flush=True)
            if failures:
                failed_groups.append(label)
    _record_timings(results, file_seconds=costs, lane="evidence")
    print(
        f"PLAYPEN_EVIDENCE_SUMMARY required={len(required)} ran={ran} "
        f"skipped={skipped} failed={failed}",
        flush=True,
    )
    if failed_groups:
        raise SystemExit(f"evidence lane failed: {failed_groups}")


_CASE_ROOTS = ("tests", "case-study")
_CASE_ROOT_PREFIXES = tuple(f"{root}/" for root in _CASE_ROOTS)
"""Where case files live: the packaged suites, and the runnable cases their loaders name."""


def _dotted_module(value: str) -> str:
    return value.removesuffix(".py").replace("/", ".")


def _case_file_facts() -> dict[str, tuple[frozenset[str] | None, frozenset[str]]]:
    return _case_file_facts_for(PLAYPEN_ROOT)


@cache
def _case_file_facts_for(root: Path) -> dict[str, tuple[frozenset[str] | None, frozenset[str]]]:
    """Per case file: its dotted imports (None when unparsable) and the `.py` names it quotes.

    The quoted names are the dependency the import graph cannot see: a thin
    loader names its runnable case by file name, and an import-closure probe
    names the product module it starts in a subprocess. Both are string
    constants, and both are dependencies.
    """

    facts: dict[str, tuple[frozenset[str] | None, frozenset[str]]] = {}
    for case_root in _CASE_ROOTS:
        for path in sorted((root / case_root).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            key = path.relative_to(root).as_posix()
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                imports: frozenset[str] | None = frozenset(
                    discover_python_imports(path, source_root=root)
                )
            except (OSError, SyntaxError, UnicodeError, ValueError):
                facts[key] = (None, frozenset())
                continue
            quoted = frozenset(
                node.value
                for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            )
            facts[key] = (imports, quoted)
    return facts


def _subtree_tests(directory: str, root: Path | None = None) -> frozenset[str]:
    prefix = f"{directory}/"
    facts = _case_file_facts() if root is None else _case_file_facts_for(root)
    return frozenset(
        key
        for key in facts
        if key.startswith(prefix)
        and key.startswith("tests/")
        and Path(key).name.startswith("test_")
    )


def _tests_depending_on(seeds: frozenset[str], root: Path | None = None) -> tuple[str, ...]:
    """Every test that reaches a seed file, transitively through case files.

    A file depends on an impacted file when it imports it by dotted package
    name (`tests.pkg.helper`, or a name below it), imports it by bare stem from
    the same directory (the pre-package spelling the routing cases still use),
    or quotes its file name as a string constant. An impacted `conftest.py` or
    `__init__.py` impacts every test below it; an unparsable file impacts its
    whole directory.
    """

    facts = _case_file_facts() if root is None else _case_file_facts_for(root)
    unparsable_directories = {
        key.rsplit("/", 1)[0] for key, (imports, _quoted) in facts.items() if imports is None
    }
    impacted = set(seeds)
    selected: set[str] = set()
    while True:
        grown = set(impacted)
        for key in impacted:
            directory = key.rsplit("/", 1)[0]
            if Path(key).name in {"conftest.py", "__init__.py"}:
                selected |= _subtree_tests(directory, root)
            imports, _quoted = facts.get(key, (frozenset(), frozenset()))
            if imports is None or directory in unparsable_directories:
                # An unreadable file beside an impacted one is an unknown
                # dependency set; no test in that directory can be shown safe.
                selected |= _subtree_tests(directory, root)
        for key, (imports, quoted) in facts.items():
            if key in grown:
                continue
            if imports is None:
                continue
            directory = key.rsplit("/", 1)[0]
            local = {name.split(".")[0] for name in imports} | {
                name.rsplit(".", 1)[-1] for name in imports
            }
            for target in impacted:
                dotted = _dotted_module(target)
                same_directory = target.rsplit("/", 1)[0] == directory
                if (
                    any(name == dotted or name.startswith(f"{dotted}.") for name in imports)
                    or (same_directory and Path(target).stem in local)
                    or Path(target).name in quoted
                ):
                    grown.add(key)
                    break
        if grown == impacted:
            break
        impacted = grown
    selected |= {
        key for key in impacted if key.startswith("tests/") and Path(key).name.startswith("test_")
    }
    if not selected:
        # A helper nobody visibly imports is not proof that nobody depends on
        # it; the directory-wide answer is the one a parse failure gets too.
        for seed in seeds:
            if not Path(seed).name.startswith("test_"):
                selected |= _subtree_tests(seed.rsplit("/", 1)[0], root)
    return tuple(sorted(selected))


def _case_tests_for_changed_path(value: str) -> tuple[str, ...]:
    """Map a changed case file to the tests that depend on it, transitively.

    A test file is its own answer. Everything else is a dependency question
    answered by `_tests_depending_on`: a helper carries the tests that import it
    (by dotted package name, or by bare stem inside its own directory), a
    `conftest.py` or `__init__.py` carries every test below it because pytest
    injects it without an import, a runnable case under `case-study/` carries
    the loaders that name its file, and a file that cannot be parsed carries its
    whole directory. Speed must never turn an unknown dependency into a silent
    skip.
    """

    path = PLAYPEN_ROOT / value
    if not value.startswith(_CASE_ROOT_PREFIXES) or path.suffix != ".py" or not path.is_file():
        return ()
    if value.startswith("tests/") and path.name.startswith("test_"):
        return (value,)
    return _tests_depending_on(frozenset({value}))


def _execution_tests(
    admitted: tuple[str, ...], *, structural_already_passed: bool
) -> tuple[str, ...]:
    """Which admitted tests this runner executes itself.

    Admission and execution are different questions. The plan always reports
    every admitted path, so a reviewer sees the full required set; only the
    structural files the gate has already run in its parallel batch are dropped,
    and only when the gate says so. Run standalone, this runner still executes
    them, which is what keeps `--run-tests` alone a complete check.
    """

    if not structural_already_passed:
        return admitted
    return tuple(value for value in admitted if value not in STRUCTURAL_TEST_PATHS)


class ProductSelectionUnavailable(RuntimeError):
    """The runner cannot say which tests cover a change, so it must not guess."""


@cache
def _product_graph(
    root: Path = PLAYPEN_ROOT,
) -> tuple[dict[str, str], dict[str, frozenset[str]], dict[str, frozenset[str] | None]]:
    """Source modules, who imports them, and what each case-study file names.

    Both halves come from `devtools.architecture.structural`, which already
    resolves relative imports and package initializers -- a second parser here
    was the defect that made `portfolio_strategy_lab/__init__.py` resolve its
    own `.contracts` one level too high. That owner speaks *logical* module
    names, with `alphalattice` and the product-area segment removed, so this
    function stays in that one space end to end rather than translating between
    two schemes.

    A source tree that cannot be parsed is not "no dependencies": it is an
    inability to plan, and it is raised as one.
    """

    source_root = root / "src"
    sources = tuple(sorted(source_root.rglob("*.py")))
    try:
        by_path = {
            path.relative_to(root).as_posix(): module
            for path, module in discover_python_modules(sources, source_root=source_root)
        }
        importers: dict[str, set[str]] = {}
        for owner, target in discover_module_dependencies(sources, source_root=source_root):
            importers.setdefault(target, set()).add(owner)
    except (OSError, SyntaxError, UnicodeError, ValueError) as error:
        raise ProductSelectionUnavailable(f"product source is unreadable: {error}") from error

    cases: dict[str, frozenset[str] | None] = {}
    for path in sorted(
        candidate for case in _CASE_ROOTS for candidate in (root / case).rglob("*.py")
    ):
        if "__pycache__" in path.parts:
            continue
        key = path.relative_to(root).as_posix()
        try:
            cases[key] = frozenset(discover_python_imports(path, source_root=root))
        except (OSError, SyntaxError, UnicodeError, ValueError):
            # Unknown, not empty. The directory answer below is what an unknown
            # dependency has to become.
            cases[key] = None
    return by_path, {name: frozenset(values) for name, values in importers.items()}, cases


def _reaching_modules(seeds: frozenset[str], root: Path | None = None) -> frozenset[str]:
    """Every module that reaches a seed by imports, transitively and unbounded.

    Not stopped at a package boundary. Architectural ownership says who may
    depend on whom; it does not say a dependency stops being real when it
    crosses the line. The Portfolio ledger is consumed by Host composition and
    by the evidence review composition, and the tests that drive those are as
    much its consumers as the ones inside its own Desk.
    """

    _by_path, importers, _cases = _product_graph() if root is None else _product_graph(root)
    reached = set(seeds)
    pending = list(seeds)
    while pending:
        current = pending.pop()
        for importer in sorted(importers.get(current, frozenset())):
            if importer not in reached:
                reached.add(importer)
                pending.append(importer)
    return frozenset(reached)


def _tests_touching(reached: frozenset[str], root: Path | None = None) -> tuple[str, ...]:
    """Case tests that reach those product modules, directly or through a case file.

    Seeds are the case files whose imports -- or whose quoted module names, for
    a probe that starts the product in a subprocess -- name a reached module;
    `_tests_depending_on` then carries helpers, loaders, conftests and
    unparsable files exactly as it does for a changed case file.
    """

    seeds: set[str] = set()
    facts = _case_file_facts() if root is None else _case_file_facts_for(root)
    for key, (imports, quoted) in facts.items():
        if imports is None:
            seeds.add(key)
            continue
        named = {_logical_product_module(value) for value in quoted}
        if (imports | named) & reached:
            seeds.add(key)
    return _tests_depending_on(frozenset(seeds), root)


def _logical_product_module(value: str) -> str:
    """The graph's logical name for a quoted `alphalattice.<group>.<package>...` string."""

    parts = value.split(".")
    if len(parts) >= 3 and parts[0] == "alphalattice":
        return ".".join(parts[2:])
    return value


def _product_tests_for_changed_path(value: str) -> tuple[str, ...]:
    """Case-study tests that consume a changed product owner.

    Alpha's classifier answers a different question -- whether a change can
    invalidate a scientific identity -- so `OUT_OF_SCOPE` there is a correct
    statement about Alpha, not a claim that the product needs no tests. A
    Portfolio ledger edit selected nothing at all for exactly that reason, and
    this closes that gap without touching Alpha's classification or its
    numerical-parity requirements.

    An ordinary module is walked back through its real importers, and a package
    initializer carries its whole package, because importing any member executes
    it and no static edge records that.

    An absent source is a refusal, not a smaller answer. The closed graph drops
    every edge whose target no longer exists, so a module removed from the tree
    has no importers to walk and its consumers are simply missing from the
    graph -- including consumers in other packages. Substituting the surviving
    package's consumers looked conservative and was not: it returned a non-empty
    list that omitted a real consumer, and a non-empty list is read as coverage.
    So a deleted module is answered from the parent commit instead, where the
    module and every edge to it still exist (V106): the tests that reached it
    there and still exist are its consumers. A path the parent does not hold
    either is still refused.
    """

    if not value.startswith("src/") or not value.endswith(".py"):
        return ()
    if not (PLAYPEN_ROOT / value).is_file():
        return _deleted_module_tests(value)
    by_path, _importers, _cases = _product_graph()
    module = by_path.get(value)
    if module is None:
        raise ProductSelectionUnavailable(
            f"cannot establish product coverage for an absent source: {value}"
        )
    seeds = {module}
    if value.endswith("/__init__.py"):
        seeds.update(name for name in by_path.values() if name.startswith(f"{module}."))
    return _tests_touching(_reaching_modules(frozenset(seeds)))


_PARENT_TREES: list[TemporaryDirectory[str]] = []


@cache
def _parent_tree(revision: str) -> Path:
    """The parent commit's sources and cases, unpacked once a process."""

    import io
    import tarfile

    holder: TemporaryDirectory[str] = TemporaryDirectory(prefix="impact-parent-")
    _PARENT_TREES.append(holder)  # kept for the process, removed at its end
    archive = subprocess.run(
        ("git", "-C", str(PLAYPEN_ROOT), "archive", "--format=tar", revision, "src", *_CASE_ROOTS),
        check=True,
        capture_output=True,
    )
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        tar.extractall(holder.name, filter="data")
    return Path(holder.name)


def _deleted_module_tests(value: str) -> tuple[str, ...]:
    """A deleted module's consumer tests, from the parent commit's graph (V106)."""

    held = subprocess.run(
        ("git", "-C", str(PLAYPEN_ROOT), "cat-file", "-e", f"{_PARENT_REVISION}:{value}"),
        capture_output=True,
        check=False,
    )
    if held.returncode != 0:  # the parent holds no such file either
        raise ProductSelectionUnavailable(
            f"cannot establish product coverage for an absent source: {value}"
        )
    parent = _parent_tree(_PARENT_REVISION)
    by_path, _importers, _cases = _product_graph(parent)
    module = by_path.get(value)
    if module is None:
        raise ProductSelectionUnavailable(
            f"cannot establish product coverage for an absent source: {value}"
        )
    seeds = {module}
    if value.endswith("/__init__.py"):
        seeds.update(name for name in by_path.values() if name.startswith(f"{module}."))
    reached = _tests_touching(_reaching_modules(frozenset(seeds), parent), parent)
    return tuple(path for path in reached if (PLAYPEN_ROOT / path).is_file())


def _moved_module_tests(value: str) -> tuple[str, ...]:
    """A moved module's own tests and the tests that name it, patching its objects (V215).

    A move changes no behaviour, so the importers that walked through it need no run;
    the tests that import the module itself, or name it in a string (a patch by path),
    are the ones a move can break.
    """

    root = PLAYPEN_ROOT if (PLAYPEN_ROOT / value).is_file() else _parent_tree(_PARENT_REVISION)
    by_path, _importers, _cases = _product_graph() if root == PLAYPEN_ROOT else _product_graph(root)
    module = by_path.get(value)
    if module is None:
        return _product_tests_for_changed_path(value)
    named = _tests_touching(frozenset({module}), None if root == PLAYPEN_ROOT else root)
    return tuple(path for path in named if (PLAYPEN_ROOT / path).is_file())


def _reach_roles(changed: tuple[str, ...]) -> tuple[str, ...]:
    """The identity roles the changed source files' areas may move, by the reach table."""

    table = PLAYPEN_ROOT / "config" / "registries" / "identity-reach.json"
    if not table.is_file():
        return ()
    areas = json.loads(table.read_text(encoding="utf-8"))["areas"]
    roles: set[str] = set()
    for value in changed:
        if not (value.startswith("src/alphalattice/") and value.endswith(".py")):
            continue
        parts = value.removesuffix(".py").split("/")[1:]
        module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        roles.update(
            areas.get(module if parts[-1] == "__init__" else module.rpartition(".")[0], ())
        )
    return tuple(sorted(roles))


def _changed_paths(*, base: str | None, staged: bool) -> tuple[str, ...]:
    args = ["git", "-c", "core.quotepath=false", "-C", str(PLAYPEN_ROOT), "diff"]
    if staged:
        args.append("--cached")
    elif base is not None:
        args.append(base)
    args.extend(("--name-only", "--diff-filter=ACDMR", "-z"))
    completed = subprocess.run(args, check=True, capture_output=True, text=True, encoding="utf-8")
    paths = {value for value in completed.stdout.split("\0") if value}
    if not staged and base is None:
        untracked = subprocess.run(
            (
                "git",
                "-c",
                "core.quotepath=false",
                "-C",
                str(PLAYPEN_ROOT),
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
            ),
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        paths.update(value for value in untracked.stdout.split("\0") if value)
    return tuple(sorted(paths))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base")
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument(
        "--structural-tests-already-passed",
        action="store_true",
        help="Do not repeat structural tests already run by check_playpen.py.",
    )
    parser.add_argument(
        "--evidence",
        action="store_true",
        help=(
            "Run the explicit real-evidence lane over the same selection instead of the "
            "routed lane: only tests marked real_evidence, every skip named."
        ),
    )
    args = parser.parse_args()
    if args.base and args.staged:
        parser.error("--base and --staged are mutually exclusive")
    global _PARENT_REVISION
    _PARENT_REVISION = args.base or "HEAD"
    all_changed = _changed_paths(base=args.base, staged=args.staged)
    previous = git_previous(PLAYPEN_ROOT, _PARENT_REVISION)
    neutral = tuple(
        verdict.path
        for verdict in neutral_verdicts(PLAYPEN_ROOT, all_changed, previous)
        if verdict.neutral
    )
    changed = tuple(value for value in all_changed if value not in neutral)
    moved = moved_files(PLAYPEN_ROOT, changed, previous)
    if neutral:
        # A syntax-equal file no code reads the text of moves no closure by rule, and
        # no byte closure tracks it, so the identity readout cannot move (GN).
        print(
            f"neutral: {len(neutral)} files, syntax equal, readout unchanged",
            file=sys.stderr,
        )
    if moved:
        print(
            f"moved: {len(moved)} files, their own tests and the tests that name them",
            file=sys.stderr,
        )
    reach = _reach_roles(changed)
    if reach:
        # The same table the gate holds (GB): what the changed areas may move.
        print(f"reach: the changed areas may move {len(reach)} roles", file=sys.stderr)
    if not changed:
        print(
            json.dumps(
                {"changed_paths": [], "neutral_paths": neutral, "required_test_paths": []},
                indent=2,
            )
        )
        return 0
    impact = alpha_verification_impact(changed)
    absent = tuple(
        value for value in impact.required_test_paths if not (PLAYPEN_ROOT / value).is_file()
    )
    if absent:
        # This used to be an `is_file()` filter, which turned a declared
        # verification test that had been moved or deleted into a silent
        # omission: the plan still looked complete and the domain simply lost
        # its coverage. A missing declaration is a stop.
        raise SystemExit(f"declared verification tests are absent: {list(absent)}")
    try:
        consumer_tests = tuple(
            sorted(
                {
                    path
                    for value in changed
                    for path in (
                        _moved_module_tests(value)
                        if value in moved and value.startswith("src/")
                        else _product_tests_for_changed_path(value)
                    )
                }
            )
        )
    except ProductSelectionUnavailable as error:
        # Refusing is the only honest answer left: an empty plan would report
        # full coverage for a change nobody can map to a test.
        raise SystemExit(f"product verification selection unavailable: {error}") from error
    required_tests = set(impact.required_test_paths)
    required_tests.update(consumer_tests)
    required_tests.update(
        test_path for value in changed for test_path in _case_tests_for_changed_path(value)
    )
    admitted_tests = tuple(sorted(required_tests))
    execution_tests = _execution_tests(
        admitted_tests, structural_already_passed=args.structural_tests_already_passed
    )
    payload = {
        "impact_hash": impact.impact_hash,
        "changed_paths": impact.changed_paths,
        # GN: the files proved neutral (no tests) and the files only moved (their own
        # tests and those naming them, not every importer's).
        "neutral_paths": neutral,
        "moved_paths": {path: sorted(names) for path, names in sorted(moved.items())},
        "reach_roles": reach,
        "domains": impact.domains,
        "required_test_paths": admitted_tests,
        # Reported separately from Alpha's domain selection: these are product
        # consumer tests for changed owners, and they carry no claim about
        # scientific invalidation. Merging the two lists would make an
        # `OUT_OF_SCOPE` change look like an Alpha-relevant one.
        "product_consumer_test_paths": consumer_tests,
        "execution_test_paths": execution_tests,
        # The files the routed lane deselects tests from and the evidence lane
        # runs. Reported in the plan so a reviewer sees what a green routed
        # lane did not prove.
        "evidence_test_paths": tuple(sorted(_files_with_real_evidence(execution_tests))),
        "full_numerical_parity_required": impact.full_numerical_parity_required,
        "fit_children_reusable": impact.fit_children_reusable,
        "current_refit_children_reusable": impact.current_refit_children_reusable,
        "domain_hashes": alpha_execution_domain_hashes(PLAYPEN_ROOT),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.run_tests and execution_tests:
        if args.evidence:
            _run_evidence_tests(execution_tests)
        else:
            _run_admitted_tests(execution_tests)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
