"""Derive and optionally execute the bounded Alpha verification plan."""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cache
from pathlib import Path, PurePosixPath
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
from devtools.architecture.git_inputs import (  # noqa: E402
    read_blobs,
    staged_changes,
    table_moves,
)
from devtools.architecture.neutral_changes import (  # noqa: E402
    git_previous,
    moved_files,
    neutral_verdicts,
)
from devtools.architecture.structural import (  # noqa: E402
    discover_module_dependencies,
    discover_python_imports,
    discover_python_modules,
    kept_fact,
    kept_facts,
)
from devtools.pipeline.stage import (  # noqa: E402
    derived_holders,
    facet_holders,
    registered_holders,
    selection_readers,
)
from devtools.testing.layout import placed_tests  # noqa: E402

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
            env=dict(os.environ),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
            | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0),
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


def _run_admitted_tests(
    test_paths: tuple[str, ...],
    *,
    workers: int | None = None,
    arguments: tuple[str, ...] = (),
) -> None:
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
        return _run_group(
            group,
            "-m",
            _ROUTED_LANE_MARKER_EXPRESSION,
            *arguments,
        )

    results = []
    costs: dict[str, float] = {}
    workers = min(workers or 8, len(groups), max(1, (os.cpu_count() or 1) // 4))
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
                imports: frozenset[str] | None = frozenset(
                    discover_python_imports(path, source_root=root)
                )
                quoted = _quoted_names(path)
            except (OSError, SyntaxError, UnicodeError, ValueError):
                facts[key] = (None, frozenset())
                continue
            facts[key] = (imports, quoted)
    return facts


_QUOTED: dict[bytes, frozenset[str]] = {}  # this process's, by content digest; errors never kept;
# it answers ahead of kept_fact's version, for a process whose loaded code a run never replaces


def _quoted_names(path: Path) -> frozenset[str]:
    """The string constants one case file quotes, parsed once per content in a process."""
    data = path.read_bytes()
    digest = hashlib.sha256(data).digest()
    if (known := _QUOTED.get(digest)) is None:
        known = _QUOTED[digest] = frozenset(
            kept_fact(
                _this(),
                "quoted",
                data,
                lambda bytes_: sorted(_quoted_of(bytes_, path)),
                lambda value: (
                    isinstance(value, list) and all(isinstance(item, str) for item in value)
                ),
            )
        )
    return known


def _quoted_of(data: bytes, path: Path) -> set[str]:
    text = data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")  # as read_text does
    return {
        node.value
        for node in ast.walk(ast.parse(text, filename=str(path)))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


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


def _depends(imports: frozenset[str], quoted: frozenset[str], directory: str, target: str) -> bool:
    """A parsable case file depends on `target`: it imports it by dotted package name (or a
    name below it), by bare stem from the same directory, or quotes its file name."""
    dotted = _dotted_module(target)
    local = {name.split(".")[0] for name in imports} | {name.rsplit(".", 1)[-1] for name in imports}
    return (
        any(name == dotted or name.startswith(f"{dotted}.") for name in imports)
        or (target.rsplit("/", 1)[0] == directory and Path(target).stem in local)
        or Path(target).name in quoted
    )


def _scan_dependents(
    facts: dict[str, tuple[frozenset[str] | None, frozenset[str]]], target: str
) -> set[str]:
    """The case files that depend on `target`, by one pass of `_depends` (a target outside
    the facts)."""
    return {
        key
        for key, (imports, quoted) in facts.items()
        if imports is not None and _depends(imports, quoted, key.rsplit("/", 1)[0], target)
    }


def _case_dependents(
    facts: dict[str, tuple[frozenset[str] | None, frozenset[str]]],
) -> dict[str, set[str]]:
    """Per case file, the case files that depend on it by `_depends`'s three rules, built in one
    pass: each import name's dotted prefixes, its same-directory stems and its quoted file names
    are looked up rather than every file being tested against every impacted file per round
    (the rescan took about 3 s of every product selection)."""
    by_dotted: dict[str, set[str]] = {}
    by_stem: dict[tuple[str, str], set[str]] = {}
    by_name: dict[str, set[str]] = {}
    for target in facts:
        by_dotted.setdefault(_dotted_module(target), set()).add(target)
        by_stem.setdefault((target.rsplit("/", 1)[0], Path(target).stem), set()).add(target)
        by_name.setdefault(Path(target).name, set()).add(target)
    dependents: dict[str, set[str]] = {target: set() for target in facts}
    for key, (imports, quoted) in facts.items():
        if imports is None:
            continue
        directory = key.rsplit("/", 1)[0]
        local = {name.split(".")[0] for name in imports} | {
            name.rsplit(".", 1)[-1] for name in imports
        }
        targets: set[str] = set()
        for name in imports:
            parts = name.split(".")
            for end in range(1, len(parts) + 1):
                targets |= by_dotted.get(".".join(parts[:end]), set())
        for stem in local:
            targets |= by_stem.get((directory, stem), set())
        for value in quoted:
            targets |= by_name.get(value, set())
        for target in targets:
            dependents[target].add(key)
    return dependents


def _anchored(node: ast.AST, path: PurePosixPath) -> PurePosixPath | None:
    """The repository path a `Path(__file__)...` expression names, read from this file's place."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and len(node.args) == 1:
        named = node.args[0]
        stated = isinstance(named, ast.Name) and named.id == "__file__"
        return path if stated and node.func.id.endswith("Path") else None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        base = _anchored(node.func.value, path)
        if base is None or node.keywords:
            return None
        if node.func.attr == "resolve" and not node.args:
            return base
        names = [arg.value for arg in node.args if isinstance(arg, ast.Constant)]
        wanted = node.func.attr == "with_name" and len(node.args) == len(names) == 1
        return base.with_name(str(names[0])) if wanted else None
    if isinstance(node, ast.Attribute) and node.attr == "parent":
        base = _anchored(node.value, path)
        return None if base is None else base.parent
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "parents"
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, int)
    ):
        base = _anchored(node.value.value, path)
        return None if base is None else base.parents[node.slice.value]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        base = _anchored(node.left, path)
        part = node.right.value if isinstance(node.right, ast.Constant) else None
        return None if base is None or not isinstance(part, str) else base / part
    return None


class _MoveRewrite(ast.NodeTransformer):
    """A file with what a move's map owns made equal: imports, anchors and the map's own names."""

    def __init__(self, path: str, names: re.Pattern[str], places: dict[str, str]) -> None:
        self.path, self.names, self.places = PurePosixPath(path), names, places

    def visit(self, node: ast.AST) -> ast.AST | None:
        if isinstance(node, ast.Import | ast.ImportFrom):
            return None
        resolved = _anchored(node, self.path) if isinstance(node, ast.expr) else None
        return ast.Constant(resolved.as_posix()) if resolved else super().visit(node)

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        if isinstance(node.value, str):
            node.value = self.names.sub(lambda found: self.places[found[0]], node.value)
        return node


def _follows_the_move(before: str, after: str, old: str, new: str, moves: dict[str, str]) -> bool:
    """Whether a moved test differs from its old self only by what the move's rewrite owns: its
    imports, its file-relative anchors where each still names the same path, and the map's names."""
    places = {key: value for key, value in moves.items()}
    places |= {k[:-3].replace("/", "."): v[:-3].replace("/", ".") for k, v in moves.items()}
    order = sorted(places, key=len, reverse=True)
    names = re.compile("|".join(re.escape(name) for name in order))
    try:
        trees = [ast.parse(before), ast.parse(after)]
    except SyntaxError:
        return False
    for path, tree in zip((old, new), trees, strict=True):
        _MoveRewrite(path, names, places).visit(tree)
    return ast.dump(trees[0]) == ast.dump(trees[1])


def _spared_moves(base: str | None) -> frozenset[str]:
    """Both paths of each moved test this change's table pairs and its rewrite alone explains:
    its nodes map one to one, so the map's proof answers for it and the stage does not repeat it."""
    if base is None:
        return frozenset()
    pairs = staged_changes(PLAYPEN_ROOT, base=base, tree="HEAD")
    config = {
        path: meta[3] for meta, path in pairs if path.startswith("config/") and meta[4] != "D"
    }
    read = read_blobs(list(config.values()), PLAYPEN_ROOT)
    moves = table_moves(pairs, {path: read[oid] for path, oid in config.items()})
    tests = {new: old for new, old in moves.items() if Path(new).name.startswith("test_")}
    specs = [f"{base}:{old}" for old in tests.values()] + [f"HEAD:{new}" for new in tests]
    blobs = read_blobs(specs, PLAYPEN_ROOT)
    return frozenset(
        path
        for new, old in tests.items()
        if f"{base}:{old}" in blobs
        and f"HEAD:{new}" in blobs
        and _follows_the_move(blobs[f"{base}:{old}"], blobs[f"HEAD:{new}"], old, new, moves)
        for path in (new, old)
    )


def _registered_fixtures(conftest: str, root: Path | None) -> dict[str, set[str]] | None:
    """The fixtures a conftest only registers, by provider file; None when it does anything else."""
    found: dict[str, set[str]] = {}
    for node in ast.parse(((root or PLAYPEN_ROOT) / conftest).read_text("utf-8")).body:
        if not isinstance(node, ast.ImportFrom) or not (node.module or "").startswith("tests."):
            return None
        if any(item.asname not in (None, item.name) or item.name == "*" for item in node.names):
            return None
        names = found.setdefault((node.module or "").replace(".", "/") + ".py", set())
        names.update(item.name for item in node.names)
    return found


@cache
def _names_in(path: Path) -> frozenset[str]:
    """Each parameter and string a test file holds: what can name a fixture."""
    nodes = tuple(ast.walk(ast.parse(path.read_text("utf-8"))))
    named = {node.arg for node in nodes if isinstance(node, ast.arg)}
    named |= {
        node.value
        for node in nodes
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    return frozenset(named)


def _fixture_users(
    conftest: str, impacted: set[str], seeds: frozenset[str], root: Path | None
) -> tuple[str, ...] | None:
    """The tests a conftest carries when it is impacted only as a registry of fixture providers:
    those naming a reached provider's fixture or an indirect dependent.
    None when it must carry every test below it."""
    registered = None if conftest in seeds else _registered_fixtures(conftest, root)
    reached = {path for path in (registered or {}) if path in impacted}
    if not reached or any(
        not path.startswith("tests/_support/") or not path.endswith("_fixtures.py")
        for path in reached
    ):
        return None
    assert registered is not None
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    automatic: set[str] = set()
    try:
        for path, names in registered.items():
            body = ast.parse(((root or PLAYPEN_ROOT) / path).read_text("utf-8")).body
            definitions = {
                node.name: node
                for node in body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if path in reached:
                if any(name.startswith("pytest_") for name in definitions.keys() | names):
                    return None
                for function in definitions.values():
                    for decorator in function.decorator_list:
                        if isinstance(decorator, ast.Call) and any(
                            keyword.arg in {None, "name"}
                            or (
                                keyword.arg == "autouse"
                                and not (
                                    isinstance(keyword.value, ast.Constant)
                                    and keyword.value.value is False
                                )
                            )
                            for keyword in decorator.keywords
                        ):
                            return None
            for name in names:
                if name.startswith("pytest_"):
                    continue
                if name not in definitions:
                    return None  # re-exported or dynamic fixture: its dependencies are unknown
                functions[name] = definitions[name]
                if any(
                    isinstance(decorator, ast.Call)
                    and any(
                        keyword.arg in {None, "autouse", "name"} for keyword in decorator.keywords
                    )
                    for decorator in definitions[name].decorator_list
                ):
                    automatic.add(name)
    except (OSError, SyntaxError, UnicodeError):
        return None
    names = {name for path in reached for name in registered[path]}
    while True:
        expanded = names | {
            name
            for name, function in functions.items()
            if names
            & {
                arg.arg
                for arg in (
                    *function.args.posonlyargs,
                    *function.args.args,
                    *function.args.kwonlyargs,
                )
            }
        }
        if expanded == names:
            break
        names = expanded
    if names & automatic:
        return None
    directory = conftest.rsplit("/", 1)[0]
    facts = _case_file_facts() if root is None else _case_file_facts_for(root)
    try:
        for child in facts:
            if (
                child != conftest
                and child.startswith(directory + "/")
                and child.endswith("/conftest.py")
            ):
                body = ast.parse(((root or PLAYPEN_ROOT) / child).read_text("utf-8")).body
                if any(
                    names
                    & {
                        arg.arg
                        for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
                    }
                    for node in body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                ):
                    return None
    except (OSError, SyntaxError, UnicodeError):
        return None
    below = _subtree_tests(directory, root)
    return tuple(test for test in below if names & _names_in((root or PLAYPEN_ROOT) / test))


def _tests_depending_on(seeds: frozenset[str], root: Path | None = None) -> tuple[str, ...]:
    """Every test that reaches a seed file, transitively through case files.

    A file depends on an impacted file when it imports it by dotted package
    name (`tests.pkg.helper`, or a name below it), imports it by bare stem from
    the same directory (the pre-package spelling the routing cases still use),
    or quotes its file name as a string constant. An impacted `conftest.py` or
    `__init__.py` impacts every test below it, except a conftest that only registers
    fixture providers, which impacts the tests naming their fixtures; an unparsable
    file impacts its whole directory.
    """

    facts = _case_file_facts() if root is None else _case_file_facts_for(root)
    unparsable_directories = {
        key.rsplit("/", 1)[0] for key, (imports, _quoted) in facts.items() if imports is None
    }
    dependents = _case_dependents(facts)
    impacted, frontier = set(seeds), set(seeds)
    while frontier:  # the closure of `dependents`, from one index, not a rescan per round
        frontier = {
            key
            for target in frontier
            for key in (dependents[target] if target in facts else _scan_dependents(facts, target))
            if key not in impacted
        }
        impacted |= frontier
    selected: set[str] = set()
    for key in impacted:
        directory = key.rsplit("/", 1)[0]
        if Path(key).name in {"conftest.py", "__init__.py"}:
            users = (
                _fixture_users(key, impacted, seeds, root) if key.endswith("conftest.py") else None
            )
            selected |= _subtree_tests(directory, root) if users is None else set(users)
        imports, _quoted = facts.get(key, (frozenset(), frozenset()))
        if imports is None or directory in unparsable_directories:
            # An unreadable file beside an impacted one is an unknown
            # dependency set; no test in that directory can be shown safe.
            selected |= _subtree_tests(directory, root)
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
    places = placed_tests(PLAYPEN_ROOT)  # a test moved since the parent still covers the module
    moved = (places.get(path, path) for path in reached)
    return tuple(path for path in moved if (PLAYPEN_ROOT / path).is_file())


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


def _registered_holders(
    paths: set[str],
    *,
    staged: bool,
    pairs: list[tuple[list[str], str]],
    changed: dict[str, str],
) -> tuple[str, ...]:
    """The same declared edges as the stage, read from this plan's own input state; ``changed``
    holds both sides of each changed Python file by object id."""
    policy_path = "config/registries/test-sizes.json"
    if staged:
        policy = read_blobs([":" + policy_path], PLAYPEN_ROOT, timeout=60).get(":" + policy_path)
        if policy is None:
            raise ValueError(f"the staged selection registry is unavailable: {policy_path}")
    else:
        policy = (PLAYPEN_ROOT / policy_path).read_text(encoding="utf-8")
    rows = selection_readers(json.loads(policy))
    needed = {
        row[key]
        for row in rows.values()
        if "census" in row
        and (
            any(path.startswith("src/alphalattice/") and path.endswith(".py") for path in paths)
            or paths.intersection(row["conditional_inputs"])
        )
        for key in ("census", "sample")
    } | {node.split("::", 1)[0] for node, row in rows.items() if "literal_root" in row}
    if staged:
        blobs = read_blobs([":" + path for path in sorted(needed)], PLAYPEN_ROOT, timeout=60)
        sources = {path: blobs[":" + path] for path in needed if ":" + path in blobs}
    else:
        sources = {
            path: (PLAYPEN_ROOT / path).read_text(encoding="utf-8")
            for path in needed
            if (PLAYPEN_ROOT / path).is_file()
        }
    return (*registered_holders(paths, sources, rows), *facet_holders(pairs, changed, rows))


def _declared_holders(
    changed: tuple[str, ...], *, base: str | None, staged: bool
) -> tuple[str, ...]:
    """Use the stage's declared owners for both the standalone and workshop plans."""
    if not changed:
        return ()
    zero = "0" * 40
    command = ["git", "-C", str(PLAYPEN_ROOT), "diff", "--raw", "-z", "--no-renames", "--no-abbrev"]
    if staged:
        command.append("--cached")
    elif base:
        command.append(base)
    raw = subprocess.run(
        command, check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout
    fields = raw.split("\0")
    if fields[-1:] == [""]:
        fields.pop()
    if len(fields) % 2:
        raise ValueError("malformed changed-file record")
    entries = {path: header.split() for header, path in zip(fields[::2], fields[1::2], strict=True)}
    paths = set(changed) | entries.keys()
    candidates = {path for path in paths if path.endswith(".py")}
    pairs = [([":000000", "000000", zero, zero, "A"], path) for path in paths - candidates]
    found = {}
    for path in candidates & entries.keys():
        meta = entries[path]
        if (
            len(meta) != 5
            or not meta[0].startswith(":")
            or meta[4] not in {"A", "D", "M", "T"}
            or meta[0][1:] not in {"000000", "100644", "100755"}
            or meta[1] not in {"000000", "100644", "100755"}
            or any(not re.fullmatch(r"[0-9a-f]{40}", oid) for oid in meta[2:4])
        ):
            raise ValueError(f"changed Python source is not a regular file: {path}")
        found[path] = meta
    # A path the diff does not list is unchanged at both ends, so it holds no changed facts;
    # only against the index is it an untracked file, new to the working tree.
    for path in candidates - found.keys():
        if not (staged or base) and (PLAYPEN_ROOT / path).is_file():
            found[path] = [":000000", "100644", zero, zero, "A"]
        else:
            pairs.append(([":000000", "000000", zero, zero, "A"], path))
    ids = {meta[2] for meta in found.values() if meta[2] != zero}
    if staged:
        ids.update(meta[3] for meta in found.values() if meta[3] != zero)
    blobs = read_blobs(sorted(ids), PLAYPEN_ROOT, timeout=60)
    if set(blobs) != ids:
        raise ValueError("changed Python blob is unavailable")
    for path, meta in found.items():
        if not staged and meta[4] != "D":
            source = (PLAYPEN_ROOT / path).read_bytes()
            oid = hashlib.sha256(source).hexdigest()
            blobs[oid] = source.decode("utf-8")
            meta[3] = oid
        pairs.append((meta, path))
    declared = _registered_holders(paths, staged=staged, pairs=pairs, changed=blobs)
    return tuple(sorted({*derived_holders(pairs), *declared}))


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
    """Select, with every per-file fact this run derives kept in the line's scan store, versioned
    by this runner and the structural owner as loaded (structural.kept_facts)."""
    this = _this()
    with kept_facts("impact-selection", this) if this is not None else contextlib.nullcontext():
        return _select()


def _this() -> types.ModuleType | None:
    """This runner as loaded, or None when it was loaded under a name it is not registered by
    (a test's own copy), which then keeps no fact."""
    module = sys.modules.get(__name__)
    return module if module is not None and vars(module) is globals() else None


def _select() -> int:
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
    spared = _spared_moves(args.base)
    changed = tuple(value for value in changed if value not in spared)
    if spared:
        print(f"spared: {len(spared) // 2} moved tests follow their map", file=sys.stderr)
    try:
        declared = _declared_holders(all_changed, base=args.base, staged=args.staged)
    except (OSError, ValueError, SyntaxError, UnicodeError, subprocess.SubprocessError) as error:
        raise SystemExit(f"declared verification selection unavailable: {error}") from error
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
        neutral_files = tuple(sorted({node.partition("::")[0] for node in declared}))
        neutral_absent = [path for path in neutral_files if not (PLAYPEN_ROOT / path).is_file()]
        if neutral_absent:
            raise SystemExit(f"declared verification tests are absent: {neutral_absent}")
        payload: dict[str, object] = {
            "changed_paths": [],
            "neutral_paths": neutral,
            "required_test_paths": neutral_files,
        }
        if declared:
            execution = _execution_tests(
                neutral_files, structural_already_passed=args.structural_tests_already_passed
            )
            payload["declared_test_nodes"] = declared
            payload["execution_test_paths"] = execution
        print(json.dumps(payload, indent=2))
        if declared and args.run_tests and execution:
            if args.evidence:
                _run_evidence_tests(execution)
            else:
                _run_admitted_tests(execution)
        return 0
    impact = alpha_verification_impact(changed)
    places = placed_tests(PLAYPEN_ROOT)
    selected = tuple(places.get(value, value) for value in impact.required_test_paths)
    absent = tuple(value for value in selected if not (PLAYPEN_ROOT / value).is_file())
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
    required_tests = set(selected)
    declared_files = {node.partition("::")[0] for node in declared}
    absent_declared = sorted(path for path in declared_files if not (PLAYPEN_ROOT / path).is_file())
    if absent_declared:
        raise SystemExit(f"declared verification tests are absent: {absent_declared}")
    required_tests.update(declared_files)
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
        # The moved tests that follow their map: the commit tier leaves them out, so a stage
        # answers for what a move actually edited. Post-merge and the nightly still run them.
        "spared_paths": tuple(sorted(spared)),
        "moved_paths": {path: sorted(names) for path, names in sorted(moved.items())},
        "reach_roles": reach,
        "domains": impact.domains,
        "required_test_paths": admitted_tests,
        "declared_test_nodes": declared,
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
