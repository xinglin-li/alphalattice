"""The daily test-health report the test manager publishes: the test rows of the quality metrics,
from the checkout, its test-size registry and the commit-stage receipts it is given, each read
once through the receipt's own reader, and the rolling per-test seconds those receipts hold."""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from bisect import insort
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from datetime import date, datetime, timedelta
from fnmatch import fnmatchcase
from hashlib import sha256
from pathlib import Path
from typing import Any, TypedDict

from pydantic import TypeAdapter, ValidationError

from devtools.pipeline.definition import DefinitionError
from devtools.pipeline.receipt import (
    CASCADE,
    CRASH,
    TOOLING_REDS,
    ReceiptError,
    _valid_bound,
    classed,
    read,
)
from devtools.testing.requirements import (
    file_holders,
    index,
    legacy_owner,
    node_matches,
)
from devtools.testing.requirements import (
    load as load_requirements,
)
from devtools.testing.sizes import load_policy

SLOW_SECONDS = 10  # the quality metrics' row: commit-stage tests of 10 s or more
ROLLING_DAYS, TREND_DAYS, RANKED = (
    28,
    7,
    10,
)  # the table's window, its trend's window, each list's length
# A run on a degraded machine is left out of every baseline. As the harness's machine-health alert
# reads it (10-28: the machine drifted 3-9x slower until a reboot), its tests take DEGRADED_RATIO
# times their best or more; here it is the median node, so one slow test never condemns a run, and
# a run with fewer than MEASURED_NODES comparable nodes is unmeasured, never healthy.
DEGRADED_RATIO, MEASURED_NODES = 2, 50
NOISE_SECONDS = 0.1  # below this a node's time is overhead, never a ratio sample
LARGE_BYTES = 100 * 1024  # the shape guard's ceiling for any tracked file under tests/


def _tracked(root: Path, top: str) -> list[Path]:
    listed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--", top],
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8")
    return [root / path for path in listed.split("\0") if path and (root / path).is_file()]


def _texts(paths: list[Path]) -> dict[Path, str]:
    """Each tracked Python file's text, read once."""
    texts = {}
    for path in paths:
        if path.suffix == ".py":
            try:
                texts[path] = path.read_bytes().decode("utf-8")
            except (OSError, UnicodeError) as error:
                raise DefinitionError(
                    "pipeline.workspace_unreadable",
                    f"{path}: {error}; restore this UTF-8 file and retry health",
                ) from None
    return texts


def tree(root: Path, today: date) -> dict[str, Any]:
    """Test bloat and the quarantine, from the checkout itself, each file read once."""
    files = _tracked(root, "tests")
    texts, source = _texts(files), _texts(_tracked(root, "src"))
    tests = [
        holder
        for path, text in sorted(texts.items())
        if path.name.startswith("test_")
        for holder in file_holders(path.relative_to(root).as_posix(), text)
    ]
    held = index(root, tests=tests)
    test_lines = sum(text.count("\n") for text in texts.values())
    source_lines = sum(text.count("\n") for text in source.values())
    quarantine = [
        {
            "node": row["node"],
            "owner": row["owner"],
            "until": row["until"],
            "days_left": (date.fromisoformat(row["until"]) - today).days,
        }
        for row in load_policy(root).get("quarantine", [])
    ]
    return {
        "test_lines": test_lines,
        "source_lines": source_lines,
        "test_to_source": round(test_lines / source_lines, 3) if source_lines else None,
        "test_functions": len(tests),
        "registered_functions": sum(1 for test in tests if test.ids),
        "requirements_with_several_holders": sorted({key for key, _ in held.duplicates()}),
        "files_over_100kb": [
            p.relative_to(root).as_posix() for p in files if p.stat().st_size > LARGE_BYTES
        ],
        "quarantine": sorted(quarantine, key=lambda row: row["days_left"]),
    }


def _stage_inputs(
    receipts: Sequence[Path], unreadable: list[dict[str, str]]
) -> Iterator[tuple[Path, list[Any], datetime]]:
    """Read each receipt identity once and retain unavailable inputs by their named refusal."""
    seen_paths: set[Path] = set()
    seen_receipts: set[bytes] = set()
    for path in receipts:
        try:
            canonical = path.resolve()
            if canonical in seen_paths:
                continue
            seen_paths.add(canonical)
            data, events = read(path)
            digest = sha256(data).digest()
            if digest in seen_receipts:
                continue
            seen_receipts.add(digest)
            moment = datetime.fromisoformat(events[-1].finished.finished_at)
            if moment.utcoffset() is None:
                raise ValueError("finished_at needs a timezone offset")
        except ReceiptError as refused:
            unreadable.append({"receipt": str(path), "code": refused.code})
            continue
        except (ValueError, OSError, RuntimeError) as error:
            unreadable.append(
                {"receipt": str(path), "code": "pipeline.receipt_malformed", "reason": str(error)}
            )
            continue
        yield path, events, moment


Observation = tuple[str, float]  # one node's first attempt in one stage: its status and seconds
Run = tuple[datetime, str, dict[str, Observation]]  # a stage's finish, head and observations
Merged = Callable[[str], bool | None]  # whether a head is in the reported history; None unknown


def _row(
    node: str, samples: list[tuple[datetime, float, str]], end: date, merged: Merged
) -> dict[str, Any]:
    """One node's passed seconds in healthy runs, oldest first: its p50 and nearest-rank p90, its
    trend (over its runs in the last TREND_DAYS, the newer half's p50 less the older half's;
    unknown below four runs), the first run at SLOW_SECONDS or more, and the stage in which it
    doubled (by 1 s or more over the median before) with at least one later run, all staying there,
    with whether that stage's head is merged, when the history can tell."""
    values = [seconds for _, seconds, _ in samples]
    ordered = sorted(values)
    week = [s for at, s, _ in samples if (end - at.date()).days < TREND_DAYS]
    half = len(week) // 2
    earlier: list[float] = []
    jump = None
    for i, (at, seconds, head) in enumerate(samples):
        if earlier:
            floor = max(2 * statistics.median(earlier), statistics.median(earlier) + 1)
            # The step must be followed by at least one confirming run.
            if i + 1 < len(values) and values[i - 1] < floor <= min(values[i:]):
                jump = {"at": at.isoformat(), "head": head, "merged": merged(head)}
        insort(earlier, seconds)
    return {
        "node": node,
        "runs": len(values),
        "p50": round(statistics.median(values), 3),
        "p90": round(ordered[math.ceil(0.9 * len(ordered)) - 1], 3),
        "trend": round(statistics.median(week[-half:]) - statistics.median(week[:half]), 3)
        if half >= 2
        else None,
        "first_slow": next((at.isoformat() for at, s, _ in samples if s >= SLOW_SECONDS), None),
        "jumped_in": jump,
    }


def _machine(window: list[Run]) -> list[tuple[str, float | None]]:
    """Each run's machine state: a node is compared only with its best passed time in the other
    runs, so no run vouches for itself; a run with under MEASURED_NODES such comparisons is
    unmeasured, and one whose median comparison is DEGRADED_RATIO or more is degraded."""
    best: dict[str, list[tuple[float, int]]] = {}  # each node's two fastest passes, by run
    for i, (_, _, seen) in enumerate(window):
        for node, (status, value) in seen.items():
            if status == "PASSED":
                best[node] = sorted([*best.get(node, []), (value, i)])[:2]
    states: list[tuple[str, float | None]] = []
    for i, (_, _, seen) in enumerate(window):
        ratios = []
        for node, (status, value) in seen.items():
            other = next((v for v, j in best.get(node, []) if j != i), None)
            if status == "PASSED" and other is not None and other >= NOISE_SECONDS:
                ratios.append(value / other)
        if len(ratios) < MEASURED_NODES:
            states.append(("unmeasured", None))
        else:
            factor = statistics.median(ratios)
            states.append(("degraded" if factor >= DEGRADED_RATIO else "healthy", round(factor, 2)))
    return states


def ranked(runs: list[Run], today: date, merged: Merged = lambda head: None) -> dict[str, Any]:
    """The rolling per-test table over the last ROLLING_DAYS of stages (finish, head, each first
    attempt's status and seconds), from healthy runs' passes only: the ten slowest by p50, the ten
    largest growers, each test whose last three observations were healthy passes of SLOW_SECONDS
    or more (a missing, failed, skipped, degraded or unmeasured run breaks the series), and every
    node's p50 in pytest-split's .test_durations form, so the same file balances workers."""
    window = sorted(
        (run for run in runs if 0 <= (today - run[0].date()).days < ROLLING_DAYS),
        key=lambda run: run[0],
    )
    states = _machine(window)
    samples: dict[str, list[tuple[datetime, float, str]]] = {}
    series: dict[str, list[bool]] = {}  # each observation, in order: a healthy slow pass or not
    degraded = []
    for (at, head, seen), (state, factor) in zip(window, states, strict=True):
        if state == "degraded":
            degraded.append({"at": at.isoformat(), "head": head, "factor": factor})
        for node in series.keys() - seen.keys():
            series[node].append(False)  # a stage without this node cannot confirm a running series
        for node, (status, value) in seen.items():
            healthy = state == "healthy" and status == "PASSED"
            series.setdefault(node, []).append(healthy and value >= SLOW_SECONDS)
            if healthy:
                samples.setdefault(node, []).append((at, value, head))
    table = {node: _row(node, rows, today, merged) for node, rows in sorted(samples.items())}
    growers = [row for row in table.values() if (row["trend"] or 0) > 0]
    return {
        "runs": sum(state == "healthy" for state, _ in states),
        "degraded": degraded,
        "unmeasured": sum(state == "unmeasured" for state, _ in states),
        "slowest": sorted(table.values(), key=lambda r: (-r["p50"], r["node"]))[:RANKED],
        "growers": sorted(growers, key=lambda r: (-r["trend"], r["node"]))[:RANKED],
        "slow_three_running": sorted(
            node for node, seen in series.items() if len(seen) >= 3 and all(seen[-3:])
        ),
        "durations": {node: row["p50"] for node, row in table.items()},
    }


def _first_attempts(
    events: list[Any], tested: list[dict[str, Any]]
) -> tuple[str, dict[str, Observation]]:
    """A stage's head and each node's first attempt: its status and seconds."""
    head = next(e.workspace for e in events if "workspace" in e.id).head
    return head, {r["node"]: (r["status"], r["seconds"]) for r in tested if r["attempt"] == 1}


def _latest_slow(path: str, tested: list[dict[str, Any]]) -> dict[str, Any]:
    """The latest passing stage's tests of SLOW_SECONDS or more: their count and time share."""
    total = sum(result["seconds"] for result in tested)
    slow = [result for result in tested if result["seconds"] >= SLOW_SECONDS]
    return {
        "slow_from": path or None,
        "slow_tests": len(slow) if total else None,
        "slow_share": round(sum(r["seconds"] for r in slow) / total, 3) if total else None,
    }


def stages(
    receipts: Sequence[Path], today: date | None = None, merged: Merged = lambda head: None
) -> dict[str, Any]:
    """Speed and reliability: results, red classes, flakes and the latest passing stage's slow
    tests, and with ``today`` the rolling per-test table. A stage with no verdict (an ERROR, or a
    receipt its reader refuses) is an error. A red stage counts toward the tooling-red share when
    its class is known and holds no defect; one with no class (written before receipts carried it)
    is unclassified, never a defect. With no receipt each rate is unknown, never zero."""
    results: Counter[str] = Counter()
    reds: list[tuple[str | None, dict[str, int]]] = []
    flakes: Counter[str] = Counter()
    nodes = 0
    unreadable: list[dict[str, str]] = []
    latest: tuple[datetime | None, str, list[dict[str, Any]]] = (None, "", [])
    runs: list[Run] = []
    for path, events, moment in _stage_inputs(receipts, unreadable):
        finished = events[-1].finished
        results[finished.result] += 1
        reds += [(finished.red_class, dict(finished.red_classes))] * (finished.result != "PASS")
        tested = [e.test_result.model_dump() for e in events if "test_result" in e.id]
        nodes += len({result["node"] for result in tested})
        runs.append((moment, *_first_attempts(events, tested)))
        flakes.update(node for node, kind in classed(tested)[1].items() if kind == "flake")
        if finished.result == "PASS" and tested and (latest[0] is None or moment > latest[0]):
            latest = (moment, str(path), tested)
    if unreadable:
        results["UNREADABLE"] = len(unreadable)
    count = sum(results.values())
    return {
        "stages": count,
        "by_result": dict(results),
        "error_rate": round((results["ERROR"] + len(unreadable)) / count, 3) if count else None,
        **_red_share(reds, count),
        "unreadable": unreadable,
        "flake_rate": round(sum(flakes.values()) / nodes, 4) if nodes else None,
        "flaked_nodes": sorted(flakes),
        **_latest_slow(latest[1], latest[2]),
    } | ({"tests": ranked(runs, today, merged)} if today else {})


def _red_share(reds: Sequence[tuple[str | None, dict[str, int]]], count: int) -> dict[str, Any]:
    """The red stages by class, the tooling-red share of all stages, and beside it the stages with
    an unknown part. A stage counts toward the share by its full causes: a flake, tooling or load
    cause and no defect; slow alone is not one. A stage with any unclassified cause, or written
    with no class at all, counts as unclassified even when another cause is known: an unknown is
    never hidden behind a known one, and never a defect."""
    classes = Counter(label or "unclassified" for label, _ in reds)
    tooling = sum(
        1 for _, causes in reds if causes.keys() & TOOLING_REDS and "defect" not in causes
    )
    return {
        "by_red_class": dict(classes),
        "tooling_red_share": round(tooling / count, 3) if count else None,
        "unclassified_reds": sum(1 for _, causes in reds if not causes or "unclassified" in causes),
    }


def _flake_error(detail: str) -> DefinitionError:
    return DefinitionError(
        "pipeline.flake_evidence_unavailable",
        detail + "; restore the named evidence before retrying",
    )


def _first_assertion(text: str, source: str, node: str) -> str:
    for line in text.splitlines():
        if re.search(r"\b(?:assert|AssertionError)\b", line):
            return line.strip()
    raise _flake_error(f"{source}: {node}: first assertion line is unavailable")


def _commit_infrastructure(
    path: Path, errors: list[Any], issues: list[dict[str, str]] | None, head: str
) -> set[str]:
    crash = {r.node for r in errors if CRASH.search((r.failure or {}).get("message", ""))}
    cascade = {r.node for r in errors if crash and CASCADE in (r.failure or {}).get("message", "")}
    if issues is not None:
        for result in errors:
            if result.node in crash or result.node in cascade:
                issues.append(
                    dict(
                        source="commit",
                        run=str(path),
                        node=result.node,
                        head=head,
                        kind="worker-crash" if result.node in crash else "workspace-cascade",
                        detail=(result.failure or {}).get("message", ""),
                    )
                )
    return crash | cascade


def _receipt_flake_node(
    path: Path,
    node: str,
    attempts: list[Any],
    head: str,
    at: datetime,
    issues: list[dict[str, str]] | None,
) -> dict[str, Any] | None:
    first, last = attempts[0], attempts[-1]
    if (
        first.attempt != 1
        or last.status != "PASSED"
        or not last.solo
        or first.status not in {"FAILED", "ERROR"}
    ):
        return None
    try:
        if first.status == "ERROR" and first.phase != "setup":
            raise _flake_error(f"{path}: {node}: non-setup error is unmeasured")
        if (
            last.timeout_seconds is None
            or not _valid_bound(last.timeout_seconds)
            or last.seconds > last.timeout_seconds
        ):
            raise _flake_error(f"{path}: {node}: solo bound is missing or exceeded")
        failure = _first_assertion((first.failure or {}).get("message", ""), str(path), node)
    except DefinitionError as error:
        if issues is None:
            raise
        issues.append(
            dict(
                source="commit",
                run=str(path),
                node=node,
                head=head,
                kind="unknown",
                detail=str(error),
            )
        )
        return None
    return dict(
        source="commit",
        head=head,
        node=node,
        at=at,
        run=str(path),
        first_failure=failure,
        failure_evidence=str(path),
        solo_evidence=str(path),
    )


def _flake_nodes(rows: list[Any]) -> dict[str, None]:
    """Candidate solo passes, including an unbounded one whose evidence needs a named refusal."""
    classes = classed([r.model_dump() for r in rows])[1]
    unbounded = {
        r.node
        for r in rows
        if r.solo and r.status == "PASSED" and not _valid_bound(r.timeout_seconds)
    }
    return dict.fromkeys(
        [*(node for node, kind in classes.items() if kind == "flake"), *sorted(unbounded)]
    )


def _receipt_flakes(
    path: Path, events: list[Any], at: datetime, issues: list[dict[str, str]] | None = None
) -> list[dict[str, Any]]:
    """Classify each product commit node without turning another node's unknown into a flake."""
    workspace = next(e.workspace for e in events if "workspace" in e.id)
    if events[0].started.stage != "commit" or workspace.repo != "product":
        return []
    rows = [e.test_result for e in events if "test_result" in e.id]
    errors = [r for r in rows if r.attempt == 1 and r.status == "ERROR"]
    if events[-1].finished.result == "ERROR" and not errors:
        if issues is None:
            raise _flake_error(f"{path}: stage error has no named node")
        issues.append(
            dict(
                source="commit",
                run=str(path),
                node="",
                kind="stage-unmeasured",
                detail="stage error has no named node",
            )
        )
    infrastructure = _commit_infrastructure(path, errors, issues, workspace.head)
    found = []
    for node in _flake_nodes(rows):
        if node in infrastructure:
            continue
        attempts = sorted((r for r in rows if r.node == node), key=lambda r: r.attempt)
        item = _receipt_flake_node(path, node, attempts, workspace.head, at, issues)
        if item is not None:
            found.append(item)
    return found


def _post_flakes(facts: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Accept transient workshop-verified facts; this is not publication authority."""
    rows = []
    required = {"source", "head", "node", "at", "run", "first_failure"}
    for fact in facts:
        if (
            not isinstance(fact, dict)
            or not required <= set(fact)
            or fact.get("source") != "post-merge"
        ):
            raise _flake_error("post-merge observation is malformed")
        extra = set(fact) - required
        if extra not in (set(), {"failure_evidence", "solo_evidence"}) or any(
            not isinstance(fact[key], str) or not fact[key].strip() for key in extra
        ):
            raise _flake_error("post-merge receipt references are malformed")
        if (
            not isinstance(fact["head"], str)
            or not re.fullmatch(r"[0-9a-f]{40}", fact["head"])
            or any(
                not isinstance(fact[key], str) or not fact[key].strip()
                for key in ("node", "at", "run", "first_failure")
            )
        ):
            raise _flake_error("post-merge observation has unknown identity or failure")
        try:
            at = datetime.fromisoformat(fact["at"])
            if at.utcoffset() is None:
                raise ValueError("timestamp has no offset")
        except ValueError as error:
            raise _flake_error(f"{fact['run']}: invalid observation time: {error}") from None
        line = _first_assertion(fact["first_failure"], fact["run"], fact["node"])
        rows.append({**fact, "at": at, "first_failure": line})
    return rows


def _flake_groups(rows: list[dict[str, Any]], today: date) -> dict[str, list[dict[str, Any]]]:
    """One head/node counts once across sources; retain the earliest dated observation."""
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for row in sorted(rows, key=lambda value: value["at"]):
        day = row["at"].date()
        if day > today:
            raise _flake_error(f"{row['run']}: observation is in the future")
        if (today - day).days > 6:
            continue
        unique.setdefault((row["head"], row["node"]), row)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in unique.values():
        groups.setdefault(row["node"], []).append(row)
    return groups


def _flake_test_path(node: str) -> str:
    path, separator, name = node.partition("::")
    parts = path.split("/")
    if (
        not separator
        or not name
        or not path.startswith("tests/")
        or not path.endswith(".py")
        or "\\" in path
        or any(char in path for char in "\n\r\x00")
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise _flake_error(f"{node}: foreign test identity")
    return path


def _flake_owner(
    root: Path,
    node: str,
    requirements: dict[str, dict[str, Any]],
    cache: dict[str, dict[str, tuple[str, ...]]],
    legacy: list[tuple[str, dict[str, int]]],
) -> tuple[str, str]:
    path = _flake_test_path(node)
    try:
        source = (root / path).resolve()
        if not source.is_relative_to(root.resolve()):
            raise ValueError("test resolves outside the checkout")
        if path not in cache:
            text = source.read_bytes().decode("utf-8")
            cache[path] = {holder.node: holder.ids for holder in file_holders(path, text)}
        ids = cache[path].get(node.partition("[")[0])
        if ids is None:
            raise ValueError("test holder is absent")
        if not ids:
            if not legacy:
                legacy.append(legacy_owner(root))
            owner, counts = legacy[0]
            actual = sum(not holder_ids for holder_ids in cache[path].values())
            if not 0 < actual <= counts.get(path, 0):
                raise ValueError("test is not admitted by the shrink-only legacy count")
            return owner, "unregistered legacy"
        if any(key not in requirements or requirements[key].get("retired") for key in ids):
            raise ValueError("test names an unknown or retired requirement")
        owners = {requirements[key]["owner"] for key in ids}
        if len(owners) != 1 or not next(iter(owners)).strip():
            raise ValueError("test has no unique registered requirement owner")
        return next(iter(owners)), "requirement"
    except (OSError, UnicodeError, SyntaxError, KeyError, TypeError, ValueError) as error:
        raise _flake_error(f"{node}: registered requirement owner unavailable: {error}") from None


def _already_held(node: str, selector: str) -> bool:
    return node_matches(node, selector) or (
        "[" not in selector.partition("::")[2] and fnmatchcase(node, selector)
    )


def _holder_at(path: str, text: str, node: str) -> Any:
    found = [holder for holder in file_holders(path, text) if node_matches(node, holder.node)]
    if len(found) != 1:
        raise ValueError("the exact holder is missing or ambiguous")
    return found[0]


class _HolderResolver:
    """Bind source test definitions to the exact current Git tree and unique file moves."""

    def __init__(self, root: Path) -> None:
        from devtools.architecture.git_inputs import (  # lazy: flake history only
            read_blobs,
            staged_changes,
            unique_moves,
        )

        self.root, self.read_blobs = root, read_blobs
        self.staged_changes, self.unique_moves = staged_changes, unique_moves
        self.blobs: dict[str, str | None] = {}
        self.moves: dict[str, dict[str, str]] = {}
        self.clean: set[str] = set()
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD^{commit}"],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
            timeout=60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.target = result.stdout.strip()
        if not re.fullmatch(r"[0-9a-f]{40}", self.target):
            raise ValueError("the publication-base commit is unavailable")

    def blob(self, revision: str, path: str) -> str | None:
        spec = f"{revision}:{path}"
        if spec not in self.blobs:
            self.blobs[spec] = self.read_blobs([spec], root=self.root, timeout=60).get(spec)
        return self.blobs[spec]

    def bind(self, source_head: str, source_node: str) -> tuple[str, str] | None:
        path = _flake_test_path(source_node)
        source = self.blob(source_head, path)
        if source is None:
            raise ValueError("the source holder blob is unavailable")
        original = _holder_at(path, source, source_node)
        target_path, target = path, self.blob(self.target, path)
        if target is None:
            if source_head not in self.moves:
                pairs = self.staged_changes(
                    self.root, tree=self.target, base=source_head, timeout=60
                )
                self.moves[source_head] = self.unique_moves(pairs)
            destinations = [new for new, old in self.moves[source_head].items() if old == path]
            if len(destinations) != 1:
                return None
            target_path = destinations[0]
            target = self.blob(self.target, target_path)
            if target is None:
                raise ValueError("the moved target holder blob is unavailable")
        mapped = target_path + "::" + source_node.partition("::")[2]
        try:
            current = _holder_at(target_path, target, mapped)
        except ValueError:
            return None
        if current.digest != original.digest:
            return None
        disk = self.root / target_path
        if disk.is_symlink() or not disk.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("the target holder resolves outside its checkout")
        if target_path not in self.clean:
            checked = subprocess.run(
                ["git", "diff", "--quiet", self.target, "--", target_path],
                cwd=self.root,
                capture_output=True,
                timeout=60,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if checked.returncode == 1:
                raise ValueError("the target holder differs from the publication-base commit")
            if checked.returncode:
                raise ValueError("the target holder's Git state is unavailable")
            self.clean.add(target_path)
        return mapped, original.digest


def _bind_holders(
    root: Path, rows: list[dict[str, Any]], issues: list[dict[str, str]] | None
) -> list[dict[str, Any]]:
    """Discard superseded definitions and keep the source identity for verified moves."""
    if not rows:
        return []
    try:
        resolver = _HolderResolver(root)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise _flake_error(f"{root}: publication-base Git identity unavailable: {error}") from None
    if issues is not None:
        for issue in issues:
            source_node, head = issue.get("node", ""), issue.get("head", "")
            if issue.get("kind") != "unknown" or not source_node or not head:
                continue
            try:
                identity = resolver.bind(head, source_node)
            except (OSError, UnicodeError, SyntaxError, ValueError, subprocess.SubprocessError):
                continue  # retain the named source unknown when its target cannot be proved
            if identity is None:
                issue["kind"] = "superseded"
                issue["detail"] += "; holder definition differs from the publication base"
            else:
                issue["source_node"] = source_node
                issue["node"], issue["holder_digest"] = identity
    bound = []
    for row in rows:
        source_node = row["node"]
        try:
            identity = resolver.bind(row["head"], source_node)
        except (
            OSError,
            UnicodeError,
            SyntaxError,
            ValueError,
            subprocess.SubprocessError,
        ) as error:
            if issues is None:
                raise _flake_error(
                    f"{source_node}: source holder identity unavailable: {error}"
                ) from None
            issues.append(
                dict(
                    source=row["source"],
                    run=row["run"],
                    node=source_node,
                    head=row["head"],
                    kind="unknown",
                    detail=f"source holder identity unavailable: {error}",
                )
            )
            continue
        if identity is None:
            if issues is not None:
                issues.append(
                    dict(
                        source=row["source"],
                        run=row["run"],
                        node=source_node,
                        kind="superseded",
                        detail="holder definition differs from the publication base",
                    )
                )
            continue
        node, digest = identity
        bound.append({**row, "node": node, "source_node": source_node, "holder_digest": digest})
    return bound


def _flake_inputs(
    root: Path,
    receipts: Sequence[Path],
    post_facts: Sequence[dict[str, Any]],
    today: date,
    include_held: bool,
    issues: list[dict[str, str]] | None,
) -> list[dict[str, Any]]:
    if (
        not isinstance(root, Path)
        or type(today) is not date
        or not isinstance(receipts, (list, tuple))
        or any(not isinstance(path, Path) for path in receipts)
        or not isinstance(post_facts, (list, tuple))
        or type(include_held) is not bool
    ):
        raise _flake_error(
            "candidate inputs need a checkout, receipt paths, post facts and a calendar date"
        )
    unreadable: list[dict[str, str]] = []
    rows = _post_flakes(post_facts)
    for path, events, at in _stage_inputs(receipts, unreadable):
        if at.date() > today:
            raise _flake_error(f"{path}: observation is in the future")
        if (today - at.date()).days <= 6:
            rows.extend(_receipt_flakes(path, events, at, issues))
    for unavailable in unreadable:
        if issues is not None and unavailable["code"] != "pipeline.receipt_missing":
            issues.append(
                dict(
                    source="commit",
                    run=unavailable["receipt"],
                    node="",
                    kind="incomplete-stage"
                    if unavailable["code"] == "pipeline.receipt_incomplete"
                    else "unusable-stage",
                    detail=unavailable["code"],
                )
            )
        else:
            raise _flake_error(f"{unavailable['receipt']}: {unavailable['code']}")
    return rows


def flake_candidates(
    root: Path,
    receipts: Sequence[Path],
    post_facts: Sequence[dict[str, Any]],
    today: date,
    *,
    include_held: bool = False,
    _issues: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Derive proposals only; publication re-verifies post facts through their workshop owner."""
    rows = _flake_inputs(root, receipts, post_facts, today, include_held, _issues)
    for row in rows:
        if row["at"].date() > today:
            raise _flake_error(f"{row['run']}: observation is in the future")
    rows = [row for row in rows if (today - row["at"].date()).days <= 6]
    rows = _bind_holders(root, rows, _issues)
    try:
        policy = load_policy(root)
    except (OSError, UnicodeError, TypeError, ValueError) as error:
        raise _flake_error(f"{root / 'config/registries/test-sizes.json'}: {error}") from None
    try:
        requirements = load_requirements(root)
    except (OSError, UnicodeError, TypeError, ValueError) as error:
        raise _flake_error(f"{root / 'config/registries/requirements.json'}: {error}") from None
    candidates = []
    holders: dict[str, dict[str, tuple[str, ...]]] = {}
    legacy: list[tuple[str, dict[str, int]]] = []
    unknown = {issue["node"] for issue in (_issues or []) if issue["kind"] == "unknown"}
    for node, observations in sorted(_flake_groups(rows, today).items()):
        if node in unknown:
            continue
        held = any(_already_held(node, q["node"]) for q in policy.get("quarantine", []))
        if len(observations) < 2 or (held and not include_held):
            continue
        try:
            owner, owner_kind = _flake_owner(root, node, requirements, holders, legacy)
        except DefinitionError as error:
            if _issues is None:
                raise
            _issues.append(
                dict(
                    source="commit",
                    run=observations[0]["run"],
                    node=node,
                    kind="unknown",
                    detail=str(error),
                )
            )
            continue
        first, second = observations[:2]
        runs = "; ".join(
            f"{row['source']} {row['head']} on {row['at'].date().isoformat()}"
            for row in (first, second)
        )
        candidates.append(
            dict(
                node=node,
                owner=owner,
                owner_kind=owner_kind,
                until=(today + timedelta(days=7)).isoformat(),
                reason=f"Automatic: flake twice in 7 days; {runs}",
                same_failure=first["first_failure"] == second["first_failure"],
                observations=[{**row, "at": row["at"].isoformat()} for row in (first, second)],
                **({"held": held} if include_held else {}),
            )
        )
    return candidates


def flake_assessment(
    root: Path,
    receipts: Sequence[Path],
    post_facts: Sequence[dict[str, Any]],
    today: date,
    *,
    post_issues: Sequence[dict[str, str]] = (),
    include_held: bool = False,
) -> dict[str, Any]:
    """Report named unknown/infrastructure nodes while retaining independent verified candidates."""
    if not isinstance(post_issues, (tuple, list)):
        raise _flake_error("post-merge issue list is malformed")
    issues: list[dict[str, str]] = []
    for issue in post_issues:
        if (
            not isinstance(issue, dict)
            or not {"source", "run", "node", "kind", "detail"} <= set(issue)
            or set(issue) - {"source", "run", "node", "kind", "detail", "head", "at"}
            or issue.get("source") != "post-merge"
            or issue.get("kind") not in {"unknown", "worker-crash", "workspace-cascade"}
            or (
                "head" in issue
                and (
                    not isinstance(issue["head"], str)
                    or not re.fullmatch(r"[0-9a-f]{40}", issue["head"])
                )
            )
            or any(
                not isinstance(issue.get(k), str) or not issue[k].strip()
                for k in ("run", "node", "detail")
            )
        ):
            raise _flake_error("post-merge issue identity is malformed")
        if "at" in issue:
            try:
                at = datetime.fromisoformat(issue["at"])
                if at.utcoffset() is None or at.date() > today:
                    raise ValueError("issue date is unmeasured or future")
            except (TypeError, ValueError):
                raise _flake_error("post-merge issue date is malformed") from None
            if (today - at.date()).days > 6:
                continue
        issues.append(dict(issue))
    candidates = flake_candidates(
        root, receipts, post_facts, today, include_held=include_held, _issues=issues
    )
    return {"candidates": candidates, "issues": issues}


def _merged(root: Path, head: str) -> bool | None:
    """Whether a stage's head is in this checkout's history; None when Git cannot tell."""
    found = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", head, "HEAD"], capture_output=True
    )
    return {0: True, 1: False}.get(found.returncode)


def report(
    root: Path,
    receipts: Sequence[Path],
    today: date,
    *,
    store: Path | None = None,
    release_base: str | None = None,
) -> dict[str, Any]:
    """The day's report: the tree's rows, the stages', then the code's."""
    try:
        return {
            "date": today.isoformat(),
            "tree": tree(root, today),
            "stages": stages(receipts, today, lambda head: _merged(root, head)),
            "code": code(root, store, release_base=release_base),
        }
    except DefinitionError:
        raise
    except (OSError, ValueError, SyntaxError, subprocess.CalledProcessError) as error:
        raise DefinitionError(
            "pipeline.workspace_unreadable",
            f"{root}: {error}; repair the checkout inputs and retry health",
        ) from None


GROWTH_BUDGET = 3_000
"""Net source lines a patch may add over the last release (0.2 plan, growth budget)."""
CODE_SECONDS = 300
"""One code-health tool's bound; one that overruns reads unmeasured."""
READ = ("config/package-architecture.json", "pyproject.toml", ".gitignore")
"""What the rows read beside src's modules: the registry, the project metadata, the ignore rules."""
ROWS = ("sizes", "boundaries", "growers", "dead_code", "dependencies", "complexity")
GRAPH = """import json, sys
import grimp
graph = grimp.build_graph(*sys.argv[1:], cache_dir=None)
print(json.dumps([[a, b, len(graph.get_import_details(importer=a, imported=b)) or 1]
                  for a in graph.modules for b in graph.find_modules_directly_imported_by(a)]))
"""
"""import-linter's graph (grimp), run on the tree's own src in its own interpreter."""
VULTURE = ("-m", "vulture", "src", "--min-confidence", "100")
FIRST = ("alphalattice", "devtools", "scripts", "tests")
DEPTRY = ("-m", "deptry", "src", *(f"--known-first-party={name}" for name in FIRST))
RUFF = (
    *("-m", "ruff", "check", "--isolated", "--select", "C901,PLR0912,PLR0915"),
    *("--output-format", "json", "--no-cache", "src"),
)


def code(
    root: Path,
    store: Path | None = None,
    *,
    day_start: str | None = None,
    release_base: str | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """The tree's code health beside its test health: sizes by package and area against the growth
    budget, module boundaries, the day's growers, dead code, dependencies and complexity.

    Information only: it refuses nothing, and only the whole tree is weighed against the budget. A
    row whose tool or input fails reads unmeasured, never zero, and is measured again next time. The
    day is the 24 hours before the report runs. Measured rows are kept in ``store`` by the tree, the
    bases and the bytes they read, and reused only whole and complete; a store or input that cannot
    be read or written is named, never raised."""
    try:
        tree = _run(root, "git", "rev-parse", "HEAD^{tree}")
        # the day starts at the first-parent commit 24 hours before now: HEAD itself on a quiet day
        if day_start is None:
            when = f"--before={int(time.time() if now is None else now) - 86_400} +0000"
            day_start = _run(root, "git", "rev-list", "-1", "--first-parent", when, "HEAD")
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return {"unmeasured": f"the checkout's Git tree cannot be read: {error}"}
    inputs: dict[str, Any] = {
        "tree": tree,
        "day_start": day_start or None,
        "release_base": release_base,
    }
    try:  # the bytes the rows read, committed or not, ignored or not, and the registry they cover
        inputs["read"], registered = _read_digest(root), _owner(root)[1]
        key = sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
        kept = store / f"{key}.json" if store is not None else None
        note = "" if kept is not None else "not kept: no store"
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        kept, registered, note = None, {}, f"not kept: its inputs cannot be read: {error}"
    prior = _kept_rows(kept, inputs, registered)
    measures = {
        "sizes": lambda: _sizes(root, inputs["day_start"], release_base),
        "boundaries": lambda: _boundaries(root),
        "growers": lambda: _growers(root, inputs["day_start"]),
        "dead_code": lambda: _dead_code(root),
        "dependencies": lambda: _dependencies(root),
        "complexity": lambda: _complexity(root),
    }
    seconds: dict[str, float] = {}
    rows = {
        name: prior[name] if name in prior else _measured(seconds, name, measures[name])
        for name in ROWS
    }
    found = {"inputs": inputs, **rows, "seconds": seconds}
    if kept is not None and seconds:
        try:
            kept.parent.mkdir(parents=True, exist_ok=True)
            kept.write_text(json.dumps(found, indent=1) + "\n", encoding="utf-8", newline="\n")
            note = "kept"
        except OSError as error:
            note = f"not kept: {error}"
    return {**found, "reused": sorted(prior), "store": note or "kept"}


def _read_digest(root: Path) -> str:
    """One digest of what the rows read: Git's index membership under src (sizes counts by it), and
    every file they read from disk: each module under src, tracked, untracked or ignored (the tools
    walk them all), the ignore rules they honour, registry and project. Each part is length-framed
    and an absent file is its own mark, so no content can stand for another state."""
    sources = [
        p
        for p in (root / "src").rglob("*")
        if p.suffix in {".py", ".pyi"} or p.name == ".gitignore"
    ]
    listed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--", "src"], capture_output=True, check=True
    ).stdout
    digest = sha256(b"%d\0" % len(listed) + listed)
    for path in sorted(sources) + [root / name for name in READ]:
        name = path.relative_to(root).as_posix().encode()
        data = path.read_bytes() if path.is_file() else None
        digest.update(b"%d\0" % len(name) + name)
        digest.update(b"-" if data is None else b"%d\0" % len(data) + data)
    return digest.hexdigest()


def _kept_rows(
    kept: Path | None, inputs: dict[str, Any], registered: dict[str, str]
) -> dict[str, Any]:
    """The rows a store holds for exactly these inputs, each whole to its own shape and covering
    every registered package and area; an unmeasured, partial or incomplete row is measured anew."""
    if kept is None:
        return {}
    try:
        row = json.loads(kept.read_text("utf-8"))
        if not isinstance(row, dict) or row.get("inputs") != inputs:
            return {}
    except (OSError, ValueError):
        return {}
    kept_rows = {}
    for name in ROWS:
        try:
            value = SHAPES[name].validate_python(row.get(name), strict=True)
        except ValidationError:
            continue
        if _covers(name, value, registered):
            kept_rows[name] = value
    return kept_rows


def _covers(name: str, row: Any, registered: dict[str, str]) -> bool:
    """A sizes or boundaries row names every registered package (and sizes every area): an empty map
    is complete only for an empty registry."""
    if name == "sizes":
        return set(registered) <= set(row["packages"]) and set(registered.values()) <= set(
            row["areas"]
        )
    if name == "boundaries":
        return set(row["packages"]) == set(registered)
    return True


class _Lines(TypedDict):
    lines: int
    day: int | None


class _Budget(TypedDict):
    net_since_release: int
    limit: int
    over: bool


class _Unmeasured(TypedDict):
    unmeasured: str


class _Sizes(TypedDict):
    source_lines: int
    day_net: int | None
    packages: dict[str, _Lines]
    areas: dict[str, _Lines]
    budget: _Budget | _Unmeasured


class _Edges(TypedDict):
    statements: int
    share: float
    fan_out: int
    fan_in: int


class _Boundaries(TypedDict):
    cross_package_statements: int
    package_edges: int
    packages: dict[str, _Edges]
    cycles: list[list[str]]


class _Grower(TypedDict):
    path: str
    net: int
    merges: list[str]


class _Named(TypedDict):
    certain: int
    named: list[str]


class _Dependencies(TypedDict):
    findings: int
    by_code: dict[str, int]
    named: list[str]


class _Complexity(TypedDict):
    findings: int
    by_rule: dict[str, int]
    packages: dict[str, int]


SHAPES: dict[str, TypeAdapter[Any]] = {
    "sizes": TypeAdapter(_Sizes),
    "boundaries": TypeAdapter(_Boundaries),
    "growers": TypeAdapter(list[_Grower]),
    "dead_code": TypeAdapter(_Named),
    "dependencies": TypeAdapter(_Dependencies),
    "complexity": TypeAdapter(_Complexity),
}
"""Each measured row's whole shape, the one a kept row must have to be reused."""


def _measured(seconds: dict[str, float], name: str, measure: Callable[[], Any]) -> Any:
    started = time.perf_counter()
    try:
        return measure()
    except Exception as error:  # a row that cannot be read is unmeasured, never zero
        return {"unmeasured": f"{type(error).__name__}: {str(error)[:300]}"}
    finally:
        seconds[name] = round(time.perf_counter() - started, 1)


def _run(
    root: Path, *argv: str, ok: tuple[int, ...] = (0,), env: dict[str, str] | None = None
) -> str:
    """One git or tool process over the checkout; an exit outside ``ok`` is named, never read."""
    if argv[0] != "git":
        argv = (sys.executable, *argv)
    else:
        argv = ("git", "-c", "core.quotepath=off", *argv[1:])
    done = subprocess.run(
        argv,
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=CODE_SECONDS,
        env={**os.environ, **(env or {})},
    )
    if done.returncode not in ok:
        raise ValueError(f"{argv[0:3]} exited {done.returncode}: {done.stderr.strip()[-300:]}")
    return done.stdout.strip("\n")


def _owner(root: Path) -> tuple[Callable[[str], str | None], dict[str, str]]:
    """A path's or module's registered package (the longest physical target it falls under), and
    every registered package's product area."""
    packages = json.loads((root / "config/package-architecture.json").read_text("utf-8"))[
        "packages"
    ]
    targets = sorted(
        ((p["physical_namespace_target"], p["package_id"]) for p in packages),
        key=lambda t: -len(t[0]),
    )

    def owner(name: str) -> str | None:
        module = name.removeprefix("src/").removesuffix(".py").replace("/", ".")
        module = module.removesuffix(".__init__")
        return next((pid for t, pid in targets if module == t or module.startswith(t + ".")), None)

    return owner, {p["package_id"]: p.get("product_area") or "unassigned" for p in packages}


def _numstat(root: Path, old: str, new: str, *paths: str) -> Iterator[tuple[str, int]]:
    """Each changed file's net lines, its path read exactly through Git's NUL-delimited form."""
    listed = _run(root, "git", "diff", "--numstat", "-z", "--no-renames", old, new, "--", *paths)
    for record in listed.split("\0"):
        if record.strip():
            added, removed, path = record.split("\t", 2)
            if added.isdigit() and removed.isdigit():  # a binary file's '-' counts no lines
                yield path, int(added) - int(removed)


def _sizes(root: Path, day_start: str | None, base: str | None) -> dict[str, Any]:
    (owner, areas), lines, day = _owner(root), Counter[str](), Counter[str]()
    for file in _tracked(root, "src"):
        if file.suffix == ".py":
            package = owner(file.relative_to(root).as_posix()) or "unowned"
            lines[package] += file.read_bytes().count(b"\n")
    for path, net in _numstat(root, day_start, "HEAD", "src") if day_start else ():
        if path.endswith(".py"):
            day[owner(path) or "unowned"] += net
    since = (
        sum(net for path, net in _numstat(root, base, "HEAD", "src") if path.endswith(".py"))
        if base
        else None
    )
    known = day_start is not None
    names = sorted(
        set(areas) | set(lines) | set(day), key=lambda pid: -lines[pid]
    )  # a removed package too
    by_area: dict[str, dict[str, int | None]] = {}
    for pid in names:
        row = by_area.setdefault(
            areas.get(pid, "unowned"), {"lines": 0, "day": 0 if known else None}
        )
        row["lines"] = (row["lines"] or 0) + lines[pid]
        row["day"] = (row["day"] or 0) + day[pid] if known else None
    return {
        "source_lines": sum(lines.values()),
        "day_net": sum(day.values()) if known else None,
        "packages": {
            pid: {"lines": lines[pid], "day": day[pid] if known else None} for pid in names
        },
        "areas": by_area,
        "budget": {
            "net_since_release": since,
            "limit": GROWTH_BUDGET,
            "over": since > GROWTH_BUDGET,
        }
        if since is not None
        else {"unmeasured": "no last release named (--release-base)"},
    }


def _boundaries(root: Path) -> dict[str, Any]:
    (owner, areas), statements, edges = _owner(root), Counter[str](), dict[str, set[str]]()
    graph = _run(root, "-c", GRAPH, *FIRST[:2], env={"PYTHONPATH": str(root / "src")})
    for importer, imported, count in json.loads(graph):
        if (a := owner(importer)) and (b := owner(imported)) and a != b:
            statements[a] += count
            edges.setdefault(a, set()).add(b)
    total, fan_in = sum(statements.values()), Counter(b for out in edges.values() for b in out)
    return {
        "cross_package_statements": total,
        "package_edges": sum(len(out) for out in edges.values()),
        "packages": {  # every registered package, a sink or an empty one too
            pid: {
                "statements": statements[pid],
                "share": round(statements[pid] / total, 3) if total else 0.0,
                "fan_out": len(edges.get(pid, ())),
                "fan_in": fan_in[pid],
            }
            for pid in sorted(areas, key=lambda pid: -statements[pid])
        },
        "cycles": _cycles(edges),
    }


def _cycles(edges: dict[str, set[str]]) -> list[list[str]]:
    """The package groups that import each other round: strongly connected, more than one."""

    def reach(start: str) -> set[str]:
        seen: set[str] = set()
        todo = [start]
        while todo:
            fresh = edges.get(todo.pop(), set()) - seen
            seen |= fresh
            todo += fresh
        return seen

    reached = {a: reach(a) for a in edges}
    groups = {frozenset(b for b in reached[a] if a in reached.get(b, ())) | {a} for a in edges}
    return sorted(sorted(group) for group in groups if len(group) > 1)


def _growers(root: Path, day_start: str | None) -> list[dict[str, Any]] | dict[str, str]:
    """The day's ten largest growers over every first-parent commit, each with what brought it: a
    card's merge by its name, or a direct commit by its subject."""
    if not day_start:
        return {"unmeasured": "no first-parent commit 24 hours before now"}
    net, behind = Counter[str](), dict[str, set[str]]()
    log = _run(root, "git", "log", "--first-parent", "--format=%H %P%x09%s", f"{day_start}..HEAD")
    for line in log.splitlines():
        shas, subject = line.split("\t", 1)
        sha, *parents = shas.split()
        by = (
            subject.removeprefix("Merge ").split(":")[0]
            if len(parents) > 1
            else f"direct: {subject[:60]}"
        )
        for path, lines in _numstat(root, f"{sha}^1", sha, "src", "tests", "scripts"):
            net[path] += lines
            behind.setdefault(path, set()).add(by)
    return [{"path": p, "net": n, "merges": sorted(behind[p])} for p, n in net.most_common(10)]


def _dead_code(root: Path) -> dict[str, Any]:
    output = _run(root, *VULTURE, ok=(0, 3)).splitlines()
    named = [line.replace("\\", "/") for line in output if line.strip()]
    return {"certain": len(named), "named": named}  # each a candidate for the weekly deletion card


def _dependencies(root: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch) / "deptry.json"
        _run(root, *DEPTRY, "--json-output", str(out), ok=(0, 1))
        rows = json.loads(out.read_text("utf-8"))
    return {
        "findings": len(rows),
        "by_code": dict(sorted(Counter(row["error"]["code"] for row in rows).items())),
        "named": sorted({f"{row['error']['code']} {row['module']}" for row in rows}),
    }


def _complexity(root: Path) -> dict[str, Any]:
    (owner, _), rows = _owner(root), json.loads(_run(root, *RUFF, ok=(0, 1)) or "[]")
    where = [Path(r["filename"]).resolve().relative_to(root.resolve()).as_posix() for r in rows]
    return {
        "findings": len(rows),
        "by_rule": dict(sorted(Counter(r["code"] for r in rows).items())),
        "packages": dict(Counter(owner(path) or "unowned" for path in where).most_common()),
    }
