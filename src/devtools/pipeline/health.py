"""The daily test-health report the test manager publishes: the test rows of the quality metrics,
from the checkout, its test-size registry and the commit-stage receipts it is given, each read
once through the receipt's own reader, and the rolling per-test seconds those receipts hold."""

from __future__ import annotations

import math
import re
import statistics
import subprocess
from bisect import insort
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from datetime import date, datetime, timedelta
from fnmatch import fnmatchcase
from hashlib import sha256
from pathlib import Path
from typing import Any

from devtools.pipeline.definition import DefinitionError
from devtools.pipeline.receipt import ReceiptError, classed, read
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
    """Speed and reliability: results, flakes and the latest passing stage's slow tests, and
    with ``today`` the rolling per-test table. A stage with no verdict (an ERROR, or a receipt
    its reader refuses) is an error; with no receipt each rate is unknown, never zero."""
    results: Counter[str] = Counter()
    flakes: Counter[str] = Counter()
    nodes = 0
    unreadable: list[dict[str, str]] = []
    latest: tuple[datetime | None, str, list[dict[str, Any]]] = (None, "", [])
    runs: list[Run] = []
    for path, events, moment in _stage_inputs(receipts, unreadable):
        finished = events[-1].finished
        results[finished.result] += 1
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
        "unreadable": unreadable,
        "flake_rate": round(sum(flakes.values()) / nodes, 4) if nodes else None,
        "flaked_nodes": sorted(flakes),
        **_latest_slow(latest[1], latest[2]),
    } | ({"tests": ranked(runs, today, merged)} if today else {})


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
    crash = {
        r.node
        for r in errors
        if re.search(
            r"worker 'gw[0-9]+' crashed while running", (r.failure or {}).get("message", "")
        )
    }
    cascade = {
        r.node
        for r in errors
        if crash and "incomplete session workspace:" in (r.failure or {}).get("message", "")
    }
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
        if not last.timeout_seconds or last.seconds > last.timeout_seconds:
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
    classes = classed([r.model_dump() for r in rows])[1]
    found = []
    for node, kind in classes.items():
        if kind != "flake" or node in infrastructure:
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


def report(root: Path, receipts: Sequence[Path], today: date) -> dict[str, Any]:
    """The day's report: the tree's rows, then the stages'."""
    try:
        return {
            "date": today.isoformat(),
            "tree": tree(root, today),
            "stages": stages(receipts, today, lambda head: _merged(root, head)),
        }
    except DefinitionError:
        raise
    except (OSError, ValueError, SyntaxError, subprocess.CalledProcessError) as error:
        raise DefinitionError(
            "pipeline.workspace_unreadable",
            f"{root}: {error}; repair the checkout inputs and retry health",
        ) from None
