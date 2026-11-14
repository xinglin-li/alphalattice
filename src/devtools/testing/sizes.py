"""Registered test sizes consumed by pytest collection and the canonical gate."""

import ast
import json
import logging
import os
import re
import stat
import sys
import threading
import time
from collections.abc import Callable, Iterable
from datetime import date
from fnmatch import fnmatchcase
from math import isfinite
from pathlib import Path, PurePosixPath
from typing import Any


def _declarations(tree: ast.Module, name: str) -> list[Any]:
    direct = [
        node.value
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
    ]
    writes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id == name
        and isinstance(node.ctx, (ast.Store, ast.Del))
    ]
    return direct if len(writes) == len(direct) else [*direct, None]


def whole_tree_scope(tree: ast.Module) -> bool:
    """The shared layout declaration is one literal whole-tree scope."""
    declared = _declarations(tree, "TEST_SCOPE")
    return (
        len(declared) == 1
        and isinstance(declared[0], ast.Constant)
        and declared[0].value == "whole-tree"
    )


def load_policy(root: Path) -> dict[str, Any]:
    """Read one collection's policy once, without a process-global cache."""
    return validate_policy(
        json.loads((root / "config/registries/test-sizes.json").read_text("utf-8"))
    )


def validate_policy(value: Any) -> dict[str, Any]:
    """Admit the same catalog from collection files or staged registry blobs."""
    if not isinstance(value, dict) or value.get("schema") != "alphalattice.test-sizes.v1":
        raise ValueError("test_size.schema")
    if value.get("sizes") != ["small", "medium", "large"]:
        raise ValueError("test_size.sizes")
    if not isinstance(value.get("defaults"), list) or not value["defaults"]:
        raise ValueError("test_size.defaults")
    if not isinstance(value.get("large"), list):
        raise ValueError("test_size.large")
    if any(
        not isinstance(r, dict)
        or not isinstance(r.get("pattern"), str)
        or not isinstance(r.get("owner"), str)
        or not r["pattern"]
        or not r["owner"]
        for r in value["defaults"] + value["large"]
    ):
        raise ValueError("test_size.rule")
    if any(r.get("size") not in {"small", "medium"} for r in value["defaults"]):
        raise ValueError("test_size.default_size")
    if "mandatory" in value and (
        not isinstance(value["mandatory"], list) or "tests/structural/*" not in value["mandatory"]
    ):
        raise ValueError("test_size.mandatory_missing")
    _post_merge_only(value)
    _reach(value)
    _whole_tree_guards(value)
    _structural_fence(value)
    _limits(value)
    _quarantine(value)
    _framework_fixtures(value)
    _unit_backends(value)
    _data_files(value)
    return value


def _unit_backends(policy: dict[str, Any]) -> None:
    """A backend is a named resource dependency, never an inferred native extension."""
    rows = policy.get("unit_backends", [])
    seen: set[str] = set()
    if not isinstance(rows, list) or ("unit_backends" in policy and not rows):
        raise ValueError("testing.layer_contract_unmeasured: restore the unit backend catalog")
    for row in rows:
        name = row.get("name") if isinstance(row, dict) else None
        if not (
            isinstance(name, str)
            and re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", name)
            and set(row) == {"kind", "name", "reason"}
            and isinstance(row.get("kind"), str)
            and row.get("kind") in {"module", "callable"}
            and isinstance(row.get("reason"), str)
            and row["reason"].strip()
            and name not in seen
        ):
            raise ValueError(
                "testing.layer_contract_unmeasured: name each backend once with its kind "
                "and reason; restore the unit backend catalog"
            )
        seen.add(name)


def _framework_fixtures(policy: dict[str, Any]) -> None:
    """Only exact named root framework fixtures with reasons may skip module inventories."""
    rows = policy.get("framework_fixtures", [])
    seen: set[str] = set()
    if not isinstance(rows, list):
        raise ValueError("test_size.framework_fixtures_invalid: use named entries with reasons")
    for row in rows:
        node = row.get("node") if isinstance(row, dict) else None
        reason = row.get("reason") if isinstance(row, dict) else None
        if not (
            isinstance(node, str)
            and re.fullmatch(r"tests/conftest\.py::[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", node)
            and isinstance(reason, str)
            and reason.strip()
            and node not in seen
        ):
            raise ValueError(
                "test_size.framework_fixtures_invalid: name each exact root fixture once "
                "with its own reason; modules and patterns are not entries"
            )
        seen.add(node)


def _data_files(policy: dict[str, Any]) -> None:
    """A data file under tests/ is registered once, by its exact path, with its own reason."""
    rows = policy.get("data_files", [])
    seen: set[str] = set()
    if not isinstance(rows, list):
        raise ValueError("test_size.data_files_invalid: use path and reason entries")
    for row in rows:
        path = row.get("path") if isinstance(row, dict) else None
        reason = row.get("reason") if isinstance(row, dict) else None
        if not (
            isinstance(path, str)
            and path.startswith("tests/")
            and not set(path) & set("*?[\\")
            and isinstance(reason, str)
            and reason.strip()
            and path not in seen
        ):
            raise ValueError(
                "test_size.data_files_invalid: name each exact tests/ path once "
                "with its own reason; patterns are not entries"
            )
        seen.add(path)


def _structural_file(pattern: str, policy: dict[str, Any]) -> bool:
    """Carry structural standing only through an exact registered file declaration."""
    if pattern.startswith("tests/structural/"):
        return True
    path = pattern.partition("::")[0].removesuffix("*")
    parsed = PurePosixPath(path)
    if (
        not path.startswith("tests/")
        or not path.endswith(".py")
        or str(parsed) != path
        or ".." in parsed.parts
        or any(char in path for char in "*?[\\:\0\r\n")
        or not all(part.isidentifier() for part in parsed.with_suffix("").parts)
    ):
        return False
    try:
        row = _rule(path, policy["defaults"])
    except ValueError:
        return False
    return bool(row and row["pattern"] == path + "*" and row["owner"] == "tests.structural")


def _foreign_structural_override(pattern: str, policy: dict[str, Any]) -> bool:
    """A more specific foreign rule may not take ownership inside a mapped reach."""
    path = pattern.partition("::")[0].removesuffix("*")
    reach_prefix = re.split(r"[*?\[]", pattern, maxsplit=1)[0]
    for row in policy["defaults"]:
        candidate = row["pattern"]
        prefix = re.split(r"[*?\[]", candidate, maxsplit=1)[0]
        if (
            row["owner"] != "tests.structural"
            and len(candidate) >= len(path + "*")
            and (
                fnmatchcase(pattern, candidate)
                if not any(char in pattern for char in "*?[")
                else reach_prefix.startswith(prefix) or prefix.startswith(reach_prefix)
            )
        ):
            return True
    return False


def _post_merge_only(value: dict[str, Any]) -> None:
    """Keep fenced obligations mandatory and reject malformed stage deferrals."""
    post = value.get("post_merge_only", [])
    fenced = value.get("structural_fence", [])
    fenced = fenced if isinstance(fenced, list) else []
    if (
        not isinstance(post, list)
        or any(
            not isinstance(p, str)
            or not p.startswith("tests/")
            or (_structural_file(p, value) and not p.startswith("tests/structural/"))
            or (
                p.startswith("tests/structural/")
                and any(
                    fnmatchcase(p, r.get("pattern", "").partition("::")[0])
                    for r in fenced
                    if isinstance(r, dict) and isinstance(r.get("pattern"), str)
                )
            )
            or not p.endswith(".py")
            or ".." in Path(p).parts
            or "\\" in p
            or any(c in p for c in "*?[")
            for p in post
        )
        or len(set(post)) != len(post)
    ):
        raise ValueError("test_size.rule: post_merge_only needs distinct non-fenced test files")


def validate_post_merge_scopes(policy: dict[str, Any], read: Callable[[str], str]) -> None:
    """A deferred structural file must carry the layout owner's whole-tree declaration."""
    paths = [p for p in policy.get("post_merge_only", []) if p.startswith("tests/structural/")]
    if not paths:
        return
    for path in paths:
        try:
            tree = ast.parse(read(path))
        except (OSError, UnicodeError, KeyError, TypeError, SyntaxError) as error:
            raise ValueError(
                f"test_size.rule: {path}: read its whole-tree declaration: {error}"
            ) from error
        if not whole_tree_scope(tree):
            raise ValueError(f"test_size.rule: {path}: declare TEST_SCOPE = 'whole-tree'")


def _reach(policy: dict[str, Any]) -> None:
    """Require structural standing, a reason and a declared source-reading scope."""
    for row in policy.get("reach", []):  # a structural node's reach: per_file or declared inputs
        pattern = row.get("pattern", "")
        structural = isinstance(pattern, str) and (
            pattern.startswith("tests/structural/")
            or (
                row.get("owner") == "tests.structural"
                and _structural_file(pattern, policy)
                and not _foreign_structural_override(pattern, policy)
            )
        )
        if not (structural and row.get("reason")) or (
            row.get("per_file") is not True and not row.get("inputs")
        ):
            raise ValueError("test_size.reach_invalid:" + str(row.get("pattern")))


FENCE_LAWS = frozenset({"public-private", "identity", "evidence", "secrets"})
"""The laws whose structural guards stay in the commit stage whole until they declare a reach."""


def _structural_fence(policy: dict[str, Any]) -> None:
    """A fenced structural guard names its law and why, once."""
    rows, seen = policy.get("structural_fence", []), set()
    if not isinstance(rows, list):
        raise ValueError("test_size.fence_invalid")
    for row in rows:
        pattern = row.get("pattern") if isinstance(row, dict) else None
        if not (
            isinstance(pattern, str)
            and pattern.startswith("tests/structural/")
            and row.get("law") in FENCE_LAWS
            and isinstance(row.get("reason"), str)
            and row["reason"]
            and pattern not in seen
        ):
            raise ValueError("test_size.fence_invalid:" + str(pattern))
        seen.add(pattern)


def guards_the_change(nodeid: str, policy: dict[str, Any], changed: frozenset[str]) -> bool:
    """A fast whole-tree guard runs at commit when the tree it reads changed; a slow one waits."""
    rule = policy.get("whole_tree_guards", {})
    return not any(fnmatchcase(nodeid, p) for p in rule.get("slow", ())) and any(
        fnmatchcase(path, read) for path in changed for read in rule.get("reads", ())
    )


def _whole_tree_guards(policy: dict[str, Any]) -> None:
    """The rule names why, what its guards read, and which are too slow for the commit stage."""
    rule = policy.get("whole_tree_guards", {})
    if rule and not (
        isinstance(rule, dict)
        and set(rule) == {"reason", "reads", "slow"}
        and isinstance(rule["reason"], str)
        and rule["reason"].strip()
        and all(
            isinstance(rule[key], list) and all(isinstance(value, str) for value in rule[key])
            for key in ("reads", "slow")
        )
        and rule["reads"]
    ):
        raise ValueError("test_size.whole_tree_guards_invalid")


def moved(nodeid: str, policy: dict[str, Any]) -> bool:
    """A whole-tree structural guard with no declared reach and no fence law: the commit stage
    leaves it to post-merge, which runs every structural node whole (the reform's rule, 0.2.md)."""
    nodeid = nodeid.replace("\\", "/")
    return (
        nodeid.startswith("tests/structural/")
        and reach(nodeid, policy) is None
        and not any(fnmatchcase(nodeid, r["pattern"]) for r in policy.get("structural_fence", []))
    )


# A standing red's class, by evidence: red everywhere (real), red then green (unstable), red on one
# platform from our own code (platform) or from a node's missing tool (environment). Only real and
# unstable leave the merge lane; a by-design platform red is marked in its test instead.
QUARANTINE_CLASSES = ("real", "unstable", "platform", "environment")


def _iso_day(value: Any) -> bool:
    """A real calendar date in ISO form, so 2026-99-99 is refused."""
    try:
        return bool(date.fromisoformat(str(value)).isoformat() == value)
    except ValueError:
        return False


def _classed(row: dict[str, Any]) -> bool:
    """A row's optional class, first red day (no later than its until) and distinct hosts."""
    hosts = row.get("seen_on")
    return bool(
        ("class" not in row or row["class"] in QUARANTINE_CLASSES)
        and ("since" not in row or (_iso_day(row["since"]) and row["since"] <= row["until"]))
        and (
            "seen_on" not in row
            or (
                type(hosts) is list
                and hosts
                and all(type(host) is str and host.strip() for host in hosts)
                and len(set(hosts)) == len(hosts)
            )
        )
    )


def _quarantine(policy: dict[str, Any]) -> None:
    """Refuse a quarantined node without an owner, a reason or an until date; a classed row also
    names its class and, when given, the day it was first seen red, no later than its until, and
    the distinct hosts it was seen red on, so a red anywhere else is not taken for it."""
    for row in policy.get("quarantine", []):  # a flaky node runs post-merge only, until its date
        reason = row.get("reason")
        if isinstance(reason, str) and re.search(
            r"(?<![A-Za-z0-9])(?:workshops?|harness(?:es)?|(?:mac|pc)[-_]?\d+)(?![A-Za-z0-9])",
            reason,
            re.IGNORECASE,
        ):
            raise ValueError(
                f"test_size.quarantine_invalid:{row.get('node')}: "
                "describe the platform without internal machine or development terms"
            )
        dated = _iso_day(row.get("until"))
        if not (
            str(row.get("node")).startswith("tests/")
            and row.get("owner")
            and row.get("reason")
            and dated
            and _classed(row)
            and (
                "renewals" not in row
                or (
                    type(row["renewals"]) is int
                    and row["renewals"] in (0, 1)
                    and isinstance(row["reason"], str)
                    and row["reason"].strip()
                    and "\n" not in row["reason"]
                    and "\r" not in row["reason"]
                )
            )
        ):
            raise ValueError("test_size.quarantine_invalid:" + str(row.get("node")))


# A Large rule's lane contract: nightly_only runs only in the nightly; release_only runs in the
# release lane at promotion (it reads the release archives), and the nightly lists it skipped.
SIZE_CONTRACTS = ("nightly_only", "release_only")


def _admission_limits(policy: dict[str, Any]) -> None:
    """Validate the optional changed-test admission budget and Large contract."""
    admission = policy.get(
        "admission_limits"
    )  # contract: require after pre-contract policy readers retire
    if admission is not None:
        if not isinstance(admission, dict) or set(admission) != {"small", "medium"}:
            raise ValueError("test_size.admission_limits")
        for row in admission.values():
            if (
                not isinstance(row, dict)
                or set(row) != {"seconds", "tmp_bytes"}
                or type(row["seconds"]) not in (int, float)
                or not isfinite(row["seconds"])
                or row["seconds"] <= 0
                or type(row["tmp_bytes"]) is not int
                or row["tmp_bytes"] < 0
            ):
                raise ValueError("test_size.admission_limits")
    if any(
        "size_contract" in row and row["size_contract"] not in SIZE_CONTRACTS
        for row in policy["large"]
    ):
        raise ValueError("test_size.large_contract")


def _growth(policy: dict[str, Any]) -> None:
    """Validate the net test lines a change may add beyond its net source."""
    growth = policy.get("growth")  # contract: require after pre-contract policy readers retire
    if growth is not None and (
        not isinstance(growth, dict)
        or set(growth) != {"floor_lines"}
        or type(growth["floor_lines"]) is not int
        or growth["floor_lines"] < 0
    ):
        raise ValueError("test_size.growth")


def _limits(policy: dict[str, Any]) -> None:
    """Refuse disabled bounds, duplicate Large declarations and unexplained resource needs."""
    limits = policy.get("limits")
    if not isinstance(limits, dict) or set(limits) != set(policy["sizes"]):
        raise ValueError("test_size.limits")
    bounds = [
        *limits.values(),
        *(r.get("timeout_seconds", limits["large"]) for r in policy["large"]),
    ]
    if any(type(n) not in (int, float) or not isfinite(n) or n <= 0 for n in bounds):
        raise ValueError("test_size.limits")
    _admission_limits(policy)
    _growth(policy)
    if len({r["pattern"] for r in policy["large"]}) != len(policy["large"]):
        raise ValueError("test_size.large_duplicate")
    for row in policy["large"]:
        reason = row.get("reason")
        if (
            not isinstance(reason, str)
            or "\n" in reason
            or "\r" in reason
            or reason.strip().casefold()
            in {"", "slow", "integration", "slow test", "integration test"}
        ):
            raise ValueError("test_size.large_reason")


def tmp_footprint(root: Path) -> int:
    """Apparent regular-file bytes retained under tmp_path; never follow links or reparse points."""
    try:
        try:
            info = root.lstat()
        except FileNotFoundError:
            return 0  # the owned root was removed before its teardown observation
        reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        if not stat.S_ISDIR(info.st_mode) or getattr(info, "st_file_attributes", 0) & reparse:
            raise ValueError("temporary root is not a physical directory")
        total, pending = 0, [root]
        while pending:
            with os.scandir(pending.pop()) as entries:
                for entry in entries:
                    info = entry.stat(follow_symlinks=False)
                    if (
                        stat.S_ISLNK(info.st_mode)
                        or getattr(info, "st_file_attributes", 0) & reparse
                    ):
                        continue  # an external or shared target is not part of tmp_path
                    if stat.S_ISDIR(info.st_mode):
                        pending.append(Path(entry.path))
                    elif stat.S_ISREG(info.st_mode):
                        total += info.st_size
                    else:
                        raise ValueError(f"unsupported temporary entry {entry.path}")
        return total
    except (OSError, ValueError) as error:
        raise ValueError(
            f"test_size.footprint_unmeasured: {root}: {error}; restore the temporary tree"
        ) from error


def tmp_entry_footprint(path: Path) -> int:
    """Measure one new base entry without following a link into shared data."""
    try:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & getattr(
            stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0
        ):
            return 0
        if stat.S_ISREG(info.st_mode):
            return info.st_size
        if stat.S_ISDIR(info.st_mode):
            return tmp_footprint(path)
        raise ValueError("unsupported temporary entry")
    except FileNotFoundError:
        return 0
    except (OSError, ValueError) as error:
        raise ValueError(
            f"test_size.footprint_unmeasured: {path}: {error}; restore the temporary tree"
        ) from error


def admission_limit(
    node: str, size: str, seconds: float, footprint: int, policy: dict[str, Any]
) -> None:
    """Judge one changed test's complete phases and retained tmp_path footprint."""
    if not isinstance(node, str) or not node or size not in {"small", "medium", "large"}:
        raise ValueError("test_size.admission_invalid: name a test and its registered size")
    if (
        type(seconds) not in (int, float)
        or not isfinite(seconds)
        or seconds < 0
        or type(footprint) is not int
        or footprint < 0
    ):
        raise ValueError(
            f"test_size.admission_unmeasured: {node}: restore complete seconds "
            "and tmp footprint at teardown"
        )
    if size == "large":
        return
    limits = policy.get("admission_limits")
    if not isinstance(limits, dict) or size not in limits:
        raise ValueError(
            f"test_size.admission_unmeasured: {node}: restore registered admission limits"
        )
    bound = limits[size]
    if seconds > bound["seconds"] or footprint > bound["tmp_bytes"]:
        upgrade = "medium" if size == "small" else "large"
        raise ValueError(
            f"test_size.admission_limit: {node}: {size} measured {seconds} s and "
            f"{footprint} bytes tmp footprint at teardown; limits {bound['seconds']} s and "
            f"{bound['tmp_bytes']} bytes; reduce the test or declare {upgrade}"
        )


def _rule(nodeid: str, rules: list[dict[str, Any]]) -> dict[str, Any] | None:
    matches = [r for r in rules if fnmatchcase(nodeid, r["pattern"])]
    if not matches:
        return None
    width = max(len(r["pattern"]) for r in matches)
    closest = [r for r in matches if len(r["pattern"]) == width]
    if len({(r["owner"], r.get("size", "large")) for r in closest}) > 1:
        raise ValueError("test_size.owner_conflict:" + nodeid)
    return min(closest, key=lambda r: r["pattern"])


def classify(
    nodeid: str, declared: Iterable[str], policy: dict[str, Any]
) -> tuple[str, str, str, float]:
    """Resolve size, owner, declaration and execution bound in one rule scan."""
    sizes = set(declared)
    if len(sizes) > 1:
        raise ValueError("test_size.conflicting:" + nodeid)
    nodeid = nodeid.replace("\\", "/")
    default = _rule(nodeid, policy["defaults"])
    if default is None:
        raise ValueError("test_size.owner_missing:" + nodeid)
    large = _rule(nodeid, policy["large"])
    if large and large["owner"] != default["owner"]:
        raise ValueError("test_size.owner_conflict:" + nodeid)
    if large and sizes and sizes != {"large"}:
        raise ValueError("test_size.conflicting:" + nodeid)
    rule = large or default
    size = next(iter(sizes)) if sizes else "large" if large else rule["size"]
    if size not in policy["sizes"] or not rule.get("owner"):
        raise ValueError("test_size.declaration_invalid:" + nodeid)
    origin = "marker:" + nodeid if sizes else rule["pattern"]
    seconds = (
        large.get("timeout_seconds", policy["limits"][size]) if large else policy["limits"][size]
    )
    return str(size), str(rule["owner"]), str(origin), float(seconds)


def changed_paths(*, floor: bool = False) -> frozenset[str] | None:
    """A scoped run's changed paths from its `git diff --name-status --no-renames` listing,
    an added or deleted one also as `+path`; None outside a scoped run."""
    named = os.environ.get("ALPHALATTICE_CHANGED_PATHS")
    rows = [r.split("\t") for r in Path(named).read_text("utf-8").splitlines()] if named else []
    if floor and (
        not rows
        or any(
            len(row) != 2
            or row[0] not in {"A", "D", "M", "T"}
            or not row[1]
            or any(c in '"\\\ufffd' or ord(c) < 32 for c in row[1])
            for row in rows
        )
    ):
        return None  # unmeasured floor: keep the whole run
    found = {f"+{r[-1]}" for r in rows if r[0] in {"A", "D"}} | {r[-1] for r in rows if r[0]}
    return frozenset(found) if named else None


def reach(nodeid: str, policy: dict[str, Any]) -> dict[str, Any] | None:
    """The declared reach of a structural node, which a scoped merge or fast run honours."""
    return _rule(nodeid.replace("\\", "/"), policy.get("reach", []))


def _triggered(nodeid: str, row: dict[str, Any], changed: frozenset[str]) -> bool:
    own = nodeid.replace("\\", "/").split("::")[0]  # its own file or a declared input changed
    return own in changed or any(fnmatchcase(p, g) for p in changed for g in row.get("inputs", ()))


def scoped(
    nodeid: str, policy: dict[str, Any], changed: frozenset[str] | None, *, floor: bool = False
) -> bool:
    """A per_file node scans changed files unless an input or the session floor changed."""
    row = reach(nodeid, policy) if changed is not None and not floor else None
    return bool(
        row and row.get("per_file") is True and not _triggered(nodeid, row, changed or frozenset())
    )


def _contracted(contract: Any, lane: str) -> bool | None:
    """A Large rule's lane contract decides alone, and the release lane runs only release_only
    rules (the nightly lists those skipped); None leaves the lane rules to decide."""
    if lane == "release" or contract == "release_only":
        return contract == "release_only" and lane in ("nightly", "release")
    if contract == "nightly_only":
        return lane == "nightly"
    return None


def included(
    nodeid: str,
    size: str,
    lane: str,
    policy: dict[str, Any],
    changed: frozenset[str] | None = None,
    *,
    floor: bool = False,
) -> bool:
    """Mandatory holders always run, except that a scoped merge run keeps a reach-declared node
    only when it is per_file or triggered, and a moved structural guard (``moved``) only when its
    own file changed; the fast lane keeps only scoped per_file nodes; each lane otherwise admits
    its declared sizes."""
    large = _rule(nodeid.replace("\\", "/"), policy.get("large", [])) if size == "large" else None
    decided = _contracted(large.get("size_contract") if large else None, lane)
    if decided is not None:
        return decided

    row = reach(nodeid, policy) if changed is not None else None
    if lane == "fast":
        return scoped(nodeid, policy, changed)
    if lane == "merge" and row:
        if floor:
            return any(fnmatchcase(nodeid, p) for p in policy["mandatory"]) or size != "large"
        return row.get("per_file") is True or _triggered(nodeid, row, changed or frozenset())
    if lane == "merge" and changed is not None and moved(nodeid, policy):
        own = nodeid.replace("\\", "/").split("::")[0]  # else post-merge runs it whole
        return own in changed or guards_the_change(nodeid, policy, changed)
    return (
        not lane
        or any(fnmatchcase(nodeid, p) for p in policy["mandatory"])
        or (size != "large" if lane == "merge" else size == "large")
    )


def stage_files(
    policy: dict[str, Any],
    stage: str,
    paths: Iterable[str],
    *,
    lane: str = "",
    read: Callable[[str], str] | None = None,
) -> list[str]:
    """Defer verified files only in commit; every other stage keeps them."""
    post = (
        frozenset(policy.get("post_merge_only", []))
        if stage == "commit" and lane not in {"candidate", "nightly"}
        else frozenset()
    )
    paths = list(paths)
    names = {path.replace("\\", "/").split("::", 1)[0] for path in paths}
    structural = [path for path in post & names if path.startswith("tests/structural/")]
    if structural:

        def read_own(path: str) -> str:
            return (Path(__file__).resolve().parents[3] / path).read_text("utf-8")

        validate_post_merge_scopes(
            {"post_merge_only": structural}, read if read is not None else read_own
        )
    return [path for path in paths if path.replace("\\", "/").split("::", 1)[0] not in post]


def carry_cost(rows: dict[str, Any], moves: dict[str, str]) -> dict[str, Any]:
    """Byte-identical path moves carry their measured standing through the same pairing."""
    inverse = {old: new for new, old in moves.items()}
    return {
        inverse.get(n.partition("::")[0], n.partition("::")[0])
        + n.partition("::")[1]
        + n.partition("::")[2]: v
        for n, v in rows.items()
    }


def cost_policy(
    policy: Any, *, code: str = "test_size.cost"
) -> tuple[float, dict[str, dict[str, float]]]:
    """Read scalar debt without headroom, or explicit measured/cap records until contraction."""
    if not isinstance(policy, dict):
        raise ValueError(f"{code}: restore the configured cost policy object")
    for option in ("record_untouched", "require_clean"):
        if option in policy and type(policy[option]) is not bool:
            raise ValueError(f"{code}: {option} must be a boolean; restore the cost policy")
    rows = policy.get("slow_baseline")
    # Old receipts have no margin; one preserves their exact scalar bound.
    margin: Any = policy.get(
        "cost_margin",
        1
        if isinstance(rows, dict) and all(type(v) in (int, float) for v in rows.values())
        else None,
    )
    if (
        type(margin) not in (int, float)
        or not isfinite(margin)
        or margin < 1
        or not isinstance(rows, dict)
    ):
        raise ValueError(
            f"{code}: invalid cost_margin or slow_baseline; "
            "restore a finite margin at least 1 and measured/cap rows"
        )
    return float(margin), {node: _cost_row(node, row, code) for node, row in rows.items()}


def _cost_row(node: Any, row: Any, code: str) -> dict[str, float]:
    if type(row) in (int, float):
        if not isfinite(row) or row <= 10:
            raise ValueError(
                f"{code}: invalid legacy timing {node}; "
                "restore its measured scalar above the absolute bound"
            )
        row = {"measured": row, "cap": row}
    if (
        not isinstance(node, str)
        or not node
        or not isinstance(row, dict)
        or set(row) != {"measured", "cap"}
        or any(type(v) not in (int, float) or not isfinite(v) for v in row.values())
        or not 0 <= row["measured"] <= row["cap"]
    ):
        raise ValueError(
            f"{code}: invalid measured/cap row {node}; restore both finite measured and cap values"
        )
    return row


def _solo_cost(measurement: Any, node: str, code: str) -> tuple[dict[str, Any], float, bool]:
    if (
        not isinstance(measurement, dict)
        or measurement.get("node") != node
        or type(measurement.get("solo_seconds")) not in (int, float)
        or not isfinite(measurement["solo_seconds"])
        or measurement["solo_seconds"] < 0
        or type(measurement.get("clean")) is not bool
        or (measurement.get("clean") is True and measurement.get("state") != "MEASURED")
        or any(
            not isinstance(measurement.get(edge), list)
            for edge in ("other_holders_before", "other_holders_after")
        )
        or (
            measurement["clean"]
            and (
                measurement["other_holders_before"]
                or measurement["other_holders_after"]
                or measurement.get("holder_errors")
            )
        )
    ):
        raise RuntimeError(f"{code}_unmeasured: {node}: restore its clean solo receipt")
    return measurement, float(measurement["solo_seconds"]), bool(measurement["clean"])


def _unchanged_cost(
    node: str,
    seconds: float,
    bound: float,
    original: Any,
    policy: dict[str, Any],
    margin: float,
    confirm: Callable[..., Any] | None,
    code: str,
) -> tuple[dict[str, Any], Any]:
    """Keep existing debt; record untouched overages only from clean solo evidence."""
    row: dict[str, Any] = {}
    if seconds > bound:
        row.update(untouched_over_bound=True, bound_seconds=bound)
    if original is not None or seconds <= bound or not policy.get("record_untouched"):
        return row, original
    measurement = confirm(node, bound, seconds) if confirm is not None else None
    measurement, cost, clean = _solo_cost(measurement, node, code)
    row.update(solo_seconds=cost, bound_seconds=bound)
    if not clean:
        row["unconfirmed_load"] = True
        return row, None
    return row, {"measured": cost, "cap": margin * cost} if cost > bound else None


def _base_cost(
    node: str,
    bound: float,
    cost: float,
    margin: float,
    baseline: Callable[..., Any] | None,
    code: str,
) -> dict[str, float]:
    first, base_cost, clean = _solo_cost(
        baseline(node, bound, cost) if baseline is not None else None, node, code
    )
    if not clean or first.get("state") != "MEASURED" or first.get("node") != node:
        raise RuntimeError(
            f"{code}_unmeasured: {node}: restore its matching passing base solo receipt"
        )
    return {"measured": base_cost, "cap": max(bound, margin * base_cost)}


def _changed_cost(
    measured: dict[str, Any],
    bound: float,
    cap: dict[str, Any] | None,
    original: Any,
    policy: dict[str, Any],
    margin: float,
    confirm: Callable[..., Any] | None,
    baseline: Callable[..., Any] | None,
    code: str,
) -> tuple[dict[str, Any], Any]:
    """Use the clean result to decide whether an existing node needs a base measurement."""
    node, seconds = measured["node"], measured["seconds"]
    row: dict[str, Any] = {}
    if seconds <= bound and policy.get("require_clean") is not True:
        return row, original if cap is not None else None
    measurement = confirm(node, bound, seconds) if confirm is not None else None
    measurement, cost, clean = _solo_cost(measurement, node, code)
    row.update(solo_seconds=cost, bound_seconds=bound)
    if not clean:
        if policy.get("require_clean") is True:
            raise RuntimeError(
                f"{code}_unmeasured: {node}: repeat the solo check without competing holders"
            )
        row.update(
            unconfirmed_load=True,
            other_holders_before=measurement.get("other_holders_before"),
            other_holders_after=measurement.get("other_holders_after"),
        )
        return row, original if cap is not None else None
    seeded = cap is None and measured.get("existing") is True and cost > bound
    if seeded:
        original = _base_cost(node, bound, cost, margin, baseline, code)
        cap, bound = original, original["cap"]
        row["bound_seconds"] = bound
    if seconds > bound:
        row["load_observation"] = cost <= bound
    if cost > bound:
        raise ValueError(
            f"{code}: {node}: {cost}s alone, bound {bound}; "
            "share setup, split the behaviour or declare Large for post-merge"
        )
    if cap is None:
        return row, None
    if isinstance(original, dict):
        narrowed = min(cap["cap"], margin * cost)
        return row, original if seeded and narrowed == cap["cap"] else {
            "measured": cost,
            "cap": narrowed,
        }
    return row, cost if cost > 10 else None  # scalar writer retires with migration


def _cost_observation(measured: Any, code: str) -> None:
    if (
        not isinstance(measured, dict)
        or not isinstance(measured.get("node"), str)
        or not measured["node"]
        or type(measured.get("seconds")) not in (int, float)
        or type(measured.get("large")) is not bool
        or type(measured.get("cached", False)) is not bool
        or measured.get("outcome") not in {"passed", "failure", "error", "skipped"}
    ):
        raise ValueError(f"{code}: malformed node observation; restore complete JUnit")
    if measured["outcome"] == "skipped":
        return
    if measured["outcome"] != "passed":
        raise RuntimeError(
            f"{code}_unmeasured: {measured['node']}: failed outcomes cannot establish cost standing"
        )
    if not isfinite(measured["seconds"]) or measured["seconds"] < 0:
        raise ValueError(f"{code}: {measured['node']}: invalid duration; restore complete JUnit")


def _cost_bound(measured: dict[str, Any], cap: dict[str, Any] | None, code: str) -> float:
    bound = (
        cap["cap"]
        if cap is not None
        else measured.get("limit_seconds", 60 if measured["large"] else 10)
    )
    if (
        type(bound) not in (int, float)
        or not isfinite(bound)
        or bound < 0
        or (cap is None and bound == 0)
    ):
        raise ValueError(f"{code}: {measured['node']}: restore its declared positive size bound")
    return float(bound)


def _cost_selected(node: str, judged: tuple[str, ...] | None) -> bool:
    return judged is None or any(
        node == name
        or node.startswith((name + "::", name + "["))
        or (
            not name.rpartition("/")[2].startswith("test_")
            and node.startswith(name.rpartition("/")[0] + "/")
        )
        for name in judged
    )


def measured_costs(
    observations: Iterable[dict[str, Any]],
    policy: dict[str, Any],
    *,
    confirm: Callable[[str, float, float], dict[str, Any]] | None = None,
    baseline: Callable[[str, float, float], dict[str, Any]] | None = None,
    judged: Iterable[str] | None = None,
    code: str = "test_size.cost",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Refuse only clean solo costs; preserve caps for loaded observations in every lane."""
    margin, caps = cost_policy(policy, code=code)
    if judged is not None:
        if isinstance(judged, str):
            raise ValueError(
                f"{code}: selected nodes must be a collection; restore the declared test selection"
            )
        judged = tuple(judged)
        if not all(isinstance(node, str) and node for node in judged):
            raise ValueError(
                f"{code}: selected nodes must be nonempty strings; "
                "restore the declared test selection"
            )
    rows: list[dict[str, Any]] = []
    lowered: dict[str, Any] = {}
    for measured in observations:
        _cost_observation(measured, code)
        if measured["outcome"] == "skipped":
            if policy.get("require_clean") is True and _cost_selected(measured["node"], judged):
                raise RuntimeError(
                    f"{code}_unmeasured: {measured['node']}: "
                    "run the selected test before deciding its cost"
                )
            continue
        node, seconds, large = measured["node"], measured["seconds"], measured["large"]
        cap, original = caps.get(node), policy["slow_baseline"].get(node)
        bound = _cost_bound(measured, cap, code)
        row = dict(
            node=node,
            seconds=seconds,
            size=measured.get("size", "large" if large else "small" if seconds < 1 else "medium"),
        )
        selected = _cost_selected(node, judged)
        if measured.get("cached", False):
            if selected:
                raise RuntimeError(
                    f"{code}_unmeasured: {node}: rerun the selected node without cached timing"
                )
            changes, kept = {"cached": True}, original
        elif selected:
            changes, kept = _changed_cost(
                measured, bound, cap, original, policy, margin, confirm, baseline, code
            )
        else:
            changes, kept = _unchanged_cost(
                node, seconds, bound, original, policy, margin, confirm, code
            )
        if kept is not None:
            lowered[node] = kept
        rows.append({**row, **changes})
    if not rows:
        raise ValueError(f"{code}: no executed timings; restore complete JUnit")
    return rows, lowered


def floor_mode(lane: str) -> bool:
    """The session floor affects only the merge lane; any nonempty flag keeps reach whole."""
    return lane == "merge" and bool(os.environ.get("ALPHALATTICE_REACH_FLOOR"))


class SessionFootprint:
    """One full-suite process's partial session-fixture and sampled basetemp evidence."""

    interval_seconds = 120.0

    def __init__(
        self, root: Path | None, directory: Path, worker: str, attempt: str, source: str
    ) -> None:

        if len(attempt) != 32 or any(ch not in "0123456789abcdef" for ch in attempt):
            raise ValueError("invalid suite attempt identity")
        self.root, self.directory, self.worker = root, directory, worker
        self.attempt, self.source = attempt, source
        self.owner_token = ""
        self.fixtures: list[dict[str, Any]] = []
        self.peak: dict[str, Any] = {"bytes": None, "contributors": [], "samples": 0}
        self.errors: list[str] = []
        self.started = time.monotonic()
        self._last_sample: float | None = self.started
        self._max_gap = 0.0
        self._sample_seconds = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._logger = logging.getLogger("devtools.testing.session_footprint")
        self._old_level, self._old_propagate = self._logger.level, self._logger.propagate
        self._handler: Any = None
        self._stream: Any = None
        try:
            self._stream = (directory / f"session-footprint-{attempt}-{worker}.log").open(
                "x", encoding="utf-8"
            )
            self._handler = logging.StreamHandler(self._stream)
            self._handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(self._handler)
            self._logger.setLevel(logging.INFO)
            self._logger.propagate = False
        except (OSError, ValueError) as error:
            self.errors.append("build log unavailable: " + str(error))
        if worker == "main" and root is not None:
            self._thread = threading.Thread(
                target=self._sample_loop, name="session-footprint", daemon=True
            )
            try:
                self._thread.start()
            except RuntimeError as error:
                self.errors.append("sampler unavailable: " + str(error))
                self._thread = None

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.sample()

    def sample(self) -> None:

        now = time.monotonic()
        if self._last_sample is not None:
            self._max_gap = max(self._max_gap, now - self._last_sample)
        self._last_sample = now
        try:
            if self.root is None:
                raise ValueError("pytest basetemp is not named")
            info = self.root.lstat()
            reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            if not stat.S_ISDIR(info.st_mode) or getattr(info, "st_file_attributes", 0) & reparse:
                raise ValueError("basetemp is not a physical directory")
            groups: list[dict[str, Any]] = []
            with os.scandir(self.root) as entries:
                top = list(entries)
            for entry in top:
                metadata = entry.stat(follow_symlinks=False)
                if (
                    re.fullmatch(r"popen-gw[0-9]+", entry.name)
                    and stat.S_ISDIR(metadata.st_mode)
                    and not getattr(metadata, "st_file_attributes", 0) & reparse
                ):
                    with os.scandir(entry.path) as children:
                        paths = [Path(child.path) for child in children]
                else:
                    paths = [Path(entry.path)]
                for path in paths:
                    groups.append(
                        {
                            "path": path.relative_to(self.root).as_posix(),
                            "bytes": tmp_entry_footprint(path),
                        }
                    )
            total = sum(row["bytes"] for row in groups)
            self.peak["samples"] += 1
            if self.peak["bytes"] is None or total > self.peak["bytes"]:
                self.peak.update(
                    bytes=total,
                    contributors=sorted(groups, key=lambda row: row["bytes"], reverse=True)[:20],
                    at_seconds=now - self.started,
                )
        except (OSError, ValueError) as error:
            self.errors.append("sample unmeasured: " + str(error))
        finally:
            self._sample_seconds += time.monotonic() - now

    def before_fixture(
        self, fixture: Any, request: Any
    ) -> tuple[float, Path | None, set[str] | None, str]:

        if self.root is None or "tmp_path_factory" not in fixture.argnames:
            return time.perf_counter(), None, None, "no attributable temporary factory root"
        try:
            base = request.config._tmp_path_factory.getbasetemp()
            before = {entry.name for entry in base.iterdir()}
            return time.perf_counter(), base, before, ""
        except (OSError, ValueError, AttributeError) as error:
            return time.perf_counter(), None, None, "fixture root unmeasured: " + str(error)

    def failed_fixture(
        self,
        fixture: Any,
        snapshot: tuple[float, Path | None, set[str] | None, str],
        error: BaseException,
    ) -> None:

        self.fixtures.append(
            {
                "name": fixture.argname,
                "owner": str(fixture.baseid),
                "build_seconds": time.perf_counter() - snapshot[0],
                "bytes": None,
                "roots": [],
                "reason": "fixture setup failed: " + type(error).__name__,
            }
        )

    def after_fixture(
        self, fixture: Any, request: Any, snapshot: tuple[float, Path | None, set[str] | None, str]
    ) -> None:

        started, base, before, reason = snapshot
        row: dict[str, Any] = {
            "name": fixture.argname,
            "owner": str(fixture.baseid),
            "build_seconds": time.perf_counter() - started,
            "bytes": None,
            "roots": [],
            "reason": reason or "fixture teardown not observed",
        }
        self.fixtures.append(row)
        if base is None or before is None:
            return
        try:
            roots = [entry for entry in base.iterdir() if entry.name not in before]
            row["roots"] = (
                [path.relative_to(self.root).as_posix() for path in roots] if self.root else []
            )
        except (OSError, ValueError) as error:
            row["reason"] = "fixture roots unmeasured: " + str(error)
            return
        if not roots:
            row["reason"] = "no uniquely attributable new root; shared or lazy data may exist"
            return

        def measure() -> None:
            try:
                row["bytes"] = sum(tmp_entry_footprint(path) for path in roots)
                row["reason"] = ""
            except (OSError, ValueError) as error:
                row["reason"] = "fixture bytes unmeasured: " + str(error)

        request.addfinalizer(measure)

    def finish(self) -> None:

        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                self.errors.append("sampler did not finish before the bounded join")
            else:
                self.sample()
        if self._handler is not None:
            self._logger.removeHandler(self._handler)
            self._handler.close()
        if self._stream is not None:
            self._stream.close()
        self._logger.setLevel(self._old_level)
        self._logger.propagate = self._old_propagate
        if self.root is None:
            self.errors.append("pytest basetemp is not named")
        if self.worker != "main":
            self.peak["reason"] = "run peak is sampled by the controller"
        value = {
            "schema": "alphalattice.session-footprint.v1",
            "attempt": self.attempt,
            "source_sha": self.source,
            "worker": self.worker,
            "status": "partial",
            "scope": "unverified until joined to the attempt manifest",
            "basis": (
                "sampled apparent bytes; no-follow; hardlinks counted per path; "
                "fixture ownership may overlap"
            ),
            "fixtures": self.fixtures,
            "peak": {
                **self.peak,
                "interval_seconds": self.interval_seconds,
                "max_gap_seconds": self._max_gap,
                "sample_seconds": self._sample_seconds,
                "elapsed_seconds": time.monotonic() - self.started,
            },
            "errors": self.errors,
        }
        try:
            with (self.directory / f"session-footprint-{self.attempt}-{self.worker}.json").open(
                "x", encoding="utf-8"
            ) as stream:
                json.dump(value, stream, allow_nan=False)
                stream.write("\n")
        except (OSError, ValueError) as error:
            print(f"test_size.session_footprint_unmeasured: {error}", file=sys.stderr)
