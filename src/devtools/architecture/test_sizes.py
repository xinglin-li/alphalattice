"""Registered test sizes consumed by pytest collection and the canonical gate."""

import json
import os
from collections.abc import Iterable
from fnmatch import fnmatchcase
from math import isfinite
from pathlib import Path
from typing import Any


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
    _reach(value)
    _limits(value)
    return value


def _reach(policy: dict[str, Any]) -> None:
    """Refuse a reach row outside tests/structural, without a reason, or declaring nothing."""
    for row in policy.get("reach", []):  # a structural node's reach: per_file or declared inputs
        if not (row.get("pattern", "").startswith("tests/structural/") and row.get("reason")) or (
            row.get("per_file") is not True and not row.get("inputs")
        ):
            raise ValueError("test_size.reach_invalid:" + str(row.get("pattern")))


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


def changed_paths() -> frozenset[str] | None:
    """A scoped run's changed paths from its `git diff --name-status --no-renames` listing,
    an added or deleted one also as `+path`; None outside a scoped run."""
    named = os.environ.get("ALPHALATTICE_CHANGED_PATHS")
    rows = [r.split("\t") for r in Path(named).read_text("utf-8").splitlines()] if named else []
    found = {f"+{r[-1]}" for r in rows if r[0] in {"A", "D"}} | {r[-1] for r in rows if r[0]}
    return frozenset(found) if named else None


def reach(nodeid: str, policy: dict[str, Any]) -> dict[str, Any] | None:
    """The declared reach of a structural node, which a scoped merge or fast run honours."""
    return _rule(nodeid.replace("\\", "/"), policy.get("reach", []))


def _triggered(nodeid: str, row: dict[str, Any], changed: frozenset[str]) -> bool:
    own = nodeid.replace("\\", "/").split("::")[0]  # its own file or a declared input changed
    return own in changed or any(fnmatchcase(p, g) for p in changed for g in row.get("inputs", ()))


def scoped(nodeid: str, policy: dict[str, Any], changed: frozenset[str] | None) -> bool:
    """A per_file node scans only the changed files, unless its own file or an input changed."""
    row = reach(nodeid, policy) if changed is not None else None
    return bool(
        row and row.get("per_file") is True and not _triggered(nodeid, row, changed or frozenset())
    )


def included(
    nodeid: str,
    size: str,
    lane: str,
    policy: dict[str, Any],
    changed: frozenset[str] | None = None,
) -> bool:
    """Mandatory holders always run, except that a scoped merge run keeps a reach-declared node
    only when it is per_file or triggered, and the fast lane keeps only scoped per_file nodes;
    each lane otherwise admits its declared sizes."""
    row = reach(nodeid, policy) if changed is not None else None
    if lane == "fast":
        return scoped(nodeid, policy, changed)
    if lane == "merge" and row:
        return row.get("per_file") is True or _triggered(nodeid, row, changed or frozenset())
    return (
        not lane
        or any(fnmatchcase(nodeid, p) for p in policy["mandatory"])
        or (size != "large" if lane == "merge" else size == "large")
    )
