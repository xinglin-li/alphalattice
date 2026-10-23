"""Registered test sizes consumed by pytest collection and the canonical gate."""

import json
from collections.abc import Iterable
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any


def load_policy(root: Path) -> dict[str, Any]:
    """Read one collection's policy once, without a process-global cache."""
    value = json.loads((root / "config/registries/test-sizes.json").read_text("utf-8"))
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
    if any(r.get("size") not in value["sizes"] for r in value["defaults"]):
        raise ValueError("test_size.default_size")
    if "mandatory" in value and (
        not isinstance(value["mandatory"], list) or "tests/structural/*" not in value["mandatory"]
    ):
        raise ValueError("test_size.mandatory_missing")
    return value


def _rule(nodeid: str, rules: list[dict[str, Any]]) -> dict[str, Any] | None:
    matches = [r for r in rules if fnmatchcase(nodeid, r["pattern"])]
    if not matches:
        return None
    width = max(len(r["pattern"]) for r in matches)
    closest = [r for r in matches if len(r["pattern"]) == width]
    if len({(r["owner"], r.get("size", "large")) for r in closest}) > 1:
        raise ValueError("test_size.owner_conflict:" + nodeid)
    return min(closest, key=lambda r: r["pattern"])


def classify(nodeid: str, declared: Iterable[str], policy: dict[str, Any]) -> tuple[str, str, str]:
    """Resolve one size, its declaring owner and the declaration that selected it."""
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
    return str(size), str(rule["owner"]), str(origin)


def included(nodeid: str, size: str, lane: str, policy: dict[str, Any]) -> bool:
    """Mandatory holders always run; each lane otherwise admits its declared sizes."""
    return (
        not lane
        or any(fnmatchcase(nodeid, p) for p in policy["mandatory"])
        or (size != "large" if lane == "merge" else size == "large")
    )
