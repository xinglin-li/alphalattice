"""Registered test sizes consumed by pytest collection and the canonical gate."""

import json
import os
import re
from collections.abc import Callable, Iterable
from datetime import date
from fnmatch import fnmatchcase
from math import isfinite
from pathlib import Path, PurePosixPath
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
    _post_merge_only(value)
    _reach(value)
    _structural_fence(value)
    _limits(value)
    _quarantine(value)
    _framework_fixtures(value)
    _unit_backends(value)
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
    """Keep structural obligations mandatory and reject malformed stage deferrals."""
    post = value.get("post_merge_only", [])
    if (
        not isinstance(post, list)
        or any(
            not isinstance(p, str)
            or not p.startswith("tests/")
            or _structural_file(p, value)
            or not p.endswith(".py")
            or ".." in Path(p).parts
            or "\\" in p
            or any(c in p for c in "*?[")
            for p in post
        )
        or len(set(post)) != len(post)
    ):
        raise ValueError("test_size.rule: post_merge_only needs distinct non-structural test files")


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


def moved(nodeid: str, policy: dict[str, Any]) -> bool:
    """A whole-tree structural guard with no declared reach and no fence law: the commit stage
    leaves it to post-merge, which runs every structural node whole (the reform's rule, 0.2.md)."""
    nodeid = nodeid.replace("\\", "/")
    return (
        nodeid.startswith("tests/structural/")
        and reach(nodeid, policy) is None
        and not any(fnmatchcase(nodeid, r["pattern"]) for r in policy.get("structural_fence", []))
    )


def _quarantine(policy: dict[str, Any]) -> None:
    """Refuse a quarantined node without an owner, a reason or an until date."""
    for row in policy.get("quarantine", []):  # a flaky node runs post-merge only, until its date
        try:  # a real calendar date in ISO form, so 2026-99-99 is refused
            dated = date.fromisoformat(str(row.get("until"))).isoformat() == row.get("until")
        except ValueError:
            dated = False
        if not (
            str(row.get("node")).startswith("tests/")
            and row.get("owner")
            and row.get("reason")
            and dated
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
    row = reach(nodeid, policy) if changed is not None else None
    if lane == "fast":
        return scoped(nodeid, policy, changed)
    if lane == "merge" and row:
        if floor:
            return any(fnmatchcase(nodeid, p) for p in policy["mandatory"]) or size != "large"
        return row.get("per_file") is True or _triggered(nodeid, row, changed or frozenset())
    if lane == "merge" and changed is not None and moved(nodeid, policy):
        return nodeid.replace("\\", "/").split("::")[0] in changed  # else post-merge runs it whole
    return (
        not lane
        or any(fnmatchcase(nodeid, p) for p in policy["mandatory"])
        or (size != "large" if lane == "merge" else size == "large")
    )


def stage_files(
    policy: dict[str, Any], stage: str, paths: Iterable[str], *, lane: str = ""
) -> list[str]:
    """Defer these files only in commit; every other stage keeps them."""
    post = (
        frozenset(policy.get("post_merge_only", []))
        if stage == "commit" and lane not in {"candidate", "nightly"}
        else frozenset()
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
