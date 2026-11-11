"""Publish resource budgets from observed receipts; missing evidence never becomes zero."""

from __future__ import annotations

import json
from pathlib import Path

from devtools.evaluation.forms import admit_replay_measurement
from devtools.evaluation.receipts import number, write


def capture_profile(out: Path, facts: dict, workspace, client, client_error) -> None:
    """Record a Goal profile only when the actual bound workspace identifies it."""
    goal = facts.get("goal")
    goal_id = goal.get("goal_id") if isinstance(goal, dict) else None
    if not goal_id or (out / "profile.json").exists():
        return
    try:
        path = workspace()
        if path is None:
            raise ValueError("evaluation.profile_workspace_unobserved")
        profile = client(path).request({"operation": "GOAL_PROFILE", "goal_id": goal_id})
    except (client_error, OSError, ValueError) as error:
        profile = dict(status="UNOBSERVED", reason=str(error))
    write(out / "profile.json", profile)


def read(path: Path, default=None):
    try:
        value = json.loads(path.read_bytes())
        return default if default is not None and not isinstance(value, type(default)) else value
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def inodes(files: dict) -> dict:
    result = {}
    for row in files.values():
        if len(row) != 4 or any(type(v) is not int or v < 0 for v in row) or not row[1]:
            raise ValueError("budget.invalid_inode_receipt")
        key = tuple(row[:2])
        if key in result and result[key] != row[2]:
            raise ValueError("budget.inconsistent_inode_receipt")
        result[key] = row[2]
    return result


def growth(before: dict, after: dict, prefixes: list) -> dict:
    old, current = inodes(before), inodes(after)
    seen, writers = set(), {}
    for path, row in sorted(after.items()):
        key = tuple(row[:2])
        if key in seen:
            continue
        seen.add(key)
        added = max(0, current[key] - old.get(key, 0))
        relative = path.split("/", 1)[-1]
        owner = next((p for p in prefixes if relative.startswith(p + "/")), relative.split("/")[0])
        if added:
            writers[owner] = writers.get(owner, 0) + added
    return writers


def _token_usage(row, keys):
    usage = row.get("usage")
    if not isinstance(usage, dict) or any(
        type(usage.get(key)) is not int or usage[key] < 0 for key in keys
    ):
        return None
    return usage if usage["cached_input_tokens"] <= usage["input_tokens"] else None


def codex_tokens(out: Path):
    """Read the native session's observed Codex turn usage."""
    session = read(out / "session.json", {}).get("session")
    if not isinstance(session, str) or not session:
        return None
    keys = ("input_tokens", "cached_input_tokens", "output_tokens")
    totals = dict.fromkeys(keys, 0)
    thread = turn = False
    completed = 0
    try:
        with (out / "lead.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if not isinstance(row, dict):
                    return None
                kind = row.get("type")
                if kind == "thread.started":
                    if turn or row.get("thread_id") != session:
                        return None
                    thread = True
                elif kind == "turn.started":
                    if not thread or turn:
                        return None
                    turn = True
                elif kind == "turn.completed":
                    usage = _token_usage(row, keys)
                    if not turn or usage is None:
                        return None
                    for key in keys:
                        totals[key] += usage[key]
                    completed += 1
                    turn = False
                elif kind in {"turn.failed", "error"}:
                    return None
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return totals if completed and not turn else None


def cost(out: Path):
    session = read(out / "session.json", {}).get("session")
    path = out / "lead.jsonl"
    values = []
    for line in path.read_text("utf-8").splitlines() if path.is_file() else []:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(row, dict):
            continue
        if (
            row.get("type") == "result"
            and not row.get("parent_tool_use_id")
            and row.get("session_id", session) == session
        ):
            values.append(row.get("total_cost_usd"))
    return values[-1] if session and values and number(values[-1]) else None


def peak(out: Path, *, rows: list | None = None):
    legs = (
        [
            [row for path in leg.glob("native-*-memory.json") for row in read(path, [{}])]
            for leg in (out, out / "daily")
        ]
        if rows is None
        else [rows]
    )
    if any(
        not any(isinstance(row, dict) and row.get("host_tree_observed") for row in leg)
        for leg in legs
    ):
        return None
    samples = [row for leg in legs for row in leg]
    if not samples or not all(
        isinstance(row, dict)
        and row.get("host_tree_complete") is True
        and number(row.get("host_private_commit_bytes"))
        for row in samples
    ):
        return None
    return max(row["host_private_commit_bytes"] for row in samples)


def profile_rows(leg: str, receipt: dict, *, admitted: bool) -> tuple:
    """Render one owner's profile; only observed finite values may become budget figures."""
    stages, coverage, projected, failures = [], [], {}, {}
    field = "profile"
    try:
        profile = receipt["profile"]
        if not admitted and (
            profile["schema"],
            profile["status"],
            profile["gaps"],
            profile["coverage"]["status"] in ("COMPLETE", "PARTIAL"),
        ) != (
            "alphalattice.goal-profile.v1",
            "GOAL_PROFILE",
            [],
            True,
        ):
            raise ValueError(field)
        field = "stages"
        if not profile["stages"]:
            raise ValueError(field)
        measured = profile["coverage"]
        coverage.append(f"| {leg} | coverage | {measured['status']} |")
        for field in (
            "rpc_seconds",
            "phase_seconds",
            "overlap_seconds",
            "host_seconds",
            "unattributed_seconds",
        ):
            if not (admitted and field == "host_seconds") and not number(measured[field]):
                raise ValueError(field)
            coverage.append(f"| {leg} | {field} | {measured[field]} |")
        field = "host_share_percent"
        window = measured["host_seconds"] + measured["unattributed_seconds"]
        if not number(window) or not window:
            raise ValueError(field)
        share = measured["host_seconds"] / window * 100
        coverage.append(f"| {leg} | Host share of Goal window (%) | {share} |")
        for row in profile["stages"]:
            names = [row[k] for k in ("namespace", "task_id", "stage_id")]
            field = "stage_scope"
            if not all(names):
                raise ValueError(field)
            label = " | ".join(
                v.replace("|", r"\|").replace("\n", " ").replace("\r", " ") for v in names
            )
            for field in ("execute_seconds", "verify_seconds"):
                if not admitted and not number(row[field]):
                    raise ValueError(field)
            stages.append(
                f"| {leg} | {label} | {row['execute_seconds']} | {row['verify_seconds']} |"
            )
        projected[leg] = dict(
            profile={k: v for k, v in profile.items() if k != "cache_counts"},
            host_share_percent=share,
        )
    except (KeyError, TypeError, AttributeError, ValueError, ArithmeticError) as error:
        field = error.args[0] if isinstance(error, KeyError) else field
        failures[leg] = f"budget.profile_missing:{leg}:{field}"
        coverage.append(f"| {leg} | {failures[leg]} | MISSING |")
    return projected, failures, stages, coverage


def publish(out: Path, result: dict, legs: dict, *, admitted: bool = False) -> dict:
    """Publish scoped stage times and Goal-window Host share without summing overlaps."""
    stages = [
        "| Leg | Namespace | Task | Stage | Execute (s) | Verify (s) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    coverage = ["| Leg | Measure | Observed |", "| --- | --- | --- |"]
    projected, failures = {}, {}
    for leg, receipt in legs.items():
        kept, failed, phase_rows, union_rows = profile_rows(leg, receipt, admitted=admitted)
        projected.update(kept)
        failures.update(failed)
        stages.extend(phase_rows)
        coverage.extend(union_rows)
    result.update(
        profiles=projected, profile_failure=failures, held=result["held"] and not failures
    )
    result["status"] = "COMPLETE" if result["held"] else "INCOMPLETE"
    result["table"] = f"Budget: {result['status']}\n{result.get('reason', '')}\n" + result["table"]
    result["table"] += "\n" + "\n".join([*stages, "", *coverage]) + "\n"
    write(out / "budget.json", result)
    (out / "budget.md").write_text(result["table"], encoding="utf-8", newline="\n")
    return result


def budget(
    out: Path, *, scenario, replay_source: str | None = None, require_replay: bool = False
) -> dict:
    """Report cumulative session cost and SI, inclusive storage/time/commit limits."""
    receipt = out / "replay-measurement.json"
    if require_replay or receipt.is_file():
        try:
            value = admit_replay_measurement(read(receipt), replay_source or "")
            return publish(
                out,
                dict(
                    held=True,
                    source_sha=value["source_sha"],
                    table="Replay baseline\nSample interval: "
                    + value.get("measurement_interval", "not declared"),
                ),
                value["legs"],
                admitted=True,
            )
        except ValueError as error:
            return publish(
                out, dict(held=False, reason=str(error), table="Replay baseline: INCOMPLETE"), {}
            )
    return _native_budget(out, scenario())


def daily_scope_complete(start_scope, end_scope):
    """Admit only the same observed workspace and fully resolved aliases at both boundaries."""
    workspace_root = start_scope.get("workspace_root")
    end_root = end_scope.get("workspace_root")
    return (
        start_scope.get("complete") is True
        and end_scope.get("complete") is True
        and isinstance(workspace_root, list)
        and len(workspace_root) == 2
        and all(type(value) is int and value >= 0 for value in workspace_root)
        and workspace_root[1] > 0
        and isinstance(end_root, list)
        and len(end_root) == 2
        and all(type(value) is int and value >= 0 for value in end_root)
        and workspace_root == end_root
        and isinstance(start_scope.get("accounted_aliases"), dict)
        and start_scope["accounted_aliases"] == end_scope.get("accounted_aliases")
        and start_scope.get("excluded_aliases") == []
        and end_scope.get("excluded_aliases") == []
    )


def _storage(out, scenario):
    limits = scenario["budgets"]
    before, after = (
        read(out / "daily" / name) for name in ("workspace-before.json", "workspace-after.json")
    )
    footprint, scope = read(out / "footprint-after.json"), read(out / "footprint-roots.json", {})
    start_scope = read(out / "daily/workspace-scope-before.json", {})
    end_scope = read(out / "daily/footprint-roots.json", {})
    error = None
    try:
        complete = daily_scope_complete(start_scope, end_scope)
        writers = (
            growth(before, after, scenario["daily"]["writer_prefixes"])
            if complete and before is not None and after is not None
            else None
        )
        if not complete:
            error = "budget.daily_scope_incomplete_or_changed"
        size = (
            sum(inodes(footprint).values())
            if footprint is not None and scope.get("complete") is True and scope.get("roots")
            else None
        )
    except (ValueError, TypeError, AttributeError) as failure:
        writers, size, error = None, None, str(failure)
    written = sum(writers.values()) if writers is not None else None
    year = (
        size + limits["trading_days"] * written
        if size is not None and written is not None
        else None
    )
    return writers, size, written, year, error


def _host_cost(out, result):
    session_host = read(out / "session.json", {}).get("host")
    leg_host = read(out.parent / "leg.json", {}).get("host")
    host_conflict = session_host is not None and leg_host is not None and session_host != leg_host
    codex = not host_conflict and (session_host or leg_host) == "codex"
    if host_conflict:
        result.update(first_use_cost_usd=None, first_use_cost_reason="budget.host_mismatch")
    if codex:
        result.update(
            first_use_cost_usd="not billed per session (subscription)",
            first_use_cost_reason="not billed per session (subscription)",
            first_use_codex_tokens=codex_tokens(out),
        )
    return codex, host_conflict


def _budget_table(result, limits, codex, host_conflict, writers):
    rows = [
        ("First use wall (s)", "first_use_wall_seconds", None),
        ("First use cost (USD)", "first_use_cost_usd", None),
        ("Daily wall (s)", "daily_wall_seconds", limits["daily_wall_seconds"]),
        ("Daily growth (bytes)", "daily_bytes", limits["daily_bytes"]),
        (
            "Host tree private commit (bytes)",
            "peak_host_private_commit_bytes",
            limits["host_private_commit_bytes"],
        ),
        ("Year footprint (bytes)", "year_bytes", limits["year_bytes"]),
    ]
    checks, lines = {}, ["| Measure | Observed | Limit | Result |", "| --- | --- | --- | --- |"]
    for label, key, limit in rows:
        value = result[key]
        checks[key] = number(value) and (limit is None or value <= limit)
        if codex and key == "first_use_cost_usd":
            checks[key] = result["first_use_codex_tokens"] is not None
        status = ("REPORTED" if limit is None else "PASS") if checks[key] else "FAIL"
        observed = value if value is not None else "MISSING"
        if codex and key == "first_use_cost_usd":
            observed = result["first_use_cost_reason"]
            label = "First use cost (USD; tokens below)"
        elif host_conflict and key == "first_use_cost_usd":
            observed = result["first_use_cost_reason"]
        bound = limit if limit is not None else "reported"
        lines.append(f"| {label} | {observed} | {bound} | {status} |")
    if codex:
        tokens = result["first_use_codex_tokens"]
        for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
            value = tokens[key] if tokens is not None else "MISSING"
            status = "REPORTED" if tokens is not None else "FAIL"
            lines.append(f"| First use {key} | {value} | reported | {status} |")
    if writers:
        lines += ["", "| Writer | Daily bytes |", "| --- | --- |"]
        lines += [f"| {owner} | {value} |" for owner, value in sorted(writers.items())]
    return checks, lines


def _native_budget(out, scenario):
    limits = scenario["budgets"]
    first, daily = read(out / "summary.json", {}), read(out / "daily/summary.json", {})
    writers, size, written, year, error = _storage(out, scenario)
    result = dict(
        first_use_wall_seconds=first.get("wall_seconds"),
        first_use_cost_usd=cost(out),
        daily_wall_seconds=daily.get("wall_seconds"),
        daily_bytes=written,
        daily_by_writer=writers,
        peak_host_private_commit_bytes=peak(out),
        first_use_footprint_bytes=size,
        year_bytes=year,
    )
    for key in result:
        if key != "daily_by_writer" and not number(result[key]):
            result[key] = None
    codex, host_conflict = _host_cost(out, result)
    checks, lines = _budget_table(result, limits, codex, host_conflict, writers)
    result.update(
        held=all(checks.values()),
        checks=checks,
        table="\n".join(lines) + "\n",
        basis=(
            "New inode bytes plus positive retained-inode growth; "
            "managed footprint includes run-owned tools/cache/temp; sampled private commit"
        ),
        storage_failure=error,
    )
    return publish(
        out,
        result,
        {
            "first_use": {"profile": read(out / "profile.json")},
            "daily": {"profile": read(out / "daily/profile.json")},
        },
    )
