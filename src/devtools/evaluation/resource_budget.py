"""Publish resource budgets from observed receipts; missing evidence never becomes zero."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from time import perf_counter

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


def _memory_samples(out: Path, rows: list | None) -> list:
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
        return []
    samples = [row for leg in legs for row in leg]
    if not samples or not all(
        isinstance(row, dict)
        and row.get("host_tree_complete") is True
        and number(row.get("host_private_commit_bytes"))
        for row in samples
    ):
        return []
    return samples


def peak(out: Path, *, rows: list | None = None):
    return max(
        (row["host_private_commit_bytes"] for row in _memory_samples(out, rows)), default=None
    )


def _commit_by_kind(row: dict) -> dict:
    unclassified = {"unclassified": row["host_private_commit_bytes"]}
    members = row.get("host_tree_members")
    processes = row.get("processes")
    if not isinstance(members, list) or not isinstance(processes, list):
        return unclassified
    live = set(members)
    totals: dict[str, int] = {}
    measured = [p for p in processes if isinstance(p, dict) and p.get("pid") in live]
    for process in measured:
        size = process.get("private")
        if type(size) is not int or size < 0:
            return unclassified
        kind = process.get("kind")
        kind = kind if isinstance(kind, str) and kind else "unclassified"
        totals[kind] = totals.get(kind, 0) + size
    if not measured or sum(totals.values()) != row["host_private_commit_bytes"]:
        return unclassified
    return totals


def memory_peaks(out: Path, *, rows: list | None = None, kinds: Iterable[str] = ()) -> dict:
    """Keep the tree peak and each kind's independent peak in the same sampled window."""
    result: dict = dict(tree=None, kinds=dict.fromkeys(kinds))
    for row in _memory_samples(out, rows):
        at = dict(at=row.get("at"), stage=row.get("stage"))
        at.update(
            {name: row[name] for name in ("run_label", "command") if row.get(name) is not None}
        )
        total, shares = row["host_private_commit_bytes"], _commit_by_kind(row)
        if result["tree"] is None or total > result["tree"]["bytes"]:
            result["tree"] = dict(bytes=total, **at, by_kind=shares)
        for kind, size in shares.items():
            high = result["kinds"].get(kind)
            if high is None or size > high["bytes"]:
                result["kinds"][kind] = dict(bytes=size, **at)
    return result


def declared_process_kinds() -> Iterable[str]:
    """The Host tree budget covers only the product's declared process kinds."""
    root = Path(__file__).resolve().parents[3]
    return read(root / "config/registries/process-budgets.json")["kinds"]


def timed_memory_operations(call, windows, operations):
    """Bind the resolved CLI operation to the measured call's monotonic window."""

    def timed(argv, session=None):
        start = perf_counter()
        answer = None
        try:
            answer = call(argv, session)
            return answer
        finally:
            if argv:
                end = perf_counter()
                name = answer.get("operation") if isinstance(answer, dict) else None
                operation = name if isinstance(name, str) and name in operations else None
                windows.append((start, end, argv[0], operation))

    return timed


def memory_sample_stage(sample: dict, instant: float, windows, label: str) -> dict:
    """Bind one sample to an observed operation only inside its unique clock window."""
    matches = [window for window in windows if window[0] <= instant < window[1]]
    command, stage = None, None
    if len(matches) == 1:
        _, _, command, *operation = matches[0]
        # Earlier replay producers recorded only the command in their windows.
        if len(operation) == 1:
            stage = operation[0]
    return sample | dict(run_label=label, command=command, stage=stage)


def replay_memory_meter(
    out: Path, rows: list, kinds: Iterable[str], *, covered: bool, aligned: bool
) -> dict:
    memory = memory_peaks(out, rows=rows, kinds=kinds)
    maximum = memory["tree"]["bytes"] if memory["tree"] else None
    complete = covered and maximum is not None and all(row["host_tree_observed"] for row in rows)
    return dict(
        host_tree_observed=any(row["host_tree_observed"] for row in rows),
        peak_host_private_commit_bytes=maximum,
        host_private_commit=memory,
        gaps=[]
        if complete
        else ["LISTENER_COVERAGE_INCOMPLETE" if aligned else "METER_CLOCK_UNOBSERVED"],
        sampling_seconds=2,
    )


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
        meter = receipt.get("meter", {})
        if leg in projected and "host_private_commit" in meter:
            projected[leg]["host_private_commit"] = meter["host_private_commit"]
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
    memory = memory_peaks(out, kinds=declared_process_kinds())
    result = dict(
        first_use_wall_seconds=first.get("wall_seconds"),
        first_use_cost_usd=cost(out),
        daily_wall_seconds=daily.get("wall_seconds"),
        daily_bytes=written,
        daily_by_writer=writers,
        peak_host_private_commit_bytes=memory["tree"]["bytes"] if memory["tree"] else None,
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
        host_private_commit=memory,
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
