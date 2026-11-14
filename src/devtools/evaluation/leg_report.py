"""What a native leg's own files say: its Host and model time, its grade's findings, and each
native release gate's verdict.

Read by the native leg's report and the release gates' Inspect task; nothing here runs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def product_reason(text: str) -> str | None:
    """Only the product's registered refusals can be identified from untyped text."""
    try:
        root = Path(__file__).resolve().parents[3]
        refusals = json.loads((root / "config/registries/refusals.json").read_bytes())["entries"]
    except (OSError, ValueError, KeyError, TypeError):
        return "native_leg.refusals_unavailable"
    return text if text in refusals else None


def native_failure(out: Path) -> str | None:
    """The latest native process's own failed result, never a filename from the transcript."""
    streams = sorted(
        out.glob("native-*-stdout.jsonl"),
        key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else -1,
    )
    if streams:
        try:
            result = {}
            with streams[-1].open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if isinstance(row, dict) and row.get("type") == "result":
                        result = row
            if result.get("is_error") is True:
                status = result.get("api_error_status")
                if type(status) is int and 400 <= status <= 599:
                    kind = (
                        "provider_authentication"
                        if "authenticat" in str(result.get("result", "")).lower()
                        else "provider_error"
                    )
                    return f"native_leg.{kind}:{status}"
                code = str(result.get("result", "")).split(":", 1)[0]
                return product_reason(code) or "native_leg.native_result_error"
        except (OSError, ValueError):
            return "native_leg.native_result_unreadable"
    return None


def log_failure(log: Path) -> str:
    """Name a registered refusal or an exception type from a failed step's log."""
    tail = log.read_text(encoding="utf-8", errors="replace")[-2000:] if log.exists() else ""
    codes = re.findall(
        r"^\w+(?:Error|Exception):\s*([a-z][a-z0-9_]*\.[a-z][a-z0-9_:-]*)(?=\s|$)", tail, re.M
    )
    named = [reason for code in codes if (reason := product_reason(code))]
    errors = re.findall(r"^(\w+(?:Error|Exception))\b", tail, re.M)
    return (
        named[-1]
        if named
        else f"native_leg.step_failed:{errors[-1]}"
        if errors
        else "native_leg.goal_incomplete"
    )


def timing(out: Path) -> dict[str, Any] | None:
    """A leg's Host seconds against its model seconds, and its slowest stages. The profile's own
    gaps are named: this table reads, where the budget's proof refuses a profile with gaps."""
    try:
        profile = json.loads((out / "profile.json").read_bytes())
        coverage = profile["coverage"]
    except (OSError, ValueError, KeyError):
        return None
    model = cost = turns = 0
    for path in sorted(out.glob("native-*-stdout.jsonl")):  # each process counts from its start
        seen = dict(model=0, cost=0)
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if '"type": "result"' not in line and '"type":"result"' not in line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            seen["model"] = max(seen["model"], row.get("duration_api_ms") or 0)
            seen["cost"] = max(seen["cost"], row.get("total_cost_usd") or 0)
            turns += 1
        # A resumed session may report its whole total, or only its own process's.
        model = seen["model"] if seen["model"] >= model else model + seen["model"]
        cost = seen["cost"] if seen["cost"] >= cost else cost + seen["cost"]

    def seconds(row: dict[str, Any]) -> float:  # a stage without a finite phase reads as a gap
        phases = (row.get("execute_seconds"), row.get("verify_seconds"))
        return sum(value for value in phases if isinstance(value, int | float))

    stages = sorted((seconds(r), r["stage_id"]) for r in profile.get("stages", []))[-5:]
    return dict(
        host_seconds=coverage.get("host_seconds"),
        phase_seconds=coverage.get("phase_seconds"),
        rpc_seconds=coverage.get("rpc_seconds"),
        unattributed_seconds=coverage.get("unattributed_seconds"),
        model_seconds=round(model / 1000, 1),
        cost_usd=round(cost, 2),
        turns=turns,
        gaps=sorted({g.get("code") for g in profile.get("gaps", [])}),
        slowest=[[stage, round(seconds, 1)] for seconds, stage in reversed(stages)],
    )


def graded(out: Path) -> str:
    """A closed Goal is not a clean grade: the grader's findings stay beside the state."""
    path = out / "summary.json"
    if not path.exists():
        return "ungraded"
    summary = json.loads(path.read_bytes())
    failures = sorted({row["code"] for row in summary.get("failures", [])})
    return ", ".join([*summary.get("findings", []), *failures]) or "clean"


GRADES = ("fresh/summary.json", "fresh/daily/summary.json", "fresh/chain-summary.json")


def gate_verdicts(run: Path, source: str) -> dict[str, bool]:
    """Each native release gate's verdict from one finished leg, by the rules the release lane
    records them with: a result counts only when the leg itself graded it (it is under
    ``grades``), and an absent result is a failed gate, never a pass. The lane's ``gate`` record
    re-checks each result against its recorded sha256, so this reader does not hash again. An
    unreadable leg.json raises."""
    leg = json.loads((run / "leg.json").read_bytes())

    def graded(name: str) -> dict[str, Any]:
        path = run / name
        return (
            json.loads(path.read_bytes())
            if name in leg.get("grades", {}) and path.is_file()
            else {}
        )

    native = leg["state"] == "DONE" and leg["source"] == source and leg.get("stop") is None
    native = native and all(graded(name).get("held") is True for name in GRADES)
    budget = graded("fresh/budget.json")
    found = {f"native-{leg['host']}": native}
    found["resource-budget"] = (
        native and budget.get("held") is True and budget.get("status") == "COMPLETE"
    )
    if leg.get("previous"):
        path = run / "acceptance-summary.json"
        summary = json.loads(path.read_bytes()) if path.is_file() else {}
        upgrade = summary.get("upgrade") or {}
        # The acceptance owner's whole verdict (held: the upgrade and the released cancel), on
        # this candidate, with each part it is made of still read.
        found["upgrade-axis"] = summary.get("candidate_sha") == source and all(
            (
                summary.get("held") is True,
                upgrade.get("held") is True,
                upgrade.get("book_verdict") == "ALL EQUAL",
                (upgrade.get("released_cancel") or {}).get("held") is True,
            )
        )
    return found
