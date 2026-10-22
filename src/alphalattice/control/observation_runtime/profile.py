"""Windowed ledger and phase occupancy; elapsed unions never assert a causal path."""

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from math import isfinite
from typing import Any


def _stamp(value: Any) -> float:
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        raise ValueError("clock has no timezone")
    return stamp.timestamp()


def _union(intervals: Iterable[tuple[float, float]], lo: float, hi: float) -> float:
    end, total = lo, 0.0
    for start, stop in sorted(intervals):
        start, stop = max(lo, start), min(hi, stop)
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return round(total, 6)


def _window(window: tuple[str, str]) -> tuple[float, float]:
    lo, hi = map(_stamp, window)
    if hi < lo:
        raise ValueError("observation.profile_window_invalid")
    return lo, hi


def _phase(row: Mapping[str, Any]) -> tuple[float, float]:
    start, seconds = _stamp(row["started_at"]), float(row["wall_seconds"])
    if not isfinite(seconds) or seconds < 0:
        raise ValueError("phase lacks duration")
    return start, start + seconds


def _overlaps(start: float, stop: float, lo: float, hi: float) -> bool:
    """A recorded point on either boundary is observed; elapsed intervals must intersect."""
    return lo <= start <= hi if start == stop else start < hi and stop > lo


def align_clock_inputs(
    rpc: list[dict[str, Any]],
    phases: list[dict[str, Any]],
    correlations: Sequence[Mapping[str, Any]],
    gaps: list[dict[str, Any]],
) -> dict[str, float]:
    """Align projection copies to UTC using only their Host's recorded clock origin."""
    bindings: dict[str, tuple[float, str] | None] = {}
    for origin in correlations:
        reference = str(origin.get("clock_ref", ""))
        try:
            if not isinstance(origin.get("clock_ref"), str) or reference in bindings:
                raise ValueError("ambiguous clock reference")
            if origin["schema"] != "alphalattice.clock-correlation.v1":
                raise ValueError("unknown clock correlation")
            if (
                any(
                    type(origin[k]) not in (int, float) or not isfinite(origin[k])
                    for k in ("monotonic_anchor", "uncertainty_seconds")
                )
                or origin["uncertainty_seconds"] < 0
            ):
                raise ValueError("invalid sampling evidence")
            offset = _stamp(origin["system_anchor"]) - _stamp(origin["replay_anchor"])
            bindings[reference] = offset, origin["source"]
        except (ValueError, KeyError, TypeError, OverflowError, OSError):
            bindings[reference] = None
            gaps.append(
                {"code": "CLOCK_CORRELATION_UNREADABLE", "source": origin.get("provenance")}
            )
    for values, translated in ((rpc, True), (phases, False)):
        kept = []
        for row in values:
            code = (
                "CLOCK_MAPPING_DRIFT" if row.get("clock_drift") is True else "CLOCK_MAPPING_MISSING"
            )
            try:
                binding = bindings.get(row.get("clock_ref", ""))
                if binding is None or row.get("clock_drift") is not False:
                    raise ValueError("clock mapping unavailable")
                if translated:
                    if row["source"] != binding[1]:
                        raise ValueError("clock belongs to another Host")
                    row["at"] = datetime.fromtimestamp(
                        _stamp(row["at"]) + binding[0], UTC
                    ).isoformat()
                row["clock_domain"] = "UTC"
                kept.append(row)
            except (ValueError, KeyError, TypeError, OverflowError, OSError):
                gaps.append({"code": code, "source": row.get("provenance")})
        values[:] = kept
    return {key: value[0] for key, value in bindings.items() if value is not None}


def _rpcs(
    rpc: Sequence[Mapping[str, Any]], gaps: list[Any], domains: set[str]
) -> list[tuple[float, float]]:
    pairs: dict[tuple[str, str], dict[str, set[float]]] = defaultdict(lambda: defaultdict(set))
    rpc_intervals: list[tuple[float, float]] = []
    for row in rpc:
        try:
            pairs[(row["source"], row["operation_ref"])][row["phase"]].add(_stamp(row["at"]))
            domains.add(row.get("clock_domain", "UNOBSERVED"))
        except (ValueError, KeyError, TypeError):
            gaps.append({"code": "RPC_ROW_UNREADABLE", "source": row.get("provenance")})
    for rpc_key, ends in sorted(pairs.items()):
        if len(ends.get("REQUESTED", ())) != 1 or len(ends.get("RETURNED", ())) != 1:
            gaps.append({"code": "UNPAIRED_RPC", "source": list(rpc_key)})
            continue
        start, stop = next(iter(ends["REQUESTED"])), next(iter(ends["RETURNED"]))
        if stop < start:
            gaps.append({"code": "RPC_CLOCK_REVERSED", "source": list(rpc_key)})
        else:
            rpc_intervals.append((start, stop))
    return rpc_intervals


def stage_profile(
    *,
    window: tuple[str, str],
    window_source: str,
    rpc: Sequence[Mapping[str, Any]],
    phases: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Project scoped owner inputs once; absent or contradictory evidence remains a gap."""
    lo, hi = _window(window)
    gaps = [dict(row) for row in diagnostics]
    domains: set[str] = set()
    rpc_intervals = _rpcs(rpc, gaps, domains)
    phase_intervals: list[tuple[float, float]] = []
    unique: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in phases:
        key = tuple(
            row.get(k)
            for k in ("namespace", "task_id", "execution_id", "stage_id", "phase", "started_at")
        )
        if any(v is not None and not isinstance(v, str) for v in key):
            gaps.append({"code": "PHASE_ROW_UNREADABLE", "source": row.get("provenance")})
        else:
            unique[key].append(row)
    stages: dict[tuple[Any, ...], dict[str, set[Any]]] = defaultdict(lambda: defaultdict(set))
    for key, copies in sorted(unique.items(), key=lambda item: str(item[0])):
        row = copies[0]
        if any(other != row for other in copies[1:]):
            gaps.append({"code": "CONFLICTING_PHASE", "source": list(key)})
            continue
        try:
            start, stop = _phase(row)
            if not all(key[:5]) or row["phase"] not in {"execute", "verify"}:
                raise ValueError("phase lacks scope")
        except (ValueError, KeyError, TypeError):
            gaps.append({"code": "UNKEPT_OR_INTERRUPTED_PHASE", "source": list(key)})
            continue
        domains.add(row.get("clock_domain", "UNOBSERVED"))
        if not _overlaps(start, stop, lo, hi):
            continue
        interval = (start, stop)
        phase_intervals.append(interval)
        stage = stages[key[:2] + key[3:4]]
        stage[row["phase"]].add(interval)
        stage["attempts"].add(key[2])
        stage["provenance"].add(str(row.get("provenance", "UNOBSERVED")))
        if row.get("dropped_spans"):
            gaps.append({"code": "DROPPED_SPANS", "source": list(key)})
    if domains != {"UTC"}:
        gaps.append({"code": "CLOCK_DOMAINS_UNALIGNED", "source": sorted(domains)})
    rpc_seconds, phase_seconds = (_union(v, lo, hi) for v in (rpc_intervals, phase_intervals))
    covered = _union(rpc_intervals + phase_intervals, lo, hi)
    overlap = round(rpc_seconds + phase_seconds - covered, 6)

    def seen(value: float, rows: Any, present: Callable[[Any], bool] = bool) -> Any:
        if present(rows):
            return value
        return {
            "status": "UNOBSERVED",
            "reason": "NO_KEPT_INTERVALS",
            "provenance": {"window_source": window_source, "gaps": gaps},
        }

    return {
        "schema": "alphalattice.goal-profile.v1",
        "window": {"start": window[0], "end": window[1], "source": window_source},
        "stages": [
            {
                "namespace": key[0],
                "task_id": key[1],
                "stage_id": key[2],
                "execute_seconds": seen(_union(stage["execute"], lo, hi), stage["execute"]),
                "verify_seconds": seen(_union(stage["verify"], lo, hi), stage["verify"]),
                "attempts": sorted(stage["attempts"]),
                "provenance": sorted(stage["provenance"]),
            }
            for key, stage in stages.items()
        ],
        "coverage": {
            "status": "PARTIAL" if gaps or covered < hi - lo else "COMPLETE",
            "rpc_seconds": seen(rpc_seconds, rpc_intervals),
            "phase_seconds": seen(phase_seconds, phase_intervals),
            "overlap_seconds": seen(overlap, (rpc_intervals, phase_intervals), all),
            "host_seconds": seen(covered, (rpc_intervals, phase_intervals), any),
            "unattributed_seconds": round(hi - lo - covered, 6),
        },
        "gaps": gaps,
        "clock_domains": sorted(domains),
        "cache_counts": {"status": "UNOBSERVED", "reason": "NO_RECORDED_CACHE_COUNTS"},
        "critical_path_unavailable_reason": "CAUSAL_LINKS_NOT_RECORDED",
    }
