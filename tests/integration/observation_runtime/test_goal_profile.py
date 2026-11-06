"""Windowed occupancy is derived from kept observations, never inferred causality."""

from datetime import UTC, datetime, timedelta
from importlib import import_module

import pytest


def _inputs():
    start = datetime(2026, 10, 1, tzinfo=UTC)

    def at(seconds):
        return (start + timedelta(seconds=seconds)).isoformat()

    rpc = [
        {
            "source": "host",
            "operation_ref": "r",
            "phase": phase,
            "at": at(t),
            "clock_domain": "UTC",
            "provenance": f"row-{t}",
        }
        for phase, t in (("REQUESTED", 0), ("RETURNED", 5))
    ]
    phases = [
        {
            "namespace": "research",
            "task_id": "t",
            "execution_id": "attempt-1",
            "stage_id": "stage",
            "phase": phase,
            "started_at": at(t),
            "wall_seconds": seconds,
            "clock_domain": "UTC",
            "provenance": f"phase-{t}",
        }
        for phase, t, seconds in (("execute", 2, 5), ("verify", 8, 2))
    ]
    return start, at, rpc, phases


def test_profile_unions_clips_and_charges_distinct_attempts_once():
    """Overlap, repeated readouts and retries cannot inflate recorded Host occupancy."""
    profile = import_module("alphalattice.control.observation_runtime.profile").stage_profile
    _, at, rpc, phases = _inputs()
    args = dict(window=(at(0), at(12)), window_source="GOAL_TERMINAL", rpc=rpc, phases=phases)
    result = profile(**args)
    assert result["coverage"] == {
        "status": "PARTIAL",
        "rpc_seconds": 5,
        "phase_seconds": 7,
        "overlap_seconds": 3,
        "host_seconds": 9,
        "unattributed_seconds": 3,
    }
    assert result["stages"][0]["execute_seconds"] == 5
    assert result["stages"][0]["verify_seconds"] == 2
    assert result["stages"][0]["attempts"] == ["attempt-1"]
    assert result["window"]["source"] == "GOAL_TERMINAL"
    assert result["cache_counts"]["status"] == "UNOBSERVED"
    assert "critical_path" not in result and result["critical_path_unavailable_reason"]
    assert profile(**args) == result
    expired = {**phases[0], "execution_id": "old-attempt", "started_at": at(-10)}
    assert profile(**{**args, "phases": [*phases, expired]})["stages"] == result["stages"]
    args["phases"] = phases + phases + [{**phases[0], "execution_id": "attempt-2"}]
    duplicate = profile(**args)
    assert duplicate["coverage"] == result["coverage"]
    assert duplicate["stages"][0]["attempts"] == ["attempt-1", "attempt-2"]
    assert len(duplicate["stages"][0]["provenance"]) == 2
    args["window"] = (at(3), at(9))
    assert profile(**args)["coverage"]["host_seconds"] == 5
    missing = profile(
        window=(at(0), at(12)),
        window_source="GOAL_TERMINAL",
        rpc=[],
        phases=[],
        diagnostics=[{"code": "PHASE_ROWS_UNREADABLE", "source": "phase-file"}],
    )
    for key in ("rpc_seconds", "phase_seconds", "host_seconds", "overlap_seconds"):
        value = missing["coverage"][key]
        assert isinstance(value, dict), key
        assert value["status"] == "UNOBSERVED" and value["reason"] and value["provenance"]
    execute = profile(
        window=(at(0), at(12)), window_source="GOAL_TERMINAL", rpc=rpc, phases=phases[:1]
    )
    assert execute["stages"][0]["verify_seconds"]["status"] == "UNOBSERVED"
    for point in (0, 8, 12):
        zero = profile(
            window=(at(0), at(12)),
            window_source="GOAL_TERMINAL",
            rpc=rpc,
            phases=[phases[0], {**phases[1], "started_at": at(point), "wall_seconds": 0}],
        )
        assert zero["stages"][0]["verify_seconds"] == 0


@pytest.mark.parametrize(
    "fault", ["return", "endpoint", "conflict", "clock", "unreadable", "scope"]
)
def test_missing_or_ambiguous_evidence_never_claims_complete(fault):
    """Missing endpoints, conflicting duplicates, bad rows and clocks remain named gaps."""
    profile = import_module("alphalattice.control.observation_runtime.profile").stage_profile
    _, at, rpc, phases = _inputs()
    diagnostics = []
    if fault == "return":
        rpc.pop()
    elif fault == "endpoint":
        phases[0]["wall_seconds"] = None
    elif fault == "conflict":
        phases.append({**phases[0], "wall_seconds": 6})
    elif fault == "clock":
        phases[0]["clock_domain"] = "simulated"
    elif fault == "scope":
        phases[0]["stage_id"] = []
    else:
        diagnostics.append({"code": "PHASE_ROWS_UNREADABLE", "source": "kept-file"})
    result = profile(
        window=(at(0), at(12)),
        window_source="GOAL_TERMINAL",
        rpc=rpc,
        phases=phases,
        diagnostics=diagnostics,
    )
    expected = {
        "return": "UNPAIRED_RPC",
        "endpoint": "UNKEPT_OR_INTERRUPTED_PHASE",
        "conflict": "CONFLICTING_PHASE",
        "clock": "CLOCK_DOMAINS_UNALIGNED",
        "unreadable": "PHASE_ROWS_UNREADABLE",
        "scope": "PHASE_ROW_UNREADABLE",
    }[fault]
    assert expected in {gap["code"] for gap in result["gaps"]}
    assert result["coverage"]["status"] == "PARTIAL"
    assert "critical_path" not in result
