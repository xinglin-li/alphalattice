"""Verified fakes: the host facts a deterministic test's fake assumes, against observations.

A fake stands for a real host's behaviour (a Job that denies breakaway, a command line the host
accepts). Each fake declares the facts it assumes, keyed by host capability and observation key.
Observations of the real hosts, by model-free probes and native runs, are recorded with their
source and host version; the newest observation of a fact confirms or contradicts each assumption.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

_OBSERVATION = ("capability", "key", "value", "host_version", "observed_at", "source")


def _when(row: Mapping[str, Any]) -> datetime:
    """An observation's instant; a time without an offset cannot be ordered and is refused."""
    missing = [field for field in _OBSERVATION if field not in row]
    if missing:
        raise ValueError(f"host_facts.observation_incomplete:{','.join(missing)}")
    when = datetime.fromisoformat(str(row["observed_at"]).replace("Z", "+00:00"))
    if when.tzinfo is None:
        raise ValueError("host_facts.observed_at_without_offset")
    return when


def latest(observations: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    """The newest observation of each (capability, key), by instant."""
    ordered = sorted(observations, key=_when)
    return {(str(row["capability"]), str(row["key"])): row for row in ordered}


def _holds(assumed: object, observed: object) -> bool:
    """A listed assumption holds when every item was observed; any other must be equal."""
    if isinstance(assumed, list):
        return isinstance(observed, list) and set(assumed) <= set(observed)
    return assumed == observed


def verify_fakes(
    fakes: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    versions: Mapping[str, str],
) -> list[dict[str, Any]]:
    """One verdict per assumption, in declaration order; only VERIFIED proves a fake.

    CONTRADICTED: the newest observation disagrees. STALE: it agrees, but under another version
    than ``versions`` gives for its surface (the row's ``surface``, else its host): re-observe.
    UNPROVED: either version unknown. UNVERIFIED: never observed.
    """
    seen = latest(observations)
    verdicts: list[dict[str, Any]] = []
    for fake in fakes:
        for capability, assumptions in fake["assumes"].items():
            for key, assumed in assumptions.items():
                row = seen.get((capability, key))
                verdict = "UNVERIFIED"
                if row is not None and not _holds(assumed, row["value"]):
                    verdict = "CONTRADICTED"
                elif row is not None:
                    current = versions.get(row.get("surface") or capability.split(".", 1)[0])
                    verdict = "UNPROVED" if None in (current, row["host_version"]) else "STALE"
                    verdict = "VERIFIED" if current and row["host_version"] == current else verdict
                verdicts.append(
                    {
                        "fake": fake["fake"],
                        "capability": capability,
                        "key": key,
                        "assumed": assumed,
                        "observed": None if row is None else row["value"],
                        "source": None if row is None else row["source"],
                        "verdict": verdict,
                    }
                )
    return verdicts


def uncovered(
    fakes: Sequence[Mapping[str, Any]], observations: Sequence[Mapping[str, Any]]
) -> list[tuple[str, str]]:
    """Observed facts no fake assumes: host behaviour the deterministic tests do not model."""
    assumed = {(c, k) for fake in fakes for c, keys in fake["assumes"].items() for k in keys}
    return sorted(set(latest(observations)) - assumed)
