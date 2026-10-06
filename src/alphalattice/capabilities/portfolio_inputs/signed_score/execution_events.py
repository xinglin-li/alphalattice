"""Project execution-owner schedule rows into Portfolio execution events."""

from __future__ import annotations

from datetime import date, datetime
from typing import Protocol, cast

import pyarrow as pa

from alphalattice.foundation.causal_outcomes.execution.methods import (
    ExecutionOutcomeMethodSeal,
)

from .contracts import PortfolioExecutionEvents


class ExecutionScheduleAuthority(Protocol):
    def resolve_method_seal(self, snapshot_hash: str) -> ExecutionOutcomeMethodSeal: ...


def manifest_snapshot_hash(manifest_ref: str) -> str:
    """Extract an exact lowercase SHA-256 handle from a manifest reference.

    Args:
        manifest_ref: Owner manifest handle whose final component is a hash with optional .json
            suffix.

    Returns:
        The validated 64-character snapshot identity.

    Raises:
        ValueError: The final reference component is not a lowercase SHA-256 identity.
    """
    value = manifest_ref.rsplit("/", maxsplit=1)[-1].removesuffix(".json")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("portfolio_inputs.execution_manifest_ref_unreadable")
    return value


def execution_events_from_schedule(
    *,
    table: pa.Table,
    authority: ExecutionScheduleAuthority,
    manifest_ref: str,
    sessions: tuple[date, ...],
) -> PortfolioExecutionEvents:
    """Bind exact owner rows and method seal; never reconstruct session offsets."""
    columns = {
        name: table[name].to_pylist()
        for name in (
            "formation_session",
            "formation_close_at",
            "entry_open_at",
            "holding_end_open_at",
        )
    }
    by_formation: dict[date, tuple[datetime, datetime, datetime]] = {}
    for index, raw_formation in enumerate(columns["formation_session"]):
        formation = cast(date, raw_formation)
        events = cast(
            tuple[datetime, datetime, datetime],
            (
                columns["formation_close_at"][index],
                columns["entry_open_at"][index],
                columns["holding_end_open_at"][index],
            ),
        )
        if by_formation.setdefault(formation, events) != events:
            raise ValueError("portfolio_inputs.execution_events_disagree")
    try:
        ordered = tuple(by_formation[value] for value in sessions)
    except KeyError as error:
        raise ValueError("portfolio_inputs.execution_events_session_absent") from error
    snapshot_hash = manifest_snapshot_hash(manifest_ref)
    seal = authority.resolve_method_seal(snapshot_hash)
    binding = seal.binding if seal.disposition == "METHOD_BOUND" else None
    return PortfolioExecutionEvents.create(
        outcome_snapshot_hash=snapshot_hash,
        outcome_manifest_ref=manifest_ref,
        method_seal_disposition=seal.disposition,
        method_binding_hash=None if binding is None else binding.binding_hash,
        execution_recipe_id=None if binding is None else binding.recipe_id,
        execution_recipe_hash=None if binding is None else binding.recipe_hash,
        ordered_formation_sessions=sessions,
        decision_at=tuple(value[0] for value in ordered),
        entry_at=tuple(value[1] for value in ordered),
        exit_at=tuple(value[2] for value in ordered),
        session_close_at={
            session: value[0] for session, value in zip(sessions, ordered, strict=True)
        },
    )


__all__ = ["execution_events_from_schedule", "manifest_snapshot_hash"]
