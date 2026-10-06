"""What a result's window can claim about time, stated from its Panel's recorded marks (V347).

The approximate point-in-time contract backfills the initial cohort before T0 and follows the
observed membership from T0; the Sector map and the provider's prices carry their own treatments.
A readback, a report or a CRO material whose window reaches before T0 says so. Each statement
here is generated from a typed mark the Panel records -- its `bootstrap_t0_session`, its initial
cohort, its `survivorship_bias_warning`, its `sector_history_treatment` -- and the workspace's
price basis, one template per mark value, so a new value (V346's Sector treatment, V345's price
basis) changes the statement with no carrier edited, and a value with no template is named rather
than dropped.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any, Final, Self

from pydantic import BaseModel, ConfigDict, Field

from alphalattice.foundation.market_data_ops.sources.membership import INITIAL_COHORT_BACKFILL
from alphalattice.kernel.shared_kernel.sector_treatment import (
    SECTOR_HISTORY_BACKFILLED,
    SECTOR_HISTORY_FORWARD,
)

_SECTOR: Final[Mapping[str, str]] = {
    SECTOR_HISTORY_BACKFILLED: (
        "Every session uses the Sector classification observed on {observed}{after_t0}; it is "
        "not point in time."
    ),
    SECTOR_HISTORY_FORWARD: (
        "Each listing uses the Sector classification first recorded for it on every session "
        "before its first reclassification, which is not point in time; {reclassified} "
        "reclassification(s) since {first_reclassified} each apply from the session whose data "
        "update observed it, and no published session moved."
    ),
}
"""The Sector treatment's statement, by the Panel's `sector_history_treatment`."""

_PRICE: Final[Mapping[str, str]] = {
    "split_adjusted": (
        "Prices and share volumes are the provider's, split-adjusted to the current share basis, "
        "and the adjusted close is back-adjusted for later splits and dividends; neither is the "
        "price as it read at each session."
    ),
    "unadjusted": (
        "Prices and share volumes are as traded; the projection applies only the splits known "
        "at each session."
    ),
}
"""The price basis's statement, by the market profile's `daily_price_basis`."""


class TemporalStatement(BaseModel):  # type: ignore[misc]
    """A window's temporal claims: the marks it rests on and the statements they generate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    t0_session: date | None
    """The last session of the workspace's first qualified Panel; none when the Panel records
    no membership."""
    initial_cohort_size: int | None = Field(default=None, ge=1)
    """The listings of the initial cohort, which stands in for every session before T0."""
    window_start: date | None
    window_end: date | None
    data_start: date | None
    """The first session the result's inputs read, its lookback and training included."""
    window_before_t0: bool | None
    data_before_t0: bool | None
    universe_basis: str
    """The membership promise before T0 (`INITIAL_COHORT_BACKFILL_NOT_POINT_IN_TIME`), or the
    Panel's universe scope when it records no T0."""
    survivorship_bias: bool
    sector_treatment: str
    sector_observed_on: date | None
    sector_reclassification_count: int = Field(default=0, ge=0)
    """The reclassifications its sessions read, each from its effective session (V346)."""
    sector_first_reclassified: date | None = None
    """The first of their effective sessions; none while no reclassification is in force."""
    price_basis: str | None
    statements: tuple[str, ...] = Field(min_length=1)

    @classmethod
    def from_panel(
        cls,
        safe_summary: Mapping[str, Any],
        *,
        window_start: date | None,
        window_end: date | None,
        data_start: date | None,
        price_basis: str | None,
    ) -> Self:
        """State a window's temporal claims from the Panel that fed it.

        Args:
            safe_summary: The Panel manifest's `safe_summary`.
            window_start: The first session the result claims (its evaluated window).
            window_end: The last.
            data_start: The first session its inputs read.
            price_basis: The workspace's market profile `daily_price_basis`.

        Returns:
            The marks and their statements.
        """
        membership = _mapping(safe_summary.get("membership"))
        risk = _mapping(safe_summary.get("temporal_risk"))
        policy = _mapping(safe_summary.get("universe_policy"))
        t0 = _date(membership.get("bootstrap_t0_session"))
        ranges: Sequence[object] = membership.get("basis_ranges") or []
        epochs: Sequence[object] = membership.get("epochs") or []
        backfilled = bool(ranges) and _mapping(ranges[0]).get("basis") == INITIAL_COHORT_BACKFILL
        cohort = int(_mapping(epochs[0])["member_count"]) if backfilled and epochs else None
        survivors = bool(policy.get("survivorship_bias_warning", True))
        sector = str(risk.get("sector_history_treatment") or "NOT_RECORDED")
        observed = _date(risk.get("sector_observed_at"))
        reclassified: Sequence[object] = (
            _mapping(safe_summary.get("lineage")).get("sector_reclassifications") or []
        )
        first_reclassified = min(
            (_date(_mapping(item).get("effective_session")) for item in reclassified),
            default=None,
            key=lambda value: value or date.max,
        )
        before = None if t0 is None or window_start is None else window_start < t0
        history_before = None if t0 is None or data_start is None else data_start < t0
        statements = [
            _membership_statement(
                t0=t0,
                cohort=cohort,
                window_start=window_start,
                data_start=data_start,
                before=before,
                history_before=history_before,
            ),
        ]
        if survivors:
            statements.append(
                "The cohort holds only the listings still listed when it was taken, so the "
                + ("history it stands in for" if t0 is not None else "whole history")
                + " carries survivorship bias."
            )
        statements.append(
            _templated(_SECTOR, sector, "Sector treatment").format(
                observed=observed.isoformat() if observed else "its latest refresh",
                after_t0=", the sessions after T0 included" if t0 is not None else "",
                reclassified=len(reclassified),
                first_reclassified=(
                    first_reclassified.isoformat() if first_reclassified else "the rule's start"
                ),
            )
        )
        statements.append(_templated(_PRICE, price_basis or "NOT_RECORDED", "price basis"))
        return cls(
            t0_session=t0,
            initial_cohort_size=cohort,
            window_start=window_start,
            window_end=window_end,
            data_start=data_start,
            window_before_t0=before,
            data_before_t0=history_before,
            universe_basis=(
                INITIAL_COHORT_BACKFILL
                if backfilled
                else str(risk.get("universe_temporal_scope") or "NOT_RECORDED")
            ),
            survivorship_bias=survivors,
            sector_treatment=sector,
            sector_observed_on=observed,
            sector_reclassification_count=len(reclassified),
            sector_first_reclassified=first_reclassified,
            price_basis=price_basis,
            statements=tuple(statements),
        )


def _membership_statement(
    *,
    t0: date | None,
    cohort: int | None,
    window_start: date | None,
    data_start: date | None,
    before: bool | None,
    history_before: bool | None,
) -> str:
    listings = f"the initial cohort of {cohort} listings" if cohort else "the initial cohort"
    if t0 is None:
        held = f"the cohort of {cohort} listings" if cohort else "the cohort"
        return f"The Panel records no T0: every session holds {held} it was built with."
    if before:
        return (
            f"The window starts {window_start}, before T0 ({t0}): until T0 every session holds "
            f"{listings}."
        )
    if history_before:
        return (
            f"The window starts at or after T0 ({t0}), but its inputs reach back to {data_start}, "
            f"and before T0 every session holds {listings}."
        )
    return (
        f"The window and its inputs start at or after T0 ({t0}): membership follows the entries "
        "and exits observed since."
    )


def _templated(table: Mapping[str, str], value: str, kind: str) -> str:
    template = table.get(value)
    if template is None:
        return f"The {kind} `{value}` has no installed statement; its mark is recorded above."
    return template


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _date(value: object) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        return date.fromisoformat(value[:10])
    return None


__all__ = ["TemporalStatement"]
