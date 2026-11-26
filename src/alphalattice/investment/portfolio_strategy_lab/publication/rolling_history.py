"""Append-only realized return extensions for one verified Portfolio report.

The rolling head keeps references to the original report and one decision-update
publication. Numerical history remains in its owners' sealed artifacts; this
module verifies the linkage and projects the joined return path without copying
publication bodies into a growing document.
"""

from __future__ import annotations

import math
from datetime import date
from itertools import pairwise
from typing import Literal, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.capabilities.portfolio_backtesting.metrics import (
    evaluate_net_simple_return_path,
)
from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (
    PortfolioDecisionCheckpoint,
    PortfolioObservedSettlement,
    PortfolioUpdatePublication,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash


class RollingHistoryError(ValueError):
    """Stable refusal for a broken rolling-report identity or return axis."""


class _RollingContract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class RollingReportObservation(_RollingContract):
    """One already-verified baseline net return, dated by formation."""

    formation_session: date
    net_simple_return: float = Field(gt=-1.0, allow_inf_nan=False)
    benchmark_simple_return: float | None = Field(default=None, gt=-1.0, allow_inf_nan=False)


class RollingReportBaseline(_RollingContract):
    """Verified REPORT inputs needed to extend its existing economic history.

    This is an in-process value, never a persisted copy of the REPORT series. The
    persisted head below carries only hashes and dates.
    """

    report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    program_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    strategy_package_id: str = Field(min_length=1, max_length=160)
    strategy_package_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    cost_bps_per_side: Literal[5, 10]
    formation_start: date
    formation_end: date
    baseline_observed_through: date
    base_last_holding_end_session: date
    first_parent_publication_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    baseline_claim: str = Field(min_length=1, max_length=240)
    observations: tuple[RollingReportObservation, ...] = Field(min_length=2)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def valid_baseline(self) -> Self:
        """Require one ordered formation axis and its observation boundary."""
        sessions = tuple(row.formation_session for row in self.observations)
        if (
            self.formation_start > self.formation_end
            or self.baseline_observed_through < self.formation_end
            or self.base_last_holding_end_session < self.formation_end
            or sessions != tuple(sorted(set(sessions)))
            or sessions[0] != self.formation_start
            or sessions[-1] != self.formation_end
        ):
            raise RollingHistoryError("portfolio_rolling.baseline_axis_invalid")
        return self


class RollingReportHead(_RollingContract):
    """Small sealed link to a baseline REPORT and one existing update publication."""

    kind: Literal["RollingPortfolioReportHead"] = "RollingPortfolioReportHead"
    report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    program_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    strategy_package_id: str = Field(min_length=1, max_length=160)
    strategy_package_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    cost_bps_per_side: Literal[5, 10]
    formation_start: date
    formation_end: date
    baseline_observed_through: date
    baseline_claim: str = Field(min_length=1, max_length=240)
    previous_head_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    previous_publication_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    increment_publication_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    increment_outcome_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    increment_outcome_count: int = Field(ge=0)
    first_increment_formation: date | None = None
    last_increment_formation: date | None = None
    increment_observed_through: date
    observed_through: date
    head_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal the normalized head under its own content hash."""
        values = {**values, "head_hash": "0" * 64}
        identity = cls.model_construct(**values).model_dump(mode="json", exclude={"head_hash"})
        return cls(**identity, head_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def identity_and_increment(self) -> Self:
        """Verify the head identity and its increment's dates and count."""
        if (
            (self.increment_outcome_count == 0)
            != (self.first_increment_formation is None and self.last_increment_formation is None)
            or (self.increment_outcome_count > 0)
            != (
                self.first_increment_formation is not None
                and self.last_increment_formation is not None
            )
            or (
                self.first_increment_formation is not None
                and self.last_increment_formation is not None
                and (
                    self.first_increment_formation <= self.formation_end
                    or self.first_increment_formation > self.last_increment_formation
                    or self.last_increment_formation > self.observed_through
                )
            )
            or self.observed_through < self.baseline_observed_through
            or self.observed_through
            != max(self.baseline_observed_through, self.increment_observed_through)
            or self.head_hash != canonical_hash(self.model_dump(mode="json", exclude={"head_hash"}))
        ):
            raise RollingHistoryError("portfolio_rolling.head_identity_invalid")
        return self


def append_rolling_report_head(
    *,
    baseline: RollingReportBaseline,
    previous_head: RollingReportHead | None,
    publication: PortfolioUpdatePublication,
    checkpoint: PortfolioDecisionCheckpoint,
) -> RollingReportHead:
    """Seal one publication reference without copying its numerical history.

    Empty increments are retained as small links too: they advance the verified
    observation watermark while leaving the realized return series unchanged.
    """
    _require_checkpoint(baseline, checkpoint)
    expected_parent = (
        baseline.first_parent_publication_hash
        if previous_head is None
        else previous_head.increment_publication_hash
    )
    if (
        publication.checkpoint_hash != checkpoint.history_hash
        or publication.parent_hash != expected_parent
        or (
            previous_head is not None
            and (
                previous_head.head_hash
                != canonical_hash(previous_head.model_dump(mode="json", exclude={"head_hash"}))
                or _head_baseline_fields(previous_head) != _baseline_head_fields(baseline)
                or publication.observed_through < previous_head.increment_observed_through
            )
        )
    ):
        raise RollingHistoryError("portfolio_rolling.publication_binding_invalid")
    outcomes = _eligible_outcomes(baseline, publication, checkpoint)
    return RollingReportHead.create(
        **_baseline_head_fields(baseline),
        previous_head_hash=None if previous_head is None else previous_head.head_hash,
        previous_publication_hash=expected_parent,
        increment_publication_hash=publication.content_hash,
        increment_outcome_hash=canonical_hash(tuple(event.content_hash for event in outcomes)),
        increment_outcome_count=len(outcomes),
        first_increment_formation=None if not outcomes else outcomes[0].formation_session,
        last_increment_formation=None if not outcomes else outcomes[-1].formation_session,
        increment_observed_through=publication.observed_through,
        observed_through=max(baseline.baseline_observed_through, publication.observed_through),
    )


def project_rolling_report(
    *,
    baseline: RollingReportBaseline,
    heads: tuple[RollingReportHead, ...],
    publications: tuple[PortfolioUpdatePublication, ...],
    checkpoint: PortfolioDecisionCheckpoint,
) -> dict[str, object]:
    """Verify and project a complete head chain over the verified baseline.

    The caller reopens REPORT, decision publications, and their ordinary store
    proofs before calling this function. This function checks that the supplied
    objects are the exact identities each compact head names.
    """
    _require_checkpoint(baseline, checkpoint)
    if len(heads) != len(publications):
        raise RollingHistoryError("portfolio_rolling.head_publication_count_mismatch")
    prior_head: RollingReportHead | None = None
    prior_publication_hash = baseline.first_parent_publication_hash
    outcomes: list[PortfolioObservedSettlement] = []
    for head, publication in zip(heads, publications, strict=True):
        if (
            head.head_hash != canonical_hash(head.model_dump(mode="json", exclude={"head_hash"}))
            or _head_baseline_fields(head) != _baseline_head_fields(baseline)
            or head.previous_head_hash != (None if prior_head is None else prior_head.head_hash)
            or head.previous_publication_hash != prior_publication_hash
            or head.increment_publication_hash != publication.content_hash
        ):
            raise RollingHistoryError("portfolio_rolling.head_chain_invalid")
        rebuilt = append_rolling_report_head(
            baseline=baseline,
            previous_head=prior_head,
            publication=publication,
            checkpoint=checkpoint,
        )
        if rebuilt != head:
            raise RollingHistoryError("portfolio_rolling.head_increment_mismatch")
        outcomes.extend(_eligible_outcomes(baseline, publication, checkpoint))
        prior_head, prior_publication_hash = head, publication.content_hash
    if (
        outcomes
        and outcomes[0].entry.schedule.entry_session != baseline.base_last_holding_end_session
    ):
        raise RollingHistoryError("portfolio_rolling.return_axis_overlap_or_gap")
    for earlier, later in pairwise(outcomes):
        if earlier.entry.schedule.holding_end_session != later.entry.schedule.entry_session:
            raise RollingHistoryError("portfolio_rolling.return_axis_overlap_or_gap")

    rows = list(baseline.observations)
    field = f"net_return_{baseline.cost_bps_per_side}bps"
    for event in outcomes:
        rows.append(
            RollingReportObservation(
                formation_session=event.formation_session,
                net_simple_return=float(getattr(event, field)),
                benchmark_simple_return=None,
            )
        )
    if tuple(row.formation_session for row in rows) != tuple(
        sorted({row.formation_session for row in rows})
    ):
        raise RollingHistoryError("portfolio_rolling.formation_axis_overlap")

    values: npt.NDArray[np.float64] = np.asarray(
        [row.net_simple_return for row in rows], dtype=np.float64
    )
    metrics: dict[str, float] = {}
    absences: dict[str, str] = {}
    metric_fields = (
        "cumulative_return",
        "annualized_return",
        "annualized_volatility",
        "maximum_drawdown",
        "sharpe",
        "sortino",
    )
    if len(values) < 2:
        absences.update((name, "INSUFFICIENT_REALIZED_OBSERVATIONS") for name in metric_fields)
    else:
        computed = evaluate_net_simple_return_path(net_simple_returns=values)
        for name in metric_fields:
            value = float(getattr(computed, name))
            if math.isfinite(value):
                metrics[name] = value
            else:
                absences[name] = (
                    "ZERO_DOWNSIDE_DEVIATION" if name == "sortino" else "NONFINITE_DERIVED_METRIC"
                )

    latest = publications[-1] if publications else None
    open_positions: list[dict[str, object]] = []
    if latest is not None:
        if latest.active_entry is not None:
            schedule = latest.active_entry.entry.schedule
            open_positions.append(
                {
                    "status": "ENTRY_SETTLED_OUTCOME_PENDING",
                    "formation_session": schedule.formation_session.isoformat(),
                    "entry_session": schedule.entry_session.isoformat(),
                    "holding_end_session": schedule.holding_end_session.isoformat(),
                }
            )
        if latest.pending_proposal is not None:
            schedule = latest.pending_proposal.schedule
            open_positions.append(
                {
                    "status": "PROPOSAL_NOT_YET_SETTLED",
                    "formation_session": schedule.formation_session.isoformat(),
                    "entry_session": schedule.entry_session.isoformat(),
                    "holding_end_session": schedule.holding_end_session.isoformat(),
                }
            )
    observed_through = (
        baseline.baseline_observed_through if not heads else heads[-1].observed_through
    )
    curve = []
    net_wealth = 100.0
    for row in rows:
        net_wealth *= 1.0 + row.net_simple_return
        curve.append(
            {
                "formation_session": row.formation_session.isoformat(),
                "net_simple_return": row.net_simple_return,
                "benchmark_simple_return": row.benchmark_simple_return,
                "value": net_wealth,
            }
        )
    return {
        "base_report_hash": baseline.report_hash,
        "base_result_hash": baseline.result_hash,
        "program_hash": baseline.program_hash,
        "strategy_package_id": baseline.strategy_package_id,
        "strategy_package_hash": baseline.strategy_package_hash,
        "cost_bps_per_side": baseline.cost_bps_per_side,
        "formation_range": {
            "start": baseline.formation_start.isoformat(),
            "end": rows[-1].formation_session.isoformat(),
            "base_end": baseline.formation_end.isoformat(),
        },
        "outcomes_observed_through": observed_through.isoformat(),
        "claim": baseline.baseline_claim,
        "execution_basis": "DAILY_BAR_QA_NOT_VERIFIED_VENUE_EXECUTION",
        "metrics": metrics,
        "metric_absences": absences,
        "curve": curve,
        "observation_count": len(rows),
        "open_positions": open_positions,
        "head_hash": None if not heads else heads[-1].head_hash,
        "outcome_count": len(outcomes),
    }


def _require_checkpoint(
    baseline: RollingReportBaseline, checkpoint: PortfolioDecisionCheckpoint
) -> None:
    if (
        checkpoint.history_hash != baseline.checkpoint_hash
        or checkpoint.package.strategy_id != baseline.strategy_package_id
        or checkpoint.package.package_hash != baseline.strategy_package_hash
    ):
        raise RollingHistoryError("portfolio_rolling.foreign_checkpoint_or_package")


def _eligible_outcomes(
    baseline: RollingReportBaseline,
    publication: PortfolioUpdatePublication,
    checkpoint: PortfolioDecisionCheckpoint,
) -> tuple[PortfolioObservedSettlement, ...]:
    positions = {session: index for index, session in enumerate(checkpoint.formation_sessions)}
    found = []
    for event in publication.events:
        if event.phase != "OUTCOME_SETTLED" or event.formation_session <= baseline.formation_end:
            continue
        schedule = event.entry.schedule
        if event.formation_session not in positions:
            raise RollingHistoryError("portfolio_rolling.outcome_axis_or_cost_invalid")
        formation = positions[event.formation_session]
        if (
            event.formation_session > publication.observed_through
            or schedule.formation_session != event.formation_session
            or schedule.actual_session_span != 2
            or positions.get(schedule.entry_session) != formation + 1
            or positions.get(schedule.holding_end_session) != formation + 2
            or schedule.holding_end_session > publication.observed_through
            or getattr(event, f"net_return_{baseline.cost_bps_per_side}bps") is None
        ):
            raise RollingHistoryError("portfolio_rolling.outcome_axis_or_cost_invalid")
        found.append(event)
    if tuple(event.formation_session for event in found) != tuple(
        sorted({event.formation_session for event in found})
    ):
        raise RollingHistoryError("portfolio_rolling.increment_axis_invalid")
    return tuple(found)


def _baseline_head_fields(baseline: RollingReportBaseline) -> dict[str, object]:
    return {
        "report_hash": baseline.report_hash,
        "result_hash": baseline.result_hash,
        "program_hash": baseline.program_hash,
        "checkpoint_hash": baseline.checkpoint_hash,
        "strategy_package_id": baseline.strategy_package_id,
        "strategy_package_hash": baseline.strategy_package_hash,
        "cost_bps_per_side": baseline.cost_bps_per_side,
        "formation_start": baseline.formation_start,
        "formation_end": baseline.formation_end,
        "baseline_observed_through": baseline.baseline_observed_through,
        "baseline_claim": baseline.baseline_claim,
    }


def _head_baseline_fields(head: RollingReportHead) -> dict[str, object]:
    return {
        "report_hash": head.report_hash,
        "result_hash": head.result_hash,
        "program_hash": head.program_hash,
        "checkpoint_hash": head.checkpoint_hash,
        "strategy_package_id": head.strategy_package_id,
        "strategy_package_hash": head.strategy_package_hash,
        "cost_bps_per_side": head.cost_bps_per_side,
        "formation_start": head.formation_start,
        "formation_end": head.formation_end,
        "baseline_observed_through": head.baseline_observed_through,
        "baseline_claim": head.baseline_claim,
    }


__all__ = [
    "RollingHistoryError",
    "RollingReportBaseline",
    "RollingReportHead",
    "RollingReportObservation",
    "append_rolling_report_head",
    "project_rolling_report",
]
