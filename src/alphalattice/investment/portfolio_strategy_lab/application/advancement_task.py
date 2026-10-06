"""One advancement adapter under the shared Task Control runner.

Seven work items, one per lane, in dependency order. That shape is the whole
recovery story: the runner executes the first item that is not yet VERIFIED, so
an interruption inside Risk resumes at Risk with Data, Feature and Alpha already
proved -- rather than restarting a walk that would refit models the workspace
already holds. A single-stage advancement could only ever resume by redoing
everything, which is exactly the double-fit this Gate forbids.

The adapter reads work; it never computes it. Every count in the sealed receipt
comes from the domain receipt its owner produced, and the overall disposition is
derived from those receipts rather than from what was requested.

No registry, runner, heartbeat, checkpoint or cancellation loop is defined here.
Task Control owns all of those; this file is the domain half of an existing seam.
"""

from __future__ import annotations

from collections.abc import Callable

from alphalattice.investment.portfolio_strategy_lab.application.advancement import (
    AdvancementLaneId,
    DomainLaneReceipt,
    OperationalReceipt,
)

ADVANCEMENT_TASK_KIND = "portfolio_public_watermark_advancement"
ADVANCEMENT_INPUT_SCHEMA = "portfolio-public-watermark-advancement-program"
ADVANCEMENT_EVIDENCE_KIND = "portfolio-public.lane-receipt"

RunLane = Callable[[AdvancementLaneId], DomainLaneReceipt]
SealOperationalReceipt = Callable[[], OperationalReceipt]
"""Evaluated when the advancement seals, not when the run is admitted.

A profile receipt describes what the work cost, so it can only be taken after the
work. Passing a value here instead of a callable would seal a measurement made
before a single lane had run.
"""


__all__ = [
    "ADVANCEMENT_EVIDENCE_KIND",
    "ADVANCEMENT_INPUT_SCHEMA",
    "ADVANCEMENT_TASK_KIND",
    "RunLane",
    "SealOperationalReceipt",
]
