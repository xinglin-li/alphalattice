"""What a Portfolio study may not declare yet, and why (V313, deferred with V309).

The catalog holds these rules, but a Portfolio study cannot run them now: each needs design of
its own, beyond this wave. The declaration refuses them already; this list says so where an
agent looks first, the Portfolio draft and `schema show`, so none is declared to be refused.
"""

from __future__ import annotations

from typing import Final

NOT_AVAILABLE: Final[tuple[dict[str, object], ...]] = (
    {
        "field": "portfolio.weight_rule",
        "values": ["mu.iv1", "mu.iv2"],
        "reason": (
            "The mu rules weigh by a causal rank-return curve over matured formations, and a "
            "Portfolio study's axis starts at its Alpha study's first scored session, so its "
            "first rebalance has no curve (`tranche_curve_not_available`); a warm-up from "
            "scores before the window is new design."
        ),
    },
    {
        "field": "portfolio.policy.family",
        "values": ["RETURN_SCALED_TOTAL_SIGNAL"],
        "reason": (
            "This family reads a per-formation market beta lane, which no kept code builds "
            "since the research campaign that used it retired."
        ),
    },
    {
        "field": "portfolio.policy.family",
        "values": ["RANK_BUFFERED_SCORE_RISK_COST", "STRATIFIED_TOP_K_EQUAL_WEIGHT"],
        "reason": (
            "The rank-buffered score/risk/cost and the stratified equal-weight recipes were "
            "a retired research campaign's (a fixed grid on normalized targets; the "
            "return-scaled family's band pool), outside the declaration a Portfolio study "
            "takes."
        ),
    },
)
"""Each rule a Portfolio study may not declare now: its declaration field, its values and
why; all three are deferred with V309's parameter search (their row is V313)."""

__all__ = ["NOT_AVAILABLE"]
