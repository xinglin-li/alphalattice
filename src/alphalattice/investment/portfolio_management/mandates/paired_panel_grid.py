"""The Portfolio grid a paired panel study runs, defined once (OW10).

Alpha's authoring check admits a paired panel request only when its Portfolio stage is one of
these grids, and the Portfolio lab runs the grid the request names, so what may be submitted and
what runs cannot differ. They live with the Portfolio Manager, a peer of both: Alpha may not
import the lab, which imports Alpha.
"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True, slots=True)
class PairedPanelPortfolioGrid:
    """One admissible Portfolio stage of a paired panel study.

    Attributes:
        risk_method_ids: The Risk surfaces the stage compares.
        portfolio_policy_ids: The policies it runs, in order.
        sector_capacities: The sector caps it compares; None is uncapped.
        cost_stress_bps: The transaction cost arms, in basis points.
        top_k: The names a policy may hold.
        name_cap: The largest weight one name may take.
        risk_aversion_grid: The optimizer's risk aversions the inner selection searches.
        turnover_regularization_grid: Its turnover penalties.
        alpha_utility_method_id: How a score enters the optimizer's utility.
        fixed_score_method_id: The fixed score every arm is paired against.
    """

    risk_method_ids: tuple[str, ...]
    portfolio_policy_ids: tuple[str, ...]
    sector_capacities: tuple[float | None, ...]
    cost_stress_bps: tuple[int, ...]
    top_k: int
    name_cap: float
    risk_aversion_grid: tuple[float, ...]
    turnover_regularization_grid: tuple[float, ...]
    alpha_utility_method_id: str
    fixed_score_method_id: str


PAIRED_PANEL_GRID_FIELDS = tuple(field.name for field in fields(PairedPanelPortfolioGrid))

PAIRED_PANEL_PORTFOLIO_GRID = PairedPanelPortfolioGrid(
    risk_method_ids=("R0", "R1"),
    portfolio_policy_ids=("TOP_K_EQUAL_WEIGHT", "MINIMUM_VARIANCE", "TOP_K_SCORE_RISK_COST"),
    sector_capacities=(None, 0.2, 0.08),
    cost_stress_bps=(0, 2, 5, 10, 20),
    top_k=100,
    name_cap=0.02,
    risk_aversion_grid=(100.0, 1000.0, 10000.0),
    turnover_regularization_grid=(0.001, 0.1, 10.0),
    alpha_utility_method_id="DIMENSIONLESS_SCORE_UTILITY",
    fixed_score_method_id="MOMENTUM_252_21_SIGNED_CROSS_SECTION",
)
"""The paired study's whole grid: two Risk surfaces, three policies, three sector caps."""

RANK_BUFFERED_TURNOVER_SUCCESSOR_GRID = PairedPanelPortfolioGrid(
    risk_method_ids=("R0",),
    portfolio_policy_ids=("RANK_BUFFERED_SCORE_RISK_COST",),
    sector_capacities=(),
    cost_stress_bps=(2, 5, 10, 20),
    top_k=100,
    name_cap=0.02,
    risk_aversion_grid=(100.0,),
    turnover_regularization_grid=(10.0,),
    alpha_utility_method_id="DIMENSIONLESS_SCORE_UTILITY",
    fixed_score_method_id="MOMENTUM_252_21_SIGNED_CROSS_SECTION",
)
"""The turnover successor's one arm: R0, the rank-buffered policy, one point of each grid."""


__all__ = [
    "PAIRED_PANEL_GRID_FIELDS",
    "PAIRED_PANEL_PORTFOLIO_GRID",
    "RANK_BUFFERED_TURNOVER_SUCCESSOR_GRID",
    "PairedPanelPortfolioGrid",
]
