"""Prove a fifth portfolio policy needs no edit to the walk-forward engine.

The four installed families now reach the deterministic path engine through
`PortfolioPolicyCatalog`. A researcher adding a policy writes one adapter module
and installs it; `evaluation/walk_forward.py`, the search engine, the backtesting
owner, and publication stay untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from alphalattice.capabilities.portfolio_backtesting.contracts import PortfolioTargetDecision
from alphalattice.investment.portfolio_strategy_lab.contracts import (
    PortfolioScoreMode,
    TopKEqualWeightPolicy,
)
from alphalattice.investment.portfolio_strategy_lab.evaluation.walk_forward import (
    PortfolioPolicyDecisionProvider,
)
from alphalattice.investment.portfolio_strategy_lab.optimizer.service import PortfolioOptimizer
from alphalattice.investment.portfolio_strategy_lab.policies.catalog import (
    PortfolioPolicyCatalog,
    build_installed_portfolio_policy_catalog,
)
from alphalattice.investment.portfolio_strategy_lab.policies.contracts import (
    BoundPolicyDecisionInput,
    PortfolioPolicyAdapterBinding,
    PortfolioPolicyRecipe,
    portfolio_adapter_implementation_hash,
)

type FloatArray = NDArray[np.float64]
type BoolArray = NDArray[np.bool_]

_LISTINGS = ("listing-a", "listing-b", "listing-c", "listing-d")
_SESSIONS = (date(2024, 3, 1), date(2024, 3, 2))


@dataclass(frozen=True, slots=True)
class _StubWorkspace:
    """Minimal workspace satisfying the walk-forward decision surface."""

    scores: dict[tuple[str, PortfolioScoreMode], FloatArray]
    covariances: FloatArray
    decision_eligible: BoolArray
    formation_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]
    sector_exposure_matrix: FloatArray
    equal_weight_sector_exposure: FloatArray


def _workspace(candidate_id: str = "ridge") -> _StubWorkspace:
    count = len(_LISTINGS)
    scores: FloatArray = np.asarray([[0.9, 0.4, 0.2, 0.1], [0.3, 0.8, 0.5, 0.2]], dtype=np.float64)
    covariances: FloatArray = np.stack([np.eye(count, dtype=np.float64) * 0.04 for _ in _SESSIONS])
    eligible: BoolArray = np.ones((len(_SESSIONS), count), dtype=np.bool_)
    sector_matrix: FloatArray = np.asarray(
        [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]], dtype=np.float64
    )
    return _StubWorkspace(
        scores={(candidate_id, PortfolioScoreMode.STOCK_ONLY): scores},
        covariances=covariances,
        decision_eligible=eligible,
        formation_sessions=_SESSIONS,
        ordered_listing_ids=_LISTINGS,
        sector_exposure_matrix=sector_matrix,
        equal_weight_sector_exposure=np.asarray([0.5, 0.5], dtype=np.float64),
    )


@dataclass(frozen=True, slots=True)
class _InverseVolatilityPolicy:
    """A fifth policy specification owned entirely by this case study."""

    top_k: int
    maximum_weight: float

    @property
    def policy_id(self) -> str:
        return "CASE_STUDY_INVERSE_VOLATILITY"


class _InverseVolatilityAdapter:
    policy_id = "CASE_STUDY_INVERSE_VOLATILITY"
    solver_backed = False

    def describe_adapter_binding(self) -> PortfolioPolicyAdapterBinding:
        """The statement every adapter makes about itself, measured from this file.

        An adapter installed from outside the Desk names the package its source
        sits under, so its content identity is measured exactly as a built-in
        adapter's is. The contract gained this method at `c6b3e6b8`; a stub that
        predates it is not a fifth policy, it is a half of one.
        """

        return PortfolioPolicyAdapterBinding.create(
            policy_id=self.policy_id,
            adapter_implementation_hash=portfolio_adapter_implementation_hash(
                ("tests", Path(__file__))
            ),
            recipe_schema_id="CASE_STUDY_INVERSE_VOLATILITY",
            solver_semantics="CLOSED_FORM_INVERSE_VOLATILITY_NO_OPTIMIZER",
            deterministic_policy={
                "selection": "stable-score-top-k",
                "allocation": "inverse-covariance-diagonal-volatility-capped-renormalised",
            },
            input_consumption_semantics="RISK_FORECAST_REQUIRED_NO_OPTIMIZER",
        )

    def decide(
        self,
        *,
        policy: PortfolioPolicyRecipe,
        inputs: BoundPolicyDecisionInput,
        optimizer: PortfolioOptimizer,
    ) -> PortfolioTargetDecision:
        del optimizer
        variance = np.diag(inputs.covariance).astype(np.float64)
        eligible = inputs.decision_eligible
        weights = np.zeros_like(inputs.reference_weights)
        inverse = np.where(eligible, 1.0 / np.sqrt(np.maximum(variance, 1e-12)), 0.0)
        order = np.argsort(-inputs.scores * eligible, kind="stable")[: policy.top_k]
        selected = np.zeros_like(inverse)
        selected[order] = inverse[order]
        total = float(selected.sum())
        if total <= 0.0:
            raise ValueError("CASE_STUDY_INVERSE_VOLATILITY_INFEASIBLE")
        weights = selected / total
        weights = np.minimum(weights, policy.maximum_weight)
        weights = weights / float(weights.sum())
        weights.setflags(write=False)
        return PortfolioTargetDecision(
            target_weights=weights,
            predicted_variance=float(weights @ inputs.covariance @ weights),
        )


def test_a_fifth_policy_runs_through_the_unmodified_decision_provider() -> None:
    catalog = PortfolioPolicyCatalog(
        (*build_installed_portfolio_policy_catalog().adapters, _InverseVolatilityAdapter())
    )
    workspace = _workspace()
    provider = PortfolioPolicyDecisionProvider(
        workspace=workspace,  # type: ignore[arg-type]
        candidate_id="ridge",
        score_mode=PortfolioScoreMode.STOCK_ONLY,
        policy=_InverseVolatilityPolicy(top_k=2, maximum_weight=0.6),  # type: ignore[arg-type]
        policies=catalog,
    )
    decision = provider(
        formation_index=0,
        reference_weights=np.zeros(len(_LISTINGS), dtype=np.float64),
    )

    assert decision.target_weights.shape == (len(_LISTINGS),)
    assert float(decision.target_weights.sum()) == pytest.approx(1.0)
    assert decision.predicted_variance > 0.0
    # The engine never learned about this policy: it is absent from the Host catalog.
    assert "CASE_STUDY_INVERSE_VOLATILITY" not in (
        build_installed_portfolio_policy_catalog().policy_ids
    )


def test_installed_catalog_is_the_frozen_families_plus_explicit_appends() -> None:
    """The installed catalog retains frozen family positions and appends capabilities in order."""

    catalog = build_installed_portfolio_policy_catalog()
    frozen = (
        "TOP_K_EQUAL_WEIGHT",
        "TOP_K_MINIMUM_VARIANCE",
        "TOP_K_SCORE_RISK_COST",
        "SECTOR_DEVIATION_PENALTY",
    )
    assert catalog.policy_ids[: len(frozen)] == frozen
    assert catalog.policy_ids == (
        *frozen,
        "CURRENT_UNIVERSE_EQUAL_WEIGHT_EVERY_FORMATION",
        "STRATIFIED_TOP_K_EQUAL_WEIGHT",
        "RETURN_SCALED_TOTAL_SIGNAL_GLOBAL_QP",
        "RANK_BUFFERED_SCORE_RISK_COST",
        "WHOLE_BOOK_HYSTERESIS_EQUAL_WEIGHT",
        "WHOLE_BOOK_HYSTERESIS_INVERSE_VOLATILITY",
        "WHOLE_BOOK_HYSTERESIS_CAUSAL_RANK_MU_DIAGONAL_TILT",
    )
    assert {
        value.policy_id: value.solver_backed for value in catalog.binding.ordered_capabilities
    } == {
        "TOP_K_EQUAL_WEIGHT": False,
        "TOP_K_MINIMUM_VARIANCE": True,
        "TOP_K_SCORE_RISK_COST": True,
        "SECTOR_DEVIATION_PENALTY": True,
        "CURRENT_UNIVERSE_EQUAL_WEIGHT_EVERY_FORMATION": False,
        "STRATIFIED_TOP_K_EQUAL_WEIGHT": False,
        "RETURN_SCALED_TOTAL_SIGNAL_GLOBAL_QP": True,
        "RANK_BUFFERED_SCORE_RISK_COST": True,
        "WHOLE_BOOK_HYSTERESIS_EQUAL_WEIGHT": False,
        "WHOLE_BOOK_HYSTERESIS_INVERSE_VOLATILITY": False,
        "WHOLE_BOOK_HYSTERESIS_CAUSAL_RANK_MU_DIAGONAL_TILT": False,
    }
    # The identity is over the *ordered* capabilities, which is why the append had
    # to be an append: the same eleven adapters in another order are a different
    # catalog, and published evidence names the hash.
    reordered = PortfolioPolicyCatalog(tuple(reversed(catalog.adapters)))
    assert set(reordered.policy_ids) == set(catalog.policy_ids)
    assert reordered.binding.catalog_hash != catalog.binding.catalog_hash


def test_frozen_policy_specifications_route_by_their_declared_identity() -> None:
    """`policy_id` is a property, so frozen artifact identity is untouched."""

    policy = TopKEqualWeightPolicy(top_k=50, maximum_weight=0.05)
    assert policy.policy_id == "TOP_K_EQUAL_WEIGHT"
    assert "policy_id" not in policy.model_dump(mode="json")
    adapter = build_installed_portfolio_policy_catalog().resolve(policy)
    assert adapter.policy_id == "TOP_K_EQUAL_WEIGHT"
    assert adapter.solver_backed is False


def test_uninstalled_policy_fails_closed_before_any_numerical_call() -> None:
    catalog = build_installed_portfolio_policy_catalog()
    with pytest.raises(ValueError, match="PORTFOLIO_POLICY_ADAPTER_NOT_INSTALLED"):
        catalog.resolve(_InverseVolatilityPolicy(top_k=2, maximum_weight=0.6))
