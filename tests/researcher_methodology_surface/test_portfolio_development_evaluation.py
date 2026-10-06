"""A Portfolio policy installed from outside the catalog, and what its binding covers.

An adapter's implementation identity covers the code that decides its weights, so a
rewrite under a stable id moves its binding; the capability here is one the installed
catalog does not carry, so what is proven is extension rather than configuration. A
declared axis admits values exactly as given. (The lab's Optuna search and its walk-forward
driver retired with the lab, RT R08.)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from alphalattice.capabilities.portfolio_backtesting.contracts import PortfolioTargetDecision
from alphalattice.investment.portfolio_strategy_lab.optimizer import service as optimizer_service
from alphalattice.investment.portfolio_strategy_lab.optimizer.service import PortfolioOptimizer
from alphalattice.investment.portfolio_strategy_lab.policies.catalog import (
    PortfolioPolicyCatalog,
    build_installed_portfolio_policy_catalog,
)
from alphalattice.investment.portfolio_strategy_lab.policies.contracts import (
    PORTFOLIO_POLICY_PACKAGE,
    BoundPolicyDecisionInput,
    PortfolioPolicyAdapterBinding,
    PortfolioPolicyRecipe,
    portfolio_adapter_implementation_hash,
)
from alphalattice.investment.portfolio_strategy_lab.policies.search_domain import SearchAxis
from alphalattice.kernel.shared_kernel.source_identity import (
    checkout_root,
    number_deciding_closure,
    number_deciding_rule,
)

CASE_PACKAGE = "researcher_methodology_surface"
"""This case study is the adapter's package for content identity purposes.

An out-of-tree adapter states its own package rather than borrowing the Desk's:
claiming ``portfolio_strategy_lab`` would describe this file as something it is
not, and the component id is part of the identity.
"""

type FloatArray = NDArray[np.float64]
type BoolArray = NDArray[np.bool_]

"""Twenty-four formations, which is a floor rather than a preference.

The path metrics run a moving-block bootstrap with a block size of 21, so a
shorter series indexes past the end of its own circular buffer. Small enough to
stay fast, long enough that the engine has a path to measure.
"""

_POLICY_ID = "CASE_STUDY_INVERSE_VOLATILITY"


@dataclass(frozen=True, slots=True)
class _InverseVolatilityPolicy:
    """A fifth policy specification owned entirely by this case study."""

    top_k: int
    maximum_weight: float

    @property
    def policy_id(self) -> str:
        return _POLICY_ID


class _InverseVolatilityAdapter:
    policy_id = _POLICY_ID
    solver_backed = False

    def describe_adapter_binding(self) -> PortfolioPolicyAdapterBinding:
        return PortfolioPolicyAdapterBinding.create(
            policy_id=self.policy_id,
            adapter_implementation_hash=portfolio_adapter_implementation_hash(
                (CASE_PACKAGE, Path(__file__))
            ),
            recipe_schema_id=_POLICY_ID,
            solver_semantics="CLOSED_FORM_INVERSE_VOLATILITY",
            deterministic_policy={"weighting": "inverse-volatility"},
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
        inverse = np.where(eligible, 1.0 / np.sqrt(np.maximum(variance, 1e-12)), 0.0)
        order = np.argsort(-inputs.scores * eligible, kind="stable")[: policy.top_k]
        selected = np.zeros_like(inverse)
        selected[order] = inverse[order]
        total = float(selected.sum())
        if total <= 0.0:
            raise ValueError("CASE_STUDY_INVERSE_VOLATILITY_INFEASIBLE")
        weights = np.minimum(selected / total, policy.maximum_weight)
        weights = weights / float(weights.sum())
        weights.setflags(write=False)
        return PortfolioTargetDecision(
            target_weights=weights,
            predicted_variance=float(weights @ inputs.covariance @ weights),
        )


class _SolverBackedInverseVolatilityAdapter:
    """The same ``policy_id``, a genuinely different implementation.

    It delegates to the Desk optimizer instead of weighting in closed form, so it
    computes different weights and its declared source closure is this file *plus*
    the optimizer. That is exactly the case a name-based identity cannot see: same
    id, different code, different numbers, and -- before an adapter binding
    existed -- the same catalog identity.
    """

    policy_id = _POLICY_ID
    solver_backed = True

    def describe_adapter_binding(self) -> PortfolioPolicyAdapterBinding:
        return PortfolioPolicyAdapterBinding.create(
            policy_id=self.policy_id,
            adapter_implementation_hash=portfolio_adapter_implementation_hash(
                # Two packages, because the closure genuinely spans two: this
                # adapter is owned by the case study and the optimizer is not.
                (CASE_PACKAGE, Path(__file__)),
                (PORTFOLIO_POLICY_PACKAGE, Path(optimizer_service.__file__)),
            ),
            recipe_schema_id=_POLICY_ID,
            solver_semantics="CONVEX_MINIMUM_VARIANCE",
            deterministic_policy={"weighting": "solver"},
        )

    def decide(
        self,
        *,
        policy: PortfolioPolicyRecipe,
        inputs: BoundPolicyDecisionInput,
        optimizer: PortfolioOptimizer,
    ) -> PortfolioTargetDecision:
        solution = optimizer.minimum_variance(
            scores=inputs.scores,
            covariance=inputs.covariance,
            reference_weights=inputs.reference_weights,
            decision_eligible=inputs.decision_eligible,
            top_k=policy.top_k,
            maximum_weight=policy.maximum_weight,
        )
        return PortfolioTargetDecision(
            target_weights=solution.weights,
            predicted_variance=solution.predicted_variance,
        )


def _catalog(adapter: object | None = None) -> PortfolioPolicyCatalog:
    return PortfolioPolicyCatalog(
        (
            *build_installed_portfolio_policy_catalog().adapters,
            adapter or _InverseVolatilityAdapter(),  # type: ignore[arg-type]
        )
    )


def test_an_adapter_rewritten_under_a_stable_id_moves_its_binding() -> None:
    """requirement: replacing an adapter's code must move its identity.

    ``PortfolioPolicyCatalogBinding`` binds ``policy_id`` and ``solver_backed``
    and nothing else, so an adapter could be rewritten completely -- different
    objective, different weights -- while the catalog identity stood still. That
    contract is left alone because frozen artifacts decode it; the adapter binding
    is where content identity lives now.
    """

    original = _InverseVolatilityAdapter().describe_adapter_binding()
    rewritten = _SolverBackedInverseVolatilityAdapter().describe_adapter_binding()

    assert original.policy_id == rewritten.policy_id == _POLICY_ID
    # The names agree and the implementations do not, which is the whole case.
    assert original.adapter_implementation_hash != rewritten.adapter_implementation_hash
    assert original.binding_hash != rewritten.binding_hash
    # And the identity is the code, not the flag: the two differ on
    # ``solver_backed`` too, but flipping only that could never be enough.
    assert original.solver_semantics != rewritten.solver_semantics

    # The catalog binding still cannot tell them apart, which is why it is not
    # the thing being relied on.
    assert (
        _catalog(_InverseVolatilityAdapter()).binding.catalog_hash
        != _catalog(_SolverBackedInverseVolatilityAdapter()).binding.catalog_hash
    )


def test_the_equal_weight_adapter_binds_the_module_that_picks_its_holdings() -> None:
    """requirement: an adapter's closure must cover what actually decides.

    ``TopKEqualWeightAdapter`` runs no solver, and its binding once hashed only its own
    module on the strength of that. But ``stable_top_k`` lives in the optimizer
    module and chooses which names are held, so rewriting it changes the holdings
    and the weights while the adapter's identity stood still -- the same
    "same id, changed code" defect the binding exists to catch, one import away.

    On the rule (LAWS.md ID3) the adapter's own module walks the optimizer module it
    imports, so the closure covers ``stable_top_k`` whether or not the optimizer is
    listed. The listed pair differed from the adapter alone only while the switch
    table kept the pair's byte value, which ended when the closure moved (RT R19).
    """

    from alphalattice.investment.portfolio_strategy_lab.policies import top_k_equal_weight
    from alphalattice.investment.portfolio_strategy_lab.policies.top_k_equal_weight import (
        TopKEqualWeightAdapter,
    )

    root = checkout_root(Path(top_k_equal_weight.__file__))
    closure = number_deciding_closure(
        (top_k_equal_weight.__name__,), root=root, rule=number_deciding_rule(root)
    )
    assert optimizer_service.stable_top_k.__module__ in closure

    with_optimizer = portfolio_adapter_implementation_hash(
        (PORTFOLIO_POLICY_PACKAGE, Path(top_k_equal_weight.__file__)),
        (PORTFOLIO_POLICY_PACKAGE, Path(optimizer_service.__file__)),
    )
    binding = TopKEqualWeightAdapter().describe_adapter_binding()
    assert binding.adapter_implementation_hash == with_optimizer


def test_the_engine_never_learned_this_policy() -> None:
    """Extension, not configuration: nothing installed knows this capability."""

    installed = build_installed_portfolio_policy_catalog()
    assert _POLICY_ID not in installed.policy_ids
    with pytest.raises(ValueError, match="PORTFOLIO_POLICY_ADAPTER_NOT_INSTALLED"):
        installed.resolve(_InverseVolatilityPolicy(top_k=2, maximum_weight=0.6))


def test_a_categorical_axis_admits_values_rather_than_converting_them() -> None:
    """requirement: ``20.9`` is not the admitted choice ``20``.

    ``admit`` used to call ``int(value)`` and then check membership, so a draw the
    domain never contained was rounded into one that it did -- and the evidence
    recorded the domain's own hash asserting the value came from it. Conversion
    *is* repair, and repairing an inadmissible value is worse than refusing
    because it succeeds.
    """

    axis = SearchAxis(name="top_k", kind="categorical", choices=(2, 3, 4))
    assert axis.admit(3) == 3

    with pytest.raises(ValueError, match="value_not_a_choice"):
        axis.admit(2.9)
    with pytest.raises(ValueError, match="value_not_a_choice"):
        axis.admit(3.0)
    # ``True`` is an ``int`` equal to one in Python, so a truth value would be
    # admitted on any axis that happened to offer 1 unless bool is excluded.
    truthy = SearchAxis(name="top_k", kind="categorical", choices=(1, 2))
    with pytest.raises(ValueError, match="value_not_a_choice"):
        truthy.admit(True)

    # The same exclusion on the float side.
    weight = SearchAxis(name="weight_cap", kind="float", low=0.0, high=2.0)
    with pytest.raises(ValueError, match="value_not_finite"):
        weight.admit(True)


def test_a_log_axis_is_refused_at_declaration_and_at_draw() -> None:
    """A log axis over a non-positive bound is a space nobody can draw from."""

    axis = SearchAxis(name="risk_aversion", kind="float", low=1e2, high=1e5, log=True)
    assert axis.admit(1e3) == 1e3
    with pytest.raises(ValueError, match="value_out_of_bounds"):
        axis.admit(-1.0)

    # Refused where the mistake was made, rather than at every later draw.
    with pytest.raises(ValueError, match="search_axis_log_bounds_invalid"):
        SearchAxis(name="risk_aversion", kind="float", low=0.0, high=1e5, log=True)
    with pytest.raises(ValueError, match="search_axis_log_bounds_invalid"):
        SearchAxis(name="risk_aversion", kind="float", low=-1.0, high=-0.5, log=True)
