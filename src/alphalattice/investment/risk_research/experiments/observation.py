"""Count numerical calls by watching them happen.

The previous recorder looped over the finished surface's formation sessions and
called ``record`` once per session. That number is a property of the result, not
of the run: a build that resumed a 900-session checkpoint and computed 100 new
formations reported 1000, identical to a cold run. It could not distinguish
computation from reuse, which is precisely the distinction the evidence claims.

This wraps the installed catalog so the adapter it hands to the frozen numerical
path is an observing delegate. Only a real ``estimate`` call increments the
count, so a resumed prefix is not counted -- it is never executed.

The wrapper is identity-preserving by construction: ``binding``, ``seal_recipe``,
and the adapter's declared identity all delegate unchanged, so installing an
observer cannot move the Program hash. Observation must not be visible in
identity, or measurement would change what it measures.
"""

from __future__ import annotations

from alphalattice.investment.risk_research.estimators.contracts import (
    BoundRiskReturnInput,
    EstimatedCovariance,
    RiskEstimatorAdapter,
    RiskEstimatorNumericalBinding,
    RiskEstimatorRecipeEnvelope,
)


class NumericalCallCounter:
    """A mutable tally the observing adapter writes to."""

    def __init__(self) -> None:
        self.estimate_calls = 0

    def observe_estimate(self) -> None:
        self.estimate_calls += 1


class ObservingAdapter:
    """Delegate every call, count only the numerical one."""

    def __init__(self, adapter: RiskEstimatorAdapter, counter: NumericalCallCounter) -> None:
        self._adapter = adapter
        self._counter = counter
        self.adapter_id = adapter.adapter_id
        self.recipe_schema_id = adapter.recipe_schema_id

    def describe_numerical_binding(self) -> RiskEstimatorNumericalBinding:
        return self._adapter.describe_numerical_binding()

    def validate_recipe(self, recipe: RiskEstimatorRecipeEnvelope) -> object:
        return self._adapter.validate_recipe(recipe)

    def estimate(
        self,
        *,
        recipe: RiskEstimatorRecipeEnvelope,
        inputs: BoundRiskReturnInput,
    ) -> EstimatedCovariance:
        self._counter.observe_estimate()
        return self._adapter.estimate(recipe=recipe, inputs=inputs)


__all__ = ["NumericalCallCounter", "ObservingAdapter"]
