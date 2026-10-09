"""Enforce the declared budget and determinism policy before any numerics.

``maximum_numerical_calls``, ``thread_limit``, ``seed``, and ``network_disabled``
were folded into Program identity and then ignored. A document could declare a
budget of one and run a thousand estimates: the number was sealed, so it looked
governed, but nothing consulted it.

Enforcement happens against a *plan* rather than a running total. A budget
checked as calls accumulate stops halfway and leaves partial artifacts behind;
checked against the window the Program already bound, an over-budget request
fails having computed nothing.

The rule itself now lives in ``protocols/research_authoring/execution_policy``,
because it was never Risk-specific and having one caller was the reason no other
Desk enforced a budget at all. What stays here is the projection from Risk's own
adapter bindings into the shared requirement, and the error strings -- which are
this Desk's stable boundary and are raised by the shared owner unchanged. The Risk
compiler calls it, so every command's seal checks it at PLAN and again at RUN;
the workflow holds every Desk's run offline.
"""

from __future__ import annotations

from alphalattice.investment.risk_research.estimators.contracts import (
    SINGLE_THREAD_NUMERICAL_CAPABILITY,
    RiskEstimatorNumericalBinding,
)
from alphalattice.protocols.research_authoring.contracts import ResearchExperimentEnvelope
from alphalattice.protocols.research_authoring.execution_policy import (
    ExecutionPlan,
    RuntimeCapabilityRequirement,
    enforce_declared_execution_policy,
)


def enforce_execution_policy(
    *,
    envelope: ResearchExperimentEnvelope,
    planned_numerical_calls: int,
    adapter_bindings: tuple[RiskEstimatorNumericalBinding, ...],
) -> None:
    """Refuse a plan whose declared budget or threads Risk's adapters cannot honour."""

    enforce_declared_execution_policy(
        envelope=envelope,
        plan=ExecutionPlan(planned_numerical_calls=planned_numerical_calls),
        requirements=tuple(
            RuntimeCapabilityRequirement(
                requires_single_thread=(
                    SINGLE_THREAD_NUMERICAL_CAPABILITY in binding.required_runtime_capabilities
                )
            )
            for binding in adapter_bindings
        ),
    )


__all__ = ["enforce_execution_policy"]
