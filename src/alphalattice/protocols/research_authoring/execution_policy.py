"""The declared execution plan, and the one owner that enforces a budget against it.

Risk wrote this rule and it was right: enforcement happens against a *plan*
rather than a running total, because a budget checked as calls accumulate stops
halfway and leaves partial artifacts behind, while a budget checked against the
window the Program already bound fails having computed nothing.

It had exactly one caller. Every other Desk either checked ``maximum_candidates``
somewhere unrelated or checked nothing at all, so a Portfolio Campaign could
declare a budget, watch it hash into an identity, and then run whatever the grid
happened to contain.

So the rule moves here and the Desks call it. Risk's caller keeps its exact error
strings, because those codes are its stable boundary and downstream readers key
on them.

**The plan is also the preflight output.** ``--preflight`` prints an
``ExecutionPlan`` and ``--run`` enforces the same object, so "what this will cost"
and "what this is allowed to cost" cannot be two numbers that disagree. That is
the whole reason the estimate is a typed value rather than a console line.

The offline rule is held rather than checked: the workflow runs every Desk's call held
offline, whatever the workspace allows, so no environment decides a run's outcome.
"""

from __future__ import annotations

from dataclasses import dataclass

from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    ResearchExperimentEnvelope,
)


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """What a run will do, known before it does any of it.

    ``planned_numerical_calls`` is the quantity a budget bounds: estimator calls
    for Risk, optimizer solves for Portfolio. The rest is what a researcher needs
    in order to decide whether to start, which is a different question from
    whether the Host will allow it, and both are answered from one object.
    """

    planned_numerical_calls: int
    planned_trial_count: int = 0
    planned_formation_count: int = 0
    estimated_seconds: float | None = None
    """``None`` where no measured per-unit cost exists yet.

    Absent rather than zero: a run whose cost nobody has measured should say so,
    not claim it is free.
    """


@dataclass(frozen=True, slots=True)
class RuntimeCapabilityRequirement:
    """One capability an installed adapter declares it needs to stay reproducible.

    A minimal projection rather than the adapter binding itself, so this owner
    does not have to import any Desk's estimator contracts to enforce a rule that
    is about none of them.
    """

    requires_single_thread: bool


def enforce_declared_execution_policy(
    *,
    envelope: ResearchExperimentEnvelope,
    plan: ExecutionPlan,
    requirements: tuple[RuntimeCapabilityRequirement, ...] = (),
) -> None:
    """Refuse a plan whose declared budget or threads the installed adapters cannot honour."""
    budget = envelope.budget
    if plan.planned_numerical_calls > budget.maximum_numerical_calls:
        raise AuthoringError("research_authoring.numerical_budget_exceeded")

    determinism = envelope.determinism
    if determinism.thread_limit < 1:
        raise AuthoringError("research_authoring.thread_limit_invalid")
    # The installed adapter declares what it needs. A covariance estimator is
    # single-threaded because BLAS thread count moves eigenvalue bits, so
    # honouring a wider request would silently change numbers rather than merely
    # run faster.
    if any(value.requires_single_thread for value in requirements) and (
        determinism.thread_limit != 1
    ):
        raise AuthoringError("research_authoring.thread_limit_violates_adapter_requirement")


__all__ = [
    "ExecutionPlan",
    "RuntimeCapabilityRequirement",
    "enforce_declared_execution_policy",
]
