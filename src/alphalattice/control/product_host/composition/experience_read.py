"""Experience read and the research run's explicit hypothesis."""

from typing import TYPE_CHECKING, Protocol

from alphalattice.control.product_host.composition.plain_refusals import refused
from alphalattice.interface.local_application.cli_contract import REQUEST_PROVENANCE
from alphalattice.kernel.data_layer import history

if TYPE_CHECKING:
    from alphalattice.interface.local_application.portfolio_research import (
        PortfolioResearchOperationRequest,
    )
    from alphalattice.protocols.actor_execution.contracts import AgentExecutionBinding


class _ExperimentRunner(Protocol):
    """The existing research admission boundary used by this route."""

    def run(
        self,
        plan_hash: str,
        *,
        caller: str,
        agent_execution: "AgentExecutionBinding | None",
        hypothesis_id: str | None,
        expected_goal: str | None,
    ) -> dict[str, object]:
        """Admit an attributed run."""
        ...


def read_experience(request: "PortfolioResearchOperationRequest") -> dict[str, object]:
    """Read bounded trials from the recorded experience view."""
    experience = history.read_experience(
        goal=str(request.goal_id) if request.goal_id else None,
        hypothesis=request.experience_hypothesis,
        method=request.experience_method,
        workspace=request.experience_workspace,
        limit=request.history_limit or 25,
    )
    if experience is None:
        return refused("experience.history_unavailable")
    trials = experience["trials"]
    return {
        "trials": trials,
        "notes": experience["notes"],
        "unattributed_in_page": sum(row["hypothesis_id"] is None for row in trials),
    }


def run_with_hypothesis(
    application: _ExperimentRunner,
    request: "PortfolioResearchOperationRequest",
    caller: str,
    agent_execution: "AgentExecutionBinding | None",
) -> dict[str, object]:
    """Admit the requested trial with its stated hypothesis and goal."""
    assert request.experiment_plan_hash is not None
    provenance = REQUEST_PROVENANCE.get()
    return application.run(
        request.experiment_plan_hash,
        caller=caller,
        agent_execution=agent_execution,
        hypothesis_id=request.experience_hypothesis,
        expected_goal=provenance.goal_id if provenance else None,
    )
