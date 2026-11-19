"""Experience read and the research run's explicit hypothesis."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from alphalattice.control.product_host.composition.plain_refusals import refused
from alphalattice.interface.local_application.cli_contract import REQUEST_PROVENANCE
from alphalattice.interface.local_application.failure_codes import public_failure
from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer import experiences, history, reading

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


def read_experience_class(request: "PortfolioResearchOperationRequest") -> dict[str, object]:
    """Every trial of the named trial's class, at its width, as the history knew it at the
    request's known time (`history.read_experience_class`); refused by name for a trial the
    history does not hold."""
    assert request.experience_trial is not None
    known = request.experience_known_at
    try:
        found = history.read_experience_class(
            trial=request.experience_trial,
            width=request.experience_width or "method_version",
            cutoff_us=None
            if known is None
            else round(datetime.fromisoformat(known).timestamp() * 1_000_000),
        )
    except DataError as error:  # the read's own code, or the history unavailable
        return refused(public_failure(error, "experience.history_unavailable"))
    return refused("experience.history_unavailable") if found is None else found


_RECALL_REFUSALS = {
    "data_layer.knowledge_model_unavailable": "experience.recall_models_missing",
    "data_layer.knowledge_query_invalid": "experience.recall_query_invalid",
}


def recall_experience(request: "PortfolioResearchOperationRequest") -> dict[str, object]:
    """The experiences most like the request's query in its workspace (the Host's when none is
    named), as known at its known time, each with its trials' classes
    (`history.recall_experiences`); refused by name when the retrieval models are not installed
    or the query is not admitted."""
    assert request.experience_query is not None
    opened = history.opened_log()
    workspace = request.experience_workspace or (opened[1] if opened else None)
    known = request.experience_known_at
    cutoff_us = (
        round(datetime.now(UTC).timestamp() * 1_000_000)
        if known is None
        else round(datetime.fromisoformat(known).timestamp() * 1_000_000)
    )
    if workspace is None:
        return refused("experience.history_unavailable")
    try:
        found = history.recall_experiences(
            query=request.experience_query,
            workspace=workspace,
            cutoff_us=cutoff_us,
            limit=request.experience_limit or 5,
        )
    except DataError as error:  # the recall's own code, or the history unavailable
        code = public_failure(error, "experience.history_unavailable")
        return refused(_RECALL_REFUSALS.get(code, code))
    return refused("experience.history_unavailable") if found is None else found


def read_experience_brief(request: "PortfolioResearchOperationRequest") -> dict[str, object]:
    """What to know before new work on the request's goal, as known at its known time, in its
    budget (`history.read_experience_brief`); refused by name for a goal the history did not
    hold then."""
    assert request.goal_id is not None
    known = request.experience_known_at
    try:
        found = history.read_experience_brief(
            goal=str(request.goal_id),
            budget=request.experience_budget or reading.BRIEF_BUDGET,
            cutoff_us=None
            if known is None
            else round(datetime.fromisoformat(known).timestamp() * 1_000_000),
        )
    except DataError as error:  # the read's own code, or the history unavailable
        return refused(public_failure(error, "experience.history_unavailable"))
    return refused("experience.history_unavailable") if found is None else found


def record_experience(
    request: "PortfolioResearchOperationRequest", caller: str
) -> dict[str, object]:
    """Record a review's note on what it names, with its verdict and its run's aggregates; the
    Host renders the experience after the reply. A note missing a part, or one on a trial or Task
    not yet ended, is refused by name and nothing is recorded."""
    assert request.review_subject and request.review_verdict and request.review_note is not None
    layer, unit = request.review_subject.split(":", 1)
    note = request.review_note
    try:
        text = experiences.note_text(
            note.get("hypothesis"), note.get("method"), note.get("conclusion"), note.get("lessons")
        )
        trials = note.get("trials")  # a review task's: the trials it reviewed
        held = history.record_review(layer, unit, request.review_verdict, text, caller, trials)
    except (DataError, ValueError) as error:  # the note's or the history's own code
        return refused(public_failure(error, "experience.history_unavailable"))
    return {
        "status": "RECORDED",
        "note_id": held["note_id"],
        "verdict": held["verdict"],
        "input_tokens": held["gen_ai.usage.input_tokens"],
        "output_tokens": held["gen_ai.usage.output_tokens"],
        "cost_usd": held["cost_usd"],
        "seconds": held["seconds"],
        "place": held.get("place", []),
        "trials": held.get("trials", []),
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
