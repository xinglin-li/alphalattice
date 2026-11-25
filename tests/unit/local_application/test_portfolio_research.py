"""A recall's request is bounded before any model runs: its query is 1 to 512 characters and its
limit 1 to 20, and its answer holds exactly the fields the recall answers with."""

from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import ValidationError

from alphalattice.interface.local_application.answers import ExperienceRecallAnswer
from alphalattice.interface.local_application.portfolio_research import (
    LocalApplicationError,
    PortfolioResearchOperationRequest,
    validate_request_document,
)
from tests.portfolio_strategy_lab.cli_support import strategy_readback_operations

TEST_SOURCE = "src/alphalattice/interface/local_application/portfolio_research.py"


@pytest.mark.proves("local_application.a_recall_query_and_limit_are_bounded_before_a_model_runs")
def test_a_recall_request_is_bounded_and_its_answer_is_exact() -> None:
    """requirement: a recall request with an empty or over-long query, or a limit outside 1 to 20,
    is refused by name before any model runs, and its answer refuses a field it does not name."""

    def recall(**fields: object) -> PortfolioResearchOperationRequest:
        return PortfolioResearchOperationRequest(operation="EXPERIENCE_RECALL", **fields)

    assert recall(experience_query="momentum", experience_limit=20).experience_limit == 20
    assert recall(experience_query="m" * 512, experience_limit=1).experience_limit == 1
    for fields in (
        {"experience_query": ""},
        {"experience_query": "m" * 513},
        {"experience_query": "momentum", "experience_limit": 0},
        {"experience_query": "momentum", "experience_limit": 21},
    ):
        with pytest.raises(LocalApplicationError, match="recall_query_invalid"):
            recall(**fields)
    answer = {"score_type": "LINEAR_FUSION", "hits": [], "unnamed": 1}
    with pytest.raises(ValidationError):
        ExperienceRecallAnswer.model_validate(answer)


@pytest.mark.proves("local_application.a_goal_or_floor_read_takes_its_wait_and_its_mark")
def test_a_goal_or_floor_read_takes_its_wait_and_its_mark() -> None:
    """requirement: a goal's narrative takes the Host's wait and the mark last read, and a floor
    read takes the wait, each refused outside its bounds; no other goal read waits."""
    goal = {"operation": "GOAL_NARRATIVE", "goal_id": str(UUID(int=1))}
    floor = {"operation": "COMMITTEE_READ", "update_task_id": str(UUID(int=2))}
    taken = validate_request_document({**goal, "goal_seen": "a" * 16, "wait_seconds": 20})
    assert (taken.goal_seen, taken.wait_seconds) == ("a" * 16, 20)
    assert validate_request_document({**floor, "wait_seconds": 0.5}).wait_seconds == 0.5
    for document in ({**goal, "goal_seen": "M3"}, {**goal, "wait_seconds": 0}):
        with pytest.raises(ValidationError):
            validate_request_document(document)
    with pytest.raises(LocalApplicationError, match="wait_seconds_invalid"):
        PortfolioResearchOperationRequest(
            operation="COMMITTEE_READ", update_task_id=UUID(int=2), wait_seconds=21
        )
    with pytest.raises(LocalApplicationError, match="operation_field_not_allowed:wait_seconds"):
        PortfolioResearchOperationRequest(
            operation="GOAL_SHOW", goal_id=UUID(int=1), wait_seconds=5
        )


@pytest.mark.proves("local_application.strategy_readback_requests_keep_their_scope")
def test_every_read_of_work_planned_per_strategy_is_keyed_by_its_strategy() -> None:
    """Control: Every read of work planned per strategy is keyed by its strategy."""

    contract = PortfolioResearchOperationRequest.field_contract
    readbacks = strategy_readback_operations()
    for operation in readbacks:
        required, allowed = contract(operation)
        assert not required and {"task_id", "strategy_package_id"} <= allowed, operation
