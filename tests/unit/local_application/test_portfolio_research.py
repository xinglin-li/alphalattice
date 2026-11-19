"""A recall's request is bounded before any model runs: its query is 1 to 512 characters and its
limit 1 to 20, and its answer holds exactly the fields the recall answers with."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from alphalattice.interface.local_application.answers import ExperienceRecallAnswer
from alphalattice.interface.local_application.portfolio_research import (
    LocalApplicationError,
    PortfolioResearchOperationRequest,
)

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
