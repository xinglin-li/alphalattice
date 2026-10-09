"""The typed reads answer what their published models say (binding plan, CLI-16)."""

from __future__ import annotations

import json

from alphalattice.control.product_host.composition.local_web_session import (
    LocalPortfolioWebSession,
)
from alphalattice.interface.local_application.answers import ANSWERS
from alphalattice.interface.local_application.cli import schema
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchRequestDocument as PortfolioResearchAgentRequest,
)
from tests.portfolio_strategy_lab.local_web_support import InstalledAgent, _json

STUDY_READS = frozenset(
    {"EXPERIMENTS", "EXPERIMENT_READBACK", "RESEARCH_INPUTS", "RESEARCH_HISTORY"}
)
"""Checked on a published study, in test_local_web_factor_experiments.py."""


def test_each_task_and_book_read_answers_its_published_model(
    live: LocalPortfolioWebSession,
) -> None:
    """contract (binding plan, CLI-16): on a workspace with a finished run, each typed read's
    answer validates against the model `alphalattice schema show` publishes for it; the study
    reads are checked beside a published study."""

    task_id = _json(live, "/api/run", method="POST", payload={})["task_id"]
    live.dispatcher.drain_for_tests()  # type: ignore[union-attr]
    bridge = InstalledAgent(live.operations)
    checked = set()
    for operation, fields in (
        ("STATUS", {"task_id": task_id}),
        ("TASKS", {}),
        ("TASK_RECOVERY", {"task_id": task_id}),
        ("TASK_GUARDIAN", {}),
        ("RESULTS", {}),
        ("PENDING_DECISIONS", {}),
        ("ACTIVITY_REFUSALS", {}),
        ("UPGRADE_OVERVIEW", {}),
    ):
        body = json.loads(
            bridge.invoke(PortfolioResearchAgentRequest(operation=operation, **fields))
        )
        assert body.get("status") != "REFUSED", (operation, body)
        ANSWERS[operation].model_validate(body)
        assert schema(operation)["answer_schema"] == ANSWERS[operation].model_json_schema()
        checked.add(operation)
    assert checked | STUDY_READS <= set(ANSWERS)


def test_every_operation_publishes_its_answer() -> None:
    """Every operation publishes its answer."""

    from alphalattice.interface.local_application.operations import OPERATIONS

    assert set(ANSWERS) == set(OPERATIONS)
    for operation, model in ANSWERS.items():
        published = schema(operation)["answer_schema"]
        assert published == model.model_json_schema(), operation
        undescribed = [
            name
            for name, field in published.get("properties", {}).items()
            if not field.get("description")
        ]
        assert not undescribed, (operation, undescribed)
