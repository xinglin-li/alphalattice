"""Shared Goal ledger fixtures and request builders."""

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import pytest

from alphalattice.control.product_host.composition.goals import GoalApplication
from alphalattice.control.product_host.publication.goals import GoalStore
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchOperationRequest as Request,
)

NOW = datetime(2026, 9, 15, tzinfo=UTC)

DECLARATION = {
    "title": "A study",
    "objective": "Does the signal survive costs?",
    "kind": "RESEARCH",
    "scope": "One admitted historical input; full declared support",
    "criteria": [{"criterion_id": "folds", "text": "Inspect all folds, including negative ones"}],
    "deliverables": [
        {"deliverable_id": "evidence", "kind": "RESULT", "description": "The exact result"}
    ],
    "budget": {"maximum_tasks": 2, "maximum_numerical_calls": 10, "maximum_model_calls": 2},
    "research": {
        "purpose": "NEW_RESEARCH",
        "comparison_design": "Same target and costs; no retuning",
        "required_stages": ["ALPHA", "RISK"],
    },
}


@pytest.fixture
def goal_app(tmp_path):
    calls = []
    tasks: dict[UUID, tuple[Any, ...]] = {}
    body = {
        "status": "EXPERIMENT_PUBLISHED",
        "task_id": str(UUID(int=2)),
        "program": {"program_hash": "a" * 64, "kind": "alpha.model-development"},
        "evidence": {"evidence_hash": "b" * 64},
        "result": {"candidates": [{"score": -1}]},
        "next_requests": {"continue": dict(operation="EXPERIMENT_DRAFT", task_id=str(UUID(int=2)))},
    }

    def read(request, caller):
        calls.append((request.operation, caller))
        assert request.operation not in {"EXPERIMENT_RUN", "RUN"}
        return deepcopy(body)

    app = GoalApplication(
        GoalStore(tmp_path, "workspace"),
        lambda: NOW,
        read,
        lambda _: NOW + timedelta(seconds=10),
        tasks.get,
        workspace=tmp_path,
    )
    opened = app.operate(
        Request(
            operation="GOAL_OPEN",
            goal_id=UUID(int=1),
            goal_declaration=DECLARATION,
            change_reason="Register before execution",
        ),
        "HUMAN",
    )
    return app, opened, calls, body, tasks


def _reference(reference_id="alpha", label="Alpha result", stage="ALPHA", **request):
    return {
        "reference_id": reference_id,
        "label": label,
        "stage": stage,
        "request": request or {"operation": "EXPERIMENT_READBACK", "task_id": str(UUID(int=2))},
    }


def _attach(app, opened, reference_id="alpha"):
    return app.operate(
        Request(
            operation="GOAL_ATTACH",
            goal_hash=opened["goal_hash"],
            change_reason="Attach exact result",
            goal_reference=_reference(reference_id),
        ),
        "EXTERNAL_AUTOMATION",
    )


def _submission(**overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "outcome": "ACHIEVED",
        "summary": "The signal survives costs on every fold.",
        "criteria": [{"criterion_id": "folds", "answer": "MET", "evidence": ["alpha"]}],
        "deliverables": [{"deliverable_id": "evidence", "references": ["alpha"]}],
    }
    return document | overrides
