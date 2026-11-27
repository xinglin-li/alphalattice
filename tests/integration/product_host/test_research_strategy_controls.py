"""The research strategy's controls: each missing component's first step, the Risk study's, and
the plan's declaration once its parents are determined."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest

from alphalattice.capabilities.domain_contracts.identity import canonical_hash
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceManifest,
    publish_research_workspace_manifest,
)
from alphalattice.control.product_host.data_preparation import research_strategy
from alphalattice.control.product_host.data_preparation.research_strategy import (
    ResearchStrategyPreparation,
)
from alphalattice.control.task_control.contracts import TaskLifecycle

RISK = {
    "lifecycle": "SUCCEEDED",
    "kind": "risk.covariance-development",
    "sessions": {"start": "2020-01-02", "end": "2026-09-10"},
}
"""A completed Risk study wide enough for any window below."""
RISK_TASK = "00000000-0000-4000-8000-000000000001"


def _owner(tmp_path: Path, studies: list[dict[str, Any]]) -> ResearchStrategyPreparation:
    publish_research_workspace_manifest(tmp_path, ResearchWorkspaceManifest.research_only("fwd"))
    return ResearchStrategyPreparation(
        SimpleNamespace(workspace=tmp_path),  # type: ignore[arg-type]
        clock=lambda: datetime(2026, 10, 2, tzinfo=UTC),
        read_experiment=lambda _task: {},
        list_experiments=lambda: {"experiments": studies},
    )


def _alpha(component: str, **fields: object) -> dict[str, object]:
    return {
        "lifecycle": "SUCCEEDED",
        "kind": "alpha.model-development",
        "component_recipe_id": component,
        **fields,
    }


def test_strategy_controls_offer_each_missing_components_first_step(tmp_path: Path) -> None:
    """regression (a fresh workspace found no way to its components): each missing component
    offers its first step, the input left to choose; a component a study holds is not missing."""
    studies: list[dict[str, Any]] = []
    owner = _owner(tmp_path, studies)
    controls = owner.controls()
    required = controls["required_components"]
    assert required and controls["missing_components"] == required
    for component in required:
        assert controls["next_requests"][f"component:{component}"] == {
            "operation": "MODEL_TRAINING_INPUT_PLAN",
            "research_input_id": None,
            "component_id": component,
        }
    studies.append(_alpha(required[0]))
    again = owner.controls()
    assert again["missing_components"] == required[1:]
    assert f"component:{required[0]}" not in again["next_requests"]
    assert again["next_requests"]["plan"]["operation"] == "RESEARCH_STRATEGY_PLAN"
    assert again["risk_windows"] == []
    assert again["next_requests"]["risk"] == {
        "operation": "EXPERIMENT_CONTROLS",
        "research_input_id": None,
        "experiment_kind": "risk.covariance-development",
    }
    studies.append(RISK)
    assert "risk" not in owner.controls()["next_requests"]


def test_strategy_controls_offer_the_risk_study_until_one_covers_the_window(
    tmp_path: Path,
) -> None:
    """requirement: the Risk study's controls are offered until a completed Risk study covers
    the window a calibrated Alpha study names; no window is known before one completes."""
    studies: list[dict[str, Any]] = []
    owner = _owner(tmp_path, studies)
    controls = owner.controls()
    assert controls["risk_windows"] == []
    assert controls["next_requests"]["risk"] == {
        "operation": "EXPERIMENT_CONTROLS",
        "research_input_id": None,
        "experiment_kind": "risk.covariance-development",
    }
    studies.append(RISK)
    assert "risk" not in owner.controls()["next_requests"]


def test_strategy_controls_fill_the_plan_once_its_parents_are_determined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement (fewer agent steps): one completed study per component and one Risk study
    covering the calibrated study's window, on one input, fill the plan's declaration
    and let it seal its output directory."""
    # The input's calendar and the Risk study's returns reach far enough before the first
    # formation.
    history = tuple(date(2015, 1, 1) + timedelta(days=day) for day in range(4300))
    monkeypatch.setattr(
        research_strategy, "read_factor_bundle", lambda *_args: SimpleNamespace(sessions=history)
    )
    monkeypatch.setattr(research_strategy, "risk_return_surface", lambda *_: (tmp_path, history))
    monkeypatch.setattr(
        research_strategy.CausalRiskReturnReader, "available_sessions", lambda _self, axis: axis
    )
    studies: list[dict[str, Any]] = []
    owner = _owner(tmp_path, studies)
    required = owner.controls()["required_components"]
    window = {"start": "2023-01-03", "end": "2026-06-30"}
    binding = canonical_hash({"input": "fixture"})
    alpha_tasks = [str(UUID(int=index + 1)) for index in range(len(required))]
    studies.extend(
        _alpha(c, task_id=task, input_binding_hash=binding, sessions=window)
        for c, task in zip(required, alpha_tasks, strict=True)
    )
    studies.append({**RISK, "task_id": RISK_TASK, "input_binding_hash": binding})
    filled = owner.controls()
    assert filled["risk_windows"]
    declaration = filled["next_requests"]["plan"]["experiment_document"]
    assert declaration == {
        "input_binding_hash": binding,
        "alpha_task_ids": alpha_tasks,
        "risk_task_id": RISK_TASK,
    }
    owner.session.task_control_registry = SimpleNamespace(
        task=lambda _: SimpleNamespace(lifecycle=TaskLifecycle.SUCCEEDED),
        tasks=lambda: (),
    )
    answer = owner.plan(declaration)
    plan = owner.last_plan
    assert plan is not None and answer["plan_hash"] == plan.plan_hash
    request_hex = canonical_hash(answer["declaration"]).split(":", 1)[1]
    assert plan.artifact_root_relative == f"artifacts/research-strategy-inputs/blake3-{request_hex}"
