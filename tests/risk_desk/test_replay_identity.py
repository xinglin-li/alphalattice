"""A sealed Risk study replays through its estimators' recorded moves, and only those (V314)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from alphalattice.investment.risk_research.estimators.catalog import (
    ESTIMATOR_CATALOG_ROLE,
    numerical_binding_role,
)
from alphalattice.investment.risk_research.experiments.contracts import (
    RISK_EXPERIMENT_KIND,
    RiskDevelopmentProgramBinding,
)
from alphalattice.investment.risk_research.experiments.replay_identity import (
    program_follows_recorded_moves,
)
from alphalattice.protocols.research_authoring.contracts import SealedResearchProgram

ADAPTER = "risk-covariance-ewma-standardized-ledoit-wolf"


def _binding(catalog: str, numerical: str, recipe: str = "c" * 64) -> RiskDevelopmentProgramBinding:
    return RiskDevelopmentProgramBinding.create(
        catalog_hash=catalog,
        selected_adapter_id=ADAPTER,
        selected_numerical_binding_hash=numerical,
        recipe_hash=recipe,
        parameter_domain_hash="d" * 64,
    )


def _program(
    binding: RiskDevelopmentProgramBinding, *, authority: str = "e" * 64
) -> SealedResearchProgram:
    return SealedResearchProgram.create(
        kind=RISK_EXPERIMENT_KIND,
        envelope_hash="f" * 64,
        desk_program_hash=binding.selected_method_binding_hash,
        resolved_sessions=(date(2024, 1, 2),),
        catalog_hash=binding.catalog_hash,
        method_binding_hash=binding.development_binding_hash,
        parameter_domain_hash=binding.parameter_domain_hash,
        authority_hash=authority,
    )


def _moves(root: Path, *moves: tuple[str, str, str]) -> Path:
    (root / "config").mkdir(parents=True)
    document = {
        "schema": "identity-successors",
        "version": 1,
        "moves": [
            {
                "role": role,
                "predecessor": predecessor,
                "successor": successor,
                "change": "a tidy-up in the estimator's closure",
                "reason": "no number moved",
            }
            for role, predecessor, successor in moves
        ],
    }
    (root / "config" / "identity-successors.json").write_text(json.dumps(document), "utf-8")
    return root


def test_a_replay_follows_its_estimators_recorded_moves_and_nothing_else(tmp_path: Path) -> None:
    """regression (V314): an edit in an estimator's closure that moved no number refused every
    sealed Risk study's replay, since the replay compared its Program by equality; the recorded
    Program is current when its catalog and its estimator's numerical binding follow recorded
    moves to the installed ones, and a recipe or an authority the moves do not explain is still
    another study."""

    old_catalog, new_catalog = "1" * 64, "2" * 64
    old_numerical, new_numerical = "3" * 64, "4" * 64
    recorded = _program(_binding(old_catalog, old_numerical))
    binding = _binding(new_catalog, new_numerical)
    installed = _program(binding)
    moved = _moves(
        tmp_path / "moved",
        (numerical_binding_role(ADAPTER), old_numerical, new_numerical),
        (ESTIMATOR_CATALOG_ROLE, old_catalog, new_catalog),
    )

    assert program_follows_recorded_moves(installed, installed, binding, root=moved)
    assert program_follows_recorded_moves(recorded, installed, binding, root=moved)
    unrecorded = _moves(tmp_path / "unrecorded")
    assert not program_follows_recorded_moves(recorded, installed, binding, root=unrecorded)
    other_recipe = _program(_binding(old_catalog, old_numerical, recipe="9" * 64))
    assert not program_follows_recorded_moves(other_recipe, installed, binding, root=moved)
    elsewhere = _program(_binding(old_catalog, old_numerical), authority="8" * 64)
    assert not program_follows_recorded_moves(elsewhere, installed, binding, root=moved)
