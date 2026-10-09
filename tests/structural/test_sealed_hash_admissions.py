"""TE12 over a whole class (NM2, V451): each equality on a hash NM1's rule moved has a reason.

NM1 moved how a recipe, strategy or package hash is computed. An admission that compares a value
sealed before the move with the installed one, by equality, refuses it. One did, unseen: the
heterogeneous closure is a worktree-based root no checkout resolved, so U0 never read it. Every
such comparison in the owners of those hashes is classified here, and the walk below must find
exactly this set. A new comparison is classified before it lands.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "alphalattice"
OWNERS = (
    "investment/alpha_research/scores",
    "investment/portfolio_strategy_lab/application",
    "investment/portfolio_strategy_lab/policies",
    "control/product_host/composition",
)
FIELDS = frozenset(
    {
        "strategy_hash",
        "recipe_hash",
        "component_recipe_hash",
        "package_hash",
        "strategy_package_hash",
    }
)

WORKSPACE = (
    "a workspace's own record against another or the installed value; a workspace prepared "
    "before NM2 stops reading (decision C), and no root the tree must read holds such a record"
)
SELF = "one object's seal against its own content, or two fields of one record"
NOT_MOVED = "a hash NM1's rule did not move: model, risk, method or final-evaluation recipes"
WRITTEN = "a record this tree writes from the installed values; nothing loads a stored one"
CONTENT = "the stored recipe's identity computed now, under the rule the installed one holds"

CLASSIFIED: dict[str, str] = {
    "control/product_host/composition/decision_advancement.py::"
    "DecisionAdvancementApplication._ew_input": WORKSPACE,
    "control/product_host/composition/decision_advancement.py::"
    "DecisionAdvancementApplication._execute": SELF,
    "control/product_host/composition/decision_advancement.py::latest_per_recipe": (
        "groups one workspace's completed scores by the recipe hashes each holds and compares "
        "their sessions; under a moved hash each group still names its own latest, so more "
        "updates are deep-verified, never fewer"
    ),
    "control/product_host/composition/portfolio_application.py::"
    "PortfolioResearchApplication.recover": SELF,
    "control/product_host/composition/portfolio_finalization.py::"
    "HostFinalizationRelease.release": NOT_MOVED,
    "control/product_host/composition/portfolio_finalization.py::"
    "HostReleasedArtifacts.open_released": NOT_MOVED,
    "control/product_host/composition/portfolio_updates.py::"
    "PortfolioUpdateApplication._checkpoint": WORKSPACE,
    "control/product_host/composition/portfolio_updates.py::"
    "PortfolioUpdateApplication.model_bindings_match": WORKSPACE,
    "control/product_host/composition/portfolio_updates.py::PortfolioUpdateApplication.plan": (
        WORKSPACE
    ),
    "control/product_host/composition/rolling_portfolio_report.py::_baseline": (
        WORKSPACE + "; V690 binds the exact sealed source book program to its checkpoint's "
        "package before extending the report; neither is inferred from an installed recipe"
    ),
    "control/product_host/composition/strategy_activation.py::StrategyActivation._book": WORKSPACE,
    "control/product_host/composition/strategy_activation.py::StrategyActivation._grant": (
        WORKSPACE
    ),
    "control/product_host/composition/strategy_activation.py::"
    "StrategyActivation._activation_book": (
        WORKSPACE + "; V614 moves the unchanged already-active package/hash guard into the shared "
        "book admission read used by activation, its offer and REPORT standing"
    ),
    "control/product_host/composition/strategy_activation.py::StrategyActivation._state": WORKSPACE,
    "control/product_host/composition/strategy_activation.py::StrategyActivation.dates": (
        WORKSPACE
        + "; V589 selects a sealed model set's fit dates against its training admission's "
        "component recipe, not an admission or reuse of numbers; both records are from that "
        "workspace and unmatched model sets are not sources of this strategy's dates"
    ),
    "control/product_host/composition/strategy_activation.py::admit_decision_checkpoint": (
        WORKSPACE
    ),
    "control/product_host/composition/strategy_calibration.py::"
    "StrategyCalibrationApplication._binding": WORKSPACE,
    "control/product_host/composition/strategy_calibration.py::"
    "StrategyCalibrationApplication._observations": WORKSPACE,
    "control/product_host/composition/strategy_calibration.py::"
    "StrategyCalibrationApplication._require": WORKSPACE,
    "control/product_host/composition/strategy_calibration.py::"
    "StrategyCalibrationApplication.plan": WORKSPACE,
    "control/product_host/composition/strategy_calibration.py::"
    "StrategyCalibrationApplication.read_input": WORKSPACE,
    "control/product_host/composition/strategy_scoring.py::"
    "StrategyScoringApplication._binding": WORKSPACE,
    "control/product_host/composition/strategy_scoring.py::"
    "StrategyScoringApplication.authority_for_formation": WORKSPACE,
    "control/product_host/composition/strategy_scoring.py::"
    "StrategyScoringApplication.published_score": WORKSPACE,
    "control/product_host/composition/strategy_scoring.py::"
    "StrategyScoringApplication._verify_step": (
        WORKSPACE + "; V691 moves the unchanged score/package equality into the shared "
        "verified score reader used by verify_step and verify_score_step"
    ),
    "investment/alpha_research/scores/frozen_inference.py::"
    "FrozenComponentInferenceAuthority.check_identity": SELF,
    "investment/alpha_research/scores/heterogeneous_product.py::"
    "HeterogeneousComponentModelSet.validate_closure": WORKSPACE,
    "investment/alpha_research/scores/heterogeneous_product.py::"
    "HeterogeneousLiveScoreClosureReceipt.validate_closure": WRITTEN,
    "investment/alpha_research/scores/heterogeneous_product.py::"
    "HeterogeneousModelSetAuthority.validate_identity": WORKSPACE,
    "investment/alpha_research/scores/heterogeneous_product.py::is_installed_component": CONTENT,
    "investment/alpha_research/scores/heterogeneous_replay.py::"
    "AlphaRuntimeHeterogeneousPredictionOwner.__call__": NOT_MOVED,
    "investment/alpha_research/scores/heterogeneous_replay.py::"
    "HeterogeneousCurrentClosureManifest.validate_identity": SELF,
    "investment/alpha_research/scores/model_renewal.py::"
    "AlphaModelLifecycleAdmission.planned_refits": NOT_MOVED,
    "investment/alpha_research/scores/model_renewal.py::"
    "AlphaModelLifecycleAdmission.validate_admission": SELF,
    "investment/alpha_research/scores/model_renewal.py::fit_alpha_refit_child": WORKSPACE,
    "investment/alpha_research/scores/model_renewal.py::read_lifecycle_child": NOT_MOVED,
    "investment/alpha_research/scores/model_renewal.py::reuse_refit_child": NOT_MOVED,
    "investment/alpha_research/scores/model_renewal.py::verify_model_set": WORKSPACE,
    "investment/portfolio_strategy_lab/application/calibration.py::"
    "PreparedPortfolioBookInput.validate_components": SELF,
    "investment/portfolio_strategy_lab/application/calibration.py::"
    "compile_portfolio_calibration": WORKSPACE,
    "investment/portfolio_strategy_lab/application/calibration.py::select_calibration_scores": (
        WORKSPACE
    ),
    "investment/portfolio_strategy_lab/application/decision_updates.py::"
    "PortfolioDecisionCheckpoint.binding": SELF,
    "investment/portfolio_strategy_lab/application/decision_updates.py::"
    "require_proposal_position": WORKSPACE,
    "investment/portfolio_strategy_lab/application/executor.py::"
    "PortfolioResearchExecutor._materialize": WORKSPACE,
    "investment/portfolio_strategy_lab/application/finalization.py::"
    "FinalPortfolioEvaluationPackage.validate_identity": NOT_MOVED,
    "investment/portfolio_strategy_lab/application/finalization_task.py::"
    "PortfolioFinalizationTaskAdapter.verify_stage": NOT_MOVED,
    "investment/portfolio_strategy_lab/application/readback.py::_handoff_failure": NOT_MOVED,
    "investment/portfolio_strategy_lab/application/readback.py::_package_failure": NOT_MOVED,
    "investment/portfolio_strategy_lab/application/readback.py::_receipt_failure": NOT_MOVED,
    "investment/portfolio_strategy_lab/application/readback.py::_release_failure": NOT_MOVED,
    "investment/portfolio_strategy_lab/application/task.py::"
    "PortfolioResearchTaskInput.validate_selection": SELF,
    "investment/portfolio_strategy_lab/application/tranche_book_execution.py::"
    "TrancheBookDecisionProvider._admit_risk_lane": NOT_MOVED,
    "investment/portfolio_strategy_lab/policies/buffered_equal_weight.py::"
    "WholeBookHysteresisEqualWeightRecipe.validate_identity": SELF,
    "investment/portfolio_strategy_lab/policies/buffered_inverse_volatility.py::"
    "WholeBookHysteresisInverseVolatilityRecipe.validate_identity": SELF,
    "investment/portfolio_strategy_lab/policies/buffered_rank_return.py::"
    "WholeBookHysteresisCausalRankMuRecipe.validate_identity": SELF,
    "investment/portfolio_strategy_lab/policies/installed_strategies.py::"
    "ComponentBookRecipe.validate_identity": SELF,
    "investment/portfolio_strategy_lab/policies/lifecycle_research.py::"
    "LifecycleScoreEvidence.validate_frozen_recipe": WORKSPACE,
    "investment/portfolio_strategy_lab/policies/lifecycle_research.py::"
    "LocalQAOutcomeBinding.verify": NOT_MOVED,
    "investment/portfolio_strategy_lab/policies/minimum_variance.py::"
    "MinimumVarianceDevelopmentRecipe.validate_identity": SELF,
    "investment/portfolio_strategy_lab/policies/return_scaled_total_signal.py::"
    "ReturnScaledTotalSignalRecipe.validate_identity": SELF,
    "investment/portfolio_strategy_lab/policies/score_risk_cost.py::"
    "RankBufferedScoreRiskCostDevelopmentRecipe.validate_identity": SELF,
    "investment/portfolio_strategy_lab/policies/score_risk_cost.py::"
    "ScoreRiskCostDevelopmentRecipe.validate_identity": SELF,
    "investment/portfolio_strategy_lab/policies/score_risk_cost.py::"
    "ScoreRiskCostTrialRecipe.validate_identity": SELF,
    "investment/portfolio_strategy_lab/policies/tranche_book.py::"
    "TrancheBookRecipe.validate_recipe": SELF,
}
"""Each comparison site, by its file and the function it sits in, with the reason it may compare
by equality."""

SEALED_ADMISSIONS = {
    "investment/alpha_research/scores/heterogeneous_replay.py::"
    "admit_heterogeneous_current_closure": (
        "strategy_recipe_is_current",
        "component_recipe_is_current",
    ),
}
"""The admissions of a sealed root the tree must read, each with the recorded-move readers it
calls in place of an equality."""


class _Walk(ast.NodeVisitor):
    """Each function's node and each equality on a moved hash, keyed by file and function."""

    def __init__(self, relative: str, found: dict[str, ast.AST]) -> None:
        self.relative = relative
        self.found = found
        self.stack: list[str] = []

    def _key(self) -> str:
        return f"{self.relative}::{'.'.join(self.stack)}"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.stack.append(node.name)
        self.found.setdefault(f"{self._key()}#def", node)
        self.generic_visit(node)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_Compare(self, node: ast.Compare) -> None:
        named = {
            value.attr if isinstance(value, ast.Attribute) else value.id
            for side in (node.left, *node.comparators)
            for value in ast.walk(side)
            if isinstance(value, ast.Attribute | ast.Name)
        }
        if named & FIELDS and any(isinstance(op, ast.Eq | ast.NotEq) for op in node.ops):
            self.found[self._key()] = node
        self.generic_visit(node)


def _sites() -> dict[str, ast.AST]:
    found: dict[str, ast.AST] = {}
    for owner in OWNERS:
        for path in sorted((SOURCE / owner).glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            _Walk(path.relative_to(SOURCE).as_posix(), found).visit(tree)
    return found


def test_every_equality_on_a_moved_hash_has_its_reason() -> None:
    """requirement (TE12, NM2, V451): the walk finds exactly the classified comparisons: one
    added unclassified, or one classified that left, fails here."""

    sites = {key for key in _sites() if not key.endswith("#def")}
    assert sites == set(CLASSIFIED)


def test_a_sealed_root_admission_reads_recorded_moves() -> None:
    """regression (NM2, V451): the heterogeneous closure, sealed before NM1, refused at its pins
    because the admission compared its strategy and recipe hashes by equality. Each admission of
    a sealed root reads the recorded moves instead, and holds no such equality."""

    sites = _sites()
    for key, readers in SEALED_ADMISSIONS.items():
        node = sites[f"{key}#def"]
        called = {
            value.func.id
            for value in ast.walk(node)
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
        }
        assert set(readers) <= called, key
        assert key not in sites, key
