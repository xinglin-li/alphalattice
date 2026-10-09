"""The public policy path loads no solver.

`DEFAULT_DESKTOP_PROCESS_IMPORTS_NO_OPTIMIZER_SOLVER_OR_OPTUNA` is a property of
a *process*, not of a source file, so every check here runs in a fresh
interpreter. Asserting it inside the pytest process would prove nothing: by the
time this module runs, another test has already imported CVXPY and
``sys.modules`` is contaminated for reasons that have nothing to do with the
public path.

The idiom follows the existing source architecture import guards.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOLVERS = ("cvxpy", "osqp", "optuna")
OPTIMIZER = "alphalattice.investment.portfolio_strategy_lab.optimizer.service"

_PSL = "alphalattice.investment.portfolio_strategy_lab"
_RISK = "alphalattice.investment.risk_research"
EXCLUDED_PACKAGES = (
    # Research lanes the delivery plan masks from the public composition. Named
    # as packages rather than modules because the property is "the public path
    # never reaches this lane", and a single module inside one is enough to
    # falsify it. `campaign` now holds only the lanes the paired research route
    # reads (the campaign itself retired with R01).
    f"{_PSL}.regularization",
    f"{_PSL}.optimizer",
    f"{_PSL}.search",
    f"{_PSL}.campaign",
    f"{_RISK}.experiments",
    f"{_RISK}.calibration",
    f"{_RISK}.estimators",
    "alphalattice.investment.sector_research",
)


def _probe(body: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", body],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "ALPHALATTICE_NETWORK_DISABLED": "1"},
        check=False,
        capture_output=True,
        text=True,
    )


_ASSERT_CLEAN = (
    "import sys; "
    f"solvers={SOLVERS!r}; "
    "loaded=sorted({name.split('.')[0] for name in sys.modules} & set(solvers)); "
    f"assert {OPTIMIZER!r} not in sys.modules, 'optimizer.service loaded'; "
    "assert not loaded, loaded"
)

_ASSERT_NO_EXCLUDED_PACKAGE = (
    "import sys; "
    f"excluded={EXCLUDED_PACKAGES!r}; "
    "reached=sorted(name for name in sys.modules "
    "if any(name == value or name.startswith(value + '.') for value in excluded)); "
    "assert not reached, reached"
)


def test_importing_the_policy_contract_loads_no_solver() -> None:
    """The method-neutral contract is the one every policy imports.

    Its two optimizer names are annotations only, so the edge is type-level and
    the process stays solver-free.
    """

    result = _probe(
        "import alphalattice.investment.portfolio_strategy_lab.policies.contracts; " + _ASSERT_CLEAN
    )
    assert result.returncode == 0, result.stderr


def test_building_a_closed_form_catalog_loads_no_solver() -> None:
    """Constructing the public catalog is the operation the plan names explicitly.

    This is the stronger claim: not merely that a contract imports cleanly, but
    that the composition a public Product Host would perform does too.
    """

    result = _probe(
        "from alphalattice.investment.portfolio_strategy_lab.policies.buffered_equal_weight import "
        "WholeBookHysteresisEqualWeightAdapter; "
        "from alphalattice.investment.portfolio_strategy_lab.policies"
        ".buffered_inverse_volatility import "
        "WholeBookHysteresisInverseVolatilityAdapter; "
        "from alphalattice.investment.portfolio_strategy_lab.policies.buffered_rank_return import "
        "WholeBookHysteresisCausalRankMuAdapter; "
        "from alphalattice.investment.portfolio_strategy_lab.policies.catalog import "
        "PortfolioPolicyCatalog; "
        "catalog = PortfolioPolicyCatalog((WholeBookHysteresisEqualWeightAdapter(), "
        "WholeBookHysteresisInverseVolatilityAdapter(), "
        "WholeBookHysteresisCausalRankMuAdapter())); "
        "assert len(catalog.adapters) == 3; "
        "assert catalog.binding.catalog_hash; " + _ASSERT_CLEAN
    )
    assert result.returncode == 0, result.stderr


def test_the_frozen_alpha_product_recipe_loads_no_solver() -> None:
    """The Gate 8A recipe is on the same public path and must stay clean."""

    result = _probe(
        "from alphalattice.investment.alpha_research.scores.product_recipe import "
        "INSTALLED_ALPHA_PRODUCT_RECIPE as r; "
        "assert r.resolve_estimator_parameters(); " + _ASSERT_CLEAN
    )
    assert result.returncode == 0, result.stderr


def test_the_probe_would_catch_a_solver_import() -> None:
    """The guard above is only worth having if it can fail.

    A test that asserts an absence is indistinguishable from a test that asserts
    nothing until you show it detecting the thing it forbids, so this imports the
    optimizer deliberately and requires the same assertion to reject it.
    """

    result = _probe(f"import {OPTIMIZER}; " + _ASSERT_CLEAN)
    assert result.returncode != 0
    assert "AssertionError" in result.stderr


@pytest.mark.parametrize("solver", SOLVERS)
def test_each_named_solver_is_actually_installed(solver: str) -> None:
    """Absence must mean "not imported", never "not installed".

    The parent project ships `cvxpy-base`, `osqp` and `optuna`, so a green guard
    could otherwise be an artefact of a missing dependency rather than evidence
    about the import graph.
    """

    result = _probe(f"import {solver}")
    assert result.returncode == 0, f"{solver} is not installed: {result.stderr}"


def test_the_public_executor_reaches_no_excluded_research_lane() -> None:
    """The strongest form: the whole public path, not one contract on it.

    `tranche_book_execution` is what a public Product Host composes -- the
    closed-form policy, the Risk allocation projection and the Alpha replay
    consumer, wired together. If any of the masked research lanes were reachable
    from there, the Gate's exit condition would be false however clean the
    individual contracts look.

    `campaign` is in the excluded list for a reason that is not solver contagion:
    what is left of it serves the paired research route, and this asserts the
    public path has no stake in that route.
    """

    result = _probe(
        "import alphalattice.investment.portfolio_strategy_lab.application."
        "tranche_book_execution as m; "
        "assert m.TrancheBookDecisionProvider is not None; "
        + _ASSERT_CLEAN
        + "; "
        + _ASSERT_NO_EXCLUDED_PACKAGE
    )
    assert result.returncode == 0, result.stderr


def test_the_public_risk_surface_reaches_no_excluded_research_lane() -> None:
    """Risk enters the public path as one decomposition module and nothing else."""

    result = _probe(
        "from alphalattice.investment.risk_research.surfaces.decomposition import "
        "RiskAllocationProjection; "
        "assert RiskAllocationProjection is not None; "
        + _ASSERT_CLEAN
        + "; "
        + _ASSERT_NO_EXCLUDED_PACKAGE
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("entry", "lanes_excluded"),
    [
        ("import alphalattice.control.product_host.composition.portfolio_application", False),
        ("import alphalattice.investment.portfolio_strategy_lab.application.task", True),
        (
            "import runpy; "
            "runpy.run_path('scripts/run_public_portfolio_research.py', run_name='probe')",
            False,
        ),
    ],
)
def test_each_complete_public_entrypoint_keeps_the_process_import_closure_clean(
    entry: str, lanes_excluded: bool
) -> None:
    """Gate 8B is a complete application closure, not just a clean policy leaf.

    No entry loads a solver. The Product Host's composition and the thin CLI serve the
    research lanes as well (the Workbench runs their studies), so only the Portfolio Task
    keeps the lanes out of its process.
    """

    checks = [_ASSERT_CLEAN, *([_ASSERT_NO_EXCLUDED_PACKAGE] if lanes_excluded else [])]
    result = _probe(f"{entry}; " + "; ".join(checks))
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "module",
    [
        "alphalattice.investment.portfolio_strategy_lab.regularization.split_policy",
        "alphalattice.investment.portfolio_strategy_lab.campaign.upstream",
    ],
)
def test_the_excluded_package_probe_would_catch_each_lane(module: str) -> None:
    """A guard asserting an absence is worth nothing until it detects a presence.

    Both named modules are import-clean leaves -- neither pulls a solver -- so
    only the package check can catch them, and this proves it does.
    """

    result = _probe(f"import {module}; " + _ASSERT_NO_EXCLUDED_PACKAGE)
    assert result.returncode != 0
    assert "AssertionError" in result.stderr


def test_the_split_policy_owner_is_singular_and_correctly_placed() -> None:
    """One owner of the anchored fold geometry, beside the contracts that read it.

    `anchored_fold_geometry` lives in `regularization/split_policy.py` because the
    regularization record contracts consume it: `regularization/contracts.py`
    imports it relatively (its runtime retired with the lab, R06; the Stage-6
    campaign's split-policy binding with R01).

    The module's own docstring records that it exists to collapse three copies of
    one geometry, which is the duplication this asserts has not returned.
    """

    src = ROOT / "src" / "alphalattice" / "investment" / "portfolio_strategy_lab"
    definitions = [
        path.relative_to(ROOT).as_posix()
        for path in src.rglob("*.py")
        if "def anchored_fold_geometry(" in path.read_text(encoding="utf-8")
    ]
    assert definitions == [
        "src/alphalattice/investment/portfolio_strategy_lab/regularization/split_policy.py"
    ]

    regularization_consumers = sorted(
        path.name
        for path in (src / "regularization").glob("*.py")
        if "from .split_policy import" in path.read_text(encoding="utf-8")
    )
    assert regularization_consumers == ["contracts.py"]


def test_the_local_application_service_reaches_no_solver_or_masked_lane() -> None:
    """The product boundary is the widest thing a Desktop imports.

    It is also where contamination would be least visible: a service that pulled
    the Campaign lane in through one convenience import would look fine in every
    unit test and load a solver on a researcher's laptop.
    """

    result = _probe(
        "from alphalattice.interface.local_application.portfolio_research import "
        "LocalPortfolioResearchService; "
        "assert LocalPortfolioResearchService is not None; "
        + _ASSERT_CLEAN
        + "; "
        + _ASSERT_NO_EXCLUDED_PACKAGE
    )
    assert result.returncode == 0, result.stderr


def test_the_control_catalog_alone_reaches_no_solver_or_masked_lane() -> None:
    """The catalog is what a UI renders before any workspace exists."""

    result = _probe(
        "from alphalattice.investment.portfolio_strategy_lab.application.controls import "
        "INSTALLED_PUBLIC_CONTROL_CATALOG as c; "
        "assert c.catalog_hash and len(c.controls) == 9; "
        + _ASSERT_CLEAN
        + "; "
        + _ASSERT_NO_EXCLUDED_PACKAGE
    )
    assert result.returncode == 0, result.stderr


_VALIDATION_RUNTIME = "alphalattice.oversight.model_validation"

_ASSERT_NO_VALIDATION_RUNTIME = (
    "import sys; "
    f"reached=sorted(n for n in sys.modules if n.startswith({_VALIDATION_RUNTIME!r})); "
    "assert not reached, reached"
)


def test_the_portfolio_finalization_seam_loads_no_validation_runtime() -> None:
    """Portfolio declares the port; it must not be able to reach an implementation.

    The whole injection story rests on this. If importing the finalization
    contracts pulled the Gate in, "composition injects it" would be a convention
    rather than a structure, and the next caller could construct one directly.
    """

    result = _probe(
        "from alphalattice.investment.portfolio_strategy_lab.application.finalization import "
        "ProtectedEvaluationPort, FrozenPortfolioCandidate; "
        "assert ProtectedEvaluationPort is not None and FrozenPortfolioCandidate is not None; "
        + _ASSERT_NO_VALIDATION_RUNTIME
    )
    assert result.returncode == 0, result.stderr


def test_the_finalization_task_adapter_loads_no_validation_runtime() -> None:
    """The adapter that consumes the port is the likeliest place to slip."""

    result = _probe(
        "from alphalattice.investment.portfolio_strategy_lab.application.finalization_task "
        "import PortfolioFinalizationTaskAdapter as a; "
        "assert a.task_kind; " + _ASSERT_NO_VALIDATION_RUNTIME
    )
    assert result.returncode == 0, result.stderr


def test_the_shared_backtesting_owner_loads_no_validation_runtime_or_storage() -> None:
    """Backtesting is mechanics: no Validation, no storage, no Strategy Lab."""

    result = _probe(
        "from alphalattice.capabilities.portfolio_backtesting.segments import "
        "run_portfolio_walk_forward_segment as run; "
        "assert run is not None; " + _ASSERT_NO_VALIDATION_RUNTIME + "; import sys; "
        "assert not [n for n in sys.modules if n.startswith("
        "'alphalattice.investment.portfolio_strategy_lab')], 'strategy lab reached'"
    )
    assert result.returncode == 0, result.stderr


def test_the_portfolio_reporting_owner_loads_no_validation_runtime() -> None:
    """A report is a projection of Portfolio facts, not of a verdict."""

    result = _probe(
        "from alphalattice.investment.portfolio_strategy_lab.reporting.static import "
        "render_portfolio_research_html as render; "
        "assert render is not None; " + _ASSERT_NO_VALIDATION_RUNTIME
    )
    assert result.returncode == 0, result.stderr


def test_the_validation_gate_loads_no_solver_and_no_numerical_path() -> None:
    """The Gate verifies frozen contracts; it owns no Backtesting invocation.

    It may import the Portfolio contracts it has to verify -- the registry admits
    that peer and the constitution names it -- but reaching the executor, the
    Backtesting engine or a solver would make it a second numerical path.
    """

    result = _probe(
        "from alphalattice.oversight.model_validation.protected_gate import "
        "ProtectedValidationGate as g; "
        "assert g is not None; " + _ASSERT_CLEAN + "; import sys; "
        "forbidden=['alphalattice.capabilities.portfolio_backtesting.engine',"
        "'alphalattice.capabilities.portfolio_backtesting.segments',"
        "'alphalattice.investment.portfolio_strategy_lab.application.executor']; "
        "reached=[n for n in forbidden if n in sys.modules]; "
        "assert not reached, reached"
    )
    assert result.returncode == 0, result.stderr


def test_the_readback_owner_loads_no_validation_runtime_and_no_product_host() -> None:
    """Readback proves other people's artifacts; it may not depend on either.

    It declares the two opener ports and composition supplies them, so importing
    it must not drag in the Gate whose receipt it checks, nor the Host that wires
    the two together. If it did, "readback verifies what the Gate said" would be
    one module verifying its own collaborator.
    """

    result = _probe(
        "from alphalattice.investment.portfolio_strategy_lab.application.readback import "
        "exact_readback, strong_replay, FinalizationArtifactOpener, UpstreamEvidenceOpener; "
        "assert exact_readback is not None and strong_replay is not None; "
        "assert FinalizationArtifactOpener is not None and UpstreamEvidenceOpener is not None; "
        + _ASSERT_NO_VALIDATION_RUNTIME
        + "; import sys; "
        "assert not [n for n in sys.modules if n.startswith("
        "'alphalattice.control.product_host')], 'product host reached'"
    )
    assert result.returncode == 0, result.stderr


def test_the_finalization_store_loads_no_validation_runtime() -> None:
    """Portfolio's registry is what the Gate reads; it must not read the Gate."""

    result = _probe(
        "from alphalattice.investment.portfolio_strategy_lab.publication.finalization_ledger "
        "import PortfolioFinalizationStore as s; "
        "assert s is not None; " + _ASSERT_NO_VALIDATION_RUNTIME
    )
    assert result.returncode == 0, result.stderr


def test_the_shared_commit_primitive_couples_to_neither_of_its_two_users() -> None:
    """The crash-window rule is stated once, below both authorities that need it.

    `CommittedIndex` sits in the content store precisely so Portfolio's
    finalization store and the Validation Gate can share it without either
    importing the other. Package `__init__` chains drag in parents regardless of
    what this module does, so the claim under test is the one that can actually
    be violated: it must not reach either of its own users, or the executor.
    """

    result = _probe(
        "from alphalattice.control.workspace_runtime.content_store "
        "import CommittedIndex; "
        "assert CommittedIndex is not None; " + _ASSERT_NO_VALIDATION_RUNTIME + "; import sys; "
        "forbidden=['alphalattice.control.product_host.composition.portfolio_finalization',"
        "'alphalattice.investment.portfolio_strategy_lab.publication.finalization_ledger',"
        "'alphalattice.investment.portfolio_strategy_lab.application.executor']; "
        "reached=[n for n in forbidden if n in sys.modules]; "
        "assert not reached, reached"
    )
    assert result.returncode == 0, result.stderr
