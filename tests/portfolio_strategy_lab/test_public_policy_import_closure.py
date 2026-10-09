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
    """Importing the policy contract loads no solver."""

    result = _probe(
        "import alphalattice.investment.portfolio_strategy_lab.policies.contracts; " + _ASSERT_CLEAN
    )
    assert result.returncode == 0, result.stderr


def test_building_a_closed_form_catalog_loads_no_solver() -> None:
    """Building a closed form catalog loads no solver."""

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
    """The probe would catch a solver import."""

    result = _probe(f"import {OPTIMIZER}; " + _ASSERT_CLEAN)
    assert result.returncode != 0
    assert "AssertionError" in result.stderr


@pytest.mark.parametrize("solver", SOLVERS)
def test_each_named_solver_is_actually_installed(solver: str) -> None:
    """Each named solver is actually installed."""

    result = _probe(f"import {solver}")
    assert result.returncode == 0, f"{solver} is not installed: {result.stderr}"


def test_the_public_executor_reaches_no_excluded_research_lane() -> None:
    """The public executor reaches no excluded research lane."""

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
    """Each complete public entrypoint keeps the process import closure clean."""

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
    """The excluded package probe would catch each lane."""

    result = _probe(f"import {module}; " + _ASSERT_NO_EXCLUDED_PACKAGE)
    assert result.returncode != 0
    assert "AssertionError" in result.stderr


def test_the_local_application_service_reaches_no_solver_or_masked_lane() -> None:
    """The local application service reaches no solver or masked lane."""

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
    """The portfolio finalization seam loads no validation runtime."""

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
    """The validation gate loads no solver and no numerical path."""

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
    """The readback owner loads no validation runtime and no product host."""

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
    """The shared commit primitive couples to neither of its two users."""

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
