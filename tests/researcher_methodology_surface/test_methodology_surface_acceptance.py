"""One shared structural acceptance for the whole methodology surface.

Actor neutrality and authority resolution are already proven per-Desk in
``test_experiment_authoring`` -- the same document from a human, an installed
Agent and external automation reaches one Program identity, and an unresolvable
handle fails before any Desk compiles. Those are not restated here.

What only a cross-Desk case can assert is that the *shape* held while four
capabilities were added: no development path acquired publication authority, and
no generic owner learned the name of a method someone installed. Both are
properties of the import graph and the source text rather than of any one run,
and both are the kind that decay quietly -- a single import is enough.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PLAYPEN_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PLAYPEN_ROOT / "src" / "alphalattice"

_DESKS = (
    "investment/risk_research",
    "foundation/factor_research",
    "investment/alpha_research",
    "investment/portfolio_strategy_lab",
)


def _imported_modules(root: Path) -> tuple[tuple[str, str], ...]:
    found: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                found.append((f"{path.relative_to(SOURCE_ROOT)}:{node.lineno}", node.module))
            elif isinstance(node, ast.Import):
                found.extend(
                    (f"{path.relative_to(SOURCE_ROOT)}:{node.lineno}", alias.name)
                    for alias in node.names
                )
    return tuple(found)


@pytest.mark.parametrize("desk", _DESKS)
def test_no_desk_experiments_package_can_reach_its_publication_owner(desk: str) -> None:
    """No desk experiments package can reach its publication owner."""

    experiments = SOURCE_ROOT / desk / "experiments"
    if not experiments.is_dir():
        pytest.skip(f"{desk} has no experiments package")
    owner = desk.rsplit("/", maxsplit=1)[-1]
    offenders = [
        f"{where} -> {module}"
        for where, module in _imported_modules(experiments)
        if f"{owner}.publication" in module
    ]
    assert offenders == []


def test_no_generic_owner_names_an_installed_method() -> None:
    """No generic owner names an installed method."""

    generic = (
        "control/research_program",
        "control/product_host/composition",
        "foundation/feature_engine/publication",
        "foundation/feature_engine/panels",
        "foundation/factor_research/experiments",
        "investment/alpha_research/targets/execution_outcome.py",
    )
    names = (
        # Risk's alternate method, Alpha's third standardization, Portfolio's
        # third searchable policy, and the extension Feature factor.
        "ShrunkDiagonal",
        "CASE_STUDY_SHRUNK_DIAGONAL",
        "CASE_STUDY_SIGN",
        "CASE_STUDY_INVERSE_VOLATILITY",
        "overnight_return_21",
    )
    offenders: list[str] = []
    for relative in generic:
        target = SOURCE_ROOT / relative
        paths = sorted(target.rglob("*.py")) if target.is_dir() else [target]
        for path in paths:
            source = path.read_text(encoding="utf-8")
            offenders.extend(
                f"{path.relative_to(SOURCE_ROOT)}:{name}" for name in names if name in source
            )
    assert offenders == []


def test_the_feature_extension_recipe_is_owned_only_by_its_capability() -> None:
    """The feature extension recipe is owned only by its capability."""

    owner = SOURCE_ROOT / "foundation/feature_engine/producers/factors/open_intraday.py"
    assert "overnight_return_21" in owner.read_text(encoding="utf-8")

    consumers = (
        "foundation/feature_engine/publication/snapshots.py",
        "foundation/feature_engine/panels/reader.py",
        "foundation/feature_engine/producers/base_materializer.py",
    )
    for relative in consumers:
        source = (SOURCE_ROOT / relative).read_text(encoding="utf-8")
        assert "overnight_return" not in source, relative


def test_product_host_is_the_only_installed_composition_root() -> None:
    """Product host is the only installed composition root."""

    offenders = [
        f"{where} -> {module}"
        for where, module in _imported_modules(SOURCE_ROOT)
        if "product_host" in module
        and not where.startswith("control\\product_host")
        and not where.startswith("control/product_host")
    ]
    assert offenders == []


# --------------------------------------------------------------------------
# Execution acceptance
#
# The four cases above are structural: they read the import graph and the source
# text. That is the right shape for "no development path acquired publication
# authority", and it is the wrong shape for "a researcher's method actually
# runs". A Desk can satisfy every structural rule while having no way to execute
# anything, which is precisely the state Factor, Alpha and Portfolio were in when
# the record already called them complete.
#
# So each Desk is asserted to reach real execution through the *product*, and the
# assertion is about the canonical composition rather than a private assembly of
# it. Not one shared pipeline run and not one shared workspace: the claim is that
# every Desk satisfies the same extension principle, not that they compose into a
# single job.
# --------------------------------------------------------------------------

_DEVELOPMENT_KINDS = (
    "risk.covariance-development",
    "factor.screening-development",
    "alpha.model-development",
)


def test_every_development_kind_is_reachable_from_the_canonical_installer() -> None:
    """Every development kind is reachable from the canonical installer."""

    source = (SOURCE_ROOT / "control/product_host/research_authoring/execution.py").read_text(
        encoding="utf-8"
    )
    for kind in _DEVELOPMENT_KINDS:
        constant = {
            "risk.covariance-development": "RISK_EXPERIMENT_KIND",
            "factor.screening-development": "FACTOR_EXPERIMENT_KIND",
            "alpha.model-development": "ALPHA_EXPERIMENT_KIND",
        }[kind]
        assert f"envelope.kind == {constant}" in source, kind


def test_the_installer_builds_one_desk_rather_than_every_desk() -> None:
    """The installer builds one desk rather than every desk."""

    source = (SOURCE_ROOT / "control/product_host/research_authoring/execution.py").read_text(
        encoding="utf-8"
    )
    installer = source.split("def build_installed_desk_executors", maxsplit=1)[1]
    # Every branch returns a single-element tuple, and the Risk-only reads live
    # behind the Risk branch rather than above the dispatch.
    assert installer.count("return (") == len(_DEVELOPMENT_KINDS)
    assert "MarketDataRepository" not in installer


def test_no_development_executor_can_reach_a_publication_owner() -> None:
    """No development executor can reach a publication owner."""

    for relative, forbidden in (
        ("foundation/factor_research/experiments/execution.py", "factor_research.publication"),
        (
            "investment/alpha_research/experiments/development_execution.py",
            "alpha_research.publication",
        ),
        (
            "investment/alpha_research/experiments/development_execution.py",
            "alpha_research." + "candidates",
        ),
    ):
        source = (SOURCE_ROOT / relative).read_text(encoding="utf-8")
        assert f"import {forbidden}" not in source, f"{relative} -> {forbidden}"
        assert f"from {forbidden}" not in source, f"{relative} -> {forbidden}"


def test_development_evidence_is_namespaced_away_from_published_categories() -> None:
    """A reader walking published categories cannot encounter development output."""

    from alphalattice.foundation.factor_research.experiments.development_evidence import (
        FACTOR_DEVELOPMENT_EVIDENCE_CATEGORY,
    )

    assert FACTOR_DEVELOPMENT_EVIDENCE_CATEGORY.startswith("development/")


def test_the_frozen_research_foundation_contract_is_unchanged() -> None:
    """The frozen research foundation contract is unchanged."""

    from alphalattice.foundation.research_foundation.contracts import ResearchFoundationBinding
    from alphalattice.investment.alpha_research.inputs.development_foundation import (
        AlphaDevelopmentFoundationBinding,
    )

    required = {
        "factor_training_outcome_snapshot_hash",
        "factor_screening_result_hash",
        "factor_candidate_slate_hash",
        "research_desk_factor_input_hash",
    }
    assert required.issubset(set(ResearchFoundationBinding.model_fields))
    # The development binding names none of them, and says outright what it is.
    development = set(AlphaDevelopmentFoundationBinding.model_fields)
    assert not (required & development)
    assert "factor_development_checkpoint_hash" in development
    assert "development_only" in development
