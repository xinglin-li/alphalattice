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
    """Development execution has no authority to move a current pointer.

    Enforced in the import graph rather than by convention: a development
    executor that can reach its publication owner is one refactor away from
    writing a current pointer, and nothing in a green test suite would notice
    until it did.

    Stated for every Desk, not just Risk. Risk proved the property; a Desk-local
    test could not stop the next Desk from reintroducing it, and Factor's
    executor is exactly the module that would have been tempted -- its
    deterministic primitives sit one package away from the production runtime
    that owns the pointer.
    """

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
    """Adding a method must not have taught any shared owner its name.

    Each capability in this milestone was proven by installing a method the
    product does not ship: an extension Feature kernel, a second Risk recipe
    schema, a third Alpha standardization, a third searchable Portfolio policy.
    If any generic owner mentioned one of them, "install a method" would mean
    "install a method and edit the shared path", and the next one would need the
    same edits again.

    The Feature extension is the exception that proves the rule and is checked
    separately below: its mathematics and its typed recipe are product-owned by
    design, so the name is expected in the capability that owns it and nowhere
    else.
    """

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
    """The one product-owned extension, and it lives in exactly one place.

    ``overnight_return`` is deliberately different from the other three: its
    mathematics and its typed ``FactorSpec`` belong in the Feature capability,
    because a recipe assembled in a test fixture would make "add a method in the
    domain directory" false. So the name is required in its owner and forbidden
    everywhere else -- including the panel publisher and the Factor consumer that
    quote the catalog summary it appears in.
    """

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
    """Nothing under ``src`` may import the composition root.

    ``product_host`` is where concrete, Desk-aware wiring is allowed to live, and
    that is only safe while it has zero legal in-degree: anything that could
    import it could reach every Desk through it, and the layering would be a
    naming convention rather than a constraint.
    """

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
    """Each Desk kind resolves to a real executor branch, not to "not installed".

    The Factor executor existed for a whole milestone while
    ``build_installed_desk_executors`` installed Risk alone, so the only thing
    that could run a Factor experiment was a test that built the executor itself.
    A kind the canonical installer does not recognize is a Desk the product
    cannot run, regardless of what its own package contains.
    """

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
    """Resolving one Desk's authority is not a precondition for another's.

    The eager installer read Risk's return surface and sector state before
    returning anything, so a Factor run in a workspace that had never published a
    return surface failed on an artifact Factor does not read. Asserted on the
    source because the failure it prevents only appears in workspaces that are
    missing something, and a green workspace cannot show it.
    """

    source = (SOURCE_ROOT / "control/product_host/research_authoring/execution.py").read_text(
        encoding="utf-8"
    )
    installer = source.split("def build_installed_desk_executors", maxsplit=1)[1]
    # Every branch returns a single-element tuple, and the Risk-only reads live
    # behind the Risk branch rather than above the dispatch.
    assert installer.count("return (") == len(_DEVELOPMENT_KINDS)
    assert "MarketDataRepository" not in installer


def test_no_development_executor_can_reach_a_publication_owner() -> None:
    """Restated for the two executors this milestone added.

    ``test_no_desk_experiments_package_can_reach_its_publication_owner`` covers
    the packages; this names the two modules that would have been tempted, since
    both sit one import away from a runtime that owns a current pointer.
    """

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
    """Alpha's development binding is beside the constitution, not inside it.

    The tempting shortcut was to fill ``ResearchFoundationBinding``'s four
    lineage fields -- a factor training outcome, a screening result, a candidate
    slate, a Research Desk factor input -- with the one Factor development hash a
    development run has. That validates and hashes, and it is false: four
    different questions answered with the same evidence. This asserts the
    constitution still asks for all four separately.
    """

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
