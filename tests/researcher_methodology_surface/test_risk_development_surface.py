"""Recipe parameters, program identity and the development surface.

The declared domain admits a second recipe parameter that reaches the real
estimator, a recipe that disagrees with the program fails before any
estimate, experiments never import publication, the compiler, writer and
verifier name nothing they must not reach, the default catalog installs only
real capabilities, changing development code moves the program identity and
every declared owner actually moves it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from alphalattice.investment.risk_research.contracts import (
    CovarianceRecipe,
    default_covariance_recipe,
    seal_contract,
)
from alphalattice.investment.risk_research.estimators.capability import (
    CANONICAL_NO_RANDOMNESS_SEED,
    RiskCapabilityError,
    RiskRecipeAdmission,
)
from alphalattice.investment.risk_research.estimators.catalog import (
    build_installed_risk_estimator_catalog,
)
from alphalattice.investment.risk_research.estimators.covariance import (
    COVARIANCE_RECIPE_SCHEMA_ID,
    CovarianceEstimatorAdapter,
)
from alphalattice.investment.risk_research.estimators.domains import (
    COVARIANCE_PARAMETER_DOMAIN,
    ParameterDomainError,
)
from alphalattice.investment.risk_research.experiments.contracts import (
    RISK_EXPERIMENT_KIND,
    RiskDevelopmentProgramBinding,
)
from alphalattice.investment.risk_research.experiments.execution import (
    RiskDevelopmentExecution,
    RiskDevelopmentExecutor,
)
from alphalattice.investment.risk_research.experiments.identity import (
    RISK_DEVELOPMENT_SOURCE_PATHS,
    RISK_NUMERICAL_SOURCE_PATHS,
    risk_development_source_closure_hash,
    risk_numerical_source_closure_hash,
)
from alphalattice.investment.risk_research.experiments.observation import NumericalCallCounter
from alphalattice.investment.risk_research.experiments.verification import RiskEvidenceVerifier
from alphalattice.investment.risk_research.experiments.window import (
    resolve_development_input_binding,
)
from alphalattice.kernel.shared_kernel.source_identity import (
    number_deciding_closure,
    number_deciding_rule,
)
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    ResearchExecutionEvidence,
    ResolvedResearchAuthority,
    SealedResearchProgram,
)
from tests.researcher_methodology_surface.real_workspace import RealRiskWorkspace
from tests.researcher_methodology_surface.risk_development_program_support import (
    _admission,
)
from tests.researcher_methodology_surface.risk_development_support import (
    PLAYPEN_ROOT,
    _binding,
    _bounded_authority,
)

RISK_ROOT = PLAYPEN_ROOT / "src" / "alphalattice" / "investment" / "risk_research"


def _sealed_recipe(**parameters: object) -> CovarianceRecipe:
    """Seal exactly what an author would have written, through the real domain."""

    admitted = COVARIANCE_PARAMETER_DOMAIN.admit(parameters)
    return seal_contract(CovarianceRecipe, "recipe_hash", **admitted)


def test_the_declared_domain_admits_a_second_recipe_parameter() -> None:
    """The declared domain admits a second recipe parameter."""

    axis = next(a for a in COVARIANCE_PARAMETER_DOMAIN.axes if a.name == "ewma_decay")
    assert len(axis.admissible) > 1
    assert axis.default == 0.94
    assert _sealed_recipe().recipe_hash == default_covariance_recipe().recipe_hash
    assert _sealed_recipe(ewma_decay=0.97).recipe_hash != default_covariance_recipe().recipe_hash
    with pytest.raises(ParameterDomainError):
        COVARIANCE_PARAMETER_DOMAIN.admit({"ewma_decay": 0.5})


def test_an_authored_recipe_parameter_reaches_the_real_estimator(
    real_risk_workspace: RealRiskWorkspace,
    tmp_path: Path,
) -> None:
    """An authored recipe parameter reaches the real estimator."""

    authority, requested = _bounded_authority(real_risk_workspace, count=4)
    input_binding, bounded = resolve_development_input_binding(
        authority=authority,
        return_surface=real_risk_workspace.return_surface,
        return_reader=real_risk_workspace.return_reader,
        freshness_probe=real_risk_workspace.freshness_probe,
    )

    def _run(admission: RiskRecipeAdmission, label: str) -> RiskDevelopmentExecution:
        return RiskDevelopmentExecutor().execute(
            binding=_binding(recipe_hash=admission.recipe_hash),
            input_binding=input_binding,
            admission=admission,
            return_surface=real_risk_workspace.return_surface,
            return_reader=real_risk_workspace.return_reader,
            bounded_sessions=bounded,
            output_workspace=tmp_path / label,
            sector_by_listing_id=real_risk_workspace.sector_by_listing_id,
        )

    default = _run(_admission(), "default")
    altered = _run(_admission(ewma_decay=0.97), "altered")

    # The adapter received the authored value: the surface records the exact
    # envelope that ran, and its parameters are the ones that were sealed.
    assert default.build.surface.recipe_envelope.parameters["ewma_decay"] == 0.94
    assert altered.build.surface.recipe_envelope.parameters["ewma_decay"] == 0.97
    # And it reached the mathematics, not just the metadata.
    assert _chunk_matrix_hashes(default) != _chunk_matrix_hashes(altered)
    assert default.surface_hash != altered.surface_hash
    assert default.build.surface.diagnostics_hash != altered.build.surface.diagnostics_hash
    assert len(requested) == 4


def _chunk_matrix_hashes(execution: RiskDevelopmentExecution) -> tuple[str, ...]:
    return tuple(value for chunk in execution.build.surface.chunks for value in chunk.matrix_hashes)


def test_a_recipe_that_disagrees_with_the_program_fails_before_any_estimate(
    real_risk_workspace: RealRiskWorkspace,
    tmp_path: Path,
) -> None:
    """A recipe that disagrees with the program fails before any estimate."""

    authority, _requested = _bounded_authority(real_risk_workspace, count=3)
    input_binding, bounded = resolve_development_input_binding(
        authority=authority,
        return_surface=real_risk_workspace.return_surface,
        return_reader=real_risk_workspace.return_reader,
        freshness_probe=real_risk_workspace.freshness_probe,
    )
    counter = NumericalCallCounter()
    with pytest.raises(AuthoringError, match="recipe_identity_mismatch"):
        RiskDevelopmentExecutor().execute(
            binding=_binding(),  # sealed over the default recipe
            input_binding=input_binding,
            admission=_admission(ewma_decay=0.97),  # a different admissible one
            return_surface=real_risk_workspace.return_surface,
            return_reader=real_risk_workspace.return_reader,
            bounded_sessions=bounded,
            output_workspace=tmp_path / "mismatch",
            sector_by_listing_id=real_risk_workspace.sector_by_listing_id,
            counter=counter,
        )
    # "Before any estimate" is the requirement, so it is asserted rather than
    # inferred from the exception type.
    assert counter.estimate_calls == 0
    assert not (tmp_path / "mismatch").exists() or not any((tmp_path / "mismatch").rglob("*.json"))


def _module_path(module: str) -> Path:
    return PLAYPEN_ROOT / "src" / (module.replace(".", "/") + ".py")


def _direct_imports(module: str) -> tuple[str, ...]:
    path = _module_path(module)
    if not path.is_file():
        return ()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append(node.module)
        elif isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
    return tuple(found)


def _reachable_modules(root: str) -> set[str]:
    """Every first-party module reachable from ``root`` by import, transitively.

    Direct imports are not enough to answer "can this reach an estimator": one
    hop through a module that itself imports the builder would satisfy a
    direct-import check while leaving the executor a single attribute access
    away.
    """

    seen: set[str] = set()
    queue = [root]
    while queue:
        current = queue.pop()
        for module in _direct_imports(current):
            if module in seen or not module.startswith("alphalattice."):
                continue
            seen.add(module)
            queue.append(module)
    return seen


def _imported_modules(root: Path) -> tuple[tuple[str, str], ...]:
    found: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                found.append((f"{path.name}:{node.lineno}", node.module))
            elif isinstance(node, ast.Import):
                found.extend((f"{path.name}:{node.lineno}", alias.name) for alias in node.names)
    return tuple(found)


def test_experiments_never_import_publication() -> None:
    """Development experiments cannot import their publication owner."""

    offenders = [
        where
        for where, module in _imported_modules(RISK_ROOT / "experiments")
        if "risk_research.publication" in module
    ]
    assert offenders == []


def test_the_compiler_names_no_recipe_contract_adapter_or_family() -> None:
    """The compiler names no recipe contract adapter or family."""

    imported = {
        module for _where, module in _imported_modules(RISK_ROOT / "experiments" / "compiler.py")
    }
    source = (RISK_ROOT / "experiments" / "compiler.py").read_text(encoding="utf-8")

    assert "alphalattice.investment.risk_research.contracts" not in imported
    assert "alphalattice.investment.risk_research.estimators.domains" not in imported
    # The last one to go: the compiler imported the covariance module for its
    # ``numerical_environment_hash`` and stamped it on every Program, whichever
    # method had been selected. So it named an estimator after all, and a method
    # depending on neither scikit-learn nor a BLAS thread policy was sealed under
    # an environment it never runs in. The selected capability declares its own.
    assert "alphalattice.investment.risk_research.estimators.covariance" not in imported
    for name in ("CovarianceRecipe", "COVARIANCE_", "installed_parameter_domain", "seal_contract"):
        assert name not in source, name
    # The one route that remains.
    assert "admit_recipe" in source


def test_the_development_writer_names_no_estimator_implementation() -> None:
    """The development writer names no estimator implementation."""

    imported = {
        module for _where, module in _imported_modules(RISK_ROOT / "experiments" / "development.py")
    }
    source = (RISK_ROOT / "experiments" / "development.py").read_text(encoding="utf-8")

    assert "alphalattice.investment.risk_research.estimators.covariance" not in imported
    for name in ("CovarianceRecipe", "COVARIANCE_", "numerical_environment_hash()"):
        assert name not in source, name


def test_the_default_host_catalog_installs_only_real_product_capabilities() -> None:
    """The default host catalog installs only real product capabilities."""

    from alphalattice.investment.risk_research.estimators.diagonal import (
        DIAGONAL_RECIPE_SCHEMA_ID,
        DiagonalShrunkCovarianceAdapter,
    )
    from alphalattice.investment.risk_research.estimators.fast_slow import (
        FAST_SLOW_RECIPE_SCHEMA_ID,
        FastSlowCovarianceAdapter,
    )

    catalog = build_installed_risk_estimator_catalog()

    assert set(catalog.adapter_ids) == {
        CovarianceEstimatorAdapter().adapter_id,
        FastSlowCovarianceAdapter().adapter_id,
        DiagonalShrunkCovarianceAdapter().adapter_id,
    }
    assert set(catalog.capability_handles) == {
        COVARIANCE_RECIPE_SCHEMA_ID,
        FAST_SLOW_RECIPE_SCHEMA_ID,
        DIAGONAL_RECIPE_SCHEMA_ID,
    }
    assert not any("case" in value.lower() for value in catalog.adapter_ids)


def test_an_uninstalled_capability_handle_is_refused() -> None:
    with pytest.raises(RiskCapabilityError, match="authoring_capability_not_installed"):
        build_installed_risk_estimator_catalog().admit_recipe(
            capability_handle="case-study.not-installed",
            parameters={},
            seed=CANONICAL_NO_RANDOMNESS_SEED,
        )


def test_the_evidence_verifier_cannot_reach_anything_that_computes() -> None:
    """The evidence verifier cannot reach anything that computes."""

    reachable = _reachable_modules("alphalattice.investment.risk_research.experiments.verification")
    forbidden = {
        # The builder, the executor, and the adapter behind them. The catalog is
        # listed too: resolving one yields an object with an ``estimate`` method,
        # which is the capability this path must not have.
        "alphalattice.investment.risk_research.surfaces.historical",
        "alphalattice.investment.risk_research.experiments.execution",
        "alphalattice.investment.risk_research.estimators.covariance",
        "alphalattice.investment.risk_research.estimators.catalog",
    }
    assert reachable & forbidden == set()
    # Guard the guard: the closure must actually be finding things, or an empty
    # reachable set would make the assertion above vacuous.
    assert "alphalattice.investment.risk_research.surfaces.artifacts" in reachable
    # And it is genuinely a Desk module now, not a Host one.
    assert not (
        PLAYPEN_ROOT
        / "src"
        / "alphalattice"
        / "control"
        / "product_host"
        / "research_authoring"
        / "verification.py"
    ).exists()


def test_the_generic_program_layer_knows_no_desk() -> None:
    """The generic program layer knows no desk."""

    forbidden = (
        "risk_research",
        "factor_research",
        "alpha_research",
        "portfolio_strategy_lab",
        "market_data_ops",
        "feature_engine",
        "product_host",
    )
    offenders = [
        f"{where} -> {module}"
        for where, module in _imported_modules(
            PLAYPEN_ROOT / "src" / "alphalattice" / "control" / "research_program"
        )
        if any(name in module for name in forbidden)
    ]
    assert offenders == []


def test_changing_development_code_moves_the_program_identity() -> None:
    """Changing development code moves the program identity."""

    baseline = risk_development_source_closure_hash(PLAYPEN_ROOT)
    assert baseline == risk_development_source_closure_hash(PLAYPEN_ROOT)
    # The declaration covers itself, so the set of development sources cannot be
    # narrowed without the identity noticing.
    assert "risk_research/experiments/identity.py" in "".join(RISK_DEVELOPMENT_SOURCE_PATHS)
    for path in (*RISK_DEVELOPMENT_SOURCE_PATHS, *RISK_NUMERICAL_SOURCE_PATHS):
        assert (PLAYPEN_ROOT / path).is_file(), path
    # Kept disjoint from the numerical closure, so the development identity and
    # the numerical one move independently.
    assert not set(RISK_DEVELOPMENT_SOURCE_PATHS) & set(RISK_NUMERICAL_SOURCE_PATHS)
    assert baseline != risk_numerical_source_closure_hash(PLAYPEN_ROOT)


def test_the_default_covariance_recipe_seal_does_not_move() -> None:
    """The default covariance recipe keeps its recorded field-based seal."""

    assert default_covariance_recipe().recipe_hash == (
        "9e095c7c46c7df9ca09e56f7c8cb2f5b08fccb3c803d0f7d6883f2bab001a21c"
    )


def test_runtime_measurement_cannot_move_a_covariance_identity() -> None:
    """Runtime measurement cannot move a covariance identity."""

    entries = tuple(
        ".".join(Path(path).with_suffix("").parts[1:]) for path in RISK_NUMERICAL_SOURCE_PATHS
    )
    closure = number_deciding_closure(
        entries, root=PLAYPEN_ROOT, rule=number_deciding_rule(PLAYPEN_ROOT)
    )
    assert "process_metrics.py" not in "".join(RISK_NUMERICAL_SOURCE_PATHS)
    assert not any(module.endswith("telemetry.process_metrics") for module in closure)


def test_every_declared_development_owner_actually_moves_the_identity(tmp_path: Path) -> None:
    """Every declared development owner actually moves the identity."""

    root = tmp_path / "playpen"
    for path in (*RISK_DEVELOPMENT_SOURCE_PATHS, "config/identity-roles.json"):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((PLAYPEN_ROOT / path).read_bytes())
    baseline = risk_development_source_closure_hash(root)

    for path in RISK_DEVELOPMENT_SOURCE_PATHS:
        target = root / path
        original = target.read_bytes()
        target.write_bytes(original + b"\n_AN_EDIT_TO_AN_ACTIVE_DEVELOPMENT_OWNER = 1\n")
        assert risk_development_source_closure_hash(root) != baseline, path
        target.write_bytes(original + b"\n# a comment is not syntax\n")
        assert risk_development_source_closure_hash(root) == baseline, path
        target.write_bytes(original)
    # And restoring every file restores the identity, so the moves above were
    # caused by the edits rather than by the copy.
    assert risk_development_source_closure_hash(root) == baseline


@pytest.mark.parametrize(
    "owner",
    [
        "experiments/development.py",
        "experiments/development_artifacts.py",
        "experiments/verification.py",
        "estimators/capability.py",
        "estimators/catalog.py",
        "estimators/contracts.py",
        "estimators/domains.py",
    ],
)
def test_the_owners_that_were_missing_are_covered(owner: str) -> None:
    """Named individually so a future narrowing of the list is a failing test."""

    assert any(path.endswith(owner) for path in RISK_DEVELOPMENT_SOURCE_PATHS), owner


def _sealed_program(
    binding: RiskDevelopmentProgramBinding,
    *,
    authority: ResolvedResearchAuthority,
    sessions: tuple[object, ...],
) -> SealedResearchProgram:
    return SealedResearchProgram.create(
        kind=RISK_EXPERIMENT_KIND,
        envelope_hash="0" * 64,
        desk_program_hash="1" * 64,
        resolved_sessions=sessions,
        catalog_hash=binding.catalog_hash,
        method_binding_hash=binding.development_binding_hash,
        parameter_domain_hash=binding.parameter_domain_hash,
        authority_hash=authority.authority_hash,
    )


def _sealed_evidence(
    program: SealedResearchProgram,
    execution: RiskDevelopmentExecution,
    *,
    artifact_uris: tuple[str, ...] | None = None,
) -> ResearchExecutionEvidence:
    return ResearchExecutionEvidence.create(
        kind=RISK_EXPERIMENT_KIND,
        program_hash=program.program_hash,
        desk_program_hash=program.desk_program_hash,
        method_binding_hash=program.method_binding_hash,
        authority_hash=program.authority_hash,
        disposition="COMPUTED",
        numerical_call_count=execution.estimate_calls,
        artifact_uris=artifact_uris or execution.artifact_uris,
        formation_sessions=tuple(execution.input_binding.formation_sessions),
        desk_input_binding_hash=execution.input_binding.input_binding_hash,
    )


def test_a_graph_from_another_program_over_the_same_authority_is_refused(
    real_risk_workspace: RealRiskWorkspace,
    tmp_path: Path,
) -> None:
    """A graph from another program over the same authority is refused."""

    authority, requested = _bounded_authority(real_risk_workspace, count=3)
    input_binding, bounded = resolve_development_input_binding(
        authority=authority,
        return_surface=real_risk_workspace.return_surface,
        return_reader=real_risk_workspace.return_reader,
        freshness_probe=real_risk_workspace.freshness_probe,
    )
    output = tmp_path / "shared"

    def _run(
        admission: RiskRecipeAdmission,
    ) -> tuple[SealedResearchProgram, ResearchExecutionEvidence, RiskDevelopmentExecution]:
        binding = _binding(recipe_hash=admission.recipe_hash)
        execution = RiskDevelopmentExecutor().execute(
            binding=binding,
            input_binding=input_binding,
            admission=admission,
            return_surface=real_risk_workspace.return_surface,
            return_reader=real_risk_workspace.return_reader,
            bounded_sessions=bounded,
            output_workspace=output,
            sector_by_listing_id=real_risk_workspace.sector_by_listing_id,
        )
        program = _sealed_program(binding, authority=authority, sessions=requested)
        return program, _sealed_evidence(program, execution), execution

    a_program, a_evidence, a_execution = _run(_admission())
    b_program, b_evidence, b_execution = _run(_admission(ewma_decay=0.97))
    verifier = RiskEvidenceVerifier()

    # The control: each graph verifies against its own Program.
    verifier.verify(
        program=a_program, evidence=a_evidence, authority=authority, output_workspace=output
    )
    verifier.verify(
        program=b_program, evidence=b_evidence, authority=authority, output_workspace=output
    )
    # The two runs really do share their input authority, which is what makes
    # every check that existed before this one pass.
    assert a_execution.input_binding.input_binding_hash == (
        b_execution.input_binding.input_binding_hash
    )
    assert a_evidence.authority_hash == b_evidence.authority_hash
    assert a_execution.artifact_uris != b_execution.artifact_uris

    # Program A's identity, B's artifacts. Every artifact individually valid.
    forged = _sealed_evidence(a_program, b_execution, artifact_uris=b_execution.artifact_uris)
    with pytest.raises(AuthoringError, match="evidence_surface_not_this_program"):
        verifier.verify(
            program=a_program, evidence=forged, authority=authority, output_workspace=output
        )
