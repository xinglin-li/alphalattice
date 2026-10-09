"""A second Risk recipe schema travels the whole chain, and nobody names it.

The claim under test is not "an adapter seam exists" -- calling
``catalog.resolve(...)`` then ``adapter.estimate(...)`` would prove only that.
The claim is that a researcher can add a method by writing an implementation, a
recipe schema, a parameter domain and a registration, and reach real bounded
evidence without editing the compiler, the development writer, the verifier, the
generic workflow or the publication owner.

So every case here goes through the authored document:

    YAML
    -> real authority resolution
    -> explicitly injected capability catalog
    -> domain admission
    -> sealed alternate recipe
    -> Program
    -> persisted input binding
    -> bounded real estimates
    -> generic development surface and evidence
    -> recursive verification
    -> exact replay with zero numerical calls

The same cases run the production covariance capability through the same
artifact path, because "both methods produce the same shape of evidence" is what
stops the next method needing a branch.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from alphalattice.control.product_host.composition.research_authoring import (
    build_research_program_workflow,
    installed_desk_verifiers,
)
from alphalattice.control.product_host.research_authoring.execution import (
    WorkspaceReturnSurfaceProvider,
)
from alphalattice.control.research_program.authoring.document import load_authoring_document
from alphalattice.investment.risk_research.estimators.capability import (
    CANONICAL_NO_RANDOMNESS_SEED,
)
from alphalattice.investment.risk_research.estimators.catalog import RiskEstimatorCatalog
from alphalattice.investment.risk_research.estimators.covariance import (
    COVARIANCE_RECIPE_SCHEMA_ID,
    CovarianceCapability,
    CovarianceEstimatorAdapter,
)
from alphalattice.investment.risk_research.experiments.compiler import RiskExperimentCompiler
from alphalattice.investment.risk_research.experiments.development_artifacts import (
    DEVELOPMENT_SURFACE_CATEGORY,
    RiskDevelopmentCovarianceSurface,
)
from alphalattice.investment.risk_research.experiments.execution import RiskExperimentExecutor
from alphalattice.investment.risk_research.surfaces.artifacts import RiskArtifactStore
from alphalattice.protocols.actor_execution.contracts import ActorKind
from alphalattice.protocols.research_authoring.contracts import NumericalCallRecorder
from tests.researcher_methodology_surface.alternate_capability import (
    ALTERNATE_RECIPE_SCHEMA_ID,
    ShrunkDiagonalAdapter,
    ShrunkDiagonalCapability,
)
from tests.researcher_methodology_surface.real_workspace import RealRiskWorkspace

CASE_ROOT = Path(__file__).resolve().parent
PLAYPEN_ROOT = CASE_ROOT.parents[1]
RISK_ROOT = PLAYPEN_ROOT / "src" / "alphalattice" / "investment" / "risk_research"
_FIXTURE = CASE_ROOT / "fixtures" / "risk_covariance_development.yaml"
_OUTPUT_WORKSPACE = "workspaces/research/risk-development"


class _Recorder:
    """Counts what the generic workflow observed, independent of the Desk."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def record(self, *, capability: str) -> None:
        self.calls.append(capability)


def _catalog_with_alternate() -> RiskEstimatorCatalog:
    """Both capabilities installed explicitly, by this case study only."""

    return RiskEstimatorCatalog(
        (CovarianceEstimatorAdapter(), ShrunkDiagonalAdapter()),  # type: ignore[arg-type]
        capabilities=(CovarianceCapability(), ShrunkDiagonalCapability()),  # type: ignore[arg-type]
    )


def _document(*, capability: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """The authored document, with only the estimator section varied."""

    document = load_authoring_document(_FIXTURE.read_text(encoding="utf-8"))
    draft = copy.deepcopy(dict(document))
    draft["risk"] = {"estimator": {"capability": capability, "parameters": parameters}}
    return draft


def _workflow(workspace: RealRiskWorkspace, root: Path) -> Any:
    catalog = _catalog_with_alternate()
    return build_research_program_workflow(
        workspace=workspace.workspace,
        workspace_root=root,
        # The injected catalog reaches both the compiler and the executor, so
        # admission and execution agree about what is installed.
        compilers=(RiskExperimentCompiler(catalog),),
        executors=(
            RiskExperimentExecutor(
                surface_provider=WorkspaceReturnSurfaceProvider(workspace.artifact_root),
                return_reader=workspace.return_reader,
                sector_by_listing_id=workspace.sector_by_listing_id,
                freshness_probe=workspace.freshness_probe,
                estimators=catalog,
            ),
        ),
    )


def _published_surface(root: Path) -> RiskDevelopmentCovarianceSurface:
    store = RiskArtifactStore(root / _OUTPUT_WORKSPACE)
    paths = sorted((store.root / DEVELOPMENT_SURFACE_CATEGORY).glob("*.json"))
    assert len(paths) == 1, paths
    import json

    return RiskDevelopmentCovarianceSurface(**json.loads(paths[0].read_text(encoding="utf-8")))


@pytest.mark.parametrize(
    ("capability", "parameters"),
    [
        (COVARIANCE_RECIPE_SCHEMA_ID, {"ewma_decay": 0.94}),
        (ALTERNATE_RECIPE_SCHEMA_ID, {"shrinkage_intensity": 0.10}),
    ],
)
def test_both_methods_reach_real_evidence_through_one_artifact_path(
    real_risk_workspace: RealRiskWorkspace,
    tmp_path: Path,
    capability: str,
    parameters: dict[str, Any],
) -> None:
    """Both methods reach real evidence through one artifact path."""

    workflow = _workflow(real_risk_workspace, tmp_path)
    document = _document(capability=capability, parameters=parameters)
    recorder: NumericalCallRecorder = _Recorder()

    evidence, _binding = workflow.run(
        document, actor_kind=ActorKind.HUMAN, actor_id="researcher", recorder=recorder
    )

    assert evidence.disposition == "COMPUTED"
    assert evidence.numerical_call_count > 0
    assert len(evidence.formation_sessions) == evidence.numerical_call_count

    surface = _published_surface(tmp_path)
    # The chain carried this capability's own schema, not a covariance recipe.
    assert surface.capability_handle == capability
    assert surface.recipe_envelope.recipe_schema_id == capability
    assert surface.identity_class == "DEVELOPMENT_EVIDENCE_ONLY_NEVER_PUBLISHED"

    # Replay reads the whole graph back and computes nothing.
    replayed, _ = workflow.replay(document, actor_kind=ActorKind.HUMAN, actor_id="researcher")
    assert replayed.disposition == "REUSED_EXACT"
    assert replayed.numerical_call_count == 0
    assert replayed.artifact_uris == evidence.artifact_uris
    assert replayed.desk_input_binding_hash == evidence.desk_input_binding_hash


def test_the_authored_parameter_reaches_the_alternate_adapter(
    real_risk_workspace: RealRiskWorkspace,
    tmp_path: Path,
) -> None:
    """The authored parameter reaches the alternate adapter."""

    outcomes = {}
    for intensity in (0.10, 0.50):
        root = tmp_path / f"run-{intensity}"
        workflow = _workflow(real_risk_workspace, root)
        document = _document(
            capability=ALTERNATE_RECIPE_SCHEMA_ID,
            parameters={"shrinkage_intensity": intensity},
        )
        evidence, _ = workflow.run(document, actor_kind=ActorKind.HUMAN, actor_id="researcher")
        surface = _published_surface(root)
        outcomes[intensity] = (
            surface.recipe_hash,
            surface.selected_method_binding_hash,
            tuple(v for chunk in surface.chunks for v in chunk.matrix_hashes),
            evidence.program_hash,
            surface.recipe_envelope.parameters["shrinkage_intensity"],
        )

    low, high = outcomes[0.10], outcomes[0.50]
    # The authored value arrived.
    assert low[4] == 0.10
    assert high[4] == 0.50
    # Recipe identity moved, method identity moved, Program moved.
    assert low[0] != high[0]
    assert low[1] != high[1]
    assert low[3] != high[3]
    # And it reached the mathematics.
    assert low[2] != high[2]


def test_an_undeclared_alternate_parameter_is_refused(
    real_risk_workspace: RealRiskWorkspace,
    tmp_path: Path,
) -> None:
    """Domain admission governs the alternate schema exactly as it governs the first."""

    workflow = _workflow(real_risk_workspace, tmp_path)
    document = _document(
        capability=ALTERNATE_RECIPE_SCHEMA_ID,
        parameters={"shrinkage_intensity": 0.31},
    )
    with pytest.raises(ValueError, match="parameter_value_not_admitted"):
        workflow.freeze(document, actor_kind=ActorKind.HUMAN, actor_id="researcher")


def test_the_alternate_method_is_not_installed_by_the_default_host(
    real_risk_workspace: RealRiskWorkspace,
    tmp_path: Path,
) -> None:
    """The alternate method is not installed by the default host."""

    workflow = build_research_program_workflow(
        workspace=real_risk_workspace.workspace,
        workspace_root=tmp_path,
        executors=(
            RiskExperimentExecutor(
                surface_provider=WorkspaceReturnSurfaceProvider(real_risk_workspace.artifact_root),
                return_reader=real_risk_workspace.return_reader,
                sector_by_listing_id=real_risk_workspace.sector_by_listing_id,
                freshness_probe=real_risk_workspace.freshness_probe,
            ),
        ),
    )
    document = _document(
        capability=ALTERNATE_RECIPE_SCHEMA_ID,
        parameters={"shrinkage_intensity": 0.10},
    )
    with pytest.raises(ValueError, match="authoring_capability_not_installed"):
        workflow.freeze(document, actor_kind=ActorKind.HUMAN, actor_id="researcher")


def test_no_capability_identity_carries_a_numerical_environment(
    real_risk_workspace: RealRiskWorkspace,
    tmp_path: Path,
) -> None:
    """No capability identity carries a numerical environment."""

    for capability, parameters in (
        (COVARIANCE_RECIPE_SCHEMA_ID, {"ewma_decay": 0.94}),
        (ALTERNATE_RECIPE_SCHEMA_ID, {"shrinkage_intensity": 0.10}),
    ):
        root = tmp_path / capability.lower()
        workflow = _workflow(real_risk_workspace, root)
        workflow.run(
            _document(capability=capability, parameters=parameters),
            actor_kind=ActorKind.HUMAN,
            actor_id="researcher",
        )
        assert _published_surface(root).numerical_environment_hash is None, capability


def test_the_host_installs_one_verifier_for_both_methods() -> None:
    """The host installs one verifier for both methods."""

    verifiers = installed_desk_verifiers()
    kinds = [value.kind for value in verifiers]

    assert kinds.count("risk.covariance-development") == 1
    assert len(set(kinds)) == len(kinds)
    assert "alpha.model-development" in kinds


def test_installing_the_alternate_does_not_move_the_production_method_identity(
    real_risk_workspace: RealRiskWorkspace,
    tmp_path: Path,
) -> None:
    """Installing the alternate does not move the production method identity."""

    document = _document(capability=COVARIANCE_RECIPE_SCHEMA_ID, parameters={"ewma_decay": 0.94})
    default_compiler = RiskExperimentCompiler()
    both_compiler = RiskExperimentCompiler(_catalog_with_alternate())
    envelope_document = document["experiment"]

    from alphalattice.control.product_host.research_authoring.authority import (
        WorkspaceResearchAuthorityResolver,
    )
    from alphalattice.protocols.research_authoring.contracts import (
        ResearchExperimentEnvelope,
    )

    envelope = ResearchExperimentEnvelope.create(**envelope_document)
    authority = WorkspaceResearchAuthorityResolver(workspace=real_risk_workspace.workspace).resolve(
        envelope
    )

    alone = default_compiler.compile_development_program(
        envelope=envelope, document=document, authority=authority
    ).binding
    beside = both_compiler.compile_development_program(
        envelope=envelope, document=document, authority=authority
    ).binding

    # Governance moved: the Host installed something else.
    assert alone.catalog_hash != beside.catalog_hash
    # Methodology did not: the same code runs on the same inputs.
    assert alone.selected_method_binding_hash == beside.selected_method_binding_hash
    assert alone.selected_adapter_id == beside.selected_adapter_id


def test_a_seed_is_refused_by_a_deterministic_method(
    real_risk_workspace: RealRiskWorkspace,
    tmp_path: Path,
) -> None:
    """A seed is refused by a deterministic method."""

    workflow = _workflow(real_risk_workspace, tmp_path)
    for capability, parameters in (
        (COVARIANCE_RECIPE_SCHEMA_ID, {"ewma_decay": 0.94}),
        (ALTERNATE_RECIPE_SCHEMA_ID, {"shrinkage_intensity": 0.10}),
    ):
        document = _document(capability=capability, parameters=parameters)
        document["experiment"] = {
            **document["experiment"],
            "determinism": {**document["experiment"]["determinism"], "seed": 20260816},
        }
        with pytest.raises(ValueError, match="authoring_seed_not_applicable"):
            workflow.freeze(document, actor_kind=ActorKind.HUMAN, actor_id="researcher")


def test_the_canonical_seed_is_admitted_for_both_methods(
    real_risk_workspace: RealRiskWorkspace,
    tmp_path: Path,
) -> None:
    """The refusal is a policy, not a blanket rejection of the field."""

    workflow = _workflow(real_risk_workspace, tmp_path)
    for capability, parameters in (
        (COVARIANCE_RECIPE_SCHEMA_ID, {"ewma_decay": 0.94}),
        (ALTERNATE_RECIPE_SCHEMA_ID, {"shrinkage_intensity": 0.10}),
    ):
        document = _document(capability=capability, parameters=parameters)
        assert document["experiment"]["determinism"]["seed"] == CANONICAL_NO_RANDOMNESS_SEED
        program, _ = workflow.freeze(document, actor_kind=ActorKind.HUMAN, actor_id="researcher")
        assert program.program_hash


def test_no_risk_capability_grew_a_seed_parameter() -> None:
    """Installing an additional risk method adds no unsupported seed parameter."""

    from alphalattice.investment.risk_research.estimators.domains import (
        COVARIANCE_PARAMETER_DOMAIN,
    )
    from tests.researcher_methodology_surface.alternate_capability import ALTERNATE_PARAMETER_DOMAIN

    for domain in (COVARIANCE_PARAMETER_DOMAIN, ALTERNATE_PARAMETER_DOMAIN):
        assert not any("seed" in axis.name for axis in domain.axes)
    assert CovarianceCapability().randomness_policy == "NONE"
    assert ShrunkDiagonalCapability().randomness_policy == "NONE"
