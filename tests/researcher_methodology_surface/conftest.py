"""One real workspace, built once, shared by every case in this package.

The session fixtures live in `session_fixtures.py` so the extension package can
register the same builders in its own process; only the bounded Risk development
run, which nothing outside this package needs, is defined here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from alphalattice.investment.risk_research.experiments.execution import (
    RiskDevelopmentExecution,
)
from alphalattice.protocols.research_authoring.contracts import (
    ResearchExecutionEvidence,
    ResolvedResearchAuthority,
    SealedResearchProgram,
)
from tests.researcher_methodology_surface.real_workspace import RealRiskWorkspace
from tests.researcher_methodology_surface.session_fixtures import (
    offline_execution_environment,
    real_risk_workspace,
)

__all__ = ["offline_execution_environment", "real_risk_workspace"]

CASE_ROOT = Path(__file__).resolve().parent
PLAYPEN_ROOT = CASE_ROOT.parents[1]


@dataclass(frozen=True, slots=True)
class RiskDevelopmentRun:
    """One real bounded build, plus the Program and evidence that describe it.

    The verification cases need all three. A verifier now proves that a graph
    belongs to a particular Program and authority, so handing it artifacts alone
    could only ever test half of what it does.
    """

    output: Path
    program: SealedResearchProgram
    evidence: ResearchExecutionEvidence
    authority: ResolvedResearchAuthority
    """What the Host resolved for this Program, carried so the cases that call the
    verifier directly can hand it the same side the workflow would."""

    execution: RiskDevelopmentExecution
    """The build itself, so cases needing a real surface do not pay for a second one."""


@pytest.fixture(scope="session")
def risk_development_run(
    real_risk_workspace: RealRiskWorkspace,
    tmp_path_factory: pytest.TempPathFactory,
) -> RiskDevelopmentRun:
    """One real bounded development build, shared by the verification cases.

    Built by the product executor over the same shared workspace rather than a
    second harness: the artifacts these cases delete and tamper with have to be
    the ones a real run actually writes, or the verifier is being tested against
    a fixture's idea of the layout.
    """

    from alphalattice.investment.risk_research.estimators.capability import (
        CANONICAL_NO_RANDOMNESS_SEED,
    )
    from alphalattice.investment.risk_research.estimators.catalog import (
        build_installed_risk_estimator_catalog,
    )
    from alphalattice.investment.risk_research.estimators.covariance import (
        COVARIANCE_RECIPE_SCHEMA_ID,
    )
    from alphalattice.investment.risk_research.experiments.compiler import RISK_EXPERIMENT_KIND
    from alphalattice.investment.risk_research.experiments.execution import RiskDevelopmentExecutor
    from alphalattice.investment.risk_research.experiments.window import (
        resolve_development_input_binding,
    )
    from tests.researcher_methodology_surface.risk_development_support import (
        _binding,
        _bounded_authority,
    )

    output = tmp_path_factory.mktemp("risk-development-run")
    authority, requested = _bounded_authority(real_risk_workspace, count=6)
    input_binding, bounded = resolve_development_input_binding(
        authority=authority,
        return_surface=real_risk_workspace.return_surface,
        return_reader=real_risk_workspace.return_reader,
        freshness_probe=real_risk_workspace.freshness_probe,
    )
    binding = _binding()
    execution = RiskDevelopmentExecutor().execute(
        binding=binding,
        input_binding=input_binding,
        admission=build_installed_risk_estimator_catalog().admit_recipe(
            capability_handle=COVARIANCE_RECIPE_SCHEMA_ID,
            parameters={},
            seed=CANONICAL_NO_RANDOMNESS_SEED,
        ),
        return_surface=real_risk_workspace.return_surface,
        return_reader=real_risk_workspace.return_reader,
        bounded_sessions=bounded,
        output_workspace=output,
        sector_by_listing_id=real_risk_workspace.sector_by_listing_id,
    )
    program = SealedResearchProgram.create(
        kind=RISK_EXPERIMENT_KIND,
        envelope_hash="0" * 64,
        desk_program_hash="1" * 64,
        resolved_sessions=requested,
        catalog_hash=binding.catalog_hash,
        method_binding_hash=binding.development_binding_hash,
        parameter_domain_hash=binding.parameter_domain_hash,
        authority_hash=authority.authority_hash,
    )
    evidence = ResearchExecutionEvidence.create(
        kind=RISK_EXPERIMENT_KIND,
        program_hash=program.program_hash,
        desk_program_hash=program.desk_program_hash,
        method_binding_hash=program.method_binding_hash,
        authority_hash=program.authority_hash,
        disposition="COMPUTED",
        numerical_call_count=execution.estimate_calls,
        artifact_uris=execution.artifact_uris,
        formation_sessions=requested,
        desk_input_binding_hash=input_binding.input_binding_hash,
    )
    return RiskDevelopmentRun(
        output=output,
        program=program,
        evidence=evidence,
        authority=authority,
        execution=execution,
    )
