"""One real workspace, built once, shared by every case in this package.

The session fixtures live in `session_fixtures.py` so the extension package can
register the same builders in its own process. This package owns its bounded Risk
development run and the shared Factor and Alpha web fixtures here.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceArtifact,
    ResearchWorkspaceManifest,
    publish_research_workspace_manifest,
    read_research_workspace_manifest,
)
from alphalattice.control.product_host.research_authoring.factor_inputs import (
    bind_factor_inputs,
    read_factor_bundle,
)
from alphalattice.investment.risk_research.experiments.execution import (
    RiskDevelopmentExecution,
)
from alphalattice.protocols.research_authoring.contracts import (
    ResearchExecutionEvidence,
    ResolvedResearchAuthority,
    SealedResearchProgram,
)
from tests.portfolio_strategy_lab.local_web_support import _json, _manifest
from tests.researcher_methodology_surface.factor_web_support import (
    _alpha_payload,
    _publish_factor,
    _session,
)
from tests.researcher_methodology_surface.real_workspace import (
    RealRiskWorkspace,
    build_real_risk_workspace,
    publish_causal_outcomes,
)
from tests.researcher_methodology_surface.session_fixtures import (
    offline_execution_environment,
    real_risk_workspace,
)
from tests.researcher_methodology_surface.session_workspace import copy_workspace, session_workspace

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


@pytest.fixture(scope="module")
def input_seed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The exact original real input build, once per run, including its bind/reuse assertions."""

    def build(root: Path) -> dict:
        real_risk_workspace = build_real_risk_workspace(
            tmp_path_factory.mktemp("factor-input-source"),
            symbols=tuple(f"F{i:03d}" for i in range(120)),
        )
        # The Portfolio artifact is deliberately not consumed by this Factor-only
        # composition harness; its typed reference must nevertheless survive bind.
        manifest = _manifest("factor-web")
        artifact = ResearchWorkspaceArtifact(
            artifact_key="BROAD_ENSEMBLE_EVIDENCE_ROOT", relative_path="retained-strategy-artifacts"
        )
        manifest = ResearchWorkspaceManifest.create(
            workspace_id=manifest.workspace_id,
            default_strategy_package_id=manifest.default_strategy_package_id,
            default_score_source_mode=manifest.default_score_source_mode,
            strategy_artifacts=(artifact,),
        )
        publish_research_workspace_manifest(root, manifest)
        outcome, _ref = publish_causal_outcomes(
            real_risk_workspace, at=datetime(2026, 8, 2, tzinfo=UTC)
        )
        first = bind_factor_inputs(
            workspace=root, source=real_risk_workspace.workspace, outcome_snapshot_hash=outcome
        )
        before = (root / "research-workspace.json").read_bytes()
        assert (
            bind_factor_inputs(
                workspace=root, source=real_risk_workspace.workspace, outcome_snapshot_hash=outcome
            )
            == first
        )
        assert (root / "research-workspace.json").read_bytes() == before
        assert read_research_workspace_manifest(root).strategy_artifacts == (artifact,)
        assert not tuple((root / "research-inputs").rglob("sealed-holdout"))
        return {}

    source, _metadata = session_workspace(tmp_path_factory, "factor_input", build)
    return source


@pytest.fixture
def inputs(request: pytest.FixtureRequest, input_seed: Path, tmp_path: Path) -> Path:
    # Only tests asking for the published study receive its records. Others
    # keep the original empty-study precondition, in their own writable copy.
    source = input_seed
    if "published" in request.fixturenames:
        source = request.getfixturevalue("factor_seed")[0]
    return copy_workspace(source, tmp_path / "factor-web")


@pytest.fixture(scope="module")
def alpha_seed(factor_seed, tmp_path_factory):
    """The same bound input and curated Factor study, published once across workers."""

    def build(root: Path) -> dict:
        copy_workspace(factor_seed[0], root)
        factor_task, report, _export = factor_seed[1:]
        source = root / report["document"]["experiment"]["baseline_workspace"]
        original = read_factor_bundle(root, source.parent.name)
        enhanced = bind_factor_inputs(
            workspace=root,
            source=source,
            input_id="alpha-development",
            panel_snapshot_hash=original.panel_snapshot_hash,
            outcome_snapshot_hash=original.outcome_snapshot_hash,
            include_alpha_handoff=True,
        )
        with _session(root) as live:
            choices = _json(live, f"/api/experiments/curation?task_id={factor_task}")
            choice = next(v for v in choices["choices"] if v["roles"])
            result = _json(
                live,
                "/api/experiments/curation",
                method="POST",
                payload={
                    "task_id": factor_task,
                    "experiment_curation": {
                        "expected_receipt_hash": choices["receipt_hash"],
                        "choices": [
                            {
                                "factor_id": choice["factor_id"],
                                "role": choice["roles"][0],
                                "rationale": "Fixed offline Alpha acceptance declaration.",
                            }
                        ],
                        "limitations_acknowledged": choices["limitations"],
                    },
                },
            )
            decision = result["decision"]["receipt_hash"]
            draft = _json(
                live,
                "/api/experiments/handoff",
                method="POST",
                payload={
                    "task_id": factor_task,
                    "curation_receipt_hash": decision,
                    "research_input_id": enhanced.input_id,
                    "input_binding_hash": enhanced.binding_hash,
                },
            )
            assert draft["status"] == "DRAFT_INCOMPLETE", draft
        return {
            "input_id": enhanced.input_id,
            "binding_hash": enhanced.binding_hash,
            "factor_task": factor_task,
            "decision": decision,
            "draft": draft["document"],
        }

    root, metadata = session_workspace(tmp_path_factory, "alpha_seed", build)
    enhanced = next(
        value
        for value in read_research_workspace_manifest(root).experiment_inputs
        if value.input_id == metadata["input_id"]
    )
    assert enhanced.binding_hash == metadata["binding_hash"]
    return root, enhanced, metadata["factor_task"], metadata["decision"], metadata["draft"]


@pytest.fixture
def alpha_case(alpha_seed, tmp_path_factory):
    root = copy_workspace(alpha_seed[0], tmp_path_factory.mktemp("alpha-web") / "workspace")
    return root, *alpha_seed[1:4], deepcopy(alpha_seed[4])


@pytest.fixture(scope="module")
def factor_seed(input_seed: Path, tmp_path_factory):
    """One actual Factor execution and its original publication/reuse assertions."""

    def build(root: Path) -> dict:
        copy_workspace(input_seed, root)
        task_id, report, exported = _publish_factor(root)
        return {"task_id": task_id, "report": report, "exported": exported}

    root, metadata = session_workspace(tmp_path_factory, "factor_publication", build)
    return root, metadata["task_id"], metadata["report"], metadata["exported"]


@pytest.fixture
def published(factor_seed):
    return deepcopy(factor_seed[1:])


@pytest.fixture(scope="module")
def sampled_alpha_seed(alpha_seed, tmp_path_factory):
    """A real completed sample study, shared by promotion and Task-authority controls."""

    def build(root: Path) -> dict:
        copy_workspace(alpha_seed[0], root)
        case = (root, *alpha_seed[1:])
        with _session(root) as live:
            sampled = _alpha_payload(case)
            document = sampled["experiment_document"]
            universe = document["experiment"]["universe_handle"]
            document["experiment"]["universe_handle"] = f"{universe}.sample-110"
            plan = _json(live, "/api/experiments/plan", method="POST", payload=sampled)
            assert plan["status"] == "PLANNED", plan
            sent = _json(
                live,
                "/api/experiments/run",
                method="POST",
                payload={"experiment_plan_hash": plan["plan_hash"]},
            )
            live.dispatcher.drain_for_tests()
            report = _json(live, f"/api/experiments/readback?task_id={sent['task_id']}")
            assert (report["status"], report["research_lane"]) == (
                "EXPERIMENT_PUBLISHED",
                "EXPLORATION",
            ), report
        return {"task_id": sent["task_id"], "report": report}

    root, metadata = session_workspace(tmp_path_factory, "sampled_alpha", build)
    return (root, *alpha_seed[1:]), metadata


@pytest.fixture
def sampled_alpha_case(sampled_alpha_seed, tmp_path_factory):
    seed, metadata = sampled_alpha_seed
    root = copy_workspace(seed[0], tmp_path_factory.mktemp("sampled-alpha") / "workspace")
    return (root, *seed[1:4], deepcopy(seed[4])), deepcopy(metadata)
