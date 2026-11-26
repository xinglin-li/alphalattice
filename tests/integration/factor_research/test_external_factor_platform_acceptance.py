"""An outside consumer drives the whole Factor platform without editing it.

This is mechanics acceptance, not science. It proves the chain the architecture
promises -- researcher-owned formula, typed Formula Specification, explicit
installed catalog, development admission, typed/YAML batch request, Host-resolved
authority, sealed Program, deterministic screening, durable evidence,
actor-neutral curation, recursive verification, exact replay -- can be walked end
to end by somebody who cannot commit to ``alphalattice``. The numbers it produces
are a seeded random walk over ten synthetic listings and mean nothing at all.

The method under test lives in ``external_factor_method``: a formula, a recipe, a
Formula Specification and a kernel registration, none of them in the product. It
is never product-installed, never product-admitted, never scientifically
selected, and never current. The last case exists to make that impossible to
blur.

One session-scoped workspace, because a Panel's factor axis is fixed when its
closure opens, and one module-scoped run over it, because the deterministic
program is the expensive part and every case after the first is asking a question
about the same graph.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from alphalattice.control.product_host.composition.research_authoring import (
    host_resolved_factor_inventory,
    installed_desk_verifiers,
)
from alphalattice.control.research_program.authoring.document import load_authoring_document
from alphalattice.foundation.factor_research.evaluation.oos_evidence import (
    FactorEvidenceClassification,
)
from alphalattice.foundation.factor_research.experiments.authoring import (
    FACTOR_EXPERIMENT_KIND,
    FactorInventoryEntry,
    factor_catalog_hash,
)
from alphalattice.foundation.factor_research.experiments.development_evidence import (
    FactorDevelopmentCurationReader,
    FactorDevelopmentInputBinding,
    FactorDevelopmentReceipt,
    FactorDevelopmentReceiptReader,
    publish_factor_development_receipt,
)
from alphalattice.foundation.factor_research.inputs.research_input import (
    FactorResearchCandidateRole,
    FactorResearchProposal,
    FactorResearchProposalChoice,
    build_factor_research_proposal,
)
from alphalattice.foundation.factor_research.research_loop.decisions import (
    FACTOR_REQUIRED_RESEARCH_LIMITATIONS,
    factor_research_decision_policy_hash,
)
from alphalattice.foundation.factor_research.research_loop.development_curation import (
    submit_factor_development_curation,
)
from alphalattice.foundation.feature_engine.catalog.authoring_surface import (
    describe_factor_authoring_surface,
)
from alphalattice.foundation.feature_engine.catalog.contracts import (
    FeatureCatalog,
    desktop_core_feature_bundle,
)
from alphalattice.foundation.feature_engine.panels.development_input import (
    ResolvedDevelopmentFeatureInput,
)
from alphalattice.foundation.feature_engine.producers.arithmetic_identity import (
    FACTOR_METHOD_FAMILY_OWNERS,
    method_family_content_hash,
)
from alphalattice.foundation.feature_engine.producers.factors.catalog import (
    default_extension_kernel_registry,
    extension_factor_specs,
)
from alphalattice.foundation.feature_engine.producers.factors.specifications import (
    FeatureFormulaSpecificationError,
    admitted_extension_factor_specs,
    build_installed_factor_formula_specifications,
    golden_example_frame,
)
from alphalattice.protocols.actor_execution.contracts import ActorKind
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    ResearchExecutionEvidence,
    ResearchExperimentEnvelope,
)
from tests._support.researcher_methodology_session_fixtures import build_external_factor_workspace
from tests._support.researcher_methodology_session_fixtures import (
    outcome_catalog_root as outcome_catalog_root,
)
from tests.researcher_methodology_surface.external_factor_method import (
    EXTERNAL_FACTOR_ID,
    EXTERNAL_IMPLEMENTATION_ID,
    EXTERNAL_METHOD_FAMILY,
    EXTERNAL_METHOD_FAMILY_OWNERS,
    external_factor_spec,
    external_feature_catalog,
    external_formula_specification,
    external_kernel_registry,
    external_specification_catalog,
)
from tests.researcher_methodology_surface.factor_evidence_support import (
    factor_inventory,
    factor_workflow,
    short_lookback_factor_input,
)
from tests.researcher_methodology_surface.real_workspace import (
    DEVELOPMENT_SYMBOLS,
    FACTOR_SCALE,
    MARKET_PROFILE_ID,
    SUFFICIENT_SYMBOLS,
    RealRiskWorkspace,
    build_real_risk_workspace,
    publish_causal_outcomes,
)

CASE_ROOT = Path(__file__).resolve().parents[3] / "tests/researcher_methodology_extension"
PLAYPEN_ROOT = CASE_ROOT.parents[1]
_REQUEST = (
    PLAYPEN_ROOT
    / "tests"
    / "researcher_methodology_surface"
    / "fixtures"
    / "factor_external_method_batch.yaml"
)
_PUBLISHED_AT = datetime(2026, 8, 2, tzinfo=UTC)


def _document() -> dict[str, Any]:
    """Read the authored batch through the real document loader."""

    return copy.deepcopy(dict(load_authoring_document(_REQUEST.read_text(encoding="utf-8"))))


@dataclass(frozen=True, slots=True)
class ExternalFactorRun:
    """One authored batch, executed once, plus everything a case needs to ask about it."""

    workflow: Any
    document: dict[str, Any]
    root: Path
    inventory: tuple[FactorInventoryEntry, ...]
    evidence: ResearchExecutionEvidence
    receipt: FactorDevelopmentReceipt
    child: Any
    authority: Any


@pytest.fixture(scope="module")
def external_outcomes(
    tmp_path_factory: pytest.TempPathFactory, outcome_catalog_root: Path
) -> RealRiskWorkspace:
    """Publish causal outcomes once for the external Factor walk on the sufficient set.
    Each consuming test declares this fixture so it can run alone on any worker.
    """

    external_factor_workspace = build_external_factor_workspace(
        tmp_path_factory.mktemp("external-factor-scale"),
        fresh=False,
        symbols=SUFFICIENT_SYMBOLS,
        standard=FACTOR_SCALE,
    )
    publish_causal_outcomes(external_factor_workspace, at=_PUBLISHED_AT)
    return external_factor_workspace


@pytest.fixture(scope="session")
def optimal_external_workspace(tmp_path_factory: pytest.TempPathFactory) -> RealRiskWorkspace:
    return build_real_risk_workspace(
        tmp_path_factory.mktemp("development-external-factor"),
        symbols=DEVELOPMENT_SYMBOLS,
        feature_catalog=external_feature_catalog(),
        feature_kernels=external_kernel_registry(),
        fresh=True,
    )


@pytest.fixture(scope="module")
def external_input(
    optimal_external_workspace: RealRiskWorkspace, outcome_catalog_root: Path
) -> ResolvedDevelopmentFeatureInput:
    publish_causal_outcomes(optimal_external_workspace, at=_PUBLISHED_AT)
    return short_lookback_factor_input(optimal_external_workspace)


def _execute(
    workspace: RealRiskWorkspace, root: Path, feature_input: ResolvedDevelopmentFeatureInput | None
) -> ExternalFactorRun:
    document = _document()
    inventory = factor_inventory(workspace, feature_input)
    workflow, authority = factor_workflow(
        workspace, root, document, inventory, feature_input, sufficient=feature_input is None
    )
    evidence, binding = workflow.run(document, actor_kind=ActorKind.HUMAN, actor_id="external")
    assert binding.actor_kind is ActorKind.HUMAN
    output = root / str(document["experiment"]["output_workspace"])
    receipt, child = FactorDevelopmentReceiptReader(output).load(evidence.artifact_uris[0])
    return ExternalFactorRun(
        workflow=workflow,
        document=document,
        root=output,
        inventory=inventory,
        evidence=evidence,
        receipt=receipt,
        child=child,
        authority=authority.resolve(ResearchExperimentEnvelope.create(**document["experiment"])),
    )


@pytest.fixture(scope="module")
def external_run(
    external_outcomes: RealRiskWorkspace, tmp_path_factory: pytest.TempPathFactory
) -> ExternalFactorRun:
    return _execute(external_outcomes, tmp_path_factory.mktemp("external-factor-run"), None)


@pytest.fixture(scope="module")
def development_run(
    optimal_external_workspace: RealRiskWorkspace,
    external_input: ResolvedDevelopmentFeatureInput,
    tmp_path_factory: pytest.TempPathFactory,
) -> ExternalFactorRun:
    return _execute(
        optimal_external_workspace, tmp_path_factory.mktemp("bounded-external-run"), external_input
    )


def _inventory_entries(context_factors: tuple[Any, ...]) -> tuple[FactorInventoryEntry, ...]:
    return tuple(
        FactorInventoryEntry(
            factor_id=value.factor_id,
            implementation_hash=value.implementation_hash,
            methodology_hash=value.methodology_hash,
        )
        for value in context_factors
    )


def test_an_external_method_reaches_a_published_panel_with_a_measured_identity(
    optimal_external_workspace: RealRiskWorkspace,
) -> None:
    """An external method reaches a published panel with a measured identity."""
    workspace = optimal_external_workspace
    inventory = host_resolved_factor_inventory(
        workspace=workspace.workspace, market_profile_id=MARKET_PROFILE_ID
    )
    by_factor = {entry.factor_id: entry for entry in inventory}
    installed = workspace.feature_catalog.factors
    assert len(installed) == len(FeatureCatalog.load().factors) + 1
    assert len(inventory) == len(installed)
    assert all(len(entry.methodology_hash) == 64 for entry in inventory)
    assert EXTERNAL_FACTOR_ID in by_factor

    entry = by_factor[EXTERNAL_FACTOR_ID]
    assert len(entry.implementation_hash) == 64
    assert len(entry.methodology_hash) == 64
    assert entry.implementation_hash != entry.methodology_hash

    # Re-derived from the injected registry rather than read back from the Panel
    # that published it: an identity that only agreed with itself would prove the
    # Panel self-consistent and nothing about what produced it.
    assert entry.implementation_hash == external_kernel_registry().implementation_hash(
        external_factor_spec(), core_bundle=desktop_core_feature_bundle()
    )

    # The family closure is measured, not declared: the product's table does not
    # contain this family, and the identity function still answers for it.
    assert EXTERNAL_METHOD_FAMILY not in FACTOR_METHOD_FAMILY_OWNERS
    measured = method_family_content_hash(EXTERNAL_METHOD_FAMILY, EXTERNAL_METHOD_FAMILY_OWNERS)
    assert len(measured) == 64

    # And the controls are untouched by the new family's presence, which is the
    # identity isolation this whole platform rests on.
    assert by_factor["amihud_21"].implementation_hash != entry.implementation_hash


def test_an_injected_specification_catalog_admits_the_external_method_only() -> None:
    injected = external_specification_catalog()
    admitted = admitted_extension_factor_specs(
        catalog=injected,
        registry=external_kernel_registry(),
        recipes=(external_factor_spec(),),
    )
    assert tuple(item.factor_id for item in admitted) == (EXTERNAL_FACTOR_ID,)
    assert injected.admit(external_factor_spec(), registry=external_kernel_registry()) == (
        external_formula_specification()
    )

    product = build_installed_factor_formula_specifications()
    # Installed extension recipes are derived from the build rather than a
    # remembered count. Session-observation kernels remain inert until an
    # admitted development overlay consumes them.
    assert set(product.factor_ids) == {item.factor_id for item in extension_factor_specs()}
    assert EXTERNAL_FACTOR_ID not in product.factor_ids
    with pytest.raises(FeatureFormulaSpecificationError, match="not_installed"):
        product.resolve(EXTERNAL_FACTOR_ID)
    # sector_leader_lag_5 stays refused for want of lagged Sector membership.
    assert {item.factor_id for item in admitted_extension_factor_specs()} == (
        set(product.factor_ids) - {"sector_leader_lag_5"}
    )

    with pytest.raises(ValueError, match="not code-owned"):
        default_extension_kernel_registry().resolve(EXTERNAL_IMPLEMENTATION_ID)


def test_the_authored_batch_reaches_real_evidence_and_replays_without_computing(
    external_run: ExternalFactorRun,
) -> None:
    """The authored batch reaches real evidence and replays without computing."""

    evidence = external_run.evidence
    assert evidence.kind == FACTOR_EXPERIMENT_KIND
    assert evidence.disposition == "COMPUTED"
    assert evidence.numerical_call_count == 1

    receipt, child = external_run.receipt, external_run.child
    # The selected axis is exactly what was authored, in the order it was
    # authored; the context axis is the whole published Panel.
    assert receipt.selected_factor_ids == tuple(external_run.document["factor"]["factor_ids"])
    assert EXTERNAL_FACTOR_ID in receipt.selected_factor_ids
    assert receipt.context_factor_ids == tuple(entry.factor_id for entry in external_run.inventory)
    assert child.program.factor_ids == tuple(sorted(receipt.context_factor_ids))
    # The FDR denominator is the whole axis, never the selection.
    assert child.evidence_report.factor_ids == child.program.factor_ids

    replayed, _ = external_run.workflow.replay(
        external_run.document, actor_kind=ActorKind.HUMAN, actor_id="external"
    )
    assert replayed.disposition == "REUSED_EXACT"
    assert replayed.numerical_call_count == 0
    assert replayed.artifact_uris == evidence.artifact_uris
    assert replayed.desk_input_binding_hash == evidence.desk_input_binding_hash


def test_the_verifier_refuses_a_fully_resealed_graph(development_run: ExternalFactorRun) -> None:
    """The verifier refuses a fully resealed graph."""

    verifier = next(
        item for item in installed_desk_verifiers() if item.kind == development_run.evidence.kind
    )
    # The healthy graph passes, or the refusals below would prove nothing.
    verifier.verify(
        program=development_run.receipt.program,
        evidence=development_run.evidence,
        authority=development_run.authority,
        output_workspace=development_run.root,
    )

    binding = development_run.receipt.input_binding
    context = binding.context_factors
    reordered = (context[1], context[0], *context[2:])
    assert factor_catalog_hash(_inventory_entries(reordered)) != factor_catalog_hash(
        _inventory_entries(context)
    )

    for label, rewritten in (
        ("axis", {"context_factors": reordered}),
        ("panel", {"feature_panel_snapshot_hash": "b" * 64}),
    ):
        values = {
            name: getattr(binding, name) for name in ("selected_factor_ids", "context_factors")
        }
        values.update(
            {
                name: getattr(binding, name)
                for name in (
                    "feature_panel_snapshot_hash",
                    "feature_panel_manifest_ref",
                    "causal_outcome_snapshot_hash",
                    "causal_outcome_manifest_ref",
                    "target_policy_hash",
                    "walk_forward_policy_hash",
                    "screening_policy_hash",
                    "redundancy_policy_hash",
                    "authority_hash",
                )
            }
        )
        values.update(rewritten)
        resealed_binding = FactorDevelopmentInputBinding.create(**values)
        resealed_receipt = FactorDevelopmentReceipt.create(
            program=development_run.receipt.program,
            input_binding=resealed_binding,
            checkpoint_hash=development_run.receipt.checkpoint_hash,
        )
        uri = publish_factor_development_receipt(development_run.root, resealed_receipt)
        resealed_evidence = ResearchExecutionEvidence.create(
            kind=development_run.evidence.kind,
            program_hash=development_run.evidence.program_hash,
            desk_program_hash=development_run.evidence.desk_program_hash,
            method_binding_hash=development_run.evidence.method_binding_hash,
            authority_hash=development_run.evidence.authority_hash,
            disposition="COMPUTED",
            numerical_call_count=1,
            artifact_uris=(uri,),
            formation_sessions=development_run.evidence.formation_sessions,
            desk_input_binding_hash=resealed_binding.binding_hash,
        )
        expected = (
            "evidence_context_axis_not_admitted"
            if label == "axis"
            else "evidence_panel_not_this_authority"
        )
        with pytest.raises(AuthoringError, match=expected):
            verifier.verify(
                program=development_run.receipt.program,
                evidence=resealed_evidence,
                authority=development_run.authority,
                output_workspace=development_run.root,
            )


def test_a_missing_deterministic_child_fails_replay(
    optimal_external_workspace: RealRiskWorkspace,
    external_input: ResolvedDevelopmentFeatureInput,
    tmp_path: Path,
) -> None:
    """A missing deterministic child fails replay."""

    document = _document()
    inventory = factor_inventory(optimal_external_workspace, external_input)
    workflow, _ = factor_workflow(
        optimal_external_workspace, tmp_path, document, inventory, external_input
    )
    evidence, _binding = workflow.run(document, actor_kind=ActorKind.HUMAN, actor_id="external")
    output = tmp_path / str(document["experiment"]["output_workspace"])
    _receipt, child = FactorDevelopmentReceiptReader(output).load(evidence.artifact_uris[0])

    stored = (
        output / "development" / "factor-deterministic-evidence" / f"{child.checkpoint_hash}.json"
    )
    assert stored.is_file()
    stored.unlink()

    with pytest.raises(AuthoringError, match="evidence_artifact_unverifiable"):
        workflow.replay(document, actor_kind=ActorKind.HUMAN, actor_id="external")


def _proposal(child: Any) -> FactorResearchProposal:
    """Use the eligible Factors and the Host-derived limits recorded in the evidence.
    The generated walk may produce mixed evidence, so the proposal follows its actual verdict.
    """

    cluster_by_factor = {
        factor_id: cluster
        for cluster in child.redundancy_structure.clusters
        for factor_id in cluster.member_factor_ids
    }
    roles = {
        FactorEvidenceClassification.POSITIVE_OOS_EVIDENCE: FactorResearchCandidateRole.CORE,
        FactorEvidenceClassification.MIXED_OOS_EVIDENCE: FactorResearchCandidateRole.CONDITIONAL,
    }
    return build_factor_research_proposal(
        evidence_report_hash=child.evidence_report.report_hash,
        redundancy_structure_hash=child.redundancy_structure.structure_hash,
        choices=tuple(
            FactorResearchProposalChoice(
                factor_id=item.factor_id,
                role=roles[item.classification],
                evidence_hash=item.evidence_hash,
                cluster_id=cluster_by_factor[item.factor_id].cluster_id,
                rationale="Mechanics acceptance only; this is not a scientific selection.",
            )
            for item in child.evidence_report.items
            if item.classification in roles
        ),
        limitations_acknowledged=FACTOR_REQUIRED_RESEARCH_LIMITATIONS,
    )


def test_human_and_external_automation_curate_identically_with_distinct_provenance(
    development_run: ExternalFactorRun,
) -> None:
    """Human and external automation curate identically with distinct provenance."""

    child = development_run.child
    proposal = _proposal(child)
    human = submit_factor_development_curation(
        evidence_root=development_run.root,
        receipt_handle=development_run.evidence.artifact_uris[0],
        proposal=proposal,
        actor_kind=ActorKind.HUMAN,
        actor_id="researcher@example.test",
    )
    external = submit_factor_development_curation(
        evidence_root=development_run.root,
        receipt_handle=development_run.evidence.artifact_uris[0],
        proposal=proposal,
        actor_kind=ActorKind.EXTERNAL_AUTOMATION,
        actor_id="external-codex",
    )

    assert human.decision.submission.submission_hash == (
        external.decision.submission.submission_hash
    )
    assert human.decision.actor_submission.binding_hash != (
        external.decision.actor_submission.binding_hash
    )
    assert human.decision.receipt_hash != external.decision.receipt_hash
    assert human.decision.actor_submission.agent_execution is None
    assert external.decision.actor_submission.agent_execution is None
    # One Host policy, derived from the installed source rather than named by
    # either actor.
    policy = factor_research_decision_policy_hash()
    assert human.decision.decision_policy_hash == policy
    assert external.decision.decision_policy_hash == policy
    # The Host recompiled the research input from the checkpoint; neither actor
    # supplied one.
    assert human.decision.submission.research_input.limitations == (
        FACTOR_REQUIRED_RESEARCH_LIMITATIONS
    )
    assert human.checkpoint.checkpoint_hash == child.checkpoint_hash

    filed = FactorDevelopmentCurationReader(development_run.root).submissions(child.checkpoint_hash)
    assert len(filed) == 2
    assert {value.actor_submission.actor_kind for value in filed} == {
        ActorKind.HUMAN,
        ActorKind.EXTERNAL_AUTOMATION,
    }
    assert len({value.submission.submission_hash for value in filed}) == 1

    # Replay walks the persisted decisions and still performs no numerical work.
    replayed, _ = development_run.workflow.replay(
        development_run.document, actor_kind=ActorKind.HUMAN, actor_id="external"
    )
    assert replayed.disposition == "REUSED_EXACT"
    assert replayed.numerical_call_count == 0


def test_a_curation_decision_the_host_never_admitted_cannot_be_persisted(
    development_run: ExternalFactorRun,
) -> None:
    """A curation decision the host never admitted cannot be persisted."""

    from alphalattice.foundation.factor_research.experiments.development_evidence import (
        publish_factor_development_curation,
    )
    from alphalattice.foundation.factor_research.programs.sealed import seal_contract
    from alphalattice.foundation.factor_research.research_loop.contracts import (
        FactorResearchReviewDecisionReceipt,
    )
    from alphalattice.foundation.factor_research.research_loop.decisions import (
        FactorResearchDecisionAuthorityError,
        admit_factor_research_curation,
    )

    child = development_run.child
    honest = admit_factor_research_curation(
        checkpoint=child,
        review_binding_hash=development_run.receipt.method_binding_hash,
        proposal=_proposal(child),
        actor_kind=ActorKind.HUMAN,
        actor_id="researcher@example.test",
    )
    resealed = seal_contract(
        FactorResearchReviewDecisionReceipt,
        "receipt_hash",
        submission=honest.submission,
        actor_submission=honest.actor_submission,
        decision_policy_hash="9" * 64,
    )
    with pytest.raises(FactorResearchDecisionAuthorityError, match="policy_not_installed"):
        publish_factor_development_curation(
            development_run.root,
            checkpoint=child,
            review_binding_hash=development_run.receipt.method_binding_hash,
            receipt=resealed,
        )


def test_no_product_branch_agent_or_admission_was_required() -> None:
    """An external methodology extension installs without a product source change, Agent edit, or
    product admission."""

    surface = describe_factor_authoring_surface()
    assert {item.factor_id for item in surface.entries} == {
        item.factor_id for item in extension_factor_specs()
    }
    assert EXTERNAL_FACTOR_ID not in {item.factor_id for item in surface.entries}
    assert set(surface.admitted_factor_ids) == (
        {item.factor_id for item in extension_factor_specs()} - {"sector_leader_lag_5"}
    )
    assert surface.refused_factor_ids == ("sector_leader_lag_5",)


def test_the_external_goldens_hold_against_an_independent_expectation() -> None:
    """The external goldens hold against an independent expectation."""

    specification = external_formula_specification()
    registry = external_kernel_registry()
    recipe = external_factor_spec()
    assert specification.clock.one_row_short_is_missing
    assert specification.minimum_ordered_source_rows == recipe.minimum_observations

    for example in specification.golden_examples:
        computed = registry.compute(golden_example_frame(example), recipe)
        observed = float(computed.iloc[-1])
        if example.expected_value is None:
            assert math.isnan(observed), example.label
            continue
        assert math.isclose(
            observed,
            example.expected_value,
            abs_tol=specification.absolute_tolerance,
            rel_tol=specification.relative_tolerance,
        ), example.label

    # The analytic expectation, restated independently of the frozen value: a
    # close exactly halfway up every session's range averages to one half.
    assert specification.golden_examples[0].expected_value == pytest.approx(0.5, abs=0.0)
