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
    build_research_program_workflow,
    host_resolved_factor_inventory,
    installed_desk_verifiers,
)
from alphalattice.control.product_host.research_authoring.authority import (
    WorkspaceResearchAuthorityResolver,
)
from alphalattice.control.product_host.research_authoring.execution import (
    build_installed_desk_executors,
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
from tests.researcher_methodology_surface.external_factor_method import (
    EXTERNAL_FACTOR_ID,
    EXTERNAL_IMPLEMENTATION_ID,
    EXTERNAL_METHOD_FAMILY,
    EXTERNAL_METHOD_FAMILY_OWNERS,
    external_factor_spec,
    external_formula_specification,
    external_kernel_registry,
    external_specification_catalog,
)
from tests.researcher_methodology_surface.factor_evidence_support import (
    DEVELOPMENT_EVIDENCE_POLICY,
    DEVELOPMENT_REDUNDANCY_POLICY,
)
from tests.researcher_methodology_surface.real_workspace import (
    MARKET_PROFILE_ID,
    RealRiskWorkspace,
    publish_causal_outcomes,
)

CASE_ROOT = Path(__file__).resolve().parent
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
    """The frozen typed/YAML batch request, exactly as authored.

    Read rather than assembled. A document built in Python here would let this
    case study quietly grant itself fields the real loader would reject.
    """

    return copy.deepcopy(dict(load_authoring_document(_REQUEST.read_text(encoding="utf-8"))))


def _compose(workspace: RealRiskWorkspace, root: Path, document: dict[str, Any]) -> Any:
    """Compose exactly what the canonical CLI composes, through Product Host.

    Nothing in this function knows the external method exists. That is the
    acceptance: the Host resolves an inventory off the published Panel, installs
    the Factor executor for the declared kind, and installs its verifiers -- and
    every one of those steps is the product's own, unbranched.

    The installed owners are forwarded, not named. A composer always holds the
    catalog revision and kernel set its workspace's writers were built with; here
    it hands them to the reader as well, because the Host verifies a Panel's clock
    authority against installed owners before resolving it. What is in them stays
    the workspace's business.
    """

    envelope = ResearchExperimentEnvelope.create(**document["experiment"])
    inventory = host_resolved_factor_inventory(
        workspace=workspace.workspace, market_profile_id=MARKET_PROFILE_ID
    )
    executors = build_installed_desk_executors(
        envelope=envelope,
        workspace=workspace.workspace,
        workspace_root=root,
        document=document,
        feature_catalog=workspace.feature_catalog,
        feature_kernels=workspace.feature_kernels,
        # Cross-section scale, not science: ten listings cannot meet the
        # production redundancy floor of a hundred common listings.
        factor_evidence_policy=DEVELOPMENT_EVIDENCE_POLICY,
        factor_redundancy_policy=DEVELOPMENT_REDUNDANCY_POLICY,
    )
    return build_research_program_workflow(
        workspace=workspace.workspace,
        workspace_root=root,
        executors=executors,
        factor_inventory=inventory,
        # The composer's existing seam, used for what it is for. The dispatcher
        # would otherwise build a resolver holding the shipped owners, and the
        # Panel this workspace published is one only its own revision answers for.
        authority=WorkspaceResearchAuthorityResolver(
            workspace=workspace.workspace,
            feature_catalog=workspace.feature_catalog,
            feature_kernels=workspace.feature_kernels,
        ),
    )


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
def external_outcomes(external_factor_workspace: RealRiskWorkspace) -> RealRiskWorkspace:
    """The external workspace with the causal outcomes a Factor program reads, published once.

    Every test that runs a Factor program here names this rather than counting on another
    test to have published them first: a scheduler that starts the longest work first
    can run such a test first on its worker, where nothing published them.
    """

    publish_causal_outcomes(external_factor_workspace, at=_PUBLISHED_AT)
    return external_factor_workspace


@pytest.fixture(scope="module")
def external_run(
    external_outcomes: RealRiskWorkspace, tmp_path_factory: pytest.TempPathFactory
) -> ExternalFactorRun:
    """Freeze the authored batch over the published outcomes, and run it once."""

    external_factor_workspace = external_outcomes
    root = tmp_path_factory.mktemp("external-factor-run")
    document = _document()
    workflow = _compose(external_factor_workspace, root, document)
    evidence, binding = workflow.run(document, actor_kind=ActorKind.HUMAN, actor_id="external")
    assert binding.actor_kind is ActorKind.HUMAN
    output = root / str(document["experiment"]["output_workspace"])
    receipt, child = FactorDevelopmentReceiptReader(output).load(evidence.artifact_uris[0])
    envelope = ResearchExperimentEnvelope.create(**document["experiment"])
    return ExternalFactorRun(
        workflow=workflow,
        document=document,
        root=output,
        inventory=host_resolved_factor_inventory(
            workspace=external_factor_workspace.workspace, market_profile_id=MARKET_PROFILE_ID
        ),
        evidence=evidence,
        receipt=receipt,
        child=child,
        authority=WorkspaceResearchAuthorityResolver(
            workspace=external_factor_workspace.workspace,
            feature_catalog=external_factor_workspace.feature_catalog,
            feature_kernels=external_factor_workspace.feature_kernels,
        ).resolve(envelope),
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
    external_run: ExternalFactorRun,
    external_factor_workspace: RealRiskWorkspace,
) -> None:
    """requirement: an outside formula is a first-class Factor on the Panel.

    Not a special case and not a declared identity. The Panel publishes this
    Factor's implementation and methodology hashes the same way it publishes the
    current base activations', and its implementation hash is measured from the bytes of a
    module outside the product -- resolved through the import system, which is
    what makes an external family measurable at all.
    """

    by_factor = {entry.factor_id: entry for entry in external_run.inventory}
    # The whole installed axis and nothing else: the shipped catalog plus this
    # one external Factor. The shipped catalog's own size is the feature-engine
    # owner's pin, not a number restated here -- a literal that has to move with
    # every catalog revision measures the revision, not the seam.
    installed = external_factor_workspace.feature_catalog.factors
    assert len(installed) == len(FeatureCatalog.load().factors) + 1
    assert len(external_run.inventory) == len(installed)
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
    """requirement: admission is granted by a catalog, never by being importable.

    Two facts in one case because they are one claim: the external consumer's own
    catalog admits their method, and the *product* catalog does not gain it. The
    second half is what stops a case study from becoming a product admission.
    """

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
    """requirement: freeze, run, verify recursively, replay at zero numerical work.

    The counts are the point of the last step. ``run`` reports one sealed
    deterministic program execution; ``replay`` resolves no executor at all and
    reports zero, so the zero is a fact about the path rather than a claim about
    a run that happened to skip its work. Between them the installed verifier
    walks the receipt, its deterministic child and every identity that ties them
    to this Program -- which is what makes ``REUSED_EXACT`` a statement rather
    than a label.
    """

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


def test_the_verifier_refuses_a_fully_resealed_graph(external_run: ExternalFactorRun) -> None:
    """requirement: self-consistency is not authority.

    The adversary is not a corrupted file. It is a receipt whose ordered context
    axis was rewritten and whose every dependent hash was recomputed, so the
    graph validates perfectly against itself and its own ``receipt_hash`` is
    correct. Two sides contradict it: the Program, sealed before the run from the
    Host-resolved Panel, and the authority resolved fresh from the workspace. The
    same construction with a substituted Panel identity is refused by the second.
    """

    verifier = next(
        item for item in installed_desk_verifiers() if item.kind == external_run.evidence.kind
    )
    # The healthy graph passes, or the refusals below would prove nothing.
    verifier.verify(
        program=external_run.receipt.program,
        evidence=external_run.evidence,
        authority=external_run.authority,
        output_workspace=external_run.root,
    )

    binding = external_run.receipt.input_binding
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
            program=external_run.receipt.program,
            input_binding=resealed_binding,
            checkpoint_hash=external_run.receipt.checkpoint_hash,
        )
        uri = publish_factor_development_receipt(external_run.root, resealed_receipt)
        resealed_evidence = ResearchExecutionEvidence.create(
            kind=external_run.evidence.kind,
            program_hash=external_run.evidence.program_hash,
            desk_program_hash=external_run.evidence.desk_program_hash,
            method_binding_hash=external_run.evidence.method_binding_hash,
            authority_hash=external_run.evidence.authority_hash,
            disposition="COMPUTED",
            numerical_call_count=1,
            artifact_uris=(uri,),
            formation_sessions=external_run.evidence.formation_sessions,
            desk_input_binding_hash=resealed_binding.binding_hash,
        )
        expected = (
            "evidence_context_axis_not_admitted"
            if label == "axis"
            else "evidence_panel_not_this_authority"
        )
        with pytest.raises(AuthoringError, match=expected):
            verifier.verify(
                program=external_run.receipt.program,
                evidence=resealed_evidence,
                authority=external_run.authority,
                output_workspace=external_run.root,
            )


def test_a_missing_deterministic_child_fails_replay(
    external_outcomes: RealRiskWorkspace, tmp_path: Path
) -> None:
    """requirement: the graph is walked, so a deleted child cannot be reused.

    The receipt survives untouched and still validates against its own hash.
    Under the previous replay -- which refused this Desk outright because no
    verifier existed -- this was indistinguishable from a healthy graph, because
    nothing opened the child at all.

    Its own run and its own root, because it destroys what it built.
    """

    document = _document()
    workflow = _compose(external_outcomes, tmp_path, document)
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
    """Propose exactly the Factors the evidence says are proposable.

    Derived rather than fixed. Whether a seeded random walk produces positive or
    mixed evidence is not something a mechanics case may assume, and the input
    compiler refuses an empty proposal when eligible candidates exist.

    The limitations come from the Host's own required set. They used to be a
    single value chosen here, which passed only because the acknowledgement was
    checked inside the Agent adapter and this route never went near it.
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
    external_run: ExternalFactorRun,
) -> None:
    """requirement: one Host route, one domain identity, two provenances.

    Both actors hand the product route a receipt handle and a proposal. Nothing
    else: the dossier, the research input, the limitations, the decision policy
    and the submission identity are all derived by the Host from the checkpoint
    the receipt points at, so identical content produces one submission identity
    by construction rather than by two callers agreeing.

    Driven through ``submit_factor_development_curation`` rather than by calling
    the sealer and the writer in order. That sequence was the only thing that had
    ever exercised development curation, which meant the case study was the
    product path -- and a gap like that reads as a green suite.
    """

    child = external_run.child
    proposal = _proposal(child)
    human = submit_factor_development_curation(
        evidence_root=external_run.root,
        receipt_handle=external_run.evidence.artifact_uris[0],
        proposal=proposal,
        actor_kind=ActorKind.HUMAN,
        actor_id="researcher@example.test",
    )
    external = submit_factor_development_curation(
        evidence_root=external_run.root,
        receipt_handle=external_run.evidence.artifact_uris[0],
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

    filed = FactorDevelopmentCurationReader(external_run.root).submissions(child.checkpoint_hash)
    assert len(filed) == 2
    assert {value.actor_submission.actor_kind for value in filed} == {
        ActorKind.HUMAN,
        ActorKind.EXTERNAL_AUTOMATION,
    }
    assert len({value.submission.submission_hash for value in filed}) == 1

    # Replay walks the persisted decisions and still performs no numerical work.
    replayed, _ = external_run.workflow.replay(
        external_run.document, actor_kind=ActorKind.HUMAN, actor_id="external"
    )
    assert replayed.disposition == "REUSED_EXACT"
    assert replayed.numerical_call_count == 0


def test_a_curation_decision_the_host_never_admitted_cannot_be_persisted(
    external_run: ExternalFactorRun,
) -> None:
    """requirement: the writer is not a caller-trusted store.

    A receipt whose decision policy is a value somebody chose is internally
    perfect and was previously written and replayed without complaint, because
    the writer took a bare checkpoint hash and the verifier only compared
    evidence references. Both now re-derive through the decision owner, so a
    decision is persisted exactly when it would also verify.
    """

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

    child = external_run.child
    honest = admit_factor_research_curation(
        checkpoint=child,
        review_binding_hash=external_run.receipt.method_binding_hash,
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
            external_run.root,
            checkpoint=child,
            review_binding_hash=external_run.receipt.method_binding_hash,
            receipt=resealed,
        )


def test_no_product_branch_agent_or_admission_was_required() -> None:
    """requirement: the platform absorbed a new method with no product change.

    Four checks because there are four ways this could have been false: a Host
    branch on the method, an Agent edit, a product admission, or a Stage 2
    numerical claim. The authoring report is the product's own answer about its
    own state, and it must be exactly what it was before this case existed.
    """

    surface = describe_factor_authoring_surface()
    assert {item.factor_id for item in surface.entries} == {
        item.factor_id for item in extension_factor_specs()
    }
    assert EXTERNAL_FACTOR_ID not in {item.factor_id for item in surface.entries}
    assert set(surface.admitted_factor_ids) == (
        {item.factor_id for item in extension_factor_specs()} - {"sector_leader_lag_5"}
    )
    assert surface.refused_factor_ids == ("sector_leader_lag_5",)

    for directory in (
        PLAYPEN_ROOT / "src" / "alphalattice" / "control" / "product_host",
        PLAYPEN_ROOT / "src" / "alphalattice" / "foundation" / "factor_research" / "agent",
    ):
        offenders = sorted(
            path.relative_to(PLAYPEN_ROOT).as_posix()
            for path in directory.rglob("*.py")
            for text in (path.read_text(encoding="utf-8"),)
            if EXTERNAL_FACTOR_ID in text or EXTERNAL_METHOD_FAMILY in text
        )
        assert offenders == [], directory.name


def test_the_external_goldens_hold_against_an_independent_expectation() -> None:
    """requirement: an external method's goldens are executable by the platform.

    The frame is built by the specification owner from the golden's named
    columns, so a consumer with three input fields needs no special handling --
    the property that used to be missing, and the reason a second method family
    would have had to widen the contract or ship without goldens.
    """

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
