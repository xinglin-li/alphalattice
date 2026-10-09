"""Factor curation terminates at Host authority, whoever submitted it.

An actor -- a person, an Installed Agent, an external automation -- supplies a
proposal and says who they are. Everything that decides whether that proposal is
admissible is derived by the Host from the deterministic checkpoint: the dossier
it answers, the research input it implies, the limitations bounding it, and the
policy identity it is judged under.

Two of those used to be caller input, and both looked like authority without
being any. The sealer accepted a ``decision_policy_hash`` argument, so any
well-formed sixty-four hex characters named the policy a decision was judged
under. And it accepted an ``authoritative_research_input`` beside the submission
that already contained one -- with the production runtime passing the *same*
object into both, because that was the only object it had. Two copies of one side
agree by construction.

These cases drive the real checkpoint this directory already computes, so the
thing being curated is the thing the deterministic program actually produced
rather than a plausible JSON file.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from alphalattice.foundation.factor_research.evaluation.oos_evidence import (
    FactorEvidenceClassification,
)
from alphalattice.foundation.factor_research.inputs.research_input import (
    FactorHorizonResearchInput,
    FactorResearchCandidateRole,
    FactorResearchProposal,
    FactorResearchProposalChoice,
    build_factor_research_proposal,
)
from alphalattice.foundation.factor_research.programs.program import (
    FactorResearchDeterministicEvidence,
)
from alphalattice.foundation.factor_research.programs.sealed import seal_contract
from alphalattice.foundation.factor_research.research_loop.contracts import (
    FactorResearchReviewDecisionReceipt,
    FactorResearchReviewSubmission,
)
from alphalattice.foundation.factor_research.research_loop.decisions import (
    FACTOR_REQUIRED_RESEARCH_LIMITATIONS,
    FactorResearchDecisionAuthorityError,
    admit_factor_research_curation,
    compile_factor_research_curation_submission,
    factor_research_decision_policy_hash,
    seal_factor_research_review_decision,
    verify_factor_research_curation,
)
from alphalattice.protocols.actor_execution import ActorKind, AgentExecutionBinding
from tests.factor_research_pipeline_correctness.case_program import (
    CASE_REVIEW_BINDING_HASH,
    build_case_deterministic_checkpoint,
)

_HASH = "a" * 64


def _checkpoint() -> FactorResearchDeterministicEvidence:
    return build_case_deterministic_checkpoint()


def _proposal(
    checkpoint: FactorResearchDeterministicEvidence,
    *,
    limitations: tuple[str, ...] = FACTOR_REQUIRED_RESEARCH_LIMITATIONS,
) -> FactorResearchProposal:
    """Propose exactly the Factors this evidence says are proposable.

    Derived from the classifications rather than fixed, because the input
    compiler refuses an empty proposal when eligible candidates exist and refuses
    a role the evidence does not support. A hard-coded slate would encode one
    run's numbers into a case about authority.
    """

    cluster_by_factor = {
        factor_id: cluster
        for cluster in checkpoint.redundancy_structure.clusters
        for factor_id in cluster.member_factor_ids
    }
    roles = {
        FactorEvidenceClassification.POSITIVE_OOS_EVIDENCE: FactorResearchCandidateRole.CORE,
        FactorEvidenceClassification.MIXED_OOS_EVIDENCE: FactorResearchCandidateRole.CONDITIONAL,
    }
    return build_factor_research_proposal(
        evidence_report_hash=checkpoint.evidence_report.report_hash,
        redundancy_structure_hash=checkpoint.redundancy_structure.structure_hash,
        choices=tuple(
            FactorResearchProposalChoice(
                factor_id=item.factor_id,
                role=roles[item.classification],
                evidence_hash=item.evidence_hash,
                cluster_id=cluster_by_factor[item.factor_id].cluster_id,
                rationale="Mechanics case only; this is not a scientific selection.",
            )
            for item in checkpoint.evidence_report.items
            if item.classification in roles
        ),
        limitations_acknowledged=limitations,
    )


def _agent_execution() -> AgentExecutionBinding:
    return AgentExecutionBinding(
        profile_id="factor-research.evidence-reviewer",
        mode="default",
        profile_hash=_HASH,
        document_hash=_HASH,
        response_protocol="factor_research_review",
        response_protocol_hash=_HASH,
        concrete_schema_hash=_HASH,
    )


def test_human_external_and_installed_actor_share_one_host_derived_submission() -> None:
    """requirement: identical domain content, identical Host authority, distinct provenance.

    All three actors hand the Host the same proposal and nothing else. The Host
    derives the submission -- so its identity is the same by construction rather
    than by three callers agreeing -- and derives the decision policy from its own
    installed source, so all three receipts are judged under one policy. What
    differs is the actor binding and therefore the receipt identity, which is what
    provenance is for.
    """

    checkpoint = _checkpoint()
    proposal = _proposal(checkpoint)
    common = {
        "checkpoint": checkpoint,
        "review_binding_hash": CASE_REVIEW_BINDING_HASH,
        "proposal": proposal,
    }
    human = admit_factor_research_curation(
        **common, actor_kind=ActorKind.HUMAN, actor_id="researcher@example.test"
    )
    external = admit_factor_research_curation(
        **common, actor_kind=ActorKind.EXTERNAL_AUTOMATION, actor_id="external-codex"
    )
    installed = admit_factor_research_curation(
        **common,
        actor_kind=ActorKind.INSTALLED_AGENT,
        actor_id="factor-research.evidence-reviewer",
        agent_execution=_agent_execution(),
    )

    assert human.submission.submission_hash == external.submission.submission_hash
    assert human.submission.submission_hash == installed.submission.submission_hash
    assert human.actor_submission.binding_hash != external.actor_submission.binding_hash
    assert external.actor_submission.binding_hash != installed.actor_submission.binding_hash
    assert len({human.receipt_hash, external.receipt_hash, installed.receipt_hash}) == 3

    policy = factor_research_decision_policy_hash()
    assert {value.decision_policy_hash for value in (human, external, installed)} == {policy}

    assert human.actor_submission.agent_execution is None
    assert external.actor_submission.agent_execution is None
    assert installed.actor_submission.agent_execution == _agent_execution()

    assert FactorResearchReviewDecisionReceipt.model_validate_json(human.model_dump_json()) == human


def test_an_installed_agent_without_execution_evidence_is_refused() -> None:
    """requirement: the Agent takes the same Host path and owes its provenance.

    There is no Agent-only sealing route to fall back to, so an Installed Agent
    submission without an ``AgentExecutionBinding`` has to fail on the one path
    everybody uses. The mirror case matters as much: a Human carrying Agent
    execution evidence is claiming a process it did not run.
    """

    checkpoint = _checkpoint()
    common = {
        "checkpoint": checkpoint,
        "review_binding_hash": CASE_REVIEW_BINDING_HASH,
        "proposal": _proposal(checkpoint),
    }
    with pytest.raises(ValueError, match="only installed Agent submissions"):
        admit_factor_research_curation(
            **common,
            actor_kind=ActorKind.INSTALLED_AGENT,
            actor_id="factor-research.evidence-reviewer",
        )
    with pytest.raises(ValueError, match="only installed Agent submissions"):
        admit_factor_research_curation(
            **common,
            actor_kind=ActorKind.HUMAN,
            actor_id="researcher@example.test",
            agent_execution=_agent_execution(),
        )


def test_a_submission_carrying_an_altered_research_input_is_refused() -> None:
    """requirement: the authoritative input is recompiled, never accepted.

    The adversary is not a malformed object. It is a submission that validates
    against itself perfectly -- a real proposal, a real research input, a
    correctly recomputed ``submission_hash`` -- whose input says something the
    checkpoint does not imply. Under the old sealer this was authoritative,
    because the same object was passed in as both the claim and the authority.
    """

    checkpoint = _checkpoint()
    proposal = _proposal(checkpoint)
    admitted = compile_factor_research_curation_submission(
        checkpoint=checkpoint,
        review_binding_hash=CASE_REVIEW_BINDING_HASH,
        proposal=proposal,
    )
    altered_input = seal_contract(
        FactorHorizonResearchInput,
        "input_hash",
        **{
            name: getattr(admitted.research_input, name)
            for name in type(admitted.research_input).model_fields
            if name not in {"input_hash", "kind"}
        }
        | {"limitations": ("CURRENT_UNIVERSE_RESEARCH_ONLY",)},
    )
    assert altered_input != admitted.research_input
    forged = seal_contract(
        FactorResearchReviewSubmission,
        "submission_hash",
        dossier_hash=admitted.dossier_hash,
        proposal=proposal,
        research_input=altered_input,
    )
    # Self-consistent: it revalidates its own identity on parse.
    assert FactorResearchReviewSubmission.model_validate_json(forged.model_dump_json()) == forged

    with pytest.raises(FactorResearchDecisionAuthorityError, match="input_authority_mismatch"):
        seal_factor_research_review_decision(
            checkpoint=checkpoint,
            review_binding_hash=CASE_REVIEW_BINDING_HASH,
            submission=forged,
            actor_kind=ActorKind.HUMAN,
            actor_id="researcher@example.test",
        )
    from alphalattice.protocols.actor_execution import seal_actor_submission

    forged_receipt = seal_contract(
        FactorResearchReviewDecisionReceipt,
        "receipt_hash",
        submission=forged,
        actor_submission=seal_actor_submission(
            actor_kind=ActorKind.HUMAN,
            actor_id="researcher@example.test",
            submission_hash=forged.submission_hash,
        ),
        decision_policy_hash="9" * 64,
    )
    with pytest.raises(FactorResearchDecisionAuthorityError, match="input_authority_mismatch"):
        verify_factor_research_curation(
            decision=forged_receipt,
            checkpoint=checkpoint,
            review_binding_hash=CASE_REVIEW_BINDING_HASH,
        )


def test_a_proposal_that_acknowledges_no_limitations_is_refused() -> None:
    """requirement: the required limitations bind every actor, not only the Agent.

    They were declared inside the Agent adapter, which made them a property of
    how one actor was prompted. A Human or an external automation could submit a
    curation limited by nothing, and there was no independent statement to check
    it against.
    """

    checkpoint = _checkpoint()
    with pytest.raises(FactorResearchDecisionAuthorityError, match="limitations_incomplete"):
        admit_factor_research_curation(
            checkpoint=checkpoint,
            review_binding_hash=CASE_REVIEW_BINDING_HASH,
            proposal=_proposal(checkpoint, limitations=("CURRENT_UNIVERSE_RESEARCH_ONLY",)),
            actor_kind=ActorKind.HUMAN,
            actor_id="researcher@example.test",
        )


def test_a_resealed_decision_naming_its_own_policy_is_refused() -> None:
    """requirement: a policy identity supplied by the judged party is not a policy.

    Everything in this receipt is internally correct: the submission is the one
    the Host derives, the actor binding names it, and ``receipt_hash`` covers the
    whole object including the substituted policy. Only re-deriving the policy
    from the installed source contradicts it -- which is why the sealer stopped
    taking it as an argument, and why the verifier recomputes it rather than
    reading it back.
    """

    checkpoint = _checkpoint()
    honest = admit_factor_research_curation(
        checkpoint=checkpoint,
        review_binding_hash=CASE_REVIEW_BINDING_HASH,
        proposal=_proposal(checkpoint),
        actor_kind=ActorKind.HUMAN,
        actor_id="researcher@example.test",
    )
    verify_factor_research_curation(
        decision=honest,
        checkpoint=checkpoint,
        review_binding_hash=CASE_REVIEW_BINDING_HASH,
    )

    resealed = seal_contract(
        FactorResearchReviewDecisionReceipt,
        "receipt_hash",
        submission=honest.submission,
        actor_submission=honest.actor_submission,
        decision_policy_hash="9" * 64,
    )
    assert (
        FactorResearchReviewDecisionReceipt.model_validate_json(resealed.model_dump_json())
        == resealed
    )
    with pytest.raises(FactorResearchDecisionAuthorityError, match="decision_policy_not_installed"):
        verify_factor_research_curation(
            decision=resealed,
            checkpoint=checkpoint,
            review_binding_hash=CASE_REVIEW_BINDING_HASH,
        )


def test_a_recorded_curation_reads_without_recompiling_or_current_admission(monkeypatch, tmp_path):
    """requirement (LAWS.md OP6): a read reopens what was sealed and computes nothing.

    A recorded readback took the decision's standing from the installed policy
    and recompiled its dossier and research input on every read; it now reopens
    the decision as sealed, while a writer and new work still re-derive it and
    require the installed policy."""

    from alphalattice.foundation.factor_research.experiments.development_evidence import (
        publish_factor_development_curation,
    )
    from alphalattice.foundation.factor_research.research_loop import decisions

    checkpoint = _checkpoint()
    honest = admit_factor_research_curation(
        checkpoint=checkpoint,
        review_binding_hash=CASE_REVIEW_BINDING_HASH,
        proposal=_proposal(checkpoint),
        actor_kind=ActorKind.HUMAN,
        actor_id="researcher",
    )
    common = dict(
        decision=honest, checkpoint=checkpoint, review_binding_hash=CASE_REVIEW_BINDING_HASH
    )
    monkeypatch.setattr(decisions, "factor_research_decision_policy_hash", lambda: "f" * 64)
    verify_factor_research_curation(**common, recorded_readback=True)
    with pytest.raises(FactorResearchDecisionAuthorityError, match="policy_not_installed"):
        verify_factor_research_curation(**common)
    with pytest.raises(FactorResearchDecisionAuthorityError, match="policy_not_installed"):
        publish_factor_development_curation(
            tmp_path,
            receipt=honest,
            checkpoint=checkpoint,
            review_binding_hash=CASE_REVIEW_BINDING_HASH,
        )
    with pytest.raises(FactorResearchDecisionAuthorityError, match="dossier_mismatch"):
        verify_factor_research_curation(**{**common, "review_binding_hash": "c" * 64})
    assert not tuple(tmp_path.iterdir())

    def recompiled(**_values):
        raise AssertionError("a recorded readback recompiled its curation")

    monkeypatch.setattr(decisions, "compile_factor_research_curation_submission", recompiled)
    verify_factor_research_curation(**common, recorded_readback=True)


def test_a_decision_about_another_review_binding_is_refused(tmp_path: Path) -> None:
    """requirement: the dossier a decision answers is the checkpoint's, under this binding.

    Same checkpoint, same proposal, a different Desk method binding: the dossier
    differs, so the submission the Host admits differs, so the decision belongs to
    a run this one is not.
    """

    checkpoint = _checkpoint()
    honest = admit_factor_research_curation(
        checkpoint=checkpoint,
        review_binding_hash=CASE_REVIEW_BINDING_HASH,
        proposal=_proposal(checkpoint),
        actor_kind=ActorKind.HUMAN,
        actor_id="researcher@example.test",
    )
    with pytest.raises(FactorResearchDecisionAuthorityError, match="dossier_mismatch"):
        verify_factor_research_curation(
            decision=honest,
            checkpoint=checkpoint,
            review_binding_hash="c" * 64,
        )
    assert not (tmp_path / "factor-research").exists()


def test_factor_submission_boundary_has_no_agent_runtime_import_closure() -> None:
    root = Path(__file__).resolve().parents[2]
    environment = {**os.environ, "PYTHONPATH": str(root / "src")}
    probe_code = "\n".join(
        (
            "import json, sys",
            "import alphalattice.foundation.factor_research.research_loop.decisions",
            "tokens = ('deepagents', 'langchain', 'agent_runtime', 'factor_research.agent')",
            "loaded = sorted(name for name in sys.modules",
            "                if any(token in name for token in tokens))",
            "print(json.dumps(loaded))",
        )
    )
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            probe_code,
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        check=True,
        text=True,
    )
    assert json.loads(probe.stdout) == []
