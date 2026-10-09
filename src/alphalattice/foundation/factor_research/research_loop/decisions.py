"""Host-owned admission and sealing for actor-neutral Factor research curation.

An actor -- a person, an Installed Agent, an external automation -- supplies a
*proposal* and says who they are. Everything that decides whether that proposal
is admissible is derived here, from the deterministic checkpoint, and nowhere
else.

That was not true before, in two specific ways that both looked like authority
and were not.

The sealer accepted a ``decision_policy_hash`` from its caller. A policy identity
supplied by whoever is being validated is a value, not a policy: a caller could
seal a decision under any well-formed sixty-four hex characters and the receipt
would validate against itself forever. It is derived from the installed source
closure now, and callers cannot name it.

The sealer also accepted an ``authoritative_research_input`` beside the
submission that contained one -- and the production runtime passed the *same*
object into both, because that is the only object it had. Two copies of one side
compared with each other agree by construction. The authoritative input is
recompiled here from the checkpoint's own evidence and redundancy structure, the
submitted proposal, and the Host's required limitations, and the submission must
match it exactly.

The required limitations live here rather than in the Agent adapter for the same
reason. They are a property of what a Factor curation *means*, not of how one
particular actor was prompted; an adapter that owned them would make them true
only for the actor that happened to read them.

Nothing here weighs the actor. Actor kind decides what provenance is admissible
-- only an Installed Agent may carry Agent execution evidence -- and never
whether the science is valid or preferred.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Final, cast

from alphalattice.foundation.factor_research.inputs.research_input import (
    FactorHorizonResearchInput,
    FactorResearchProposal,
    compile_factor_horizon_research_input,
)
from alphalattice.foundation.factor_research.programs.program import (
    FactorResearchDeterministicEvidence,
)
from alphalattice.foundation.factor_research.programs.sealed import seal_contract
from alphalattice.foundation.factor_research.research_loop.contracts import (
    FactorResearchReviewDecisionReceipt,
    FactorResearchReviewSubmission,
)
from alphalattice.foundation.factor_research.research_loop.decision_dossier import (
    build_factor_research_decision_dossier,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.identity_successors import is_current
from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root
from alphalattice.kernel.shared_kernel.source_identity import source_rule_closure_hash
from alphalattice.protocols.actor_execution import (
    ActorKind,
    AgentExecutionBinding,
    seal_actor_submission,
)

FACTOR_REQUIRED_RESEARCH_LIMITATIONS: Final[tuple[str, ...]] = (
    "ALPHA_SCIENTIFIC_STOP_PRESERVED",
    "CURRENT_UNIVERSE_RESEARCH_ONLY",
    "RISK_RESEARCH_NOT_ADMITTED",
    "SEALED_HOLDOUT_UNREAD",
)
"""What every Factor curation is limited by, whoever produced it.

Sorted, and the order is load-bearing: it is the tuple compiled into the
authoritative research input, so a re-derivation has to produce the same
sequence to produce the same identity.

Owned here rather than in the Agent adapter. An adapter that owned them would
make them a property of one actor's prompt, so a Human or an external automation
could submit a curation limited by nothing and the Host would have no
independent statement to check it against.
"""

_INSTALLED_PLAYPEN_ROOT: Final = resolve_playpen_root(Path(__file__))
"""Where this installed module actually sits, used when no root is named.

The policy identity is a statement about the code that validates a decision, so
the code being measured has to be the code that is running. Derived from
``__file__`` rather than accepted as an argument, because an argument is exactly
how a caller supplies its own policy.
"""


class FactorResearchDecisionAuthorityError(ValueError):
    """A Factor submission failed deterministic Host authority validation."""


DECISION_POLICY_ROLE = "factor_research.review_decision_policy"
"""The role this policy's identity is recorded under in the identity successors: a
curation sealed under a value recorded there as a predecessor is current authority."""


@cache
def _policy_hash_for(root: Path) -> str:
    return cast(
        str,
        canonical_hash(
            {
                "sources": source_rule_closure_hash(
                    root=root,
                    tracked_paths=(
                        "src/alphalattice/foundation/factor_research/inputs/research_input.py",
                        "src/alphalattice/foundation/factor_research/research_loop/contracts.py",
                        "src/alphalattice/foundation/factor_research/research_loop/decision_dossier.py",
                        "src/alphalattice/foundation/factor_research/research_loop/decisions.py",
                        "src/alphalattice/protocols/actor_execution/contracts.py",
                    ),
                    semantic_owner="factor_research.review_decision",
                    numerical_role="HOST_DECISION_POLICY",
                ),
            }
        ),
    )


def factor_research_decision_policy_hash(playpen_root: Path | None = None) -> str:
    """Bind Host decision validation without binding an optional Agent runtime.

    ``playpen_root`` is optional and exists for the callers that already hold a
    resolved root and want to state it. Omitting it measures the installed
    module's own tree, which is what a sealer must do: a policy identity handed
    in by the thing being validated is not a policy.

    Args:
        playpen_root: Explicit installed source root; None selects this module's installation.

    Returns:
        Measured Host validation-policy identity without optional actor-runtime composition.
    """
    return _policy_hash_for(
        _INSTALLED_PLAYPEN_ROOT if playpen_root is None else Path(playpen_root).resolve()
    )


def compile_factor_research_curation_submission(
    *,
    checkpoint: FactorResearchDeterministicEvidence,
    review_binding_hash: str,
    proposal: FactorResearchProposal,
) -> FactorResearchReviewSubmission:
    """Derive the one admissible submission for this checkpoint and proposal.

    Everything except the proposal comes from the checkpoint. The dossier is
    rebuilt from it, the research input is recompiled from its own evidence
    report and redundancy structure under the Host's required limitations, and
    the submission identity follows from those. An actor's own submission is
    admissible exactly when it equals this one.

    The proposal's acknowledged limitations are checked against the required set
    here rather than trusted. That check used to live only in the Agent adapter,
    which made it a rule about one actor rather than about Factor curation.

    Args:
        checkpoint: Host-validated deterministic evidence to interpret.
        review_binding_hash: Review binding retained in the rebuilt dossier.
        proposal: Actor choices and acknowledged research limitations.

    Returns:
        The canonical submission re-derived from the checkpoint and proposal.

    Raises:
        FactorResearchDecisionAuthorityError: A required research limitation was not acknowledged.
        ValueError: A checkpoint, proposal, or derived research input fails validation.
    """
    checkpoint = FactorResearchDeterministicEvidence.model_validate(checkpoint)
    proposal = FactorResearchProposal.model_validate(proposal)
    missing = tuple(
        value
        for value in FACTOR_REQUIRED_RESEARCH_LIMITATIONS
        if value not in proposal.limitations_acknowledged
    )
    if missing:
        raise FactorResearchDecisionAuthorityError(
            f"factor_research.review_limitations_incomplete:{missing[0]}"
        )
    dossier = build_factor_research_decision_dossier(
        checkpoint, review_binding_hash=review_binding_hash
    )
    research_input = _recompiled_research_input(checkpoint=checkpoint, proposal=proposal)
    return cast(
        FactorResearchReviewSubmission,
        seal_contract(
            FactorResearchReviewSubmission,
            "submission_hash",
            dossier_hash=dossier.dossier_hash,
            proposal=proposal,
            research_input=research_input,
        ),
    )


def _recompiled_research_input(
    *,
    checkpoint: FactorResearchDeterministicEvidence,
    proposal: FactorResearchProposal,
) -> FactorHorizonResearchInput:
    """Re-run the deterministic compiler the Desk already owns.

    Wrapped only to turn its boundary failures into this module's refusal: a
    proposal that names an unknown factor or a stale evidence hash is a rejected
    curation, not a Host error escaping from an input compiler.
    """
    try:
        return compile_factor_horizon_research_input(
            evidence=checkpoint.evidence_report,
            redundancy=checkpoint.redundancy_structure,
            proposal=proposal,
            limitations=FACTOR_REQUIRED_RESEARCH_LIMITATIONS,
        )
    except ValueError as error:
        raise FactorResearchDecisionAuthorityError(
            f"factor_research.review_input_not_derivable:{error}"
        ) from error


def seal_factor_research_review_decision(
    *,
    checkpoint: FactorResearchDeterministicEvidence,
    review_binding_hash: str,
    submission: FactorResearchReviewSubmission,
    actor_kind: ActorKind,
    actor_id: str,
    agent_execution: AgentExecutionBinding | None = None,
) -> FactorResearchReviewDecisionReceipt:
    """Seal one valid domain curation without invoking Agent machinery.

    The submission is admitted by *reproduction*, not by inspection: the Host
    derives the only submission this checkpoint and this proposal admit and
    requires the actor's to equal it, field for field. That closes the two ways
    an actor used to be able to supply its own authority -- carrying a research
    input the Host merely echoed back, and naming the decision policy under which
    it would be judged.

    Args:
        checkpoint: Evidence authority used to re-derive the admissible domain content.
        review_binding_hash: Qualified review binding used by the rebuilt dossier.
        submission: Actor submission required to equal the Host-derived submission.
        actor_kind: Declared actor category for provenance.
        actor_id: Actor identity retained in the submission binding.
        agent_execution: Qualified execution binding when the actor is an agent.

    Returns:
        Host-sealed decision with domain content, actor provenance, and installed policy identity.

    Raises:
        FactorResearchDecisionAuthorityError: Dossier, input, submission, or required limits
            disagree.
        ValueError: A domain or actor binding is invalid.
    """
    submission = FactorResearchReviewSubmission.model_validate(submission)
    admitted = compile_factor_research_curation_submission(
        checkpoint=checkpoint,
        review_binding_hash=review_binding_hash,
        proposal=submission.proposal,
    )
    if submission.dossier_hash != admitted.dossier_hash:
        raise FactorResearchDecisionAuthorityError("factor_research.review_dossier_mismatch")
    if submission.research_input != admitted.research_input:
        raise FactorResearchDecisionAuthorityError(
            "factor_research.review_input_authority_mismatch"
        )
    if submission.submission_hash != admitted.submission_hash:
        # Reachable only when the two differ in something the two checks above
        # do not cover, which means this contract grew a field. Refusing is the
        # correct default for an identity nobody re-derived.
        raise FactorResearchDecisionAuthorityError(
            "factor_research.review_submission_authority_mismatch"
        )
    actor_submission = seal_actor_submission(
        actor_kind=actor_kind,
        actor_id=actor_id,
        submission_hash=admitted.submission_hash,
        agent_execution=agent_execution,
    )
    return cast(
        FactorResearchReviewDecisionReceipt,
        seal_contract(
            FactorResearchReviewDecisionReceipt,
            "receipt_hash",
            submission=admitted,
            actor_submission=actor_submission,
            decision_policy_hash=factor_research_decision_policy_hash(),
        ),
    )


def admit_factor_research_curation(
    *,
    checkpoint: FactorResearchDeterministicEvidence,
    review_binding_hash: str,
    proposal: FactorResearchProposal,
    actor_kind: ActorKind,
    actor_id: str,
    agent_execution: AgentExecutionBinding | None = None,
) -> FactorResearchReviewDecisionReceipt:
    """The route for an actor that supplies a proposal and nothing else.

    Which is what an actor should supply. The domain content beyond the proposal
    -- the dossier it answers, the research input it implies, the limitations it
    is bounded by -- is derivable from the checkpoint, so asking an actor for it
    only creates something to disagree about.

    Args:
        checkpoint: Host evidence from which domain content is derived.
        review_binding_hash: Qualified review binding for the dossier.
        proposal: Actor's complete candidate choices and acknowledged limits.
        actor_kind: Provenance category of the proposing actor.
        actor_id: Identity retained in that provenance.
        agent_execution: Qualified binding required by an agent actor.

    Returns:
        Decision receipt sealed after Host derivation and reproduction of the domain submission.

    Raises:
        ValueError: Proposal, research-input authority, required limits, or actor provenance is
            invalid.
    """
    return seal_factor_research_review_decision(
        checkpoint=checkpoint,
        review_binding_hash=review_binding_hash,
        submission=compile_factor_research_curation_submission(
            checkpoint=checkpoint,
            review_binding_hash=review_binding_hash,
            proposal=proposal,
        ),
        actor_kind=actor_kind,
        actor_id=actor_id,
        agent_execution=agent_execution,
    )


def verify_factor_research_curation(
    *,
    decision: FactorResearchReviewDecisionReceipt,
    checkpoint: FactorResearchDeterministicEvidence,
    review_binding_hash: str,
    recorded_readback: bool = False,
) -> None:
    """Re-derive a persisted decision from the checkpoint that it judges.

    A receipt validates its own ``receipt_hash`` over its own contents, which
    proves it was not edited and proves nothing about whether it was ever
    admissible. This is the check that makes the difference: the dossier, the
    research input, the submission identity and the decision policy are all
    recomputed from the checkpoint and the installed source, and the receipt has
    to agree with every one of them.

    Writers, current reuse and handoff require it, with the installed policy. A
    recorded readback reopens the decision as it was sealed and computes nothing
    (LAWS.md OP6): its hashes and its actor's binding were checked as it
    loaded, the file it was read from is filed under the checkpoint it answers,
    and its caller labels the recorded policy historical, not current authority.
    Recompiling what a read reopened is the verify-all operation's work.

    Args:
        decision: Recorded receipt whose admissibility is to be checked.
        checkpoint: Host evidence authority used to re-derive its domain submission.
        review_binding_hash: Binding used to reproduce the decision dossier.
        recorded_readback: Skip current derivation for an explicitly historical reopened record.

    Raises:
        FactorResearchDecisionAuthorityError: Current dossier, research input, submission,
            installed decision policy, or actor binding disagrees with the recorded receipt.
        ValueError: Re-derived proposal or evidence authority fails validation.
    """
    if recorded_readback:
        return
    admitted = compile_factor_research_curation_submission(
        checkpoint=checkpoint,
        review_binding_hash=review_binding_hash,
        proposal=decision.submission.proposal,
    )
    if decision.submission.dossier_hash != admitted.dossier_hash:
        raise FactorResearchDecisionAuthorityError("factor_research.review_dossier_mismatch")
    if decision.submission.research_input != admitted.research_input:
        raise FactorResearchDecisionAuthorityError(
            "factor_research.review_input_authority_mismatch"
        )
    if decision.submission.submission_hash != admitted.submission_hash:
        raise FactorResearchDecisionAuthorityError(
            "factor_research.review_submission_authority_mismatch"
        )
    if not is_current(
        DECISION_POLICY_ROLE, decision.decision_policy_hash, factor_research_decision_policy_hash()
    ):
        raise FactorResearchDecisionAuthorityError(
            "factor_research.review_decision_policy_not_installed"
        )
    if decision.actor_submission.submission_hash != admitted.submission_hash:
        # The contract already refuses a binding that names another submission;
        # this catches the remaining case, a binding that names a *valid* one
        # belonging to somebody else's curation.
        raise FactorResearchDecisionAuthorityError("factor_research.review_actor_binding_mismatch")


__all__ = [
    "FACTOR_REQUIRED_RESEARCH_LIMITATIONS",
    "FactorResearchDecisionAuthorityError",
    "admit_factor_research_curation",
    "compile_factor_research_curation_submission",
    "factor_research_decision_policy_hash",
    "seal_factor_research_review_decision",
    "verify_factor_research_curation",
]
