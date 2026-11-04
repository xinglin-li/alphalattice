"""Prepare and admit one explicitly bound external Analyst answer."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import Any, cast

from alphalattice.control.product_host.composition.answer_turns import (
    SUBMISSION_DEADLINE,
    bounded_answer_turn,
)
from alphalattice.evidence.alternative_evidence.analysis.submissions import (
    accepted_analyst_answer,
    screened_analysis,
)
from alphalattice.evidence.alternative_evidence.runtime.task_adapter import (
    SubmittedEvidenceAnalysis,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.oversight.chief_risk_officer.decision.book_evidence import (
    PortfolioEvidenceReviewError,
)
from alphalattice.protocols.actor_execution import ActorKind, seal_actor_submission
from alphalattice.protocols.actor_execution.answers import AnswerVerdict, answer_view


def answer_refusal_action(body: Any) -> str:
    """A no-effect busy refusal can be resent; other refusals keep their existing route."""
    return (
        "RETRY_THE_SAME_ANSWER"
        if body.get("failure_code") == "agent_bundle.admission_busy"
        else "TELL_YOUR_LEAD"
    )


def submit_analyst_analysis(
    self: Any,
    dispatcher: Any,
    selector: Any,
    task_id: Any,
    context_hash: Any,
    answer: Any,
    caller: Any,
    unit_id: Any,
    read_files: Any,
    agent_run: Any,
    admit: Any,
    review_type: Any,
    command_type: Any,
    correction_fn: Any,
    submitted_fn: Any,
) -> Any:
    """Read and screen outside the workspace gate; file and admit only after rechecking."""
    if caller not in {"HUMAN", "EXTERNAL_AUTOMATION"}:
        raise PortfolioEvidenceReviewError(
            "alternative_evidence.external_submission_entry_required"
        )
    chosen = self._review_selector(selector)
    if isinstance(chosen, review_type):
        return chosen
    if self.evidence_task_adapter is None:
        raise PortfolioEvidenceReviewError("product_host.evidence_task_adapter_absent")
    adapter = self.evidence_task_adapter
    registry, listing, resources = self.registry, self.listing_authority, adapter.resources
    prepared = self.session.task_control_registry.task(task_id)
    # The packet this request verified serves screening and the admission's check;
    # the Task verifies it again at execution, since the materials may change between.
    packet, context, resolved, *_ = self._book_packet(chosen, task_id=task_id, unit_id=unit_id)
    if context["analysis_context_hash"] != context_hash:
        raise PortfolioEvidenceReviewError("alternative_evidence.external_analysis_binding_changed")
    entity_ids = packet.request.ordered_entity_ids
    screened, unread, left = screened_analysis(answer, packet)
    view_of = partial(answer_view, left_out=left)
    binding: dict[str, object] = {
        "role": "alternative_evidence.analyst",
        "task_id": str(task_id),
        "unit_id": unit_id,
        "analysis_context_hash": context_hash,
    }

    def require_same_binding() -> None:
        # Work done before the gate cannot authorize a changed Task, default book or
        # replacement of the installed packet and listing authorities.
        if (
            self.evidence_task_adapter is not adapter
            or adapter.resources is not resources
            or self.registry is not registry
            or self.listing_authority is not listing
            or self.default_selector(selector) != chosen
            or self.session.task_control_registry.task(task_id).record_hash != prepared.record_hash
        ):
            raise PortfolioEvidenceReviewError(
                "alternative_evidence.external_analysis_binding_changed"
            )

    deadline = SUBMISSION_DEADLINE.get()
    turn = (
        self._answer_turn(binding)
        if deadline is None
        else bounded_answer_turn(self, binding, deadline)
    )
    with turn:
        record = self._next_answer(
            binding, answer, screened, read_files=read_files, agent_run=agent_run
        )
        if record.verdict is AnswerVerdict.CORRECT:

            def file_correction() -> dict[str, object]:
                require_same_binding()
                self._file_answer(record)
                return cast(
                    dict[str, object], correction_fn("alternative_evidence.answer_problems", record)
                )

            return file_correction() if admit is None else admit(file_correction)
        return _prepare_analysis(
            self,
            dispatcher,
            selector,
            task_id,
            caller,
            unit_id,
            admit,
            command_type,
            submitted_fn,
            packet,
            context,
            resolved,
            screened,
            unread,
            view_of,
            record,
            prepared,
            chosen,
            adapter,
            require_same_binding,
            entity_ids,
        )


def _prepare_analysis(
    self: Any,
    dispatcher: Any,
    selector: Any,
    task_id: Any,
    caller: Any,
    unit_id: Any,
    admit: Any,
    command_type: Any,
    submitted_fn: Any,
    packet: Any,
    context: Any,
    resolved: Any,
    screened: Any,
    unread: Any,
    view_of: Any,
    record: Any,
    prepared: Any,
    chosen: Any,
    adapter: Any,
    require_same_binding: Any,
    entity_ids: Any,
) -> Any:
    """Compile the sealed command before the bounded filing and admission."""
    accepted = accepted_analyst_answer(screened, entity_ids=entity_ids)
    dropped = (screened.problems if record.verdict is AnswerVerdict.DONE else ()) + unread
    submitted = SubmittedEvidenceAnalysis(
        prepared_task_id=task_id,
        prepared_unit_id=unit_id,
        packet_hash=cast(str, context["packet_hash"]),
        analysis_policy_hash=cast(str, context["analysis_policy_hash"]),
        decision_policy_hash=cast(str, context["decision_policy_hash"]),
        answer=accepted,
        dropped=dropped,
        actor_submission=seal_actor_submission(
            actor_kind=ActorKind(caller),
            actor_id="local-research-external"
            if caller == "EXTERNAL_AUTOMATION"
            else "local-web-human",
            submission_hash=canonical_hash(accepted.model_dump(mode="json")),
        ),
    )
    adapter.validate_submission(submitted, now=self.clock(), packet=packet)
    if unit_id is None:
        # The packet may be a continuation's: the answer is admitted under
        # its request, obligation and admission, never as another session.
        command = command_type.recover(application=self, task=prepared)
        command.prepare_only = False
        command.continuation = None
        command.submitted_analysis = submitted
        command.subject = chosen
        command.subject_explicit = selector is not None
        command.subject_book = resolved.book
    else:
        # One unit's answer is admitted as the request, question and
        # admission that unit was prepared under, so the same answer to
        # the same packet is the same Task input wherever it is submitted.
        authority = adapter.unit_authority(prepared, unit_id)
        command = command_type(
            application=self,
            obligation=authority.obligation,
            request=authority.request,
            admission=authority.admission,
            prepare_only=False,
            submitted_analysis=submitted,
            subject=chosen,
            subject_explicit=selector is not None,
            subject_book=resolved.book,
        )
    envelope, goal, plan = command.contract()
    command._admission_contract = (envelope, goal, plan)

    action = partial(
        _file_and_admit,
        self,
        dispatcher,
        selector,
        task_id,
        unit_id,
        submitted_fn,
        resolved,
        view_of,
        record,
        chosen,
        adapter,
        require_same_binding,
        accepted,
        command,
        envelope,
        plan,
    )
    return action() if admit is None else admit(action)


def _file_and_admit(
    self: Any,
    dispatcher: Any,
    selector: Any,
    task_id: Any,
    unit_id: Any,
    submitted_fn: Any,
    resolved: Any,
    view_of: Any,
    record: Any,
    chosen: Any,
    adapter: Any,
    require_same_binding: Any,
    accepted: Any,
    command: Any,
    envelope: Any,
    plan: Any,
) -> Any:
    """Reuse only the completed exact Task, otherwise admit and file the answer."""
    require_same_binding()
    published = self._completed_task(envelope.input_hash, plan.plan_hash)
    if published is not None:
        self.record_task_admission(
            published.task_id,
            chosen,
            now=self.clock(),
            explicit=selector is not None,
            book=resolved.book,
        )
        publication = adapter.published_analysis(published.task_id, now=self.clock())
        return {
            "disposition": "REUSED_EXACT",
            "task_id": None,
            "publication_task_id": str(published.task_id),
            "analysis_publication_hash": publication.publication.publication_hash,
            **({} if unit_id is None else {"evidence_unit_id": unit_id}),
            "external_host_usage": "UNAVAILABLE",
            "answer": view_of(
                record,
                self.accepted_answer_delivery(
                    record, accepted.model_dump(mode="json"), task_id=published.task_id
                ),
                record_filed=False,
            ),
        }
    submission = dispatcher.submit(command)
    newly_filed = False
    if submission.admitted:
        newly_filed = self._file_answer(record)
    delivery = (
        self.accepted_answer_delivery(
            record,
            accepted.model_dump(mode="json"),
            task_id=submission.task_id,
            newly_filed=newly_filed,
        )
        if submission.admitted and submission.task_id is not None
        else None
    )
    outcome = replace(
        submitted_fn(submission),
        answer=view_of(record, delivery, record_filed=submission.admitted),
    )
    # The unit is the answer's own field, as a published answer says; a next request
    # is one the Host accepts.
    return replace(outcome, evidence_unit_id=unit_id)
