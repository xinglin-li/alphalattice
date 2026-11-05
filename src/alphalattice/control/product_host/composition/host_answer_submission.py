"""One keyed agent answer and its bounded Analyst admission."""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from time import monotonic
from typing import Any, cast

from pydantic import ValidationError

from alphalattice.control.product_host.composition import committee
from alphalattice.control.product_host.composition.answer_turns import (
    SUBMISSION_DEADLINE,
    SUBMISSION_WAIT_SECONDS,
    AnswerTurnBusy,
    bounded_answer_turn,
)
from alphalattice.control.product_host.composition.committee_answer import legacy_acceptance
from alphalattice.control.product_host.composition.evidence_review_application import (
    ANSWER_CATEGORY,
    ReviewOutcome,
    agent_answer_result,
)
from alphalattice.control.product_host.composition.evidence_review_bundles import (
    ANSWER_FIELDS,
    EvidenceReviewBundles,
    agent_bundle_refusal,
    bundle_answered,
    bundle_book,
    bundle_refused,
    sealed_bounds_exceeded,
)
from alphalattice.control.product_host.composition.plain_refusals import (
    STALE_PACKET,
    stale_packet_refusal,
)
from alphalattice.interface.local_application.activity import OperationSpan, refusal_code
from alphalattice.interface.local_application.cli_contract import worded
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchRequestDocument,
)
from alphalattice.oversight.chief_risk_officer.decision.book_evidence import (
    PortfolioEvidenceReviewError,
)
from alphalattice.protocols.actor_execution.answers import (
    AgentAnswerRecord,
    answer_digest,
    answer_view,
)
from alphalattice.protocols.actor_execution.bundles import answer_read

CURRENT_OPERATION: ContextVar[OperationSpan | None] = ContextVar(
    "host_current_operation", default=None
)


def submit_agent_answer(self: Any, request: Any, caller: Any) -> dict[str, object]:
    """Authenticate a bundle, serialize its retry and return its first filed result."""
    deadline = monotonic() + SUBMISSION_WAIT_SECONDS if request.committee_key is not None else None
    scope = SUBMISSION_DEADLINE.set(deadline)
    bundle = None
    try:
        # The bundle's own submission and answer.
        assert request.bundle_directory is not None
        bundle = EvidenceReviewBundles(self.review).agent_bundle(request.bundle_directory)
        if bundle is None:
            missing: dict[str, object] = agent_bundle_refusal("agent_bundle.not_prepared")
            return missing
        binding = {"role": "bundle-submit", "bundle_reference": bundle.record_hash}
        turn = (
            self.review._answer_turn(binding)
            if deadline is None
            else bounded_answer_turn(self.review, binding, deadline)
        )
        with turn:
            # Files it names as read whole are its word, kept as provenance (OP11).
            answer, read = answer_read(request.agent_answer or {})
            agent_run = self._agent_run()
            store, who = self.goals.store, (bundle.role, bundle.record_hash, answer)
            if isinstance(thread := committee.admit(store, who, request.committee_key), dict):
                return thread
            if request.committee_key is not None and isinstance(thread, committee.Floor):
                replay = committee.answer_replay(thread, who)
                if replay is not None:
                    reference, response = replay
                    record = self.review.artifacts.load(
                        ANSWER_CATEGORY, reference, AgentAnswerRecord
                    )
                    if (
                        record.answer_digest != answer_digest(answer)
                        or response.get("status") != record.verdict.value
                    ):
                        raise ValueError("agent_bundle.answer_replay_inconsistent")
                    retained: dict[str, object] = response
                    return retained
                legacy = legacy_acceptance(thread, who)
                if legacy is not None:
                    reference, task_id = legacy
                    record = self.review.artifacts.load(
                        ANSWER_CATEGORY, reference, AgentAnswerRecord
                    )
                    if record.answer_digest != answer_digest(
                        answer
                    ) or record.verdict.value not in {
                        "ACCEPTED",
                        "DONE",
                    }:
                        raise ValueError("agent_bundle.answer_replay_inconsistent")
                    # Old ACCEPTANCE rows never stored the admission-time state.
                    # Keep their receipt and ask the Task owner for its state now.
                    replayed = agent_answer_result(
                        bundle.role,
                        {
                            "disposition": "ADMITTED",
                            "task_id": task_id,
                            "answer": answer_view(record),
                        },
                        bundle_book(bundle.submission),
                    )
                    replayed["bundle_reference"] = bundle.record_hash
                    replayed["task_lifecycle"] = "UNKNOWN"
                    replayed["message"] = str(replayed["message"]) + (
                        " The original Task state was not recorded; "
                        "read the Task for its current state."
                    )
                    committee.answered(
                        store,
                        thread,
                        who,
                        replayed,
                        self.dispatcher.clock(),
                        self._file_floor,
                    )
                    retained = replayed
                    return retained
            return cast(
                dict[str, object],
                _perform_answer(
                    self, request, caller, bundle, answer, read, agent_run, store, who, thread
                ),
            )
    except AnswerTurnBusy:
        return {
            "status": "REFUSED",
            "answer_status": "REFUSED",
            **({} if bundle is None else {"agent_role": bundle.role}),
            "failure_code": "agent_bundle.admission_busy",
            "message": (
                "The Host is busy admitting another workspace change; resend the same answer."
            ),
            "next_action": "RETRY_THE_SAME_ANSWER",
        }
    finally:
        SUBMISSION_DEADLINE.reset(scope)


def _perform_answer(
    self: Any,
    request: Any,
    caller: Any,
    bundle: Any,
    answer: Any,
    read: Any,
    agent_run: Any,
    store: Any,
    who: Any,
    thread: Any,
) -> Any:
    """File an admissible answer and its own thread outcome."""
    if bundle.role not in ANSWER_FIELDS:
        try:
            with self.workspace_session.mutation_gate.hold():
                generic_body = self.review.submit_specialist_answer(
                    bundle=bundle,
                    answer=answer,
                    read_files=read,
                    agent_run=agent_run,
                )
                accepted_receipt = self._observe(
                    lambda: self.capture_accepted_answer_context(bundle, generic_body)
                )
        except PortfolioEvidenceReviewError as error:
            if str(error) not in {
                "agent_bundle.specialist_task_changed",
                "agent_bundle.answer_settled",
            }:
                raise
            return agent_bundle_refusal(str(error), role=bundle.role)
        body: dict[str, object] | None = self.observe_accepted_answer(
            generic_body, bundle, accepted_receipt=accepted_receipt
        )
    else:
        bound = PortfolioResearchRequestDocument.model_validate(
            {**bundle.submission, ANSWER_FIELDS[bundle.role]: answer}
        ).to_operation_request()
        try:
            body = self._review_operation(
                bound,
                caller=caller,
                read_files=read,
                agent_run=agent_run,
                accepted_bundle=bundle,
            )
        except ValidationError as error:
            # A record past its sealed bounds: named by place, count and bound.
            over = sealed_bounds_exceeded(error)
            if not over:
                raise
            return agent_bundle_refusal(
                "agent_bundle.seal_bound_exceeded", role=bundle.role, bounds=over
            )
        except PortfolioEvidenceReviewError as error:
            return bundle_refused(self, error, bundle, thread, request, caller)
        except ValueError as error:
            # A packet whose authority an install replaced is refused by what moved.
            if not str(error).startswith(STALE_PACKET):
                raise
            return stale_packet_refusal(
                str(error),
                role=bundle.role,
                book=bundle_book(bundle.submission),
                unit_id=bundle.submission.get("evidence_unit_id"),
            )
    assert body is not None
    answered = bundle_answered(self, bundle, body, thread, request, caller)
    if (
        isinstance(thread, committee.Floor)
        and request.committee_key is not None
        and body.get("failure_code") == "agent_bundle.admission_busy"
        and isinstance(body.get("waited_seconds"), (int, float))
    ):
        # Only the keyed, admitted review member may report this Goal's terminal stall.
        answered.update(
            status="REFUSED_REVIEW_STALLED",
            next_action="REPORT_REVIEW_STALLED",
            message_id="agent_bundle.review_stalled_message",
            message=worded("agent_bundle.review_stalled_message"),
            blocker={"kind": "WORKSPACE_MUTATION_GATE", "holder": "UNMEASURED"},
            waited_seconds=body["waited_seconds"],
            next_requests={
                "activity": {"operation": "ACTIVITY_RECENT"},
                "tasks": {"operation": "TASKS"},
            },
        )
    if answered["status"] in {"ACCEPTED", "DONE"}:
        answered["bundle_reference"] = bundle.record_hash
    if agent_run is not None and answered["status"] not in {"ACCEPTED", "DONE"}:
        answered["recorded_agent"] = agent_run.model_dump(mode="json")
    if "conversation" in body:
        answered["conversation"] = body["conversation"]
    answer_payload = body.get("answer")
    correction_reference = (
        answer_payload.get("answer_reference")
        if answered["status"] == "CORRECT" and isinstance(answer_payload, dict)
        else None
    )
    committee.answered(
        store,
        thread,
        who,
        answered,
        self.dispatcher.clock(),
        self._file_floor,
        correction_reference=correction_reference,
    )
    return answered


def submit_evidence_analysis(
    self: Any,
    request: Any,
    selector: Any,
    caller: Any,
    read_files: Any,
    agent_run: Any,
    accepted_bundle: Any,
    body_of: Any,
) -> dict[str, object]:
    """Bound only filing and Task admission, and record the held interval."""
    assert request.task_id is not None
    assert request.analysis_context_hash is not None
    assert request.analysis_answer is not None
    accepted_receipt = None
    deadline = SUBMISSION_DEADLINE.get() or (monotonic() + SUBMISSION_WAIT_SECONDS)

    def admit_analysis(
        action: Callable[[], ReviewOutcome | dict[str, object]],
    ) -> dict[str, object]:
        nonlocal accepted_receipt
        # The standard local client waits 120 seconds. Reserve 30 seconds for
        # screening, observation and the return; a no-effect busy refusal is
        # retryable like queue-full, and never becomes a filed outcome.
        started_wait = monotonic()
        remaining = deadline - started_wait
        if remaining <= 0:
            raise AnswerTurnBusy  # no gate attempt: its holder and wait are not proved
        entered = False
        if remaining > 0:
            try:
                with self.workspace_session.mutation_gate.try_hold(timeout_seconds=remaining):
                    entered = True
                    hold_started = monotonic()
                    try:
                        analysis = body_of(action())
                        accepted_receipt = self._observe(
                            lambda: self.capture_accepted_answer_context(accepted_bundle, analysis)
                        )
                        return cast(dict[str, object], analysis)
                    finally:
                        current = CURRENT_OPERATION.get()
                        if current is not None:
                            current.subject["analyst_admission_gate_hold_us"] = int(
                                (monotonic() - hold_started) * 1_000_000
                            )
            except TimeoutError:
                if entered:
                    raise
        waited = max(0.0, monotonic() - started_wait)
        current = CURRENT_OPERATION.get()
        if current is not None:
            current.subject.update(
                review_stall_blocker="WORKSPACE_MUTATION_GATE",
                review_stall_holder="UNMEASURED",
                review_stall_wait_us=int(waited * 1_000_000),
            )
        return {
            "status": "REFUSED",
            "failure_code": "agent_bundle.admission_busy",
            "waited_seconds": waited,
            "detail": (
                "The Host is busy admitting another workspace change; resend the same answer."
            ),
        }

    analysis_body = body_of(
        self.review.submit_analysis(
            dispatcher=self.dispatcher,
            selector=selector,
            task_id=request.task_id,
            context_hash=request.analysis_context_hash,
            answer=request.analysis_answer,
            caller=caller,
            unit_id=request.evidence_unit_id,
            read_files=read_files,
            agent_run=agent_run,
            admit=admit_analysis if accepted_bundle is not None else None,
        )
    )
    return cast(
        dict[str, object],
        self.observe_accepted_answer(
            analysis_body, accepted_bundle, accepted_receipt=accepted_receipt
        ),
    )


def finish_observation(
    self: Any,
    observer: Any,
    span: Any,
    request: Any,
    body: Any,
    caller: Any,
    scope: Any,
) -> Any:
    """File the return and durable refusal count before leaving this operation's context."""
    if observer is not None and span is not None:
        self._observe(lambda: observer.returned(span, body))
    refusal = refusal_code(body)
    if observer is not None and refusal is not None:
        self._observe(lambda: observer.refused(request.operation, refusal, caller=caller))
    CURRENT_OPERATION.reset(scope)
    return body
