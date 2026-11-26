"""Freezing a protected candidate: one published development result, registered by a receipt.

The candidate is frozen from the development result as published and its freeze registered before
any permit is asked for, so the Gate admits a candidate by opening this receipt. The development
Task is verified, not quoted: it must exist, be a Portfolio development Task, have completed and
have published exactly this result, each a refusal of its own.
"""

from __future__ import annotations

from datetime import UTC, datetime

from alphalattice.investment.portfolio_strategy_lab.application.finalization import (
    CompletedDevelopmentRun,
    CompletedDevelopmentTaskRegistry,
    FrozenPortfolioCandidate,
    PortfolioCandidateFreezeReceipt,
    SealedPreProtectedState,
)
from alphalattice.investment.portfolio_strategy_lab.application.task import (
    PORTFOLIO_PUBLIC_TASK_KIND,
)
from alphalattice.investment.portfolio_strategy_lab.publication.finalization_ledger import (
    PortfolioFinalizationStore,
)
from alphalattice.investment.portfolio_strategy_lab.publication.portfolio_ledger import (
    PortfolioLedgerStore,
)


class PortfolioFinalizationCompositionError(ValueError):
    """Stable refusal for a Host finalization composition failure."""


def freeze_candidate(
    ledger: PortfolioLedgerStore,
    store: PortfolioFinalizationStore,
    *,
    workspace_id: str,
    result_hash: str,
    development_task_id: str,
    tasks: CompletedDevelopmentTaskRegistry,
    frozen_at: datetime | None = None,
) -> FrozenPortfolioCandidate:
    """Freeze one already-published development result and register the freeze.

    Reads the development artifacts, computes nothing, and *publishes* the
    candidate together with a freeze receipt. Registration is the point: the Gate
    admits a candidate by opening this receipt, so freezing has to be a separate,
    earlier act than asking for a permit. A finalization that registered its own
    candidate on the way past would be manufacturing the prerequisite it is about
    to be checked against.

    The task is verified, not quoted. `development_task_id` used to be a free
    string that travelled into the candidate and the receipt unexamined -- so a
    candidate could name a task that never ran, never finished, or published a
    different result entirely, and nothing downstream could tell. Now it is
    resolved through a read-only Task Control port and checked against the
    artifacts being frozen.
    """
    # What is being frozen first, then who produced it. The artifact question is
    # answerable on its own, and answering it first means a caller freezing a
    # predecessor path is told that rather than being sent to look at a task.
    result = ledger.load_result(result_hash)
    program = ledger.load_program(result.program_hash)
    if not program.replayable:
        # The fourth reuse route. A protected finalization stands on a
        # development path harder than any of the others -- it continues it --
        # so a Program that cannot name its numerical inputs cannot be frozen as
        # a candidate at all.
        raise PortfolioFinalizationCompositionError(
            "product_host.candidate_program_assembly_absent"
        )
    run = _require_completed_development_run(
        tasks, task_id=development_task_id, result_hash=result_hash
    )
    if run.program_hash != program.program_hash or run.spec_hash != result.spec_hash:
        raise PortfolioFinalizationCompositionError(
            "product_host.development_task_published_another_program"
        )
    execution = ledger.load_execution(result.execution_ledger_hash)
    candidate = FrozenPortfolioCandidate.create(
        workspace_id=workspace_id,
        spec_hash=result.spec_hash,
        holdings_spec_hash=program.holdings_spec_hash,
        program_hash=program.program_hash,
        numerical_input_assembly_hash=program.numerical_input_assembly_hash,
        development_task_id=development_task_id,
        development_run_hash=run.projection_hash,
        development_result_hash=result.result_hash,
        execution_ledger_hash=execution.ledger_hash,
        economic_ledger_hash=result.economic_ledger_hash,
        report_hash=result.report_hash,
        control_receipt_hash=ledger.load_report(result.report_hash).control_receipt.receipt_hash,
        pre_protected_state=SealedPreProtectedState.of(execution),
        frozen_at=frozen_at or datetime.now(UTC),
    )
    store.publish_freeze(candidate=candidate, receipt=PortfolioCandidateFreezeReceipt.of(candidate))
    return candidate


def _require_completed_development_run(
    tasks: CompletedDevelopmentTaskRegistry, *, task_id: str, result_hash: str
) -> CompletedDevelopmentRun:
    """Three separate questions, three separate refusals.

    Existence, completion and authorship are genuinely different failures and a
    caller that gets one of them back knows which mistake it made. Collapsing
    them into "invalid task" would hide the interesting one: a task that ran
    fine and published something else.
    """
    run = tasks.open_completed_run(task_id)
    if run is None:
        raise PortfolioFinalizationCompositionError("product_host.development_task_not_found")
    if run.task_kind != PORTFOLIO_PUBLIC_TASK_KIND:
        raise PortfolioFinalizationCompositionError("product_host.development_task_wrong_kind")
    if not run.completed:
        raise PortfolioFinalizationCompositionError(
            "product_host.development_task_not_completed:" + run.lifecycle
        )
    if result_hash not in run.published_result_hashes:
        raise PortfolioFinalizationCompositionError(
            "product_host.development_task_did_not_publish_this_result"
        )
    return run
