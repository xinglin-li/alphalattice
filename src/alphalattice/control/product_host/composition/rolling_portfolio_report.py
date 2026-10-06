"""Join a verified installed REPORT to its immutable daily outcome heads."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING, Any, cast

from alphalattice.interface.local_application.portfolio_research import (
    LocalPortfolioResearchService,
)
from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (
    PortfolioDecisionCheckpoint,
    PortfolioUpdatePublication,
)
from alphalattice.investment.portfolio_strategy_lab.publication.rolling_history import (
    RollingHistoryError,
    RollingReportBaseline,
    RollingReportHead,
    RollingReportObservation,
    append_rolling_report_head,
    project_rolling_report,
)

if TYPE_CHECKING:
    from alphalattice.control.product_host.composition.portfolio_application import (
        PortfolioResearchApplication,
    )


def _baseline(
    application: PortfolioResearchApplication,
    checkpoint: PortfolioDecisionCheckpoint,
    publications: tuple[PortfolioUpdatePublication, ...],
) -> tuple[RollingReportBaseline, list[dict[str, Any]]]:
    """Reopen the exact source book and the single declared sealed cost lane."""
    manifest = application.pipeline.find_for_task(checkpoint.book_task_id)
    if manifest is None:
        raise RollingHistoryError("portfolio_rolling.source_book_missing")
    service = LocalPortfolioResearchService(application=application)
    result = service.open_result(manifest.result_hash)
    report = service.report_of(result)
    program = application.ledger.load_program(result.program_hash)
    if (
        program.strategy_package_hash != checkpoint.package.package_hash
        or report.program_hash != result.program_hash
        or checkpoint.initial_book.schedule.formation_session != report.window_guard.selected_end
    ):
        raise RollingHistoryError("portfolio_rolling.source_book_binding_invalid")
    execution = application.ledger.load_execution(result.execution_ledger_hash)
    economics = application.ledger.load_economics(report.economic_ledger_hash)
    cost = Decimal(economics.cost_bps_per_side)
    if cost not in (Decimal(5), Decimal(10)):
        raise RollingHistoryError("portfolio_rolling.cost_lane_not_recorded")
    rows = cast(
        list[dict[str, Any]],
        service.path_readback_of(report, execution=execution, economics=economics)["series"],
    )
    seam = checkpoint.initial_book.schedule.holding_end_session
    baseline = RollingReportBaseline(
        report_hash=report.report_hash,
        result_hash=result.result_hash,
        program_hash=program.program_hash,
        checkpoint_hash=checkpoint.history_hash,
        strategy_package_id=checkpoint.package.strategy_id,
        strategy_package_hash=checkpoint.package.package_hash,
        cost_bps_per_side=cast(Any, int(cost)),
        formation_start=report.window_guard.selected_start,
        formation_end=report.window_guard.selected_end,
        baseline_observed_through=seam,
        base_last_holding_end_session=seam,
        first_parent_publication_hash=None if not publications else publications[0].parent_hash,
        baseline_claim="DEVELOPMENT_EVIDENCE_WITH_POST_OBSERVED_QA_EXTENSION",
        observations=tuple(
            RollingReportObservation(
                formation_session=date.fromisoformat(row["session"]),
                net_simple_return=row["net_simple_return"],
                benchmark_simple_return=row.get("benchmark_simple_return"),
            )
            for row in rows
        ),
    )
    return baseline, rows


def _heads(
    baseline: RollingReportBaseline,
    checkpoint: PortfolioDecisionCheckpoint,
    publications: tuple[PortfolioUpdatePublication, ...],
) -> tuple[RollingReportHead, ...]:
    previous = None
    result = []
    for publication in publications:
        previous = append_rolling_report_head(
            baseline=baseline,
            previous_head=previous,
            publication=publication,
            checkpoint=checkpoint,
        )
        result.append(previous)
    return tuple(result)


def publish_rolling_report_heads(
    *,
    application: PortfolioResearchApplication,
    checkpoint: PortfolioDecisionCheckpoint,
    publications: tuple[PortfolioUpdatePublication, ...],
) -> tuple[str, ...]:
    """Publish only small missing heads after their decision publications commit.

    Idempotent recovery also admits legacy publications. It neither repeats a
    science run nor writes another copy of the report or its return history.
    """
    baseline, _ = _baseline(application, checkpoint, publications)
    hashes = []
    for head in _heads(baseline, checkpoint, publications):
        if not application.ledger.has_rolling_report_head(head.head_hash):
            application.ledger.publish_rolling_report_head(head)
        else:
            application.ledger.load_rolling_report_head(head.head_hash)
        hashes.append(head.head_hash)
    return tuple(hashes)


def read_rolling_report(
    *,
    application: PortfolioResearchApplication,
    checkpoint: PortfolioDecisionCheckpoint,
    publications: tuple[PortfolioUpdatePublication, ...],
) -> dict[str, object]:
    """Read an exact verified prefix; never persist anything during a page read."""
    baseline, _ = _baseline(application, checkpoint, publications)
    heads = _heads(baseline, checkpoint, publications)
    persisted = True
    for head in heads:
        if application.ledger.has_rolling_report_head(head.head_hash):
            if application.ledger.load_rolling_report_head(head.head_hash) != head:
                raise RollingHistoryError("portfolio_rolling.head_chain_invalid")
        else:
            persisted = False  # Legacy sealed publications can still be read without mutation.
    body = project_rolling_report(
        baseline=baseline, heads=heads, publications=publications, checkpoint=checkpoint
    )
    body["head_storage"] = "SEALED" if persisted else "LEGACY_DERIVED_NOT_PERSISTED"
    return body


__all__ = ["publish_rolling_report_heads", "read_rolling_report"]
