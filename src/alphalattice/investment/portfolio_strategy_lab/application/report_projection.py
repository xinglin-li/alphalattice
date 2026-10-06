"""The declared-path report, built once and consumed by both writers of it.

The executor builds this report when it publishes a path. A strong replay has to
rebuild the *same* report from the sealed children and land on the same hash --
and if the two did that through two pieces of code, a match would only say the
two agreed with each other, and a drift between them would show up as a replay
failure on a path that was fine.

So there is one owner. The executor calls it with the numbers it just produced;
the replay calls it with the numbers it just rederived. Neither passes a report
value in, which is the point: every field below is computed here from the path
and the spec, so nothing the sealed report says can be used to prove the sealed
report right.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import numpy.typing as npt

from alphalattice.investment.portfolio_strategy_lab.application.advancement import (
    PortfolioLedgerCoverage,
    ordered_listing_axis_hash,
)
from alphalattice.investment.portfolio_strategy_lab.application.contracts import (
    HELD_WEIGHT_EPSILON,
    PortfolioBenchmarkComparison,
    PortfolioBookPosition,
    PortfolioControlReceipt,
    PortfolioDeclaredPathReport,
    PortfolioEconomicLedger,
    PortfolioExecutionLedger,
    PortfolioExecutionProgram,
    PortfolioResearchSpec,
    PortfolioSupportCoverage,
    PortfolioWindowEndBook,
    build_schedule_guard,
    build_study_window_guard,
)
from alphalattice.investment.portfolio_strategy_lab.reporting.units import (
    project_report_unit_rows,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

DECLARED_PATH_LIMITATIONS: tuple[str, ...] = (
    "DEVELOPMENT_REPLAY; no Alpha fit ran.",
    "Current-universe research surface; membership is not point-in-time.",
    "Risk GMV threshold was not met and this path is not validated for QP use.",
    "Capacity is NOT_MODELED; ADV is a concentration descriptor only.",
    "Sector Forecast is not consumed.",
    "The study window is DESCRIPTIVE_SUBWINDOW and carries no claim authority.",
)
"""What this report may not be read as claiming.

Part of the report identity, so it lives beside the projection that seals it
rather than in the caller. A limitation a rebuild forgot would be a report that
hashes differently while saying something stronger.
"""


def project_declared_path_report(
    *,
    spec: PortfolioResearchSpec,
    program: PortfolioExecutionProgram,
    ledger: PortfolioExecutionLedger,
    economics: PortfolioEconomicLedger,
    comparison: PortfolioBenchmarkComparison,
    coverage: PortfolioSupportCoverage,
    net_simple_returns: npt.NDArray[np.float64],
    one_way_turnovers: npt.NDArray[np.float64],
    executed_weights: npt.NDArray[np.float64],
    opening_reference_weights: npt.NDArray[np.float64],
) -> PortfolioDeclaredPathReport:
    """Select the window, then describe the book *at that window's own end*.

    Reading the headline facts off the path end instead would produce a report
    that says "window" and describes something else -- which is why the window
    rows are computed first and every fact below is taken through them.

    ``opening_reference_weights`` is the book the path opened on, read from the
    sealed boundary lane by whoever calls this. It is a parameter rather than an
    inference because the alternative was inferring it from a row index, and a
    first row is not evidence of an empty book: a continuation opens on a frozen
    one, and reporting that as flat turns every carried name into a new position.
    """
    sessions = tuple(ledger.formation_sessions)
    window_start = spec.study_start or sessions[0]
    window_end = spec.study_end or sessions[-1]
    window_rows = tuple(
        index for index, value in enumerate(sessions) if window_start <= value <= window_end
    )
    if not window_rows:
        raise ValueError("portfolio_application.study_window_selects_no_formation")
    window_sessions = tuple(sessions[index] for index in window_rows)
    window_index: npt.NDArray[np.int64] = np.asarray(window_rows, dtype=np.int64)
    window_guard = build_study_window_guard(
        requested_start=window_start,
        requested_end=window_end,
        selected_start=window_sessions[0],
        selected_end=window_sessions[-1],
        selected_session_count=len(window_sessions),
        coverage=coverage,
        ledger_sessions=sessions,
    )
    unit_rows = project_report_unit_rows(
        report_unit=spec.report_unit,
        sessions=window_sessions,
        net_simple_returns=np.asarray(net_simple_returns, dtype=np.float64)[window_index],
        anchor_simple_returns=np.asarray(ledger.anchor_simple_returns, dtype=np.float64)[
            window_index
        ],
        benchmark_beta=economics.benchmark_beta,
    )
    weights = np.asarray(executed_weights, dtype=np.float64)
    window_end_weights = np.ascontiguousarray(weights[window_index[-1]], dtype=np.float64)
    held = window_end_weights > HELD_WEIGHT_EPSILON
    window_end_book = _project_window_end_book(
        ledger=ledger,
        weights=weights,
        formation_row=int(window_index[-1]),
        listing_axis_hash=ordered_listing_axis_hash(tuple(ledger.ordered_listing_ids)),
        opening_reference=np.asarray(opening_reference_weights, dtype=np.float64),
    )
    hhi = float(np.square(window_end_weights).sum())
    window_cap_counts = tuple(ledger.aggregate_cap_binding_counts[index] for index in window_rows)
    identity: dict[str, object] = {
        "kind": "PortfolioDeclaredPathReport",
        "program_hash": program.program_hash,
        "execution_ledger_hash": ledger.ledger_hash,
        "economic_ledger_hash": economics.economic_ledger_hash,
        "benchmark_comparison_hash": comparison.comparison_hash,
        "report_unit": spec.report_unit,
        "ledger_coverage": PortfolioLedgerCoverage.of(
            program_hash=program.program_hash,
            ledger_hash=ledger.ledger_hash,
            formation_sessions=ledger.formation_sessions,
            ordered_listing_ids=ledger.ordered_listing_ids,
            source_coverage_hash=coverage.coverage_hash,
        ),
        "control_receipt": PortfolioControlReceipt.of(spec),
        "window_end_book": window_end_book,
        "schedule_guard": build_schedule_guard(tranches=spec.tranches),
        "window_guard": window_guard,
        "risk_facts": tuple(ledger.risk_facts[index] for index in window_rows),
        "window_end_distinct_names": int(held.sum()),
        "window_end_effective_n": 0.0 if hhi <= 0.0 else 1.0 / hhi,
        "window_cumulative_net_wealth": float(
            np.prod(1.0 + np.asarray(net_simple_returns, dtype=np.float64)[window_index])
        ),
        "mean_one_way_turnover": float(
            np.asarray(one_way_turnovers, dtype=np.float64)[window_index].mean()
        ),
        "aggregate_cap_binding_sessions": sum(value > 0 for value in window_cap_counts),
        "aggregate_cap_binding_names_total": sum(window_cap_counts),
        "capacity_disposition": "NOT_MODELED",
        "window_end_median_holding_adv20_dollar_volume": (
            ledger.median_holding_adv20_by_formation[window_rows[-1]]
        ),
        "window_unit_rows": unit_rows,
        "limitations": DECLARED_PATH_LIMITATIONS,
    }
    payload = PortfolioDeclaredPathReport.model_construct(
        **identity, report_hash="0" * 64
    ).model_dump(mode="json", exclude={"report_hash"})
    return PortfolioDeclaredPathReport(**payload, report_hash=canonical_hash(payload))


def _project_window_end_book(
    *,
    ledger: PortfolioExecutionLedger,
    weights: npt.NDArray[np.float64],
    formation_row: int,
    listing_axis_hash: str,
    opening_reference: npt.NDArray[np.float64],
) -> PortfolioWindowEndBook:
    """The book at one formation, and its change from the book before it.

    The boundary is the *immediately preceding formation of the path*, not the
    previous row inside the selected window: what produced this book is a fact
    about the path, and a window that starts later does not change it.

    At the path's first formation there is no earlier row, and the answer comes
    from the sealed opening boundary rather than from the row number. Two
    different things live there. A development path opens empty, so every
    position is genuinely an open. A continuation opens on a frozen book, whose
    formation is real, earlier, and not this path's -- and reporting *that* as an
    empty opening was the defect: every carried name would read as newly opened,
    every reduction as an open, and an exit could not appear at all.
    """

    listings = tuple(ledger.ordered_listing_ids)
    ending = np.ascontiguousarray(weights[formation_row], dtype=np.float64)
    if formation_row > 0:
        preceding = np.ascontiguousarray(weights[formation_row - 1], dtype=np.float64)
        boundary_kind: str = "PRECEDING_FORMATION"
        preceding_session: date | None = ledger.formation_sessions[formation_row - 1]
    else:
        preceding = np.ascontiguousarray(opening_reference, dtype=np.float64)
        if preceding.shape != ending.shape:
            raise ValueError("portfolio_application.book_opening_axis_invalid")
        sealed = ledger.initial_boundary
        if sealed is not None and sealed.formation_session < ledger.formation_sessions[0]:
            # A boundary that sits after an *earlier* formation is a carried
            # book, whatever it holds. Its session is a real predecessor and the
            # report names it.
            boundary_kind = "SEALED_CONTINUATION_BOUNDARY"
            preceding_session = sealed.formation_session
        else:
            # The path opens at its own first formation. The only book that can
            # honestly precede that is an empty one; anything else means the
            # boundary and the lane disagree about where this path began, and
            # inventing a story for it would be the misreport again.
            if bool((preceding > HELD_WEIGHT_EPSILON).any()):
                raise ValueError("portfolio_application.book_opening_not_flat")
            boundary_kind = "FLAT_PATH_OPENING"
            preceding_session = None

    positions: list[PortfolioBookPosition] = []
    for index, listing_id in enumerate(listings):
        weight = float(ending[index])
        before = float(preceding[index])
        held_now = weight > HELD_WEIGHT_EPSILON
        held_before = before > HELD_WEIGHT_EPSILON
        if not held_now and not held_before:
            continue
        change = weight - before
        disposition = (
            "OPENED"
            if not held_before
            else "EXITED"
            if not held_now
            else "INCREASED"
            if change > HELD_WEIGHT_EPSILON
            else "REDUCED"
            if change < -HELD_WEIGHT_EPSILON
            else "HELD"
        )
        positions.append(
            PortfolioBookPosition(
                listing_id=listing_id,
                weight=weight,
                preceding_weight=before,
                weight_change=change,
                disposition=disposition,
            )
        )
    # Largest ending weight first, then by listing id: a total order over the
    # rows, so two runs of the same path emit the same table rather than the
    # same set.
    ordered = tuple(sorted(positions, key=lambda value: (-value.weight, value.listing_id)))
    return PortfolioWindowEndBook.create(
        formation_session=ledger.formation_sessions[formation_row],
        change_boundary=boundary_kind,
        preceding_formation_session=preceding_session,
        positions=ordered,
        held_count=sum(value.weight > HELD_WEIGHT_EPSILON for value in ordered),
        opened_count=sum(value.disposition == "OPENED" for value in ordered),
        exited_count=sum(value.disposition == "EXITED" for value in ordered),
        # Summed over the rows the report actually shows, in the order it shows
        # them, so the figure above the table is the table's own total rather
        # than a second reduction that could differ from it.
        absolute_weight_change_total=sum(abs(value.weight_change) for value in ordered),
        listing_axis_hash=listing_axis_hash,
    )


@dataclass(frozen=True)
class PortfolioDatedPosition:
    """Exact formation-indexed read values; not a new result or an execution."""

    book: PortfolioWindowEndBook
    ordered_listing_ids: tuple[str, ...]
    weights: tuple[float, ...]
    targets: tuple[float, ...] | None
    decision_mode: str
    one_way_turnover: float
    gross_simple_return: float
    net_simple_return: float
    benchmark_simple_return: float
    cost_fraction: float
    cash: float
    hhi: float

    def body(self) -> dict[str, object]:
        """Project a formation-dated position and its exact retained economic/read-model lanes.

        Returns:
            Mapping of ordered positions/targets, book, support, turnover, return/cost, cash and
            concentration; the date basis is formation, not execution.
        """
        return {
            "session": self.book.formation_session.isoformat(),
            "date_basis": "FORMATION_SESSION_NOT_EXECUTION_DATE",
            "weights": list(self.weights),
            "targets": None if self.targets is None else list(self.targets),
            "ordered_listing_ids": list(self.ordered_listing_ids),
            "book": self.book.model_dump(mode="json"),
            "holding_count": self.book.held_count,
            "decision_mode": self.decision_mode,
            "one_way_turnover": self.one_way_turnover,
            "gross_simple_return": self.gross_simple_return,
            "net_simple_return": self.net_simple_return,
            "benchmark_simple_return": self.benchmark_simple_return,
            "cost_fraction": self.cost_fraction,
            "cash": self.cash,
            "hhi": self.hhi,
        }


def project_dated_position(
    *,
    ledger: PortfolioExecutionLedger,
    economics: PortfolioEconomicLedger,
    formation_row: int,
    executed_weights: bytes,
    target_weights: bytes | None,
    opening_reference: npt.NDArray[np.float64],
) -> PortfolioDatedPosition:
    """Reuse the report's book/change owner; never round or rerun the path."""
    shape = ledger.executed_weights_shape
    expected_bytes = int(np.prod(shape)) * 8
    if (
        economics.execution_ledger_hash != ledger.ledger_hash
        or len(economics.net_simple_returns) != shape[0]
        or not 0 <= formation_row < shape[0]
        or len(executed_weights) != expected_bytes
        or (target_weights is not None and len(target_weights) != expected_bytes)
    ):
        raise ValueError("portfolio_application.position_axis_mismatch")
    weights: npt.NDArray[np.float64] = np.frombuffer(executed_weights, dtype="<f8").reshape(shape)
    targets: npt.NDArray[np.float64] | None = (
        None
        if target_weights is None
        else np.frombuffer(target_weights, dtype="<f8").reshape(shape)
    )
    if not np.isfinite(weights).all() or (targets is not None and not np.isfinite(targets).all()):
        raise ValueError("portfolio_application.position_values_invalid")
    book = _project_window_end_book(
        ledger=ledger,
        weights=weights,
        formation_row=formation_row,
        listing_axis_hash=ordered_listing_axis_hash(ledger.ordered_listing_ids),
        opening_reference=opening_reference,
    )
    selected = weights[formation_row]
    gross = ledger.gross_simple_returns[formation_row]
    net = economics.net_simple_returns[formation_row]
    return PortfolioDatedPosition(
        book=book,
        ordered_listing_ids=ledger.ordered_listing_ids,
        weights=tuple(map(float, selected)),
        targets=None if targets is None else tuple(map(float, targets[formation_row])),
        decision_mode=ledger.decision_modes[formation_row],
        one_way_turnover=ledger.one_way_turnovers[formation_row],
        gross_simple_return=gross,
        net_simple_return=net,
        benchmark_simple_return=ledger.anchor_simple_returns[formation_row],
        cost_fraction=gross - net,
        cash=max(0.0, 1.0 - float(selected.sum())),
        hhi=float(np.square(selected).sum()),
    )


__all__ = [
    "DECLARED_PATH_LIMITATIONS",
    "PortfolioDatedPosition",
    "project_dated_position",
    "project_declared_path_report",
]
