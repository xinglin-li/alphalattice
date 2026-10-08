"""Display values from a verified authored Portfolio; never execute or select research."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal, TypedDict

if TYPE_CHECKING:
    from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (
        PortfolioDecisionCheckpoint,
        PortfolioUpdatePublication,
    )


class HoldingDisplay(TypedDict):
    """Owner weights remain fractions; legacy target/executed values are percentages."""

    ticker: str
    name: None
    listing_id: str
    sector: Any
    mapped: None
    target: float | None
    executed: float | None
    preceding_weight: float | None
    current_weight: float | None
    weight_change: float | None
    disposition: str | None


class ForwardHoldingsDisplay(TypedDict):
    """One exact issued research publication, separate from the historical selection."""

    status: Literal["RECORDED_FORWARD_HOLDINGS"]
    available: Literal[True]
    basis: Literal["CONDITIONAL_ESTIMATE", "OBSERVED_RESEARCH_ENTRY"]
    comparison_basis: Literal[
        "FORMATION_CLOSE_ESTIMATE", "OBSERVED_PRETRADE_WEIGHTS", "PRECEDING_NOT_RECORDED"
    ]
    formation_session: str
    entry_session: str
    holding_end_session: str
    observed_through: str
    publication_hash: str
    position_hash: str
    checkpoint_hash: str
    source_book_task_id: str
    strategy_package_id: str
    reading_task_id: str
    publication_is_previous: bool
    cash: float | None
    holding_count: int
    sectors: dict[str, Any]
    holdings: list[HoldingDisplay]


def forward_holdings_view(
    *,
    checkpoint: PortfolioDecisionCheckpoint,
    publication: PortfolioUpdatePublication,
    history: tuple[PortfolioUpdatePublication, ...],
    source_book_task_id: str,
    strategy_package_id: str,
    reading_task_id: str,
    publication_is_previous: bool = False,
    sectors: dict[str, Any] | None = None,
) -> ForwardHoldingsDisplay:
    """Display the owner's exact position facts without inferring venue execution.

    Args:
        checkpoint: Verified listing axis and source book of this publication.
        publication: Exact verified update selected by the composition.
        history: Its verified retained prefix, for recorded pretrade weights.
        source_book_task_id: Historical book whose page requested this reading.
        strategy_package_id: Historical book's installed package.
        reading_task_id: Task whose readback supplied the publication.
        publication_is_previous: Whether that task reads its previously issued publication.
        sectors: Verified source-book classification at the publication's formation.

    Returns:
        Fractional owner position facts with their dates and comparison basis.

    Raises:
        ValueError: The checkpoint belongs to another source book/package or the axes differ.
    """
    from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (
        portfolio_update_positions,
    )

    if (
        str(checkpoint.book_task_id) != source_book_task_id
        or checkpoint.package.strategy_id != strategy_package_id
        or publication.checkpoint_hash != checkpoint.history_hash
    ):
        raise ValueError("workbench.forward_holdings_source_mismatch")
    positions = portfolio_update_positions(publication, history)
    ids, labels = checkpoint.ordered_listing_ids, checkpoint.listing_labels
    changes = positions.changes
    preceding = positions.preceding
    if len({len(ids), len(labels), len(positions.weights)}) != 1 or (
        preceding is not None and len(preceding) != len(ids)
    ):
        raise ValueError("workbench.portfolio_listing_axis_mismatch")
    rows: list[HoldingDisplay] = []
    for index, (listing_id, label, weight) in enumerate(
        zip(ids, labels, positions.weights, strict=True)
    ):
        before = None if preceding is None else preceding[index]
        if weight == 0 and (before is None or before == 0):
            continue
        rows.append(
            {
                "ticker": label,
                "name": None,
                "listing_id": listing_id,
                "sector": (sectors or {}).get("by_listing", {}).get(listing_id),
                "mapped": None,
                "target": None,
                "executed": None,
                "preceding_weight": before,
                "current_weight": weight,
                "weight_change": None if changes is None else changes[index],
                "disposition": None,
            }
        )
    return {
        "status": "RECORDED_FORWARD_HOLDINGS",
        "available": True,
        "basis": positions.basis,
        "comparison_basis": (
            "FORMATION_CLOSE_ESTIMATE"
            if positions.basis == "CONDITIONAL_ESTIMATE"
            else "OBSERVED_PRETRADE_WEIGHTS"
            if preceding is not None
            else "PRECEDING_NOT_RECORDED"
        ),
        "formation_session": positions.schedule.formation_session.isoformat(),
        "entry_session": positions.schedule.entry_session.isoformat(),
        "holding_end_session": positions.schedule.holding_end_session.isoformat(),
        "observed_through": publication.observed_through.isoformat(),
        "publication_hash": publication.content_hash,
        "position_hash": positions.position_hash,
        "checkpoint_hash": checkpoint.history_hash,
        "source_book_task_id": source_book_task_id,
        "strategy_package_id": strategy_package_id,
        "reading_task_id": reading_task_id,
        "publication_is_previous": publication_is_previous,
        # A proposal records close cash, not an estimated cash allocation.
        "cash": publication.book.cash if positions.basis == "OBSERVED_RESEARCH_ENTRY" else None,
        "holding_count": sum(weight != 0 for weight in positions.weights),
        "sectors": sectors or {"status": "UNAVAILABLE", "by_listing": {}},
        "holdings": rows,
    }


def portfolio_view(body: dict[str, Any], sectors: dict[str, Any] | None = None) -> dict[str, Any]:
    """Project a verified Portfolio answer into display-only Workbench values.

    Args:
        body: The authored Portfolio result or installed reading.
        sectors: Optional sector labels for the authored result.

    Returns:
        Values the Workbench may display without selecting or executing work.

    Raises:
        ValueError: If the answer is not a completed authored Portfolio.
    """
    canonical = body.get("reading_kind") == "INSTALLED_RESULT"
    if not canonical and (
        body.get("status") != "EXPERIMENT_PUBLISHED" or not body.get("portfolio_source")
    ):
        raise ValueError("workbench.completed_authored_portfolio_required")
    if canonical:
        context = body["reading_context"]
        sectors = context.get("sectors")
        position, result = body["position"], body["selected_window_metrics"]
        ids = position["ordered_listing_ids"]
        labels = [(context.get("listing_labels") or {}).get(v, v) for v in ids]
        spec, origin = body["spec"], body["originating_task_id"]
        recorded = context.get("source") or {}
        declaration = {k: spec[k] for k in ("top_k", "tranches", "exit_rank", "weight_rule")}
        declaration["cost_bps_per_side"] = spec["cost_bps_per_side"]
        declaration["unavailable_return_policy"] = recorded.get("unavailable_return_policy")
        support = {"start": body["window"]["selected_start"], "end": body["window"]["selected_end"]}
        task_id, receipt_hash = origin, None
        input_hash, input_date = recorded.get("input_binding_hash"), recorded.get("input_end")
        input_id = recorded.get("input_id") or ("Bound historical input" if input_hash else None)
        title = spec["strategy_package_id"]
        count, excluded = recorded.get("source_listing_count"), recorded.get("data_exclusion_count")
        quality = {
            "source_listing_count": count,
            "effective_listing_count": None
            if count is None or excluded is None
            else count - excluded,
            "policy": recorded.get("unavailable_return_policy"),
            "notice": (
                "Recorded source axis after declared quarantine; "
                "daily recipe eligibility may be smaller."
            ),
        }
        source_projection = {**recorded, "task_id": task_id, "result_hash": body["result_hash"]}
        review_selector = (
            {"result_hash": body["result_hash"]}
            if position["session"] == body["window"]["selected_end"]
            else None
        )
        limitations = [
            *body.get("limitations", []),
            *((context.get("strategy") or {}).get("claim_limits") or []),
        ]
    else:
        position, result = body["position"], body["result"]
        ids = body["receipt"]["source"]["ordered_listing_ids"]
        labels = body.get("listing_labels") or ids
        declaration = body["document"]["portfolio"]
        study = body["document"]["experiment"]
        quality = body.get("data_quality", {})
        source = body["portfolio_source"]
        task_id, receipt_hash = body["task_id"], body["receipt"]["receipt_hash"]
        input_id, input_hash = body["research_input_id"], body["input_binding_hash"]
        input_date, support = study["sessions"]["as_of"]["session"], study["sessions"]
        title = None  # an authored book has no owner's label: the workbench names it by its policy
        excluded = (
            len(body["receipt"]["data_exclusions"])
            if "data_exclusions" in body["receipt"]
            else None
        )
        source_projection = {
            "task_id": task_id,
            "receipt_hash": receipt_hash,
            "input_binding_hash": input_hash,
            "alpha_task_id": source["alpha_task_id"],
            "candidate_id": source["candidate_id"],
            "target_recipe_id": source["target_recipe_id"],
            "foundation_admission_hash": source.get("foundation_admission_hash"),
            # The Risk study the weights read, as the source bound it (V340, U41).
            "risk_task_id": source.get("risk_task_id"),
            "risk_surface_hash": source.get("risk_surface_hash"),
            "origin_task_id": body.get("origin_task_id"),
        }
        review_selector, limitations = body.get("review_selector"), body.get("limitations", [])
    # The sector of a holding is the one the book's execution classified it with (the Sector
    # map sealed with its Panel), handed in by the composition; absent means not resolved,
    # never "no sector".
    sector_map = sectors or {"status": "UNAVAILABLE", "reason": None, "by_listing": {}}
    by_listing = sector_map.get("by_listing") or {}
    weights, targets = position["weights"], position["targets"]
    if targets is None:
        targets = [None] * len(ids)
    if len({len(ids), len(labels), len(weights), len(targets)}) != 1:
        raise ValueError("workbench.portfolio_listing_axis_mismatch")

    book = position.get("book") or {}
    preceding = body.get("preceding_position")
    if not canonical and not book and preceding is not None:
        before = preceding["weights"]
        if len(before) != len(ids) or any(
            not isinstance(value, int | float) or not math.isfinite(value)
            for value in (*weights, *before)
        ):
            raise ValueError("workbench.portfolio_listing_axis_mismatch")
        sessions = [row["session"] for row in body["series"]]
        selected_index = next(
            (index for index, session in enumerate(sessions) if session == position["session"]),
            None,
        )
        if (
            selected_index is None
            or selected_index == 0
            or preceding.get("session") != sessions[selected_index - 1]
        ):
            raise ValueError("workbench.portfolio_session_axis_invalid")
        # Authored replay's verified preceding row supplies the same formation-end
        # comparison used by the CRO's authored book reading, never traded turnover.
        book = {
            "formation_session": position["session"],
            "preceding_formation_session": preceding["session"],
            "change_boundary": "PRECEDING_FORMATION",
            "positions": [
                {
                    "listing_id": listing,
                    "weight": ending,
                    "preceding_weight": previous,
                    "weight_change": ending - previous,
                }
                for listing, ending, previous in zip(ids, weights, before, strict=True)
                if ending > 0 or previous > 0
            ],
        }
    book_rows = {row["listing_id"]: row for row in book.get("positions", [])}
    if len(book_rows) != len(book.get("positions", [])) or not set(book_rows) <= set(ids):
        raise ValueError("workbench.portfolio_listing_axis_mismatch")

    def percent(value: Any) -> float | None:
        return (
            float(value) * 100 if isinstance(value, int | float) and math.isfinite(value) else None
        )

    holdings: list[HoldingDisplay] = []
    for listing_id, label, target, weight in zip(ids, labels, targets, weights, strict=True):
        owner_row = book_rows.get(listing_id)
        if owner_row is None and weight == 0 and (target is None or target == 0):
            continue
        holdings.append(
            {
                "ticker": label,
                "name": None,
                "listing_id": listing_id,
                "sector": by_listing.get(listing_id),
                "mapped": None,
                "target": percent(target),
                "executed": percent(weight),
                "preceding_weight": None
                if owner_row is None
                else owner_row.get("preceding_weight"),
                "current_weight": weight if owner_row is None else owner_row.get("weight"),
                "weight_change": None if owner_row is None else owner_row.get("weight_change"),
                "disposition": None if owner_row is None else owner_row.get("disposition"),
            }
        )

    # A chart coordinate system, not a new performance metric. Values are
    # indexed from 100 immediately before the first published return interval.
    # A missing return breaks the index; it is never treated as a zero return.
    indexed: list[dict[str, Any]] = []
    wealth: float | None = 100.0
    benchmark: float | None = 100.0
    previous = None
    for row in body["series"]:
        session = row["session"]
        if previous is not None and session <= previous:
            raise ValueError("workbench.portfolio_session_axis_invalid")
        previous = session
        daily, reference = row.get("net_simple_return"), row.get("benchmark_simple_return")
        wealth = None if wealth is None or percent(daily) is None else wealth * (1 + daily)
        benchmark = (
            None if benchmark is None or percent(reference) is None else benchmark * (1 + reference)
        )
        indexed.append(
            {
                "date": session,
                "daily": percent(daily),
                "benchmarkDaily": percent(reference),
                "value": wealth,
                "benchmark": benchmark,
            }
        )
    return {
        "schema": "verified-portfolio-display",
        "subject": {
            "task_id": task_id,
            "receipt_hash": receipt_hash,
            "source_kind": "INSTALLED_RESULT" if canonical else "AUTHORED_EXPERIMENT",
            "result_hash": body.get("result_hash") if canonical else None,
            # an installed result's owner label (its strategy package); an authored study has none
            "title": title,
            # the installed strategy package the book ran (its controls say whether it runs forward,
            # U73); an authored book has none
            "strategy_package_id": title,
            # the policy the study declared -- names per sleeve, sleeves, the weight rule -- from
            # which the workbench names an authored book, as its lists name it (law 134: one name)
            "policy": {
                "top_k": declaration.get("top_k"),
                "tranches": declaration.get("tranches"),
                "weight_rule": declaration.get("weight_rule"),
                # A declared catalog policy runs in place of the tranche book, whose fields its
                # document leaves out (V323): the policy names the book.
                "catalog_policy": declaration.get("policy"),
            },
            "input_id": input_id,
            "input_hash": input_hash,
            "input_date": input_date,
            "session": position["session"],
            "support": support,
            "cost_per_side": declaration["cost_bps_per_side"],
        },
        "notice": (
            "Published historical research; not independent validation, "
            "current advice or trade authority."
        ),
        "seriesBasis": (
            "Display index 100 before first interval, derived from published returns; "
            "metrics remain the owner's values."
        ),
        "series": indexed,
        "sessions": [row["session"] for row in body["series"]],
        "metrics": {
            "total": percent(result.get("cumulative_return")),
            "annual": percent(result.get("annualized_return")),
            "vol": percent(result.get("annualized_volatility")),
            "drawdown": percent(result.get("maximum_drawdown")),
            "sharpe": result.get("sharpe"),
            "sortino": result.get("sortino"),
            "informationRatio": result.get("information_ratio"),
            "turnover": percent(position.get("one_way_turnover")),
            "costBps": result.get("cost_bps"),
            # Whole-report facts beside the benchmark the owner published with the book;
            # each is the owner's value or absent, never derived here.
            "benchmarkRelative": percent(result.get("benchmark_relative_return")),
            "beta": result.get("beta"),
            "trackingError": percent(result.get("tracking_error")),
            "jensenAlpha": percent(result.get("zero_cash_jensen_alpha")),
        },
        # The shown session's own facts (the book at one date), apart from the whole-report
        # metrics above: what the policy decided that day, how much it turned over, what the
        # cost fraction and returns of that one session were. Absent facts stay absent.
        "metricAbsences": body.get("selected_window_metric_absences", {}),
        "metricProvenance": body.get("selected_window_metric_provenance"),
        "position": {
            "session": position.get("session"),
            "decisionMode": position.get("decision_mode"),
            "holdingCount": position.get("holding_count"),
            "cash": percent(position.get("cash")),
            "concentration": position.get("hhi"),
            "turnover": percent(position.get("one_way_turnover")),
            "costFraction": percent(position.get("cost_fraction")),
            "netReturn": percent(position.get("net_simple_return")),
            "grossReturn": percent(position.get("gross_simple_return")),
            "benchmarkReturn": percent(position.get("benchmark_simple_return")),
        },
        # Each position metric's unit by its path, as the book's readback names it (V343, U58).
        "metricUnits": body.get("metric_units"),
        # What the book can claim, one statement a mark, comparison first (V368, U59).
        "standing": body.get("standing"),
        "review_standing": body.get("review_standing"),
        "holdings": holdings,
        "holdings_basis": {
            "basis": "HISTORICAL_REPLAY",
            "comparison_basis": book.get("change_boundary", "PRECEDING_NOT_RECORDED"),
            "preceding_formation_session": book.get("preceding_formation_session"),
            "formation_session": book.get("formation_session", position.get("session")),
            "entry_session": None,
            "observed_through": position.get("session"),
        },
        "sectors": {
            "status": sector_map.get("status"),
            "sector_revision": sector_map.get("sector_revision"),
            "coverage_hash": sector_map.get("coverage_hash"),
            "reason": sector_map.get("reason"),
        },
        "universe": {
            "eligible": quality.get("effective_listing_count"),
            "total": quality.get("source_listing_count"),
            "quarantine": excluded,
            "definition": quality.get("notice", ""),
        },
        "limitations": limitations,
        "timing": body.get("timing"),
        "reviewSelector": review_selector,
        "positionClock": body.get("reading_context", {}).get("position_clock"),
        "bookPolicy": body.get("reading_context", {}).get("book_policy"),
        "riskDisposition": (body.get("reading_context", {}).get("strategy") or {}).get(
            "risk_disposition"
        ),
        # An installed book's statements of its window and survivorship (V347, U52); an
        # authored book's are its timing's.
        "temporalScope": body.get("reading_context", {}).get("temporal_scope"),
        # The authored policy and the recorded sources, read from this same verified
        # body so that showing them needs no second readback. Small references only:
        # the source arrays, series and resolved sessions stay with the owner.
        "declaration": {
            "top_k": declaration.get("top_k"),
            "tranches": declaration.get("tranches"),
            "exit_rank": declaration.get("exit_rank"),
            "weight_rule": declaration.get("weight_rule"),
            "catalog_policy": declaration.get("policy"),
            "cost_bps_per_side": declaration["cost_bps_per_side"],
            # The authored document omits the default policy; the receipt records the
            # one that governed the published result.
            "unavailable_return_policy": quality.get("policy"),
        },
        "source": source_projection,
    }
