"""Eligibility and units for comparing verified authored research books."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any, Final, Literal

from alphalattice.interface.local_application.portfolio_research import (
    PortfolioComparisonDimension,
    PortfolioComparisonMetric,
)

POSITION_UNITS: Final[Mapping[str, str]] = {
    "holding_count": "names",
    "cash": "portfolio weight",
    "hhi": "squared portfolio weight",
    "one_way_turnover": "portfolio fraction",
    "cost_fraction": "portfolio fraction",
    "gross_simple_return": "fraction",
    "net_simple_return": "fraction",
}
"""Each Portfolio position metric's unit: what a comparison states, and what a book's readback
names beside its position (`metric_units`, V343)."""


def _units(*keys: str) -> tuple[tuple[str, str], ...]:
    return tuple((key, POSITION_UNITS[key]) for key in keys)


def compare_experiment_reports(left: dict[str, Any], right: dict[str, Any]) -> dict[str, object]:
    """Compare verified authored readouts, using the existing comparison vocabulary."""
    if any(
        v.get("status") != "EXPERIMENT_PUBLISHED" or not v.get("portfolio_source")
        for v in (left, right)
    ):
        raise ValueError("portfolio_research.comparison_completed_books_required")
    source_keys = (
        "input_binding_hash",
        "panel_snapshot_hash",
        "universe_revision",
        "ordered_listing_ids",
        "formation_sessions",
    )
    if any(
        left["portfolio_source"][key] != right["portfolio_source"][key] for key in source_keys
    ) or (
        left["receipt"]["market_binding"] != right["receipt"]["market_binding"]
        or left["position"]["session"] != right["position"]["session"]
    ):
        raise ValueError("portfolio_research.comparison_input_or_support_mismatch")
    dimensions = []
    groups: tuple[
        tuple[
            Literal["HOLDINGS", "CONCENTRATION", "TURNOVER", "COST", "PERFORMANCE"],
            tuple[tuple[str, str], ...],
        ],
        ...,
    ] = (
        ("HOLDINGS", _units("holding_count", "cash")),
        ("CONCENTRATION", _units("hhi")),
        ("TURNOVER", _units("one_way_turnover")),
        ("COST", _units("cost_fraction")),
        ("PERFORMANCE", _units("gross_simple_return", "net_simple_return")),
    )
    for name, fields in groups:
        dimensions.append(
            PortfolioComparisonDimension(
                name,
                tuple(
                    PortfolioComparisonMetric(
                        key, unit, left["position"][key], right["position"][key]
                    )
                    for key, unit in fields
                ),
            )
        )
    dimensions.append(
        PortfolioComparisonDimension(
            "PERFORMANCE",
            tuple(
                PortfolioComparisonMetric(
                    "platform_one_way_cost_bps" if key == "cost_bps" else key,
                    "bps on platform one-way turnover"
                    if key == "cost_bps"
                    else "ratio"
                    if key in {"sharpe", "sortino", "beta", "information_ratio"}
                    else "annualized fraction"
                    if key
                    in {
                        "annualized_return",
                        "annualized_volatility",
                        "tracking_error",
                        "zero_cash_jensen_alpha",
                    }
                    else "fraction",
                    value,
                    right["result"][key],
                )
                for key, value in left["result"].items()
            ),
        )
    )
    return {
        "disposition": "DECLARED_PATH_COMPARISON_NO_SELECTION",
        "left": {
            k: left[k]
            for k in ("task_id", "receipt", "document", "position", "series", "limitations")
        },
        "right": {
            k: right[k]
            for k in ("task_id", "receipt", "document", "position", "series", "limitations")
        },
        "dimensions": [asdict(v) for v in dimensions],
        "support": left["portfolio_source"]["formation_sessions"],
        "risk": {
            "status": "NOT_ALLOCATION_RISK",
            "detail": "Separately linked Risk reports are references, not inferred portfolio risk.",
        },
        "claim": (
            "Observed research comparison; no winner, refit, publication or independent validation."
        ),
    }
