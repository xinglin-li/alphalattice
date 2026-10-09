"""A result's time and survivorship, stated from its Panel's marks (the user,
2026-09-30)."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from alphalattice.control.product_host.research_authoring.timing import temporal_scope
from alphalattice.foundation.feature_engine.publication.temporal_statement import (
    TemporalStatement,
)
from alphalattice.foundation.market_data_ops.sources.membership import INITIAL_COHORT_BACKFILL


def _summary(
    *,
    t0: str | None = "2026-08-14",
    cohort: int = 466,
    treatment: str = "CURRENT_CLASSIFICATION_BACKFILLED",
    survivors: bool = True,
    membership: bool = True,
) -> dict[str, Any]:
    return {
        "membership": (
            {
                "bootstrap_t0_session": t0,
                "basis_ranges": [
                    {
                        "first_session": "2016-09-12",
                        "last_session": "2026-08-13",
                        "basis": INITIAL_COHORT_BACKFILL,
                    }
                ],
                "epochs": [
                    {
                        "first_session": "2016-09-12",
                        "last_session": "2026-08-13",
                        "member_count": cohort,
                        "membership_hash": "a" * 64,
                    }
                ],
            }
            if membership
            else None
        ),
        "temporal_risk": {
            "universe_temporal_scope": "CURRENT_ACTIVE_SURVIVORS",
            "sector_history_treatment": treatment,
            "sector_observed_at": "2026-08-14T23:13:21+00:00",
        },
        "universe_policy": {"survivorship_bias_warning": survivors},
    }


def _state(summary: dict[str, Any], window: str, data: str, basis: str = "split_adjusted"):
    return TemporalStatement.from_panel(
        summary,
        window_start=date.fromisoformat(window),
        window_end=date(2026, 8, 17),
        data_start=date.fromisoformat(data),
        price_basis=basis,
    )


@pytest.mark.parametrize(
    ("window", "data", "before", "history", "opening"),
    [
        ("2021-01-04", "2016-09-12", True, True, "The window starts 2021-01-04, before T0"),
        ("2026-08-17", "2026-01-02", False, True, "The window starts at or after T0"),
        ("2026-08-17", "2026-08-17", False, False, "The window and its inputs start at or"),
    ],
)
def test_a_window_says_where_it_stands_to_t0(
    window: str, data: str, before: bool, history: bool, opening: str
) -> None:
    """Requirement: the statement names T0 and the initial cohort's size from the Panel's
    marks, and says whether the window, or only its inputs, reach before T0."""

    statement = _state(_summary(), window, data)
    assert statement.t0_session == date(2026, 8, 14)
    assert statement.initial_cohort_size == 466
    assert (statement.window_before_t0, statement.data_before_t0) == (before, history)
    assert statement.universe_basis == INITIAL_COHORT_BACKFILL
    assert statement.statements[0].startswith(opening)
    assert ("466 listings" in statement.statements[0]) == history
    assert "survivorship bias" in statement.statements[1]


def test_every_mark_value_generates_its_statement_and_an_unknown_one_is_named() -> None:
    """Requirement: the Sector treatment and the price basis each come from a template
    keyed by the mark, so before every session uses the current classification, T0 onward
    included; a mark with no template is named, never dropped; no survivorship sentence where
    the Panel records none, and no T0 where it records no membership."""

    current = _state(_summary(), "2021-01-04", "2016-09-12")
    assert current.statements[2] == (
        "Every session uses the Sector classification observed on 2026-08-14, the sessions after "
        "T0 included; it is not point in time."
    )
    assert "current share basis" in current.statements[3]
    assert "as traded" in _state(_summary(), "2021-01-04", "2016-09-12", "unadjusted").statements[3]
    unknown = _state(_summary(treatment="A_NEW_TREATMENT"), "2021-01-04", "2016-09-12", "other")
    assert "`A_NEW_TREATMENT` has no installed statement" in unknown.statements[2]
    assert "`other` has no installed statement" in unknown.statements[3]
    unflagged = _state(_summary(survivors=False), "2021-01-04", "2016-09-12")
    assert not any("survivorship" in line for line in unflagged.statements)
    legacy = _state(_summary(membership=False), "2021-01-04", "2016-09-12")
    assert (legacy.t0_session, legacy.universe_basis) == (None, "CURRENT_ACTIVE_SURVIVORS")
    assert legacy.statements[0] == (
        "The Panel records no T0: every session holds the cohort it was built with."
    )
    assert "the sessions after T0" not in legacy.statements[2]


def test_the_host_states_a_panel_with_the_installed_price_basis() -> None:
    """Requirement: the Host's statement reads the installed market profile's price
    basis, so a readback names it without a carrier writing it."""

    scope = temporal_scope(
        {"safe_summary": _summary()},
        window_start="2021-01-04",
        window_end="2026-08-17",
        data_start="2016-09-12",
    )
    assert scope["status"] == "RECORDED"
    assert (scope["price_basis"], scope["t0_session"]) == ("split_adjusted", "2026-08-14")
    assert "current share basis" in scope["statements"][-1]
