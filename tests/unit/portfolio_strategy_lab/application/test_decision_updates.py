"""A review names a decision publication by its sealed content apart from its rendering."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from alphalattice.capabilities.domain_contracts.identity import canonical_hash
from alphalattice.foundation.causal_outcomes.execution.contracts import (
    CausalExecutionSchedulePoint,
)
from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (
    PortfolioEntryBook,
    PortfolioUpdatePublication,
)
from tests._support.identity_fixtures import fixture_digest

TEST_SOURCE = "src/alphalattice/investment/portfolio_strategy_lab/application/decision_updates.py"
FORMED, ENTERED, HELD = date(2026, 9, 4), date(2026, 9, 8), date(2026, 9, 15)


def _at(day: date, hour: int) -> datetime:
    return datetime(day.year, day.month, day.day, hour, tzinfo=UTC)


def _publication(**changes: object) -> PortfolioUpdatePublication:
    point = CausalExecutionSchedulePoint(
        sequence=1,
        formation_session=FORMED,
        formation_close_at=_at(FORMED, 20),
        entry_session=ENTERED,
        entry_open_at=_at(ENTERED, 13),
        holding_end_session=HELD,
        holding_end_open_at=_at(HELD, 13),
        actual_session_span=5,
    )
    book = PortfolioEntryBook.create(
        schedule=point, next_position=1, weights=(0.5, 0.5), cash=0.0, sleeves=((0.5, 0.5),)
    )
    fields: dict[str, object] = {
        "plan_hash": fixture_digest("plan"),
        "checkpoint_hash": fixture_digest("checkpoint"),
        "parent_hash": None,
        "observed_through": FORMED,
        "published_at": _at(ENTERED, 23),
        "book": book,
        "pending_proposal": None,
        "active_entry": None,
        "events": (),
        "source_snapshot_hash": fixture_digest("source"),
        "html_hash": fixture_digest("rendered"),
    }
    return PortfolioUpdatePublication.create(**(fields | changes))


def _stored_before(value: PortfolioUpdatePublication) -> PortfolioUpdatePublication:
    """The publication as a store holds it from before its rendering-free digest was minted."""
    payload = value.model_dump(mode="json", exclude={"content_hash", "publication_hash"})
    return PortfolioUpdatePublication.model_validate(
        {**payload, "content_hash": canonical_hash(payload)}
    )


@pytest.mark.proves(
    "portfolio_strategy_lab.a_review_names_its_publication_apart_from_its_rendering"
)
@pytest.mark.parametrize(
    ("remake", "same_review"),
    [
        pytest.param(
            lambda value: PortfolioUpdatePublication.create(
                **(value.model_dump() | {"html_hash": fixture_digest("rendered again")})
            ),
            True,
            id="re-rendered",
        ),
        pytest.param(
            lambda value: _publication(published_at=value.published_at + timedelta(seconds=1)),
            False,
            id="sealed-field-changed",
        ),
        pytest.param(_stored_before, False, id="stored-before"),
    ],
)
def test_a_review_names_the_publication_by_its_content_apart_from_its_rendering(
    remake, same_review
):
    """A re-render keeps the review's name, a sealed change mints a new one, and a publication
    stored before still reads under its content hash."""
    value = _publication()
    other = remake(value)
    assert other.content_hash != value.content_hash
    assert (other.review_name == value.review_name) is same_review
    assert other.names(value.review_name) is same_review
    assert other.names(other.content_hash) and other.names(other.review_name)
    assert (other.publication_hash is None) is (other.review_name == other.content_hash)
    if other.publication_hash is not None:
        forged = other.model_dump(mode="json", exclude={"content_hash"})
        forged["publication_hash"] = fixture_digest("forged")
        with pytest.raises(ValueError, match="identity_invalid"):
            PortfolioUpdatePublication.model_validate(
                {**forged, "content_hash": canonical_hash(forged)}
            )
