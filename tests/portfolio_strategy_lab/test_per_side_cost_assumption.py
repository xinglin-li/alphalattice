"""The per-side cost owner: one arithmetic authority, exact ladder, frozen identities.

The delivery plan asked for a fixed-point per-side assumption and warned that the
predecessor "accepts integer platform bps and must not be treated as if it
already implements exact per-side fixed-point values such as 2.5". These tests
cover what that claim depends on: that the ladder is exact and that an ambiguous
quote is refused rather than rounded. (The baseline package whose frozen hashes
the extraction was also checked against retired with the Stage-6 campaign, R01.)
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from alphalattice.capabilities.portfolio_backtesting.contracts import (
    PortfolioPerSideCostAssumption,
    PortfolioWalkForwardError,
    platform_one_way_cost_from_per_side,
)


@pytest.mark.parametrize(
    ("per_side", "platform"),
    [("0", 0), ("2.5", 5), ("5", 10), ("10", 20), ("20", 40)],
)
def test_the_admitted_ladder_is_exact_in_both_units(per_side: str, platform: int) -> None:
    """The plan's 0/2.5/5/10/20 ladder, and the float the engine actually consumes.

    The float check is the load-bearing half. A per-side type that could not hand
    the engine an exact rate would have moved the ambiguity rather than removed
    it; the frozen 0.5 increment is what makes every platform rate a whole
    number of basis points.
    """

    assumption = PortfolioPerSideCostAssumption.from_bps_per_side(per_side)
    assert assumption.cost_bps_per_side == Decimal(per_side)
    assert assumption.platform_one_way_cost_bps == Decimal(platform)
    assert assumption.platform_cost_bps == float(platform)
    assert assumption.platform_cost_bps == float(assumption.platform_one_way_cost_bps)


def test_a_quote_finer_than_the_increment_is_refused_not_rounded() -> None:
    """0.25 bps per side is a tenth of a basis point of platform cost, and is refused."""

    with pytest.raises(PortfolioWalkForwardError, match="per_side_cost_not_fixed_point"):
        PortfolioPerSideCostAssumption.from_bps_per_side("0.25")


def test_a_value_off_the_frozen_increment_is_refused() -> None:
    """0.3 is expressible in tenths but is not on the admitted 0.5 grid.

    It is also the value that motivates the grid: no float holds 0.3 exactly, so
    admitting it would put an inexact rate into a content hash.
    """

    with pytest.raises(ValidationError, match="per_side_cost_increment_invalid"):
        PortfolioPerSideCostAssumption.from_bps_per_side("0.3")


def test_the_control_range_is_bounded_at_twenty_per_side() -> None:
    with pytest.raises(ValidationError):
        PortfolioPerSideCostAssumption.from_bps_per_side("20.5")


def test_a_negative_per_side_quote_is_refused_by_the_shared_mapping() -> None:
    with pytest.raises(PortfolioWalkForwardError, match="per_side_cost_negative"):
        platform_one_way_cost_from_per_side(-1)


def test_identity_is_tamper_evident() -> None:
    """A changed rate carrying the old hash is refused, not silently accepted."""

    assumption = PortfolioPerSideCostAssumption.from_bps_per_side("5")
    payload = assumption.model_dump(mode="json")
    payload["cost_bps_per_side_tenths"] = 100
    with pytest.raises(ValidationError, match="per_side_cost_identity_invalid"):
        PortfolioPerSideCostAssumption.model_validate(payload)


def test_json_round_trip_preserves_the_exact_rate() -> None:
    """Fixed point survives serialisation; that is why it is a scaled integer."""

    assumption = PortfolioPerSideCostAssumption.from_bps_per_side("2.5")
    restored = PortfolioPerSideCostAssumption.model_validate_json(assumption.model_dump_json())
    assert restored == assumption
    assert restored.cost_bps_per_side == Decimal("2.5")


def test_the_payload_never_shows_one_rate_without_its_basis() -> None:
    """A report that prints 10 without saying which lane is the documented trap."""

    payload = PortfolioPerSideCostAssumption.from_bps_per_side("5").researcher_payload()
    assert payload["quote_basis"] == "PER_SIDE"
    assert payload["cost_bps_per_side"] == "5"
    assert payload["platform_one_way_cost_bps"] == "10"
    assert "2 * cost_bps_per_side" in payload["mapping"]
