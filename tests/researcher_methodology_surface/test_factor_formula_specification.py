"""Complete Factor formulas, exact boundaries, and separate development admission."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from alphalattice.foundation.feature_engine.producers.factors.catalog import (
    default_extension_kernel_registry,
    extension_factor_specs,
)
from alphalattice.foundation.feature_engine.producers.factors.specifications import (
    admit_factor_development_capabilities,
    admitted_extension_factor_specs,
    build_installed_factor_formula_specifications,
    golden_example_frame,
)
from alphalattice.kernel.quant.factor_contracts import FactorSpec


@pytest.mark.parametrize("recipe", extension_factor_specs(), ids=lambda value: value.factor_id)
def test_every_installed_candidate_has_executable_boundary_goldens(recipe: FactorSpec) -> None:
    """requirement: all fifteen methods are specified and one row short is missing."""

    catalog = build_installed_factor_formula_specifications()
    registry = default_extension_kernel_registry()
    specification = catalog.validate_registration(recipe, registry=registry)
    by_label = {value.label: value for value in specification.golden_examples}
    assert specification.status == "COMPLETE"
    assert by_label["first-valid-source-boundary"].row_count == recipe.minimum_observations
    assert by_label["one-row-short-is-missing"].row_count == recipe.minimum_observations - 1
    for label, example in by_label.items():
        frame = golden_example_frame(example)
        values = registry.compute(frame, recipe)
        selected = values[frame["listing_id"] == "GOLDEN00"]
        if example.expected_value is None:
            # A candidate that requires a single row has no one-row-short frame
            # with a row in it: `minimum_observations - 1` is zero, so the source
            # is empty and the kernel produces no value at all. That is a
            # stronger missing than a NaN and it is the honest assertion here;
            # indexing the empty result instead would raise.
            if example.row_count == 0:
                assert selected.empty, label
                continue
            assert math.isnan(float(selected.iloc[-1])), label
        else:
            assert float(selected.iloc[-1]) == pytest.approx(
                example.expected_value,
                abs=specification.absolute_tolerance,
                rel=specification.relative_tolerance,
            )


@pytest.mark.parametrize(
    "recipe",
    tuple(value for value in extension_factor_specs() if value.factor_id != "sector_leader_lag_5"),
    ids=lambda value: value.factor_id,
)
def test_first_valid_golden_survives_interleaved_listing_axis(recipe: FactorSpec) -> None:
    """regression: source labels are not positions in a listing-ordered kernel array."""

    specification = build_installed_factor_formula_specifications().resolve(recipe.factor_id)
    example = next(
        value
        for value in specification.golden_examples
        if value.label == "first-valid-source-boundary"
    )
    assert example.expected_value is not None
    left = golden_example_frame(example).assign(listing_id="GOLDEN00")
    right = golden_example_frame(example).assign(listing_id="GOLDEN01")
    interleaved = pd.concat((left, right), ignore_index=True).sort_values(
        ["session_date", "listing_id"], kind="mergesort"
    )
    observed = default_extension_kernel_registry().compute(interleaved, recipe)
    for listing_id in ("GOLDEN00", "GOLDEN01"):
        listing_value = float(observed[interleaved["listing_id"] == listing_id].iloc[-1])
        assert listing_value == pytest.approx(
            example.expected_value,
            abs=specification.absolute_tolerance,
            rel=specification.relative_tolerance,
        )


def test_intraday_amplitude_is_the_absolute_intraday_move_on_a_second_session() -> None:
    """regression: the boundary golden is degenerate, so pin the formula elsewhere.

    `intraday_amplitude` is `|log(close/open)|`, and its first-valid-source golden
    is one row long. The declared open series is `price * (1 + 0.002 * sin(index))`,
    and `sin(0)` is zero, so on that single row open equals close and both
    `intraday` and its amplitude are exactly zero. The boundary golden therefore
    asserts `|0| == 0` and constrains nothing about the formula. Session index 1
    is the first row where the two series separate, and 0.0016815274096384609 is
    the value the expectation table carried before 2026-09-02, when it had been
    transcribed from a two-row frame onto a one-row golden. It is kept here, at
    the row it actually describes, so the number stays load-bearing.
    """

    recipe = next(
        value for value in extension_factor_specs() if value.factor_id == "intraday_amplitude"
    )
    intraday_recipe = next(
        value for value in extension_factor_specs() if value.factor_id == "intraday"
    )
    rows = 2
    prices = [
        100.0 * math.exp(0.0002 * index + 0.03 * math.sin(index / 13)) for index in range(rows)
    ]
    frame = pd.DataFrame(
        {
            "listing_id": ["GOLDEN00"] * rows,
            "session_date": pd.date_range("2024-01-01", periods=rows, freq="D").date,
            "close_split_adjusted": prices,
            "open_split_adjusted": [
                price * (1.0 + 0.002 * math.sin(index)) for index, price in enumerate(prices)
            ],
        }
    )
    registry = default_extension_kernel_registry()
    amplitude = registry.compute(frame, recipe)
    intraday = registry.compute(frame, intraday_recipe)

    assert float(amplitude.iloc[-1]) == pytest.approx(0.0016815274096384609, abs=1e-15, rel=1e-15)
    # The amplitude is the magnitude of the signed move, row for row, and the
    # signed move on this row is negative, so the two are not the same number.
    assert float(intraday.iloc[-1]) == pytest.approx(-0.0016815274096384609, abs=1e-15, rel=1e-15)
    assert float(amplitude.iloc[0]) == 0.0
    assert float(intraday.iloc[0]) == 0.0


def test_the_corrected_amplitude_golden_rotates_only_the_validation_identity() -> None:
    """requirement: the golden correction moved a contract identity, and it is pinned.

    Correcting the stale expectation is not free. `specification_hash` covers
    the golden examples, so it moved, and `FactorDevelopmentCapability` folds it
    into `capability_hash`, which the admission receipt folds into
    `receipt_hash`. All three rotated for this one candidate.

    What did *not* move is the executable half. `implementation_hash` is derived
    from the kernel, not from the expectation table, so it is unchanged, and so
    are the values the formula produces. That is the whole claim: the
    validation contract was corrected, the arithmetic was not touched, and this
    test fails if either half of that stops being true.
    """

    specification = build_installed_factor_formula_specifications().resolve("intraday_amplitude")
    receipt = next(
        value
        for value in admit_factor_development_capabilities()
        if value.factor_id == "intraday_amplitude"
    )

    # Rotated on 2026-09-02 by the golden correction. C8's docstrings in kernel/quant/
    # factor_formulas.py moved these on 2026-09-27; that file's bytes were restored, since a
    # byte closure reads it (V231), until W8 puts the factor identities on the rule.
    assert (
        specification.specification_hash
        == "9f52d8a0f2f769b8cd45715b7cc8c25b7f155cba6744a7b0dac790b00706324f"
    )
    assert (
        receipt.capability_hash
        == "d0c951d2c1a922a2d4b9a3fda76040b20f63817a70f97ff3f2f1229bf5f5b713"
    )
    assert (
        receipt.receipt_hash == "2bd6d469cd6cf3298c720fe44b8bbd6eca39a7ecd5940632cab5d155c9ba577f"
    )
    # The superseded identity must not come back.
    assert (
        specification.specification_hash
        != "543ef5a4f897f782301bcb18335378d9f5e4e3607fd31e5e0148b39368b5f033"
    )

    # Unchanged controls: the executable formula and the admission outcome.
    assert (
        specification.implementation_hash
        == "bfb043d078ea190ad2ef972f1882df31da52bf2701d0cafaf878fc1af170192b"
    )
    assert receipt.disposition == "ADMITTED"

    # And exactly one candidate moved: every other receipt is untouched, which
    # is what makes this a corrected expectation rather than a table rewrite.
    assert len(admit_factor_development_capabilities()) == 33


def test_admission_is_independent_and_derives_every_installed_candidate() -> None:
    receipts = admit_factor_development_capabilities()
    installed_ids = {item.factor_id for item in extension_factor_specs()}
    assert {item.factor_id for item in receipts} == installed_ids
    refused = [value for value in receipts if value.disposition == "REFUSED"]
    assert [(value.factor_id, value.reason) for value in refused] == [
        ("sector_leader_lag_5", "LAGGED_SECTOR_MEMBERSHIP_UNAVAILABLE")
    ]
    assert {item.factor_id for item in receipts if item.disposition == "ADMITTED"} == (
        installed_ids - {"sector_leader_lag_5"}
    )
    admitted = admitted_extension_factor_specs(receipts=receipts)
    assert tuple(value.factor_id for value in admitted) == tuple(
        value.factor_id
        for value in extension_factor_specs()
        if value.factor_id != "sector_leader_lag_5"
    )


def _tail_frame(prices: list[float], market: list[float]) -> pd.DataFrame:
    rows = len(prices)
    return pd.DataFrame(
        {
            "listing_id": ["GOLDEN00"] * rows,
            "session_date": pd.date_range("2024-01-01", periods=rows, freq="D").date,
            "market_return_log": market,
            "provider_adjusted_close": prices,
        }
    )


def test_tail_resilience_normalization_is_hand_derivable() -> None:
    """requirement: the approved idio_std * sqrt(6) normalization, transcribed."""

    import numpy as np

    rows = 253
    market = [0.004 * math.sin(index / 3) + 0.001 * ((index % 5) - 2) for index in range(rows)]
    prices = [
        100.0 * math.exp(0.0002 * index + 0.03 * math.sin(index / 13)) for index in range(rows)
    ]
    recipe = next(
        value for value in extension_factor_specs() if value.factor_id == "tail_resilience_252"
    )
    observed = float(
        default_extension_kernel_registry().compute(_tail_frame(prices, market), recipe).iloc[-1]
    )
    # Independent transcription of the approved formula, not the kernel. The OLS
    # window ends at the observation session, so the terminal row is included.
    stock = np.full(rows, np.nan)
    stock[1:] = np.log(np.asarray(prices[1:]) / np.asarray(prices[:-1]))
    x, y = stock[-252:], np.asarray(market)[-252:]
    beta = float(np.dot(x - x.mean(), y - y.mean()) / np.dot(y - y.mean(), y - y.mean()))
    residual = x - x.mean() - beta * (y - y.mean())
    idio_std = float(np.std(residual, ddof=1))
    order = np.lexsort((np.arange(247), y[:247]))
    episodes = np.asarray([residual[start : start + 6].sum() for start in order[:25]])
    expected = float(episodes.mean() / (idio_std * math.sqrt(6.0)))
    assert observed == pytest.approx(expected, abs=1e-12, rel=1e-12)
    assert observed * idio_std * math.sqrt(6.0) == pytest.approx(float(episodes.mean()))


def test_tail_resilience_zero_idiosyncratic_scale_is_missing() -> None:
    """requirement: a non-positive idiosyncratic scale yields the missing value."""

    rows = 253
    market = [0.004 * math.sin(index / 3) + 0.001 * ((index % 5) - 2) for index in range(rows)]
    # A constant price makes every stock log return exactly zero, so the fitted
    # beta and every OLS residual are exactly zero and the declared
    # normalization scale collapses rather than merely shrinking.
    prices = [100.0] * rows
    recipe = next(
        value for value in extension_factor_specs() if value.factor_id == "tail_resilience_252"
    )
    observed = float(
        default_extension_kernel_registry().compute(_tail_frame(prices, market), recipe).iloc[-1]
    )
    assert math.isnan(observed)


def test_tail_and_leader_clocks_are_mechanical_and_explicit() -> None:
    catalog = build_installed_factor_formula_specifications()
    tail = catalog.resolve("tail_resilience_252")
    assert tail.clock.source_interval == "[t-252,t]"
    assert tail.clock.estimation_interval == "OLS returns [t-251,t]"
    assert tail.clock.event_interval == "worst 25 Market sessions in [t-251,t-5], date tie-break"
    assert tail.clock.recovery_interval == "[s,s+5], latest end t"
    assert tail.minimum_ordered_source_rows == 253
    assert tail.clock.observation_clock.formula_skip_sessions == 0
    assert tail.clock.observation_clock.source_interval_rendered == tail.clock.source_interval

    leader = catalog.resolve("sector_leader_lag_5")
    assert leader.clock.source_interval == "[t-25,t]"
    assert "ADV21 [t-25,t-5]" in (leader.clock.estimation_interval or "")
    # The leader-formation offset is the Formula's own construction and stays;
    # only the session of execution safety at the end of the interval is gone.
    assert "five-session return (t-5,t]" in (leader.clock.estimation_interval or "")
    assert leader.minimum_ordered_source_rows == 26
    assert leader.clock.observation_clock.formula_skip_sessions == 0
