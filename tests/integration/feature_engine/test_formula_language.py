"""The factor formula language: one spelling per formula, causality first, the numeric rules
(EX, its design's section 3; the user, 2026-09-30)."""

from __future__ import annotations

import re

import numpy as np
import pytest

from alphalattice.foundation.feature_engine.producers.factors.formula_language import (
    FormulaError,
    canonical,
    check_causality,
    evaluate,
    lookback,
    parse,
)

LEAVES = frozenset({"close", "volume", "ret", "market_return", "sector_label"})


def _canonical(text: str) -> str:
    return canonical(parse(text, leaves=LEAVES))


def test_two_spellings_of_one_formula_share_one_canonical_form() -> None:
    """requirement (EX): the canonical form normalizes spacing, parentheses, numbers, keywords
    and aliases, keeps operand order, and reads back as itself."""

    assert _canonical("ts_mean( close , n = 21.0 )/close-1") == _canonical(
        "(ts_mean(close, 21) / close) - 1"
    )
    assert _canonical("delay(ret, 5)") == "lag(ret, 5)"
    assert _canonical("clip(ret, hi=0.1, lo=-0.1)") == "clip(ret, -0.1, 0.1)"
    assert _canonical("close + volume") != _canonical("volume + close"), "operand order kept"
    for text in (
        "close - (volume - ret)",
        "-(close * 2) + close * -2",
        "where(ret > 0 and not volume > 1 or ret < 0, 1, -1)",
        "rank(-ts_corr(close, volume, 10))",
        "ewm(ret, 2.5)",
    ):
        once = _canonical(text)
        assert _canonical(once) == once, text


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("close.real", "factor_formula.syntax_not_held:Attribute"),
        ("future(close)", "factor_formula.operator_unknown:future"),
        ("open_tomorrow", "factor_formula.leaf_unknown:open_tomorrow"),
        ("lag(close, -1)", "factor_formula.window_invalid:lag:-1:1-504"),
        ("ts_mean(close, 505)", "factor_formula.window_invalid:ts_mean:505:1-504"),
        ("ts_std(close, 1)", "factor_formula.window_invalid:ts_std:1:2-504"),
        ("ts_mean(close, ret)", "factor_formula.argument_not_constant:ts_mean:n"),
        ("ts_mean(close)", "factor_formula.argument_missing:ts_mean:n"),
        ("ts_mean(5, 3)", "factor_formula.series_of_a_constant:ts_mean"),
        ("close < ret < volume", "factor_formula.comparison_invalid"),
        ("ewm(ret, 60)", "factor_formula.halflife_invalid:60"),
    ],
)
def test_a_formula_the_language_does_not_hold_is_refused_by_name(text: str, code: str) -> None:
    """requirement (EX, OP12): a refusal names the node and what it found."""

    with pytest.raises(FormulaError, match=re.escape(code)):
        parse(text, leaves=LEAVES)


def test_causality_is_checked_before_anything_computes() -> None:
    """requirement (EX): a node reads sessions up to T and never later; the lookback counts
    each lag and window and fits the history floor; a leaf must come from a point-in-time
    source."""

    tree = parse("ts_mean(lag(ret, 5), 21) + delta(close, 3)", leaves=LEAVES)
    assert lookback(tree) == 5 + 20
    assert check_causality(tree, point_in_time=LEAVES) == 25
    deep = parse("ts_mean(ts_mean(ret, 300), 300)", leaves=LEAVES)
    with pytest.raises(
        FormulaError, match=re.escape("factor_formula.lookback_exceeds_floor:599>504")
    ):
        check_causality(deep, point_in_time=LEAVES)
    with pytest.raises(
        FormulaError, match=re.escape("factor_formula.leaf_not_point_in_time:close")
    ):
        check_causality(tree, point_in_time=LEAVES - {"close"})


def _panel(rows: list[list[float]]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float64)


def test_numeric_edges_give_nan_never_an_infinity() -> None:
    """requirement (EX, the user's rules): division by zero, log or sqrt outside its domain,
    and a zero deviation give NaN; NaN propagates; logic is three-valued and `where(NaN, a, b)`
    is NaN."""

    x = _panel([[1.0, 0.0, -1.0, np.nan]])
    y = _panel([[0.0, 2.0, 4.0, 1.0]])

    def run(text: str) -> list[float]:
        return list(evaluate(parse(text, leaves=LEAVES), {"close": x, "ret": y})[0])

    assert np.isnan(run("close / ret")[0]) and run("close / ret")[1] == 0.0
    assert [np.isnan(v) for v in run("log(close)")] == [False, True, True, True]
    assert [np.isnan(v) for v in run("sqrt(close)")] == [False, False, True, True]
    assert run("close > 0")[:3] == [1.0, 0.0, 0.0] and np.isnan(run("close > 0")[3])
    assert np.isnan(run("close > 0 and ret > 0")[3]), "NaN in, NaN out"
    assert np.isnan(run("where(close, 1, 2)")[3]) and run("where(close, 1, 2)")[:2] == [1.0, 2.0]
    flat = _panel([[3.0, 3.0, 3.0]])
    assert all(np.isnan(evaluate(parse("zscore(close)", leaves=LEAVES), {"close": flat})[0]))


def test_ranks_give_ties_their_average_over_the_values_present() -> None:
    """requirement (EX): `rank` and `ts_rank` give tied values their average rank, divided by
    the count of non-NaN values, so each lies in (0, 1]; a NaN stays NaN."""

    x = _panel([[3.0, 1.0, 3.0, np.nan, 2.0]])
    ranks = evaluate(parse("rank(close)", leaves=LEAVES), {"close": x})[0]
    assert list(ranks[[0, 1, 2, 4]]) == [3.5 / 4, 1 / 4, 3.5 / 4, 2 / 4]
    assert np.isnan(ranks[3])
    series = _panel([[1.0], [3.0], [2.0], [3.0]])
    ts_rank = evaluate(parse("ts_rank(close, 3)", leaves=LEAVES), {"close": series})[:, 0]
    assert np.isnan(ts_rank[:2]).all(), "a window missing a session is NaN"
    assert list(ts_rank[2:]) == [2 / 3, 2.5 / 3]


def test_a_window_reads_backward_and_a_formula_is_deterministic() -> None:
    """requirement (EX): time-series operators read each listing's own past, never a later
    session; the same panel gives the same bytes."""

    rng = np.random.default_rng(7)
    panel = {name: rng.standard_normal((60, 8)) for name in ("close", "ret", "market_return")}
    tree = parse("ts_beta(ret, market_return, 20) + ts_corr(close, ret, 10)", leaves=LEAVES)
    first = evaluate(tree, panel)
    assert first.tobytes() == evaluate(tree, panel).tobytes()
    changed = {name: value.copy() for name, value in panel.items()}
    changed["ret"][40:] = rng.standard_normal((20, 8))
    later = evaluate(tree, changed)
    assert np.array_equal(first[:40], later[:40], equal_nan=True), "no session reads the future"
