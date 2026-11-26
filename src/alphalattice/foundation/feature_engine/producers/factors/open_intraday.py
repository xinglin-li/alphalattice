"""The open/intraday method family: what the session's own seam is worth.

Every maintained control measures a move between two closes, so the overnight
and intraday halves of a session return are nowhere separated. This family is
about that seam. It holds two formulas today:

``overnight_return``      only the half realized while the market is shut.
``mean_adjusted_return``  the undecomposed full-session mean the halves sum to,
                          on the provider-adjusted basis.

The second is the family's control rather than a stray demonstration kernel. It
is the only other extension kernel this build owns, it is the quantity the gap is
a component of, and giving one demonstration formula a module of its own would be
the empty-placeholder pattern the structure plan explicitly forbids. It reads a
dividend-adjusted basis where the gap reads a split-only one, and that difference
is one of the two open scientific questions about the gap rather than an accident
to be tidied away.

Nothing here is installed by being written. A kernel computes nothing until
``catalog`` registers it and some catalog revision names its ``formula_ref``; the
shipped desktop catalog names neither. ``overnight_return_21`` in particular
carries two unsettled Human decisions and is deliberately outside every admitted
axis.

Declarations live beside the code they describe -- each kernel's algorithm
summary, numeric domain and missing-value rules are stated here and assembled
into a registry entry by ``catalog``. The alternative, declaring them at the
composition site, is how a summary comes to describe a formula that was edited
somewhere else.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import numpy as np
import pandas as pd

from alphalattice.foundation.feature_engine.catalog.observation_clock import (
    NO_ECONOMIC_SKIP,
)
from alphalattice.kernel.quant.factor_contracts import FactorFamily, FactorSpec, FactorTrack

OPEN_INTRADAY_METHOD_FAMILY: Final = "OPEN_INTRADAY"
"""This family's key in ``producers.arithmetic_identity``.

Named here rather than only there so a reader of the formulas can see which
implementation closure they belong to without opening the identity module."""

MEAN_ADJUSTED_RETURN_ID: Final = "factor.desktop.experimental.mean_adjusted_return.v1"
OVERNIGHT_RETURN_ID: Final = "factor.desktop.experimental.overnight_return.v1"

OVERNIGHT_RETURN_FACTOR_ID: Final = "overnight_return_21"
OVERNIGHT_INTRADAY_TUG_OF_WAR_FACTOR_ID: Final = "overnight_intraday_tug_of_war_63"
GAP_ABSORPTION_FACTOR_ID: Final = "gap_absorption_63"

MEAN_ADJUSTED_RETURN_REQUIRED_FIELDS: Final = ("provider_adjusted_close",)
MEAN_ADJUSTED_RETURN_DECLARATION: Final[Mapping[str, str]] = {
    "implementation_id": MEAN_ADJUSTED_RETURN_ID,
    "algorithm": "mean(log(p_t/p_t-1), window=spec.window_sessions, skip=spec.lag_sessions)",
    "numeric_domain": "float64",
    "grouping": "listing_id",
    "ordering": "session_date",
}

OVERNIGHT_RETURN_REQUIRED_FIELDS: Final = ("close_split_adjusted", "open_split_adjusted")
OVERNIGHT_RETURN_DECLARATION: Final[Mapping[str, str]] = {
    "implementation_id": OVERNIGHT_RETURN_ID,
    "algorithm": (
        "mean(log(open_t/close_t-1), window=spec.window_sessions, skip=spec.lag_sessions)"
    ),
    # Screened before the division, so an infinite quotient never reaches `log`.
    # Declared as two rules because it is two: the legs must be finite and
    # positive, and the quotient must be finite as well.
    "non_finite_or_non_positive_leg": "nan",
    "non_finite_ratio": "nan",
    "numeric_domain": "float64",
    "grouping": "listing_id",
    "ordering": "session_date",
    "price_basis": "split_adjusted",
}

OVERNIGHT_REVERSAL_5_ID: Final = "factor.desktop.experimental.overnight_reversal_5.v1"
INTRADAY_REVERSAL_5_ID: Final = "factor.desktop.experimental.intraday_reversal_5.v1"

OVERNIGHT_REVERSAL_5_FACTOR_ID: Final = "overnight_reversal_5"
INTRADAY_REVERSAL_5_FACTOR_ID: Final = "intraday_reversal_5"

OVERNIGHT_REVERSAL_5_DECLARATION: Final[Mapping[str, str]] = {
    "implementation_id": OVERNIGHT_REVERSAL_5_ID,
    "algorithm": "-sum(log(open_t/close_t-1)) over 5 sessions ending at t",
    "non_finite_or_non_positive_leg": "nan",
    "non_finite_ratio": "nan",
    "numeric_domain": "float64",
    "grouping": "listing_id",
    "ordering": "session_date",
    "price_basis": "split_adjusted",
}
INTRADAY_REVERSAL_5_DECLARATION: Final[Mapping[str, str]] = {
    "implementation_id": INTRADAY_REVERSAL_5_ID,
    "algorithm": "-sum(log(close_t/open_t)) over 5 sessions ending at t",
    "non_finite_or_non_positive_leg": "nan",
    "non_finite_ratio": "nan",
    "numeric_domain": "float64",
    "grouping": "listing_id",
    "ordering": "session_date",
    "price_basis": "split_adjusted",
}

OVERNIGHT_INTRADAY_TUG_OF_WAR_ID: Final = (
    "factor.desktop.experimental.overnight_intraday_tug_of_war.v1"
)
GAP_ABSORPTION_ID: Final = "factor.desktop.experimental.gap_absorption.v1"
OPEN_INTRADAY_DECOMPOSITION_REQUIRED_FIELDS: Final = (
    "close_split_adjusted",
    "open_split_adjusted",
)
OVERNIGHT_INTRADAY_TUG_OF_WAR_DECLARATION: Final[Mapping[str, str]] = {
    "implementation_id": OVERNIGHT_INTRADAY_TUG_OF_WAR_ID,
    "algorithm": "(mean(gap)-mean(intraday))/sample_std(gap+intraday), window=63, skip=0",
    "price_basis": "split_adjusted",
    "numeric_domain": "float64",
}
GAP_ABSORPTION_DECLARATION: Final[Mapping[str, str]] = {
    "implementation_id": GAP_ABSORPTION_ID,
    "algorithm": "-sample_cov(gap,intraday)/sample_var(gap), window=63, skip=0",
    "price_basis": "split_adjusted",
    "numeric_domain": "float64",
}


def overnight_return_factor_spec() -> FactorSpec:
    """The typed recipe that names the overnight kernel, owned by this package.

    A kernel is mathematics; a ``FactorSpec`` decides *which* window, lag and
    input fields that mathematics runs with, and it is the thing a catalog
    revision actually installs. Both belong to the Feature capability, so this
    lives beside the kernel. A spec assembled in a test fixture would make "a
    researcher adds a method here" false the moment a second caller wanted the
    same factor, and would leave the recipe unreviewable in the product.

    Declared once and completely rather than derived by copying a shipped factor
    and overriding fields: a spec built with ``model_copy`` silently inherits
    whatever the donor happened to say about tolerances, track and literature,
    which is how a recipe acquires values nobody chose.

    ``family`` is ``MOMENTUM`` because the aggregation is a windowed mean return.
    What distinguishes this factor is *which part* of the return it measures --
    only the move realized while the market is shut -- not how it aggregates.
    """
    return FactorSpec(
        factor_id=OVERNIGHT_RETURN_FACTOR_ID,
        family=FactorFamily.MOMENTUM,
        formula_ref=OVERNIGHT_RETURN_ID,
        formula="mean(log(open_t/close_t-1)) over 21 sessions ending at t",
        window_sessions=21,
        # No economic skip. The gap that closes session t is realized at that
        # session's own open, so it belongs to observation session t.
        lag_sessions=NO_ECONOMIC_SKIP,
        # Split-adjusted on both legs, so the ratio survives a split. Deliberately
        # not the provider-adjusted series: that one is dividend-adjusted, and an
        # ex-dividend open would carry the drop into the gap.
        return_convention="split_adjusted_overnight_log_return",
        required_fields=OVERNIGHT_RETURN_REQUIRED_FIELDS,
        literature_sources=("https://doi.org/10.1093/rfs/hhl024",),
        # Ordered *source rows*, not gaps: 21 rolling gap observations ending at
        # the observation session, over a series whose first row has no prior
        # close and so yields no gap at all -- window + skip + 1 = 22. The three
        # counts measure three different things and only look contradictory when
        # the unit is left off, which is exactly how they were once read as a
        # conflict.
        minimum_observations=22,
        absolute_tolerance=1e-10,
        relative_tolerance=1e-10,
        track=FactorTrack.MODEL,
        # Never an anchor: an anchor is a factor the core bundle pins, and this
        # one is reachable only through a catalog revision that asks for it.
        core_anchor=False,
    )


def open_intraday_factor_specs() -> tuple[FactorSpec, ...]:
    """Every typed recipe this family can install, in factor-id order.

    ``mean_adjusted_return`` has no recipe here on purpose: it is registered as a
    kernel a catalog revision may name, and no recipe in this build asks for it.
    Installation stays explicit and stays someone else's decision.
    """
    return (
        gap_absorption_factor_spec(),
        intraday_reversal_5_factor_spec(),
        overnight_intraday_tug_of_war_factor_spec(),
        overnight_reversal_5_factor_spec(),
        overnight_return_factor_spec(),
    )


def overnight_reversal_5_factor_spec() -> FactorSpec:
    """Five sessions of the overnight leg alone, negated so the long side pays.

    ``overnight_return_21`` already measures this leg, and this is deliberately
    not a window variation of it. Measured on the installed target window the two
    behave differently in the way that matters: the twenty-one session mean has
    lost four fifths of its magnitude between sample halves while this one is
    flat, which is what a decaying published effect and a live one look like side
    by side. Five sessions is short enough that the crowding has not reached it.

    Names that have accumulated overnight gains give them back, so the quantity
    that predicts positively is the negation and that is what is published. The
    axis already reads this way -- ``residual_reversal_vol_scaled_5`` is
    ``-sum_5(...)`` for the same reason -- and curation admits only a strictly
    positive net long-short spread, so a Factor published on its losing sign is
    refused rather than held short.
    """
    return FactorSpec(
        factor_id=OVERNIGHT_REVERSAL_5_FACTOR_ID,
        family=FactorFamily.REVERSAL,
        formula_ref=OVERNIGHT_REVERSAL_5_ID,
        formula="-sum(log(open_t/close_t-1)) over 5 sessions ending at t",
        window_sessions=5,
        lag_sessions=NO_ECONOMIC_SKIP,
        # Split-adjusted, never provider-adjusted: the provider series is
        # dividend-adjusted and would carry an ex-dividend drop into the gap leg,
        # which is exactly the leg this factor is about.
        return_convention="split_adjusted_overnight_log_return",
        required_fields=OVERNIGHT_RETURN_REQUIRED_FIELDS,
        literature_sources=("https://doi.org/10.1093/rfs/hhl024",),
        # Five rolling gaps ending at the observation session, over a series
        # whose first row has no prior close and yields no gap: window + skip + 1.
        minimum_observations=6,
        absolute_tolerance=1e-10,
        relative_tolerance=1e-10,
        track=FactorTrack.MODEL,
        core_anchor=False,
    )


def intraday_reversal_5_factor_spec() -> FactorSpec:
    """Five sessions of the intraday leg alone, negated, the half nothing else measures.

    Every other past-return factor on the axis is built from close-to-close
    moves, which sum the two legs and report their difference. The legs carry
    opposite predictive content, so a close-to-close history is a blend of two
    opposing signals -- the same dilution the Target suffers, committed on the
    feature side.
    """
    return FactorSpec(
        factor_id=INTRADAY_REVERSAL_5_FACTOR_ID,
        family=FactorFamily.REVERSAL,
        formula_ref=INTRADAY_REVERSAL_5_ID,
        formula="-sum(log(close_t/open_t)) over 5 sessions ending at t",
        window_sessions=5,
        lag_sessions=NO_ECONOMIC_SKIP,
        return_convention="split_adjusted_open_intraday_log_return",
        required_fields=OPEN_INTRADAY_DECOMPOSITION_REQUIRED_FIELDS,
        literature_sources=("https://doi.org/10.1093/rfs/hhl024",),
        # The intraday leg needs no prior close, but the shared component builder
        # screens both legs together, so the same first row is unusable.
        minimum_observations=6,
        absolute_tolerance=1e-10,
        relative_tolerance=1e-10,
        track=FactorTrack.MODEL,
        core_anchor=False,
    )


def overnight_intraday_tug_of_war_factor_spec() -> FactorSpec:
    """Describe the standardized difference between overnight and intraday mean returns.

    Returns:
        The 63-session tug-of-war specification on split-adjusted open and close prices. Recipes
        retain their declared source fields, economic skips,
        minimum ordered observations, tolerances, and implementation references.
    """
    return FactorSpec(
        factor_id=OVERNIGHT_INTRADAY_TUG_OF_WAR_FACTOR_ID,
        family=FactorFamily.MOMENTUM,
        formula_ref=OVERNIGHT_INTRADAY_TUG_OF_WAR_ID,
        formula="(mean(g)-mean(d))/std(g+d), g=ln(O_j/C_j-1), d=ln(C_j/O_j)",
        window_sessions=63,
        lag_sessions=NO_ECONOMIC_SKIP,
        return_convention="split_adjusted_open_intraday_log_return",
        required_fields=OPEN_INTRADAY_DECOMPOSITION_REQUIRED_FIELDS,
        literature_sources=("https://doi.org/10.1093/rfs/hhl024",),
        minimum_observations=64,
        absolute_tolerance=1e-10,
        relative_tolerance=1e-10,
        track=FactorTrack.MODEL,
        core_anchor=False,
    )


def gap_absorption_factor_spec() -> FactorSpec:
    """Describe the negative intraday-on-overnight regression slope.

    Returns:
        The 63-session gap-absorption specification on split-adjusted prices. Recipes retain their
        declared source fields, economic skips,
        minimum ordered observations, tolerances, and implementation references.
    """
    return FactorSpec(
        factor_id=GAP_ABSORPTION_FACTOR_ID,
        family=FactorFamily.REVERSAL,
        formula_ref=GAP_ABSORPTION_ID,
        formula="-cov(g,d)/var(g), window=63, skip=0",
        window_sessions=63,
        lag_sessions=NO_ECONOMIC_SKIP,
        return_convention="split_adjusted_open_intraday_log_return",
        required_fields=OPEN_INTRADAY_DECOMPOSITION_REQUIRED_FIELDS,
        literature_sources=("https://doi.org/10.1093/rfs/hhl024",),
        minimum_observations=64,
        absolute_tolerance=1e-10,
        relative_tolerance=1e-10,
        track=FactorTrack.MODEL,
        core_anchor=False,
    )


def _open_intraday_components(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    ordered = source.assign(_source_position=np.arange(len(source), dtype=int)).sort_values(
        ["listing_id", "session_date"], kind="mergesort"
    )
    groups = ordered["listing_id"]
    opens = pd.to_numeric(ordered["open_split_adjusted"], errors="coerce").astype(float)
    closes = pd.to_numeric(ordered["close_split_adjusted"], errors="coerce").astype(float)
    prior_close = closes.groupby(groups, sort=False).shift(1)
    valid = (
        np.isfinite(opens)
        & np.isfinite(closes)
        & np.isfinite(prior_close)
        & (opens > 0.0)
        & (closes > 0.0)
        & (prior_close > 0.0)
    )
    gap_ratio = opens / prior_close.where(valid)
    day_ratio = closes / opens.where(valid)
    gap = np.log(gap_ratio.where(np.isfinite(gap_ratio) & (gap_ratio > 0.0)))
    intraday = np.log(day_ratio.where(np.isfinite(day_ratio) & (day_ratio > 0.0)))
    return ordered, gap, intraday


def _rolling_pair(
    source: pd.DataFrame,
    specification: FactorSpec,
    evaluator: object,
) -> pd.Series:
    ordered, gap, intraday = _open_intraday_components(source)
    values: np.ndarray = np.full(len(ordered), np.nan, dtype=float)
    for _listing, positions in ordered.groupby("listing_id", sort=False).indices.items():
        indexes: np.ndarray = np.asarray(positions, dtype=np.intp)
        g = gap.iloc[indexes].to_numpy(dtype=float)
        d = intraday.iloc[indexes].to_numpy(dtype=float)
        first = specification.window_sessions + specification.lag_sessions - 1
        for end in range(first, len(indexes)):
            # The window landing on observation session ``end`` ends at the
            # source row the recipe's declared economic skip names.
            stop = end - specification.lag_sessions + 1
            gw = g[stop - specification.window_sessions : stop]
            dw = d[stop - specification.window_sessions : stop]
            if not (np.isfinite(gw).all() and np.isfinite(dw).all()):
                continue
            value = float(evaluator(gw, dw))  # type: ignore[operator]
            if np.isfinite(value):
                values[int(indexes[end])] = value
    ordered["_computed"] = values
    restored = ordered.sort_values("_source_position", kind="mergesort")
    return pd.Series(restored["_computed"].to_numpy(dtype=float), index=source.index)


def overnight_intraday_tug_of_war(source: pd.DataFrame, specification: FactorSpec) -> pd.Series:
    """Standardize the overnight-minus-intraday mean difference by total-return volatility.

    Args:
        source: Listing/session rows with split-adjusted open and close prices.
        specification: Recipe whose window and economic skip select complete finite source pairs.

    Returns:
        The mean-leg difference divided by sample deviation of their sum; zero scale is missing.
        Results follow the input row axis; unavailable windows remain missing.

    Raises:
        KeyError: A required listing, session, or source column is absent.
    """

    def evaluate(gap: np.ndarray, intraday: np.ndarray) -> float:
        total_scale = float(np.std(gap + intraday, ddof=1))
        if total_scale <= 0.0:
            return float("nan")
        return float((np.mean(gap) - np.mean(intraday)) / total_scale)

    return _rolling_pair(source, specification, evaluate)


def overnight_reversal_5(source: pd.DataFrame, specification: FactorSpec) -> pd.Series:
    """Negate the summed overnight log returns over the recipe's window.

    Args:
        source: Listing/session rows with split-adjusted open and close prices.
        specification: Recipe whose window and economic skip select complete finite source pairs.

    Returns:
        Negated gap returns with the recipe's declared economic skip. Results follow the input row
        axis; unavailable windows remain missing.

    Raises:
        KeyError: A required listing, session, or source column is absent.
    """

    def evaluate(gap: np.ndarray, _intraday: np.ndarray) -> float:
        return -float(np.sum(gap))

    return _rolling_pair(source, specification, evaluate)


def intraday_reversal_5(source: pd.DataFrame, specification: FactorSpec) -> pd.Series:
    """Negate the summed intraday log returns over the recipe's window.

    Args:
        source: Listing/session rows with split-adjusted open and close prices.
        specification: Recipe whose window and economic skip select complete finite source pairs.

    Returns:
        Negated open-to-close returns with the recipe's declared economic skip. Results follow the
        input row axis; unavailable windows remain missing.

    Raises:
        KeyError: A required listing, session, or source column is absent.
    """

    def evaluate(_gap: np.ndarray, intraday: np.ndarray) -> float:
        return -float(np.sum(intraday))

    return _rolling_pair(source, specification, evaluate)


def gap_absorption(source: pd.DataFrame, specification: FactorSpec) -> pd.Series:
    """Measure the negative covariance slope of intraday returns on overnight gaps.

    Args:
        source: Listing/session rows with split-adjusted open and close prices.
        specification: Recipe whose window and economic skip select complete finite source pairs.

    Returns:
        Negative sample covariance of the legs divided by sample gap variance; zero variance is
        missing. Results follow the input row axis; unavailable windows remain missing.

    Raises:
        KeyError: A required listing, session, or source column is absent.
    """

    def evaluate(gap: np.ndarray, intraday: np.ndarray) -> float:
        variance = float(np.var(gap, ddof=1))
        if variance <= 0.0:
            return float("nan")
        covariance = float(np.cov(gap, intraday, ddof=1)[0, 1])
        return -covariance / variance

    return _rolling_pair(source, specification, evaluate)


def mean_adjusted_return(source: pd.DataFrame, specification: FactorSpec) -> pd.Series:
    """The undecomposed full-session mean log return, on the adjusted basis."""
    ordered = source.assign(_source_position=np.arange(len(source), dtype=int)).sort_values(
        ["listing_id", "session_date"]
    )

    prices = pd.to_numeric(ordered["provider_adjusted_close"], errors="coerce")
    returns = np.log(prices / prices.groupby(ordered["listing_id"], sort=False).shift(1))
    values = returns.groupby(ordered["listing_id"], sort=False).transform(
        lambda item: (
            item.shift(specification.lag_sessions)
            .rolling(
                specification.window_sessions,
                min_periods=specification.window_sessions,
            )
            .mean()
        )
    )
    ordered["_computed"] = values.to_numpy(dtype=float)
    restored = ordered.sort_values("_source_position")
    return pd.Series(restored["_computed"].to_numpy(dtype=float), index=source.index)


def overnight_return(source: pd.DataFrame, specification: FactorSpec) -> pd.Series:
    """Mean overnight log return: the close-to-open gap, averaged over the window.

    Every catalog factor measures a move between two closes, so the overnight and
    intraday halves of a return are nowhere separated. This measures only the
    half realized while the market is shut.

    Both legs are split-adjusted, so the ratio is clean across a split. It is not
    dividend-adjusted -- an ex-dividend open carries the drop -- which is a
    property of the split basis this reads, not of the aggregation.

    Every leg is screened before the division, not after it. A zero prior close
    with a positive open divides to ``+inf``, which passes a ``ratio > 0`` test
    and reaches ``log`` as an infinite gap, and a single infinite gap makes the
    whole rolling mean infinite. A Factor value that is not finite is not a
    measurement, so the ingredients are required finite and strictly positive and
    the quotient is required finite as well -- the last because two finite,
    positive legs can still overflow.
    """
    ordered = source.assign(_source_position=np.arange(len(source), dtype=int)).sort_values(
        ["listing_id", "session_date"]
    )
    opens = pd.to_numeric(ordered["open_split_adjusted"], errors="coerce").astype(float)
    closes = pd.to_numeric(ordered["close_split_adjusted"], errors="coerce").astype(float)
    prior_close = closes.groupby(ordered["listing_id"], sort=False).shift(1)
    # A non-positive or non-finite leg is missing evidence for a log gap, not a
    # value to shift into the domain: fail that session to NaN and let the
    # eligibility store record it, matching how the core materializer treats
    # zero volume.
    usable = np.isfinite(opens) & np.isfinite(prior_close) & (opens > 0.0) & (prior_close > 0.0)
    ratio = opens / prior_close.where(usable)
    overnight = np.log(ratio.where(np.isfinite(ratio) & (ratio > 0.0)))
    values = overnight.groupby(ordered["listing_id"], sort=False).transform(
        lambda item: (
            item.shift(specification.lag_sessions)
            .rolling(
                specification.window_sessions,
                min_periods=specification.window_sessions,
            )
            .mean()
        )
    )
    ordered["_computed"] = values.to_numpy(dtype=float)
    restored = ordered.sort_values("_source_position")
    return pd.Series(restored["_computed"].to_numpy(dtype=float), index=source.index)


__all__ = [
    "GAP_ABSORPTION_DECLARATION",
    "GAP_ABSORPTION_FACTOR_ID",
    "GAP_ABSORPTION_ID",
    "INTRADAY_REVERSAL_5_DECLARATION",
    "INTRADAY_REVERSAL_5_FACTOR_ID",
    "INTRADAY_REVERSAL_5_ID",
    "MEAN_ADJUSTED_RETURN_DECLARATION",
    "MEAN_ADJUSTED_RETURN_ID",
    "MEAN_ADJUSTED_RETURN_REQUIRED_FIELDS",
    "OPEN_INTRADAY_METHOD_FAMILY",
    "OVERNIGHT_INTRADAY_TUG_OF_WAR_DECLARATION",
    "OVERNIGHT_INTRADAY_TUG_OF_WAR_FACTOR_ID",
    "OVERNIGHT_INTRADAY_TUG_OF_WAR_ID",
    "OVERNIGHT_RETURN_DECLARATION",
    "OVERNIGHT_RETURN_FACTOR_ID",
    "OVERNIGHT_RETURN_ID",
    "OVERNIGHT_RETURN_REQUIRED_FIELDS",
    "OVERNIGHT_REVERSAL_5_DECLARATION",
    "OVERNIGHT_REVERSAL_5_FACTOR_ID",
    "OVERNIGHT_REVERSAL_5_ID",
    "gap_absorption",
    "gap_absorption_factor_spec",
    "intraday_reversal_5",
    "intraday_reversal_5_factor_spec",
    "mean_adjusted_return",
    "open_intraday_factor_specs",
    "overnight_intraday_tug_of_war",
    "overnight_intraday_tug_of_war_factor_spec",
    "overnight_return",
    "overnight_return_factor_spec",
    "overnight_reversal_5",
    "overnight_reversal_5_factor_spec",
]
