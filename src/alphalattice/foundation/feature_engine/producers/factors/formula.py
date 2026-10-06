"""The factor formula kernel: one callable computes every declared formula (EX, V88).

A formula factor is data. Its `FactorSpec` names `formula_ref: factor.formula` and carries its
canonical expression, and this one callable computes it, so adding a factor adds a catalog entry
and moves no code identity (a factor's identity is its id, its kernel's and its numerical spec).
A formula reading the Sector leaf is kept under a second registration of the same callable,
`factor.formula.sector`, whose fields add the Sector's: the plan derives which (V359), and the
price kernel, with every formula kept before, stays as it was.

The user's scale-invariant rule (2026-09-30): no source the product holds is point in time, so
the leaves -- the provider's split-basis `open`, `high`, `low`, `close` and `volume`, and
`adjusted_close` -- are admitted only in forms no later split or dividend changes. A later split
scales a listing's whole history by one factor (its prices by 1/r, its volumes by r) and a later
dividend scales its adjusted closes by one factor, so each node's exponents over those three
scales are checked, and the result must hold price and volume at equal exponents and the adjusted
close at none. A kernel sees one listing, so the cross-section is the preprocessing step's, never
a formula's; a window counts the listing's own rows, as every kernel's does; the lookback and the
skip fit the catalog's invalidation budget. The Sector leaf, `sector_return`, reads the day's
equal-weight log return of the listing's Sector (`sector_return_log`, built with the Sector in force
at each session since V346): point in time from T0, the first recorded classification backfilled
before it, which a study's temporal statement says (V347). It is a research leaf until the daily
build carries its Sector child (V359); the market and registered-factor leaves wait for
point-in-time sources (V345).

The point-in-time leaves (V345) -- `open_pit`, `high_pit`, `low_pit`, `close_pit` and
`volume_pit` -- read each session's values as they traded, re-based to the session the value is
for: a price at session t as it read at T is its as-traded price times the product of the
recorded ratios to t over that product to T, a share volume the reverse over the share splits
alone (a fractional ratio is a price adjustment, V348). No later event moves them, so any form of
them is admitted, a level and its bound included; mixed with a split-basis leaf, the scale rule
holds for that leaf. A formula reading one is kept under `factor.formula.point_in_time` (or
`factor.formula.sector.point_in_time` with the Sector leaf); the daily build carries the
as-traded fields for a catalog that reads them (V395), so it is activated as any formula is.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from functools import cache
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from alphalattice.foundation.feature_engine.catalog.observation_clock import formula_skip_sessions
from alphalattice.kernel.quant.factor_contracts import FactorSpec

from .formula_language import (
    EWM_SPAN,
    Constant,
    FormulaError,
    Leaf,
    Node,
    Tree,
    Units,
    canonical,
    check_causality,
    evaluate,
    leaves_of,
    lookback,
    parse,
    signatures,
    units,
)

FORMULA_ID: Final = "factor.formula"
FORMULA_SECTOR_ID: Final = "factor.formula.sector"
"""The kernel a formula reading the Sector leaf is kept under (V359)."""
FORMULA_POINT_IN_TIME_ID: Final = "factor.formula.point_in_time"
"""The kernel a formula reading a point-in-time leaf is kept under (V345)."""
FORMULA_SECTOR_POINT_IN_TIME_ID: Final = "factor.formula.sector.point_in_time"
"""The kernel a formula reading the Sector leaf and a point-in-time leaf is kept under (V345)."""
FORMULA_IDS: Final = (
    FORMULA_ID,
    FORMULA_SECTOR_ID,
    FORMULA_POINT_IN_TIME_ID,
    FORMULA_SECTOR_POINT_IN_TIME_ID,
)
FORMULA_POINT_IN_TIME_IDS: Final = (FORMULA_POINT_IN_TIME_ID, FORMULA_SECTOR_POINT_IN_TIME_ID)
"""The kernels whose formulas read a point-in-time leaf (V345)."""
FORMULA_RESEARCH_IDS: Final = (FORMULA_SECTOR_ID, FORMULA_SECTOR_POINT_IN_TIME_ID)
"""The kernels whose leaves the daily build does not carry yet: a formula reading the Sector leaf,
a research factor until the daily build carries its Sector child (V359)."""
FORMULA_METHOD_FAMILY: Final = "FACTOR_FORMULA"
FORMULA_LEAVES: Final[Mapping[str, str]] = {
    "open": "open_split_adjusted",
    "high": "high_split_adjusted",
    "low": "low_split_adjusted",
    "close": "close_split_adjusted",
    "volume": "volume_raw",
    "adjusted_close": "provider_adjusted_close",
    "sector_return": "sector_return_log",
    "open_pit": "open_as_traded",
    "high_pit": "high_as_traded",
    "low_pit": "low_as_traded",
    "close_pit": "close_as_traded",
    "volume_pit": "volume_as_traded",
}
"""Each leaf's source field: the provider's split basis (its OHLCV arrive split-adjusted as of
the download), its adjusted close, the day's Sector return (V359), and each session's values as
they traded (V345)."""
FORMULA_POINT_IN_TIME_INDEX: Final[Mapping[str, tuple[str, int]]] = {
    "open_pit": ("price_adjustment_index", 1),
    "high_pit": ("price_adjustment_index", 1),
    "low_pit": ("price_adjustment_index", 1),
    "close_pit": ("price_adjustment_index", 1),
    "volume_pit": ("share_count_index", -1),
}
"""Each point-in-time leaf's index and its exponent: the leaf at session t as read at T is its
as-traded value times (index(t) / index(T)) to that power, so a price is divided by the ratios
recorded after t up to T and a share volume multiplied by the share splits among them (V345)."""
FORMULA_SECTOR_FIELD: Final = "sector_return_log"
"""The Sector leaf's field: built across the cross-section before a kernel sees one listing
(`producers/sector_aggregates.py`), so a formula names it only where it reads it."""
FORMULA_REQUIRED_FIELDS: Final = tuple(
    sorted(
        FORMULA_LEAVES[leaf]
        for leaf in ("open", "high", "low", "close", "volume", "adjusted_close")
    )
)
FORMULA_SECTOR_REQUIRED_FIELDS: Final = tuple(
    sorted((*FORMULA_REQUIRED_FIELDS, FORMULA_SECTOR_FIELD))
)
FORMULA_POINT_IN_TIME_FIELDS: Final = tuple(
    sorted(
        {
            *(FORMULA_LEAVES[leaf] for leaf in FORMULA_POINT_IN_TIME_INDEX),
            *(index for index, _power in FORMULA_POINT_IN_TIME_INDEX.values()),
        }
    )
)
"""The as-traded fields and their indices (`publication/projection.py` `AsTradedBar`)."""
FORMULA_POINT_IN_TIME_REQUIRED_FIELDS: Final = tuple(
    sorted((*FORMULA_REQUIRED_FIELDS, *FORMULA_POINT_IN_TIME_FIELDS))
)
FORMULA_SECTOR_POINT_IN_TIME_REQUIRED_FIELDS: Final = tuple(
    sorted((*FORMULA_SECTOR_REQUIRED_FIELDS, *FORMULA_POINT_IN_TIME_FIELDS))
)
FORMULA_HISTORY_FLOOR: Final = 276
"""The most rows a formula reads, its skip included: the catalog's invalidation budget."""
FORMULA_RETURN_CONVENTION: Final = "scale_invariant_provider_formula"
FORMULA_PREPROCESSING_RECIPES: Final = (
    "ROBUST_SECTOR_NEUTRAL_Z",
    "ROBUST_UNIVERSE_Z",
    "TIME_SERIES_ABSOLUTE_STATE_ROBUST",
)
"""The preprocessing recipes a formula factor's declaration may choose: the installed ones that
take one factor (the joint-primary and interaction recipes need a verified child). The choice is
the author's: a quantity whose Sector level is structural (liquidity, turnover) takes the Sector
demean; a return, a price or a move size is better universe-centred (`ROBUST_UNIVERSE_Z`,
V346); the absolute-state recipe standardizes a listing over its own trailing year."""
FORMULA_DECLARATION: Final[Mapping[str, str]] = {
    "implementation_id": FORMULA_ID,
    "algorithm": "evaluate(spec.formula) over each listing's ordered rows, then skip",
    "numeric_domain": "float64",
    "grouping": "listing_id",
    "ordering": "session_date",
}
FORMULA_SECTOR_DECLARATION: Final[Mapping[str, str]] = {
    **FORMULA_DECLARATION,
    "implementation_id": FORMULA_SECTOR_ID,
}
FORMULA_POINT_IN_TIME_DECLARATION: Final[Mapping[str, str]] = {
    **FORMULA_DECLARATION,
    "implementation_id": FORMULA_POINT_IN_TIME_ID,
    "algorithm": (
        "evaluate(spec.formula) over each listing's ordered rows, each point-in-time leaf "
        "re-based to the session the value is for, then skip"
    ),
}
FORMULA_SECTOR_POINT_IN_TIME_DECLARATION: Final[Mapping[str, str]] = {
    **FORMULA_POINT_IN_TIME_DECLARATION,
    "implementation_id": FORMULA_SECTOR_POINT_IN_TIME_ID,
}
_ONE, _NONE = Fraction(1), Fraction(0)
_LEAF_UNITS: Final[Mapping[str, Units]] = {
    "open": (_ONE, _NONE, _NONE),
    "high": (_ONE, _NONE, _NONE),
    "low": (_ONE, _NONE, _NONE),
    "close": (_ONE, _NONE, _NONE),
    "volume": (_NONE, _ONE, _NONE),
    "adjusted_close": (_NONE, _NONE, _ONE),
    "sector_return": (_NONE, _NONE, _NONE),
    **dict.fromkeys(FORMULA_POINT_IN_TIME_INDEX, (_NONE, _NONE, _NONE)),
}
"""Exponents over the split-basis price, the split-basis volume and the adjusted close: none for a
point-in-time leaf, which no later event rescales."""
_SECTION: Final = frozenset({"rank", "zscore", "demean", "sector_demean", "winsorize"})


def formula_tree(formula: str, *, skip: int = 0) -> Tree:
    """A formula this kernel computes, checked before anything computes.

    Args:
        formula: The formula as written.
        skip: The factor's economic skip, in rows.

    Returns:
        Its tree.

    Raises:
        FormulaError: The language's refusals; `factor_formula.source_absent` for a formula that
            reads no leaf; `factor_formula.section_is_preprocessing:<op>` for a cross-section
            operator; `factor_formula.not_scale_invariant` for a result a later split or
            dividend would change; the causality refusals, the skip included.
    """
    tree = parse(formula, leaves=frozenset(FORMULA_LEAVES))
    if not leaves_of(tree):
        raise FormulaError("factor_formula.source_absent")
    for node in _nodes(tree):
        if node.operator in _SECTION:
            raise FormulaError(f"factor_formula.section_is_preprocessing:{node.operator}")
    price, volume, adjusted = units(tree, _LEAF_UNITS)
    if price != volume or adjusted != 0:
        raise FormulaError("factor_formula.not_scale_invariant")
    check_causality(
        tree, point_in_time=frozenset(FORMULA_LEAVES), floor=FORMULA_HISTORY_FLOOR - skip
    )
    return tree


def _nodes(tree: Tree) -> tuple[Node, ...]:
    if not isinstance(tree, Node):
        return ()
    return (tree, *(node for value in tree.arguments for node in _nodes(value)))


def formula_controls() -> dict[str, object]:
    """What an author needs to write a formula factor, as `feature controls` states it.

    Returns:
        The leaves and their fields, the operators a formula may use, the rules and what the
        plan derives.
    """
    kinds = signatures()
    return {
        "formula_ref": FORMULA_ID,
        "leaves": dict(FORMULA_LEAVES),
        "research_leaves": {
            "sector_return": (
                "the day's equal-weight log return of the listing's Sector, with the Sector in "
                "force at each session (V346); point in time from T0 and the first recorded "
                "classification backfilled before it, as the study's statement says; a factor "
                "reading it is tried and reviewed, and is activated once the daily build carries "
                "its Sector child (V359); the plan keeps it under factor.formula.sector"
            ),
            **{
                leaf: (
                    f"the session's {leaf.removesuffix('_pit')} as it traded, re-based to the "
                    "session the value is for, so no later split moves it and a level form (a "
                    "price floor, a dollar threshold) reads the price as it stood (V345); "
                    + (
                        "the share volume re-based by share splits alone: a fractional split "
                        "ratio is a price adjustment, and the provider's share volume before one "
                        "is assumed scaled by it as by a split, which cannot be checked offline "
                        "(V348); "
                        if leaf == "volume_pit"
                        else ""
                    )
                    + "the daily build carries the as-traded fields for a catalog that reads them "
                    "(V395), so a factor reading it is tried, reviewed and activated as any "
                    "formula is; the plan keeps it under factor.formula.point_in_time"
                )
                for leaf in FORMULA_POINT_IN_TIME_INDEX
            },
        },
        "operators": [*kinds["ELEMENT"], *kinds["SERIES"]],
        "infix": ["+", "-", "*", "/", "-x", ">", "<", ">=", "<=", "==", "and", "or", "not"],
        "history_rows": FORMULA_HISTORY_FLOOR,
        "rules": [
            "Read the split-basis leaves only in forms no later split or dividend changes: price "
            "and volume at equal exponents and the adjusted close at none (returns, ratios, "
            "dollar volume over its own mean); a level, its log or its bound is refused. A "
            "point-in-time leaf (open_pit, high_pit, low_pit, close_pit, volume_pit) is admitted "
            "in any form, a level and its bound included.",
            "A window counts the listing's own rows and reads none after its session; the "
            f"lookback plus the skip fits {FORMULA_HISTORY_FLOOR} rows.",
            "The cross-section belongs to preprocessing: "
            f"{', '.join(sorted(_SECTION))} are refused in a formula.",
            "A missing value, x / 0 and a value outside log's or sqrt's domain are missing and "
            "propagate; comparisons and logic give 1, 0 or missing.",
        ],
        "preprocessing_recipes": list(FORMULA_PREPROCESSING_RECIPES),
        "preprocessing_rule": (
            "Name `preprocessing_recipe` in the plan's edit, one of preprocessing_recipes: a "
            "quantity whose Sector level is structural (liquidity, turnover) takes the Sector "
            "demean (ROBUST_SECTOR_NEUTRAL_Z); TIME_SERIES_ABSOLUTE_STATE_ROBUST standardizes a "
            "listing over its own trailing year; a return, a price or a move size is better "
            "universe-centred (ROBUST_UNIVERSE_Z)."
        ),
        "derived": [
            "formula_ref (factor.formula; factor.formula.sector for a formula reading the Sector "
            "leaf, factor.formula.point_in_time for one reading a point-in-time leaf, "
            "factor.formula.sector.point_in_time for one reading both)",
            "formula (its canonical spelling)",
            "required_fields",
            "window_sessions",
            "minimum_observations",
            "return_convention",
        ],
        "example": "ts_corr(close / lag(close, 1), volume / lag(volume, 1), 20)",
    }


def formula_preprocessing(recipe: str | None) -> str:
    """The preprocessing recipe a formula factor's declaration chose, checked.

    Args:
        recipe: The declared recipe.

    Returns:
        It, one of `FORMULA_PREPROCESSING_RECIPES`.

    Raises:
        FormulaError: `factor_formula.preprocessing_recipe_required:<the choices>` when none is
            declared, `factor_formula.preprocessing_recipe_not_admitted:<recipe>` for another.
    """
    if recipe is None:
        raise FormulaError(
            "factor_formula.preprocessing_recipe_required:"
            + ",".join(FORMULA_PREPROCESSING_RECIPES)
        )
    if recipe not in FORMULA_PREPROCESSING_RECIPES:
        raise FormulaError(f"factor_formula.preprocessing_recipe_not_admitted:{recipe}")
    return recipe


def formula_specification(specification: FactorSpec) -> FactorSpec:
    """The spec a formula factor is kept as: its canonical formula and what the formula decides.

    The author states the id, the formula, the family, the skip, the literature, the tolerances
    and the track; the formula decides its kernel and the kernel's fields (V359), the window, the
    minimum rows and the return convention.

    Args:
        specification: The declared spec, `formula_ref` one of `FORMULA_IDS`.

    Returns:
        The spec with the canonical formula and the derived fields.

    Raises:
        FormulaError: As `formula_tree` refuses.
    """
    skip = formula_skip_sessions(specification)
    tree = formula_tree(specification.formula, skip=skip)
    window = lookback(tree) + 1
    derived: FactorSpec = FactorSpec.model_validate(
        {
            **specification.model_dump(mode="python"),
            "formula_ref": _formula_ref(tree),
            "formula": canonical(tree),
            "required_fields": formula_required_fields(tree),
            "window_sessions": window,
            "minimum_observations": window + skip,
            "return_convention": FORMULA_RETURN_CONVENTION,
        }
    )
    return derived


_KERNELS: Final[Mapping[tuple[bool, bool], tuple[str, tuple[str, ...]]]] = {
    (False, False): (FORMULA_ID, FORMULA_REQUIRED_FIELDS),
    (True, False): (FORMULA_SECTOR_ID, FORMULA_SECTOR_REQUIRED_FIELDS),
    (False, True): (FORMULA_POINT_IN_TIME_ID, FORMULA_POINT_IN_TIME_REQUIRED_FIELDS),
    (True, True): (FORMULA_SECTOR_POINT_IN_TIME_ID, FORMULA_SECTOR_POINT_IN_TIME_REQUIRED_FIELDS),
}
"""The kernel and its fields, by whether a formula reads the Sector leaf and a point-in-time one."""


def formula_required_fields(tree: Tree) -> tuple[str, ...]:
    """The fields a formula requires: its kernel's (V359, V345).

    Args:
        tree: The formula's tree.

    Returns:
        The price fields every formula before required, with the Sector field for a formula
        reading the Sector leaf and the as-traded fields for one reading a point-in-time leaf.
    """
    return _KERNELS[(_reads_sector(tree), _reads_point_in_time(tree))][1]


def _formula_ref(tree: Tree) -> str:
    return _KERNELS[(_reads_sector(tree), _reads_point_in_time(tree))][0]


def _reads_sector(tree: Tree) -> bool:
    return any(FORMULA_LEAVES[leaf] == FORMULA_SECTOR_FIELD for leaf in leaves_of(tree))


def _reads_point_in_time(tree: Tree) -> bool:
    return any(leaf in FORMULA_POINT_IN_TIME_INDEX for leaf in leaves_of(tree))


def compute_formula(source: pd.DataFrame, specification: FactorSpec) -> pd.Series:
    """Compute a formula factor over its listings' ordered rows.

    Each listing's k-th row sits in row k of one panel, so a window reads the listing's own
    rows and every operator runs once for all listings; the rows past a listing's end are empty
    and, the formula being causal, reach no earlier value.

    Args:
        source: `listing_id`, `session_date` and the kernel's fields.
        specification: The factor's spec, its formula canonical.

    Returns:
        One value per source row, in its order; NaN where the formula gives none and on each
        listing's rows short of its minimum observations.

    Raises:
        FormulaError: `factor_formula.not_canonical` for a formula kept in another spelling.
    """
    skip = formula_skip_sessions(specification)
    tree = formula_tree(specification.formula, skip=skip)
    if canonical(tree) != specification.formula:
        raise FormulaError("factor_formula.not_canonical")
    output = np.full(len(source), np.nan)
    if source.empty:
        return pd.Series(output, index=source.index, dtype=float)
    ordered = source.assign(_row=np.arange(len(source))).sort_values(
        ["listing_id", "session_date"], kind="mergesort"
    )
    depth = ordered.groupby("listing_id", sort=False).cumcount().to_numpy()
    column, listings = pd.factorize(ordered["listing_id"])
    shape = (int(depth.max()) + 1, len(listings))
    panels = {}
    for leaf in sorted(leaves_of(tree)):
        panel = np.full(shape, np.nan)
        field = pd.to_numeric(ordered[FORMULA_LEAVES[leaf]], errors="coerce")
        panel[depth, column] = field.to_numpy(dtype=float)
        panels[leaf] = panel
    if _reads_point_in_time(tree):
        indices = {}
        for name in sorted(
            {FORMULA_POINT_IN_TIME_INDEX[v][0] for v in panels if v in FORMULA_POINT_IN_TIME_INDEX}
        ):
            index = np.full(shape, np.nan)
            index[depth, column] = pd.to_numeric(ordered[name], errors="coerce").to_numpy(
                dtype=float
            )
            indices[name] = index
        result = _rebased_to_each_session(tree, panels, indices, skip)
    else:
        result = evaluate(tree, panels)
        if skip:
            result = np.vstack([np.full((skip, shape[1]), np.nan), result])[: shape[0]]
    result[: specification.minimum_observations - 1] = np.nan
    output[ordered["_row"].to_numpy()] = result[depth, column]
    return pd.Series(output, index=source.index, dtype=float)


def _rebased_to_each_session(
    tree: Tree,
    panels: Mapping[str, npt.NDArray[np.float64]],
    indices: Mapping[str, npt.NDArray[np.float64]],
    skip: int,
) -> npt.NDArray[np.float64]:
    """A formula's values with each point-in-time leaf re-based to the session a value is for.

    A listing's indices are constant between its recorded ratios, so its sessions fall into
    stretches; one evaluation per stretch re-bases every leaf to the stretch's indices, and each
    session takes its own stretch's value, read `skip` rows earlier as the factor's skip says.
    A missing index carries the one before it, so a missing row starts no stretch.

    Args:
        tree: The formula.
        panels: Each leaf's rows by listing, as `compute_formula` lays them out.
        indices: Each point-in-time index's rows, laid out the same way.
        skip: The factor's economic skip, in rows.

    Returns:
        The values, in the same layout, the skip applied.
    """
    carried = {name: pd.DataFrame(index).ffill().to_numpy() for name, index in indices.items()}
    rows, columns = next(iter(panels.values())).shape
    started = np.zeros((rows, columns), dtype=bool)
    for index in carried.values():
        started[1:] |= ~np.isclose(index[1:], index[:-1], rtol=0.0, atol=0.0, equal_nan=True)
    stretch = np.cumsum(started, axis=0)
    result = np.full((rows, columns), np.nan)
    every = np.arange(columns)
    for number in range(int(stretch.max()) + 1):
        member = stretch == number
        first = np.argmax(member, axis=0)
        held = member.any(axis=0)
        rebased = dict(panels)
        for leaf, (name, power) in FORMULA_POINT_IN_TIME_INDEX.items():
            if leaf not in panels:
                continue
            at = np.where(held, carried[name][first, every], np.nan)
            rebased[leaf] = panels[leaf] * (carried[name] / at) ** power
        values = evaluate(tree, rebased)
        if skip:
            values = np.vstack([np.full((skip, columns), np.nan), values])[:rows]
        result = np.where(member, values, result)
    return result


# The reference: the language's rules for one listing, in plain Python, sharing nothing with the
# numpy evaluation. A formula's goldens state its values from it, so the kernel is checked against
# an independent statement of each (`specifications.py`).


def reference_value(
    tree: Tree, series: Mapping[str, Sequence[float]], *, skip: int = 0
) -> float | None:
    """A formula's value on one listing's last row, by the language's rules.

    Args:
        tree: A formula this kernel computes.
        series: Each leaf's values for the listing, oldest first, all one length.
        skip: The factor's economic skip, in rows.

    Returns:
        The value, or None where the rules give none and short of the minimum rows.
    """
    length = len(next(iter(series.values())))
    if length < lookback(tree) + 1 + skip:
        return None

    @cache
    def at(node: Tree, row: int) -> float:
        return _reference(node, series, row, at, session=length - 1)

    value = at(tree, length - 1 - skip)
    return value if math.isfinite(value) else None


def _reference(
    tree: Tree,
    series: Mapping[str, Sequence[float]],
    row: int,
    at: Callable[[Tree, int], float],
    *,
    session: int,
) -> float:
    nan = math.nan
    if isinstance(tree, Leaf):
        values = series[tree.name]
        if not 0 <= row < len(values):
            return nan
        if tree.name not in FORMULA_POINT_IN_TIME_INDEX:
            return float(values[row])
        name, power = FORMULA_POINT_IN_TIME_INDEX[tree.name]
        index = series[name]
        return float(values[row]) * math.pow(float(index[row]) / float(index[session]), power)
    if isinstance(tree, Constant):
        return tree.value
    name, arguments = tree.operator, tree.arguments

    def value(index: int, back: int = 0) -> float:
        return at(arguments[index], row - back)

    def window(index: int, n: int) -> list[float] | None:
        values = [value(index, back) for back in range(n - 1, -1, -1)]
        return None if any(math.isnan(item) for item in values) else values

    def constant(index: int) -> float:
        node = arguments[index]
        assert isinstance(node, Constant)
        return node.value

    def truth(item: float) -> bool:
        return item != 0.0

    if name in {"neg", "not", "abs", "log", "sqrt", "sign", "clip"}:
        a = value(0)
        if math.isnan(a):
            return nan
        if name == "neg":
            return -a
        if name == "not":
            return 0.0 if truth(a) else 1.0
        if name == "abs":
            return abs(a)
        if name == "log":
            return math.log(a) if a > 0.0 else nan
        if name == "sqrt":
            return math.sqrt(a) if a >= 0.0 else nan
        if name == "sign":
            return 1.0 if a > 0.0 else -1.0 if a < 0.0 else 0.0
        return min(max(a, constant(1)), constant(2))
    if name == "where":
        condition = value(0)
        return nan if math.isnan(condition) else value(1) if truth(condition) else value(2)
    if name in _BINARY:
        a, b = value(0), value(1)
        return nan if math.isnan(a) or math.isnan(b) else _BINARY[name](a, b)
    if name in {"lag", "delta"}:
        earlier = value(0, int(constant(1)))
        return earlier if name == "lag" else value(0) - earlier
    if name == "ewm":
        halflife = constant(1)
        span = math.ceil(EWM_SPAN * halflife)
        rows = window(0, span)
        if rows is None:
            return nan
        weights = [math.pow(0.5, (span - 1 - k) / halflife) for k in range(span)]
        return math.fsum(w * v for w, v in zip(weights, rows, strict=True)) / math.fsum(weights)
    if name in {"ts_corr", "ts_cov", "ts_beta"}:
        return _moment(name, window(0, int(constant(2))), window(1, int(constant(2))))
    rows = window(0, int(constant(1)))
    return nan if rows is None else _WINDOW[name](rows)


def _divide(a: float, b: float) -> float:
    return math.nan if b == 0.0 else a / b


_BINARY: Final[Mapping[str, Callable[[float, float], float]]] = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": _divide,
    ">": lambda a, b: float(a > b),
    "<": lambda a, b: float(a < b),
    ">=": lambda a, b: float(a >= b),
    "<=": lambda a, b: float(a <= b),
    "==": lambda a, b: float(a == b),
    "and": lambda a, b: float(a != 0.0 and b != 0.0),
    "or": lambda a, b: float(a != 0.0 or b != 0.0),
    "max": lambda a, b: a if a >= b else b,
    "min": lambda a, b: a if a <= b else b,
}


def _moment(name: str, x: list[float] | None, y: list[float] | None) -> float:
    if x is None or y is None:
        return math.nan
    n = len(x)
    mean_x, mean_y = sum(x) / n, sum(y) / n
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y, strict=True)) / (n - 1)
    var_x = sum((a - mean_x) ** 2 for a in x) / (n - 1)
    var_y = sum((b - mean_y) ** 2 for b in y) / (n - 1)
    if name == "ts_cov":
        return cov
    if name == "ts_beta":  # ts_beta(y, x, n): the first argument's slope on the second
        return _divide(cov, var_y)
    return _divide(cov, math.sqrt(var_x * var_y))


def _std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((item - mean) ** 2 for item in values) / (len(values) - 1))


def _ts_rank(values: list[float]) -> float:
    last = values[-1]
    below = sum(1 for item in values if item < last)
    tied = sum(1 for item in values if item == last)
    return (below + (tied + 1) / 2.0) / len(values)


_WINDOW: Final[Mapping[str, Callable[[list[float]], float]]] = {
    "ts_mean": lambda values: sum(values) / len(values),
    "ts_sum": sum,
    "ts_min": min,
    "ts_max": max,
    "ts_std": _std,
    "ts_rank": _ts_rank,
}


def golden_series(rows: int) -> dict[str, tuple[float, ...]]:
    """Deterministic source rows for a formula's goldens: every field varies and none is zero.

    Args:
        rows: How many rows.

    Returns:
        Each source field's values, oldest first, the fields in order.
    """
    close = [100.0 * math.exp(0.001 * i + 0.01 * math.sin(0.7 * i)) for i in range(rows)]
    open_ = [value * (1.0 - 0.002 * math.cos(1.3 * i)) for i, value in enumerate(close)]
    return {
        "close_split_adjusted": tuple(close),
        "high_split_adjusted": tuple(max(o, c) * 1.003 for o, c in zip(open_, close, strict=True)),
        "low_split_adjusted": tuple(min(o, c) * 0.997 for o, c in zip(open_, close, strict=True)),
        "open_split_adjusted": tuple(open_),
        "provider_adjusted_close": tuple(value * 0.98 for value in close),
        "volume_raw": tuple(
            1e6 * (1.0 + 0.3 * math.sin(0.5 * i) + 0.1 * math.cos(1.7 * i)) for i in range(rows)
        ),
    }


def _point_in_time_series(source: Mapping[str, tuple[float, ...]]) -> dict[str, tuple[float, ...]]:
    """As-traded rows for the goldens, from the split-basis rows a provider would have stored.

    A 1.25 price adjustment a third of the way in and a 2-for-1 share split two thirds of the
    way in, each undone as the provider applied it: the prices times the ratios after a row, the
    volume divided by them; the price index the product of the ratios to a row, the share index
    the product of the share split alone (V345, V348).
    """
    rows = len(source["close_split_adjusted"])
    events = ((rows // 3, 1.25, False), (2 * rows // 3, 2.0, True))
    later = [math.prod(r for at, r, _ in events if at > row) for row in range(rows)]
    prices = {
        f"{name}_as_traded": tuple(
            value * scale
            for value, scale in zip(source[f"{name}_split_adjusted"], later, strict=True)
        )
        for name in ("open", "high", "low", "close")
    }
    return {
        **prices,
        "volume_as_traded": tuple(
            value / scale for value, scale in zip(source["volume_raw"], later, strict=True)
        ),
        "price_adjustment_index": tuple(
            math.prod(r for at, r, _ in events if at <= row) for row in range(rows)
        ),
        "share_count_index": tuple(
            math.prod(r for at, r, share in events if share and at <= row) for row in range(rows)
        ),
    }


def _sector_series(rows: int) -> tuple[float, ...]:
    """A Sector return series for the goldens: varying, never zero, its own frequencies."""
    return tuple(0.004 * math.sin(0.9 * i) + 0.0015 * math.cos(2.1 * i) + 1e-4 for i in range(rows))


def formula_goldens(
    specification: FactorSpec,
) -> tuple[tuple[str, str, dict[str, tuple[float, ...]], float | None], ...]:
    """The two boundary goldens a formula factor's specification states.

    Args:
        specification: The factor's spec, derived by `formula_specification`.

    Returns:
        For the first valid row and the row short of it: the label, the rationale, the source
        rows and the value the reference states (None for the short one).
    """
    skip = formula_skip_sessions(specification)
    tree = formula_tree(specification.formula, skip=skip)
    rows = specification.minimum_observations
    goldens = []
    for label, count, rationale in (
        (
            "first-valid-source-boundary",
            rows,
            "The first row with the formula's whole window and skip; the value is the "
            "language's reference evaluation, independent of the kernel.",
        ),
        (
            "one-row-short-is-missing",
            rows - 1,
            "One source row short of the window and skip leaves the formula missing.",
        ),
    ):
        source = golden_series(count)
        if FORMULA_SECTOR_FIELD in specification.required_fields:
            # The Sector leaf's rows, only for a formula that reads it (V359).
            source = dict(sorted({**source, FORMULA_SECTOR_FIELD: _sector_series(count)}.items()))
        if "close_as_traded" in specification.required_fields:
            # The as-traded rows, only for a formula that reads a point-in-time leaf (V345).
            source = dict(sorted({**source, **_point_in_time_series(source)}.items()))
        by_leaf = {leaf: source[field] for leaf, field in FORMULA_LEAVES.items() if field in source}
        by_leaf.update(
            {
                name: source[name]
                for name, _power in FORMULA_POINT_IN_TIME_INDEX.values()
                if name in source
            }
        )
        goldens.append((label, rationale, source, reference_value(tree, by_leaf, skip=skip)))
    return tuple(goldens)


__all__ = [
    "FORMULA_DECLARATION",
    "FORMULA_HISTORY_FLOOR",
    "FORMULA_ID",
    "FORMULA_IDS",
    "FORMULA_LEAVES",
    "FORMULA_METHOD_FAMILY",
    "FORMULA_POINT_IN_TIME_DECLARATION",
    "FORMULA_POINT_IN_TIME_FIELDS",
    "FORMULA_POINT_IN_TIME_ID",
    "FORMULA_POINT_IN_TIME_IDS",
    "FORMULA_POINT_IN_TIME_INDEX",
    "FORMULA_POINT_IN_TIME_REQUIRED_FIELDS",
    "FORMULA_PREPROCESSING_RECIPES",
    "FORMULA_REQUIRED_FIELDS",
    "FORMULA_RESEARCH_IDS",
    "FORMULA_RETURN_CONVENTION",
    "FORMULA_SECTOR_DECLARATION",
    "FORMULA_SECTOR_FIELD",
    "FORMULA_SECTOR_ID",
    "FORMULA_SECTOR_POINT_IN_TIME_DECLARATION",
    "FORMULA_SECTOR_POINT_IN_TIME_ID",
    "FORMULA_SECTOR_POINT_IN_TIME_REQUIRED_FIELDS",
    "FORMULA_SECTOR_REQUIRED_FIELDS",
    "compute_formula",
    "formula_controls",
    "formula_goldens",
    "formula_preprocessing",
    "formula_required_fields",
    "formula_specification",
    "formula_tree",
    "golden_series",
    "reference_value",
]
