"""The factor formula language: a factor declared as data (EX, its design's section 3).

A formula is an expression over leaves -- the observation fields, the Sector label, the market
reference return and numeric constants -- and a fixed set of operators. It is parsed into a tree
the operators' table checks, written back in one canonical form (the factor's identity: operand
order kept, the format normalized), checked for causality before anything computes, and
evaluated over a panel of sessions by listings with numpy, deterministically.

The rules the user confirmed (2026-09-30):

- element-wise: `+ - * /`, `-x`, `abs`, `log`, `sqrt`, `sign`, `clip(x, lo, hi)`, `max(a, b)`,
  `min(a, b)`, `where(c, a, b)`; comparisons `> < >= <= ==` and `and`, `or`, `not`;
- time series, per listing and backward only: `lag(x, n)`, `delta(x, n)`, `ts_mean`, `ts_std`,
  `ts_sum`, `ts_min`, `ts_max`, `ts_rank` (each `(x, n)`), `ts_corr(x, y, n)`,
  `ts_cov(x, y, n)`, `ts_beta(y, x, n)`, `ewm(x, halflife)`; `n` is a whole number from 1 to 504
  (from 2 for the statistics), and a window missing any value is NaN;
- cross-section, per session over that session's members: `rank`, `zscore`, `demean`,
  `sector_demean`, `winsorize(x, k)`;
- numeric edges: `x / 0`, `log` or `sqrt` outside its domain and a statistic whose deviation is
  0 give NaN, never an infinity; NaN propagates; comparisons and logic are three-valued (NaN in,
  NaN out, and `where(NaN, a, b)` is NaN); the cross-section operators read a session's non-NaN
  values and leave NaN NaN; `rank` and `ts_rank` give ties their average rank over the count of
  non-NaN values, so each lies in (0, 1];
- causality: a node consumes sessions up to T, never later (there is no lead); `lag` and
  `delta` reach back, a window reaches back its length less one, and the whole lookback fits
  the history floor (504 sessions).
"""

from __future__ import annotations

import ast
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

import numpy as np
import numpy.typing as npt

type Panel = npt.NDArray[np.float64]
"""Values by session (rows, oldest first) and listing (columns)."""

type Units = tuple[Fraction, ...]
"""A value's exponents over its leaves' scales, one per scale, in the caller's order."""

HISTORY_FLOOR: Final = 504
"""The most sessions a formula may reach back, its window's length included."""

EWM_SPAN: Final = 10
"""An `ewm` reads `EWM_SPAN` half-lives of history, rounded up: a bounded, causal window."""

_ALIASES: Final = {"delay": "lag", "ts_stddev": "ts_std", "cs_rank": "rank"}


@dataclass(frozen=True, slots=True)
class _Operator:
    kind: str
    """ELEMENT, SERIES or SECTION."""
    parameters: tuple[str, ...]
    """Each parameter's name, in its written order."""

    @property
    def roles(self) -> tuple[str, ...]:
        """Each parameter's role: `x` an expression, `n` a window, `h` a half-life, `k` a
        constant."""
        return tuple(_ROLES.get(name, "x") for name in self.parameters)


_ROLES: Final = {"n": "n", "halflife": "h", "k": "k", "lo": "k", "hi": "k"}
_OPERATORS: Final[dict[str, _Operator]] = {
    "abs": _Operator("ELEMENT", ("x",)),
    "log": _Operator("ELEMENT", ("x",)),
    "sqrt": _Operator("ELEMENT", ("x",)),
    "sign": _Operator("ELEMENT", ("x",)),
    "clip": _Operator("ELEMENT", ("x", "lo", "hi")),
    "max": _Operator("ELEMENT", ("a", "b")),
    "min": _Operator("ELEMENT", ("a", "b")),
    "where": _Operator("ELEMENT", ("c", "a", "b")),
    "lag": _Operator("SERIES", ("x", "n")),
    "delta": _Operator("SERIES", ("x", "n")),
    "ts_mean": _Operator("SERIES", ("x", "n")),
    "ts_std": _Operator("SERIES", ("x", "n")),
    "ts_sum": _Operator("SERIES", ("x", "n")),
    "ts_min": _Operator("SERIES", ("x", "n")),
    "ts_max": _Operator("SERIES", ("x", "n")),
    "ts_rank": _Operator("SERIES", ("x", "n")),
    "ts_corr": _Operator("SERIES", ("x", "y", "n")),
    "ts_cov": _Operator("SERIES", ("x", "y", "n")),
    "ts_beta": _Operator("SERIES", ("y", "x", "n")),
    "ewm": _Operator("SERIES", ("x", "halflife")),
    "rank": _Operator("SECTION", ("x",)),
    "zscore": _Operator("SECTION", ("x",)),
    "demean": _Operator("SECTION", ("x",)),
    "sector_demean": _Operator("SECTION", ("x",)),
    "winsorize": _Operator("SECTION", ("x", "k")),
}
_STATISTICS: Final = frozenset({"ts_std", "ts_corr", "ts_cov", "ts_beta"})


class FormulaError(ValueError):
    """A formula refused by name: the node, what it found and what is allowed (OP12)."""


# The tree: a leaf, a constant, or an operator over its arguments.


@dataclass(frozen=True, slots=True)
class Leaf:
    """A field, the Sector label or the market reference, by its formula name."""

    name: str


@dataclass(frozen=True, slots=True)
class Constant:
    """A numeric constant."""

    value: float


@dataclass(frozen=True, slots=True)
class Node:
    """An operator over its arguments, in their written order."""

    operator: str
    arguments: tuple[Leaf | Constant | Node, ...]


type Tree = Leaf | Constant | Node

_BINARY: Final = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
_COMPARE: Final = {ast.Gt: ">", ast.Lt: "<", ast.GtE: ">=", ast.LtE: "<=", ast.Eq: "=="}
_PRECEDENCE: Final = {
    "or": 1,
    "and": 2,
    "not": 3,
    ">": 4,
    "<": 4,
    ">=": 4,
    "<=": 4,
    "==": 4,
    "+": 5,
    "-": 5,
    "*": 6,
    "/": 6,
    "neg": 7,
}


def parse(text: str, *, leaves: frozenset[str]) -> Tree:
    """Parse a formula into its tree, refusing what the language does not hold.

    Args:
        text: The formula as written.
        leaves: The leaf names a formula may read.

    Returns:
        The tree.

    Raises:
        FormulaError: A syntax the language does not hold, an unknown name or operator, or an
            argument of the wrong role, each naming the node.
    """
    try:
        expression = ast.parse(text.strip(), mode="eval").body
    except SyntaxError as error:
        raise FormulaError(f"factor_formula.syntax_invalid:{error.offset or 0}") from error
    return _tree(expression, leaves)


def _tree(node: ast.expr, leaves: frozenset[str]) -> Tree:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        if isinstance(node.value, bool) or not math.isfinite(node.value):
            raise FormulaError("factor_formula.constant_invalid")
        return Constant(float(node.value))
    if isinstance(node, ast.Name):
        if node.id not in leaves:
            raise FormulaError(f"factor_formula.leaf_unknown:{node.id}")
        return Leaf(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _tree(node.operand, leaves)
        if isinstance(inner, Constant):
            return Constant(-inner.value)
        return Node("neg", (inner,))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return Node("not", (_tree(node.operand, leaves),))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return Node(_BINARY[type(node.op)], (_tree(node.left, leaves), _tree(node.right, leaves)))
    if isinstance(node, ast.BoolOp):
        name = "and" if isinstance(node.op, ast.And) else "or"
        values = [_tree(value, leaves) for value in node.values]
        tree = values[0]
        for value in values[1:]:
            tree = Node(name, (tree, value))
        return tree
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or type(node.ops[0]) not in _COMPARE:
            raise FormulaError("factor_formula.comparison_invalid")
        return Node(
            _COMPARE[type(node.ops[0])],
            (_tree(node.left, leaves), _tree(node.comparators[0], leaves)),
        )
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return _call(node, leaves)
    raise FormulaError(f"factor_formula.syntax_not_held:{type(node).__name__}")


def _call(node: ast.Call, leaves: frozenset[str]) -> Node:
    assert isinstance(node.func, ast.Name)
    name = _ALIASES.get(node.func.id, node.func.id)
    operator = _OPERATORS.get(name)
    if operator is None:
        raise FormulaError(f"factor_formula.operator_unknown:{node.func.id}")
    written: dict[str, ast.expr] = dict(zip(operator.parameters, node.args, strict=False))
    if len(node.args) > len(operator.parameters):
        raise FormulaError(
            f"factor_formula.arity:{name}:{len(node.args)}>{len(operator.parameters)}"
        )
    for keyword in node.keywords:
        if keyword.arg not in operator.parameters or keyword.arg in written:
            raise FormulaError(f"factor_formula.argument_unknown:{name}:{keyword.arg}")
        written[keyword.arg] = keyword.value
    missing = [value for value in operator.parameters if value not in written]
    if missing:
        raise FormulaError(f"factor_formula.argument_missing:{name}:{missing[0]}")
    arguments: list[Tree] = []
    for parameter, role in zip(operator.parameters, operator.roles, strict=True):
        tree = _tree(written[parameter], leaves)
        if role != "x" and not isinstance(tree, Constant):
            raise FormulaError(f"factor_formula.argument_not_constant:{name}:{parameter}")
        if role == "n":
            assert isinstance(tree, Constant)
            low = 2 if name in _STATISTICS else 1
            if not float(tree.value).is_integer() or not low <= tree.value <= HISTORY_FLOOR:
                raise FormulaError(
                    f"factor_formula.window_invalid:{name}:{_number(tree.value)}:"
                    f"{low}-{HISTORY_FLOOR}"
                )
        if role == "h":
            assert isinstance(tree, Constant)
            if not tree.value > 0.0 or math.ceil(EWM_SPAN * tree.value) > HISTORY_FLOOR:
                raise FormulaError(f"factor_formula.halflife_invalid:{_number(tree.value)}")
        if role == "x" and isinstance(tree, Constant) and operator.kind != "ELEMENT":
            raise FormulaError(f"factor_formula.series_of_a_constant:{name}")
        arguments.append(tree)
    return Node(name, tuple(arguments))


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def signatures() -> dict[str, tuple[str, ...]]:
    """Each named operator as written, by kind: ELEMENT, SERIES and SECTION.

    Returns:
        `name(parameters)` in name order; `n` is a whole-number window, `halflife` a positive
        constant, `lo`, `hi` and `k` constants, every other parameter an expression.
    """
    kinds: dict[str, tuple[str, ...]] = {}
    for name, operator in sorted(_OPERATORS.items()):
        signature = f"{name}({', '.join(operator.parameters)})"
        kinds[operator.kind] = (*kinds.get(operator.kind, ()), signature)
    return kinds


def canonical(tree: Tree) -> str:
    """The one spelling of a formula: its identity (operand order kept).

    Args:
        tree: A parsed formula.

    Returns:
        The canonical expression: single spaces around infix operators, parentheses only where
        precedence needs them, numbers in one spelling, keywords as positions, aliases resolved.
    """
    return _print(tree, 0)


def _print(tree: Tree, context: int) -> str:
    if isinstance(tree, Leaf):
        return tree.name
    if isinstance(tree, Constant):
        text = _number(tree.value)
        return f"({text})" if tree.value < 0 and context > 0 else text
    name = tree.operator
    if name in _OPERATORS:
        return f"{name}({', '.join(_print(value, 0) for value in tree.arguments)})"
    level = _PRECEDENCE[name]
    if name == "neg":
        text = f"-{_print(tree.arguments[0], level)}"
    elif name == "not":
        text = f"not {_print(tree.arguments[0], level)}"
    else:
        left, right = tree.arguments
        # Left-associative: a right operand of equal precedence keeps its parentheses.
        text = f"{_print(left, level)} {name} {_print(right, level + 1)}"
    return f"({text})" if level < context else text


def lookback(tree: Tree) -> int:
    """How many sessions before T a formula reads: its causal reach.

    Args:
        tree: A parsed formula.

    Returns:
        The sessions before T the deepest leaf is read at; 0 for a formula of session T alone.
    """
    if not isinstance(tree, Node):
        return 0
    inner = max((lookback(value) for value in tree.arguments), default=0)
    if tree.operator in {"lag", "delta"}:
        return inner + int(_window(tree))
    if tree.operator == "ewm":
        return inner + math.ceil(EWM_SPAN * _window(tree)) - 1
    if _OPERATORS.get(tree.operator, _Operator("ELEMENT", ())).kind == "SERIES":
        return inner + int(_window(tree)) - 1
    return inner


def _window(tree: Node) -> float:
    constant = tree.arguments[-1]
    assert isinstance(constant, Constant)
    return constant.value


def leaves_of(tree: Tree) -> frozenset[str]:
    """The leaves a formula reads.

    Args:
        tree: A parsed formula.

    Returns:
        Their names.
    """
    if isinstance(tree, Leaf):
        return frozenset({tree.name})
    if isinstance(tree, Node):
        return frozenset().union(*(leaves_of(value) for value in tree.arguments))
    return frozenset()


def check_causality(
    tree: Tree, *, point_in_time: frozenset[str], floor: int = HISTORY_FLOOR
) -> int:
    """Refuse a formula that could read what session T's close does not know.

    Args:
        tree: A parsed formula.
        point_in_time: The leaves the caller admits (their sources point in time, or read in
            forms their revisions cannot change).
        floor: The most sessions the formula may read, its own session included.

    Returns:
        Its lookback, within the history floor.

    Raises:
        FormulaError: A leaf from a source that is not point in time, or a lookback past the
            floor, naming what it found and what is allowed.
    """
    for name in sorted(leaves_of(tree)):
        if name not in point_in_time:
            raise FormulaError(f"factor_formula.leaf_not_point_in_time:{name}")
    reach = lookback(tree)
    if reach + 1 > floor:
        raise FormulaError(f"factor_formula.lookback_exceeds_floor:{reach + 1}>{floor}")
    return reach


def units(tree: Tree, leaf_units: Mapping[str, Units]) -> Units:
    """A formula's exponents over its leaves' scales, refusing what mixes them.

    Sums, differences, the larger or smaller of two values, a choice and a comparison take
    operands of one unit; a product and a quotient add and subtract exponents; `sqrt` halves
    them; a constant, a truth value, a rank and a correlation have none; `log` and `clip` read a
    value without units, since a level's log or bound moves when its scale does.

    Args:
        tree: A parsed formula.
        leaf_units: Each leaf's exponents, one per scale.

    Returns:
        The formula's exponents.

    Raises:
        FormulaError: `factor_formula.units_differ:<operator>` for operands of two units, or
            `factor_formula.level_use:<operator>` for a level where the operator reads none.
    """
    width = len(next(iter(leaf_units.values()), ()))
    none: Units = tuple(Fraction(0) for _ in range(width))
    if isinstance(tree, Leaf):
        return leaf_units[tree.name]
    if isinstance(tree, Constant):
        return none
    name = tree.operator
    parts = [
        units(value, leaf_units)
        for value, role in zip(tree.arguments, _roles(tree), strict=True)
        if role == "x"
    ]

    def same(*values: Units) -> Units:
        if len(set(values)) != 1:
            raise FormulaError(f"factor_formula.units_differ:{name}")
        return values[0]

    if name in {"+", "-", "max", "min", "demean", "sector_demean", "winsorize"}:
        return same(*parts)
    if name in {">", "<", ">=", "<=", "=="}:
        same(*parts)
        return none
    if name in {"and", "or", "not"}:
        same(none, *parts)
        return none
    if name == "where":
        same(none, parts[0])
        return same(parts[1], parts[2])
    if name == "*":
        return tuple(a + b for a, b in zip(parts[0], parts[1], strict=True))
    if name == "/":
        return tuple(a - b for a, b in zip(parts[0], parts[1], strict=True))
    if name == "sqrt":
        return tuple(value / 2 for value in parts[0])
    if name in {"log", "clip"}:
        if parts[0] != none:
            raise FormulaError(f"factor_formula.level_use:{name}")
        return none
    if name in {"sign", "ts_rank", "ts_corr", "rank", "zscore"}:
        return none
    if name == "ts_cov":
        return tuple(a + b for a, b in zip(parts[0], parts[1], strict=True))
    if name == "ts_beta":
        return tuple(a - b for a, b in zip(parts[0], parts[1], strict=True))
    # neg, abs, lag, delta, ts_mean, ts_std, ts_sum, ts_min, ts_max, ewm keep their unit.
    return parts[0]


# Evaluation: numpy over sessions by listings; NaN is the one missing value.


def evaluate(
    tree: Tree,
    leaves: Mapping[str, Panel],
    *,
    sectors: npt.NDArray[np.int64] | None = None,
) -> Panel:
    """Compute a formula over a panel, deterministically.

    Args:
        tree: A parsed formula.
        leaves: Each leaf's values, all of one shape (sessions, listings), oldest session first.
        sectors: Each cell's Sector as of its session, a code (-1 for none), for
            `sector_demean`.

    Returns:
        The formula's values, NaN where the rules give none.

    Raises:
        FormulaError: `sector_demean` with no Sector labels.
    """
    shape = next(iter(leaves.values())).shape
    with np.errstate(all="ignore"):
        value = _evaluate(tree, leaves, sectors, shape)
    value = np.where(np.isfinite(value), value, np.nan)
    return np.asarray(value, dtype=np.float64)


def _evaluate(
    tree: Tree,
    leaves: Mapping[str, Panel],
    sectors: npt.NDArray[np.int64] | None,
    shape: tuple[int, ...],
) -> Panel:
    if isinstance(tree, Leaf):
        return np.asarray(leaves[tree.name], dtype=np.float64)
    if isinstance(tree, Constant):
        return np.full(shape, tree.value, dtype=np.float64)
    args = [
        _evaluate(value, leaves, sectors, shape)
        for value, role in zip(tree.arguments, _roles(tree), strict=True)
        if role == "x"
    ]
    constants = [
        value.value
        for value, role in zip(tree.arguments, _roles(tree), strict=True)
        if role in {"n", "h", "k"} and isinstance(value, Constant)
    ]
    name = tree.operator
    if name in _INFIX:
        return _INFIX[name](*args)
    if name in _ELEMENT:
        return _ELEMENT[name](*args, *constants)
    if name == "sector_demean":
        if sectors is None:
            raise FormulaError("factor_formula.sector_labels_absent")
        return _sector_demean(args[0], sectors)
    if name in _SECTION:
        return _SECTION[name](*args, *constants)
    return _SERIES[name](*args, int(constants[-1]) if name != "ewm" else constants[-1])


def _roles(tree: Node) -> tuple[str, ...]:
    if tree.operator in _OPERATORS:
        return _OPERATORS[tree.operator].roles
    return ("x",) * len(tree.arguments)


def _nan_if(condition: npt.NDArray[np.bool_], value: Panel) -> Panel:
    return np.where(condition, np.nan, value)


def _three(value: Panel) -> Panel:
    """A truth value as 1.0, 0.0 or NaN."""
    return np.where(np.isnan(value), np.nan, (value != 0.0).astype(np.float64))


def _logic(left: Panel, right: Panel, both: Callable[..., npt.NDArray[np.bool_]]) -> Panel:
    missing = np.isnan(left) | np.isnan(right)
    return _nan_if(missing, both(_three(left) == 1.0, _three(right) == 1.0).astype(np.float64))


def _compare(left: Panel, right: Panel, relation: Callable[..., npt.NDArray[np.bool_]]) -> Panel:
    return _nan_if(np.isnan(left) | np.isnan(right), relation(left, right).astype(np.float64))


_INFIX: Final[dict[str, Callable[..., Panel]]] = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: _nan_if(b == 0.0, a / np.where(b == 0.0, 1.0, b)),
    "neg": lambda a: -a,
    ">": lambda a, b: _compare(a, b, np.greater),
    "<": lambda a, b: _compare(a, b, np.less),
    ">=": lambda a, b: _compare(a, b, np.greater_equal),
    "<=": lambda a, b: _compare(a, b, np.less_equal),
    "==": lambda a, b: _compare(a, b, np.equal),
    "and": lambda a, b: _logic(a, b, np.logical_and),
    "or": lambda a, b: _logic(a, b, np.logical_or),
    "not": lambda a: _nan_if(np.isnan(a), 1.0 - _three(a)),
}

_ELEMENT: Final[dict[str, Callable[..., Panel]]] = {
    "abs": np.abs,
    "log": lambda a: _nan_if(~(a > 0.0), np.log(np.where(a > 0.0, a, 1.0))),
    "sqrt": lambda a: _nan_if(~(a >= 0.0), np.sqrt(np.where(a >= 0.0, a, 0.0))),
    "sign": np.sign,
    "clip": lambda a, lo, hi: _nan_if(np.isnan(a), np.clip(a, lo, hi)),
    "max": lambda a, b: np.maximum(a, b),
    "min": lambda a, b: np.minimum(a, b),
    "where": lambda c, a, b: _nan_if(np.isnan(c), np.where(_three(c) == 1.0, a, b)),
}


def _windows(value: Panel, n: int) -> npt.NDArray[np.float64]:
    """Each session's backward window of `n` sessions (the first `n - 1` padded with NaN)."""
    padded = np.vstack([np.full((n - 1, value.shape[1]), np.nan), value])
    return np.lib.stride_tricks.sliding_window_view(padded, n, axis=0)


def _complete(window: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    return ~np.isnan(window).any(axis=-1)


def _ts_rank(value: Panel, n: int) -> Panel:
    window = _windows(value, n)
    last = window[..., -1:]
    below = (window < last).sum(axis=-1)
    tied = (window == last).sum(axis=-1)
    rank = (below + (tied + 1) / 2.0) / n
    return _nan_if(~_complete(window), rank)


def _moment(x: Panel, y: Panel, n: int, kind: str) -> Panel:
    wx, wy = _windows(x, n), _windows(y, n)
    complete = _complete(wx) & _complete(wy)
    dx = wx - wx.mean(axis=-1, keepdims=True)
    dy = wy - wy.mean(axis=-1, keepdims=True)
    cov = (dx * dy).sum(axis=-1) / (n - 1)
    vx = (dx * dx).sum(axis=-1) / (n - 1)
    vy = (dy * dy).sum(axis=-1) / (n - 1)
    if kind == "cov":
        result, zero = cov, np.zeros_like(cov, dtype=bool)
    elif kind == "beta":  # ts_beta(y, x, n): the slope of y on x
        result, zero = cov / np.where(vy == 0.0, 1.0, vy), vy == 0.0
    else:
        denominator = np.sqrt(vx * vy)
        result, zero = cov / np.where(denominator == 0.0, 1.0, denominator), denominator == 0.0
    return _nan_if(~complete | zero, result)


def _ewm(value: Panel, halflife: float) -> Panel:
    span = math.ceil(EWM_SPAN * halflife)
    window = _windows(value, span)
    weights = 0.5 ** (np.arange(span - 1, -1, -1, dtype=np.float64) / halflife)
    return _nan_if(~_complete(window), (window * weights).sum(axis=-1) / weights.sum())


def _series(reduce: Callable[..., npt.NDArray[np.float64]]) -> Callable[[Panel, int], Panel]:
    def run(value: Panel, n: int) -> Panel:
        window = _windows(value, n)
        return _nan_if(~_complete(window), reduce(window))

    return run


def _lag(value: Panel, n: int) -> Panel:
    return np.vstack([np.full((n, value.shape[1]), np.nan), value[:-n]])[: value.shape[0]]


_SERIES: Final[dict[str, Callable[..., Panel]]] = {
    "lag": _lag,
    "delta": lambda value, n: value - _lag(value, n),
    "ts_mean": _series(lambda w: w.mean(axis=-1)),
    "ts_std": lambda value, n: _std(value, n),
    "ts_sum": _series(lambda w: w.sum(axis=-1)),
    "ts_min": _series(lambda w: w.min(axis=-1)),
    "ts_max": _series(lambda w: w.max(axis=-1)),
    "ts_rank": _ts_rank,
    "ts_corr": lambda x, y, n: _moment(x, y, n, "corr"),
    "ts_cov": lambda x, y, n: _moment(x, y, n, "cov"),
    "ts_beta": lambda y, x, n: _moment(y, x, n, "beta"),
    "ewm": _ewm,
}


def _std(value: Panel, n: int) -> Panel:
    window = _windows(value, n)
    return _nan_if(~_complete(window), window.std(axis=-1, ddof=1))


def _section_rank(value: Panel) -> Panel:
    from scipy.stats import rankdata  # type: ignore[import-untyped]

    result = np.full(value.shape, np.nan)
    for row in range(value.shape[0]):
        present = ~np.isnan(value[row])
        count = int(present.sum())
        if count:
            # Ties take their average rank, over the count of the session's values.
            result[row, present] = rankdata(value[row, present], method="average") / count
    return result


def _section(reduce: Callable[[Panel, npt.NDArray[np.float64]], Panel]) -> Callable[..., Panel]:
    def run(value: Panel, *constants: float) -> Panel:
        result = np.full(value.shape, np.nan)
        for row in range(value.shape[0]):
            present = ~np.isnan(value[row])
            if present.any():
                result[row, present] = reduce(value[row, present], np.asarray(constants))
        return result

    return run


def _zscore(values: npt.NDArray[np.float64], _k: npt.NDArray[np.float64]) -> Panel:
    if len(values) < 2:
        return np.full(values.shape, np.nan)
    std = values.std(ddof=1)
    return np.full(values.shape, np.nan) if std == 0.0 else (values - values.mean()) / std


def _winsorize(values: npt.NDArray[np.float64], k: npt.NDArray[np.float64]) -> Panel:
    if len(values) < 2:
        return values.copy()
    mean, std = values.mean(), values.std(ddof=1)
    return np.clip(values, mean - float(k[0]) * std, mean + float(k[0]) * std)


_SECTION: Final[dict[str, Callable[..., Panel]]] = {
    "rank": lambda value: _section_rank(value),
    "zscore": _section(_zscore),
    "demean": _section(lambda values, _k: values - values.mean()),
    "winsorize": _section(_winsorize),
}


def _sector_demean(value: Panel, sectors: npt.NDArray[np.int64]) -> Panel:
    result = np.full(value.shape, np.nan)
    for row in range(value.shape[0]):
        for sector in np.unique(sectors[row]):
            if sector < 0:
                continue
            members = (sectors[row] == sector) & ~np.isnan(value[row])
            if members.any():
                result[row, members] = value[row, members] - value[row, members].mean()
    return result


__all__ = [
    "EWM_SPAN",
    "HISTORY_FLOOR",
    "Constant",
    "FormulaError",
    "Leaf",
    "Node",
    "Panel",
    "Tree",
    "Units",
    "canonical",
    "check_causality",
    "evaluate",
    "leaves_of",
    "lookback",
    "parse",
    "signatures",
    "units",
]
