"""Vectorized pairwise cross-sectional rank-correlation primitives."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]


def pairwise_rank_correlations(ranked: FloatArray) -> FloatArray:
    """Compute common-finite correlations among ranked factor columns.

    Args:
        ranked: Rows by factor columns, with non-finite entries excluded pairwise.

    Returns:
        Symmetric factor correlation matrix; unresolved pairs are NaN.
    """
    finite = np.isfinite(ranked)
    mask = finite.astype(np.float64)
    values = np.where(finite, ranked, 0.0)
    count = mask.T @ mask
    left_sum = values.T @ mask
    square_sum = (values * values).T @ mask
    cross = values.T @ values
    with np.errstate(divide="ignore", invalid="ignore"):
        covariance = cross - left_sum * left_sum.T / count
        left_variance = square_sum - left_sum * left_sum / count
        denominator = np.sqrt(left_variance * left_variance.T)
        result = covariance / denominator
    result[(count < 2) | ~np.isfinite(result)] = np.nan
    return result


def average_ranks_vectorized(values: FloatArray) -> FloatArray:
    """Vectorized equivalent of the formal stable average-rank owner."""
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 1 or source.size == 0 or not np.isfinite(source).all():
        raise ValueError("average ranks require a non-empty finite vector")
    order = np.argsort(source, kind="stable")
    sorted_values = source[order]
    starts = np.flatnonzero(
        np.concatenate((np.asarray([True]), sorted_values[1:] != sorted_values[:-1]))
    )
    stops = np.concatenate((starts[1:], np.asarray([source.size])))
    averages = ((starts + 1) + stops) / 2.0
    ranks: FloatArray = np.empty(source.size, dtype=np.float64)
    ranks[order] = np.repeat(averages, stops - starts)
    ranks.setflags(write=False)
    return ranks


def stable_column_orders(values: FloatArray) -> npt.NDArray[np.intp]:
    """The stable ascending order of every column of one cross-section, NaN last.

    Sorted once per period: a subset of a column's rows sorts, stably, in the
    order this permutation lists them, so the ranks over any row mask are read
    off it without sorting again (``average_ranks_over_rows``).
    """
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 2:
        raise ValueError("column orders require a matrix")
    return np.argsort(source, axis=0, kind="stable")


def average_ranks_over_rows(
    values: FloatArray,
    orders: npt.NDArray[np.intp],
    mask: npt.NDArray[np.bool_],
    columns: Sequence[int],
) -> FloatArray:
    """``average_ranks_vectorized`` of ``values[mask, column]`` for every column, at once.

    Row ``i`` of the result is the rank vector of column ``columns[i]`` over
    the masked rows, in their row order, as its own contiguous vector -- the
    operand the one-column owner hands a reduction. The ranks are bitwise the
    one-column ones: filtering the column's stable order to the masked rows
    lists those rows exactly as the stable sort of the masked values alone
    does (removing rows changes no comparison between the rows that remain
    and keeps their original order among ties), every tie group takes the
    same ``((start + 1) + stop) / 2`` average, and the arithmetic is on
    exact integers. Every masked row of every requested column must be
    finite.
    """
    source = np.asarray(values, dtype=np.float64)
    selected: npt.NDArray[np.intp] = np.asarray(columns, dtype=np.intp)
    count = int(mask.sum())
    if source.ndim != 2 or selected.size == 0 or count == 0:
        raise ValueError("average ranks over rows require a matrix, columns and rows")
    column_orders = np.ascontiguousarray(orders[:, selected].T)
    kept = mask[column_orders]
    # Each column keeps exactly the masked rows; their positions in the
    # column's order are read in that order, giving the masked rows sorted.
    sorted_positions = np.nonzero(kept)[1].reshape(selected.size, count)
    sorted_rows = np.take_along_axis(column_orders, sorted_positions, axis=1)
    sorted_values = np.take_along_axis(
        np.ascontiguousarray(source[:, selected].T), sorted_rows, axis=1
    )
    if not np.isfinite(sorted_values).all():
        raise ValueError("average ranks over rows require finite masked values")
    new_group = np.ones(sorted_values.shape, dtype=np.bool_)
    new_group[:, 1:] = sorted_values[:, 1:] != sorted_values[:, :-1]
    positions: npt.NDArray[np.int64] = np.arange(count, dtype=np.int64)[None, :]
    starts = np.maximum.accumulate(np.where(new_group, positions, -1), axis=1)
    next_starts = np.minimum.accumulate(np.where(new_group, positions, count)[:, ::-1], axis=1)[
        :, ::-1
    ]
    stops = np.concatenate(
        (next_starts[:, 1:], np.full((selected.size, 1), count, dtype=np.int64)), axis=1
    )
    compact: npt.NDArray[np.intp] = np.cumsum(mask, dtype=np.intp) - 1
    ranks: FloatArray = np.empty((selected.size, count), dtype=np.float64)
    np.put_along_axis(ranks, compact[sorted_rows], ((starts + 1) + stops) / 2.0, axis=1)
    ranks.setflags(write=False)
    return ranks


def ordinal_percentile_columns(values: FloatArray) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
    """Per column: ``(ordinal rank + 0.5) / finite count`` of the finite entries.

    The stable ordinal ranks the turnover statistic is built on, for every
    column of a matrix at once. The ranks are taken among a column's finite
    entries alone, as the one-column rule takes them: every non-finite entry
    (NaN, ``+inf`` and ``-inf``, whichever side of the finite values it would
    sort to) is sorted as NaN, which a stable sort places last, so position
    ``i`` of the order is the column's ``i``-th finite value in the order the
    stable sort of the finite entries alone gives it -- the finite entries
    keep every comparison between them and their original order among ties.
    Non-finite entries stay NaN in the result and False in the mask; the
    source values are not touched.
    """
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 2:
        raise ValueError("ordinal percentile columns require a matrix")
    finite = np.isfinite(source)
    result: FloatArray = np.full(source.shape, np.nan, dtype=np.float64)
    if source.size:
        order = np.argsort(np.where(finite, source, np.nan), axis=0, kind="stable")
        positions = np.arange(source.shape[0], dtype=np.float64)[:, None]
        ranked = np.full(source.shape, np.nan, dtype=np.float64)
        np.put_along_axis(ranked, order, np.broadcast_to(positions, source.shape), axis=0)
        counts = finite.sum(axis=0).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(finite, (ranked + 0.5) / counts, np.nan)
    result.setflags(write=False)
    return result, finite


def centered_ranks(ranks: FloatArray) -> tuple[FloatArray, float]:
    """One side of a rank correlation: the centered ranks and their squared norm.

    Computed once per rank vector and reused against every partner, so a
    vector correlated with many others is centered once; the values are the
    ones the pairwise formula computed inline.
    """
    centered = ranks - float(np.mean(ranks))
    return centered, float(np.dot(centered, centered))


def correlation_from_centered_ranks(
    left_centered: FloatArray, left_norm: float, right_centered: FloatArray, right_norm: float
) -> float | None:
    """Correlate two pre-centered rank vectors from their squared norms.

    Args:
        left_centered: First centered rank vector.
        left_norm: Squared norm of the first vector.
        right_centered: Second centered rank vector.
        right_norm: Squared norm of the second vector.

    Returns:
        Clipped correlation, or None when the denominator is unusable.
    """
    denominator = math.sqrt(left_norm * right_norm)
    if denominator <= 0.0 or not math.isfinite(denominator):
        return None
    value = float(np.dot(left_centered, right_centered) / denominator)
    if not math.isfinite(value):
        return None
    clipped = max(-1.0, min(1.0, value))
    return 0.0 if clipped == 0.0 else clipped


def decile_spread_from_ranks(ranks: FloatArray, finite_targets: FloatArray) -> float | None:
    """The formal decile spread from average ranks already taken over the common rows."""
    percentiles = (ranks - 0.5) / ranks.size
    bottom = finite_targets[percentiles <= 0.1]
    top = finite_targets[percentiles >= 0.9]
    if bottom.size == 0 or top.size == 0:
        return None
    result = float(np.mean(top) - np.mean(bottom))
    if not math.isfinite(result):
        return None
    return 0.0 if result == 0.0 else result


def period_pair_correlation_matrix(
    values: FloatArray, *, minimum_common_count: int
) -> tuple[FloatArray, npt.NDArray[np.int64]]:
    """Pairwise common-finite Spearman of one period's columns, as a matrix.

    ``correlations[left, right]`` for ``left < right`` is the correlation of
    the two columns over the rows where both are finite, NaN when the pair is
    unresolved (fewer than ``minimum_common_count`` common rows, or a constant
    side); ``counts`` carries the common row count. Columns sharing one
    finite-row mask are ranked as a block and correlated by the matrix
    formula; a pair across two masks is ranked over the intersection and
    reduced pair by pair -- the two formulas the pairwise owner always used
    for those two cases, on the same operands.
    """
    finite = np.isfinite(values)
    groups: dict[bytes, list[int]] = defaultdict(list)
    packed = np.packbits(finite, axis=0)
    for column in range(values.shape[1]):
        groups[packed[:, column].tobytes()].append(column)
    columns = values.shape[1]
    correlations: FloatArray = np.full((columns, columns), np.nan, dtype=np.float64)
    counts = np.zeros((columns, columns), dtype=np.int64)
    group_values = list(groups.values())
    orders = stable_column_orders(values)
    for members in group_values:
        mask = finite[:, members[0]]
        common_count = int(mask.sum())
        if common_count < minimum_common_count or len(members) < 2:
            continue
        block = pairwise_rank_correlations(
            np.ascontiguousarray(average_ranks_over_rows(values, orders, mask, members).T)
        )
        index: npt.NDArray[np.intp] = np.asarray(members, dtype=np.intp)
        correlations[index[:, None], index[None, :]] = block
        counts[index[:, None], index[None, :]] = common_count

    for left_group_position, left_members in enumerate(group_values):
        for right_members in group_values[left_group_position + 1 :]:
            common = finite[:, left_members[0]] & finite[:, right_members[0]]
            common_count = int(common.sum())
            if common_count < minimum_common_count:
                continue
            # One ranking per side over the common rows and one centering per
            # column; each column is handed on as its own contiguous vector,
            # the operand the one-column path gave the reduction.
            left_ranks = [
                centered_ranks(ranks)
                for ranks in average_ranks_over_rows(values, orders, common, left_members)
            ]
            right_ranks = [
                centered_ranks(ranks)
                for ranks in average_ranks_over_rows(values, orders, common, right_members)
            ]
            for left_position, left in enumerate(left_members):
                left_centered, left_norm = left_ranks[left_position]
                for right_position, right in enumerate(right_members):
                    value = correlation_from_centered_ranks(
                        left_centered, left_norm, *right_ranks[right_position]
                    )
                    if value is not None:
                        correlations[left, right] = correlations[right, left] = value
                        counts[left, right] = counts[right, left] = common_count
    return correlations, counts


__all__ = [
    "average_ranks_over_rows",
    "average_ranks_vectorized",
    "centered_ranks",
    "correlation_from_centered_ranks",
    "decile_spread_from_ranks",
    "ordinal_percentile_columns",
    "pairwise_rank_correlations",
    "period_pair_correlation_matrix",
    "stable_column_orders",
]
