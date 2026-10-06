"""LightGBM's threads: the operator's, behind a sealed canary (binding plan, B3).

With ``deterministic`` and ``force_col_wise`` LightGBM grows the same trees at any thread
count; two things it writes still move with the count. Its model text records
``num_threads`` in a parameter section, so an adapter stores the model text without that
section (`model_trees`) and the estimator's content hash covers the trees. Its built-in
metrics sum in parallel, so the last bits of a validation l2 move with the count (1-16
threads measured, 2026-09-26); the l2 early stopping reads is computed here in the order
LightGBM sums at one thread (`sequential_l2`), which reproduces that curve bit for bit.

A session proves the canary the first time it fits at a thread count other than one: a
fixed training set, grown and early-stopped at that count, must give the sealed trees,
curve and predictions (`LIGHTGBM_THREAD_CANARIES`, sealed at one thread), or the fit is
refused by name. A LightGBM with no sealed canary of its own first grows the canary on one
thread: when that equals a sealed version's, it grows that version's trees and is proven
against it like one (a dependency change keeps its threads, PA3); otherwise it fits on one
thread.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import version
from threading import Lock
from typing import Any, Final

import numpy as np
import numpy.typing as npt

LIGHTGBM_THREAD_CANARIES: Final[dict[str, str]] = {
    # lightgbm version: SHA-256 of the canary's trees, validation curve and predictions,
    # fitted on one thread; 2, 4, 8 and 16 threads gave the same bytes when sealed.
    "4.7.0": "cfdb82d09be8eb356a80097a631e43adbc7c0e20223d0a115426e02713915337",
}
"""The sealed canary of each LightGBM version this build has proven."""

_PARAMETER_SECTION = re.compile(r"\nparameters:\n.*?\nend of parameters\n", re.S)
_FIT_THREADS: ContextVar[int] = ContextVar("lightgbm_fit_threads", default=1)
_PROVEN: set[int] = {1}
_PROVING = Lock()
_INHERITED: dict[str, str | None] = {}
"""An unsealed version's one-thread canary when it equals a sealed one's, else None."""


class LightGBMThreadCanaryMismatch(ValueError):
    """Signal that LightGBM grew different trees at the requested thread count.

    A refusal whose text is its code, as a study stage records it: the Task blocks under
    this name rather than reading as an interruption a recovery would only meet again.
    """

    failure_class = "OPERATIONAL_FAILURE"
    code = "alpha_modeling.lightgbm_thread_canary_mismatch"

    def __init__(self, threads: int) -> None:
        """Record the refused thread count and stable failure code."""
        self.threads = threads
        super().__init__(self.code)


@dataclass(frozen=True, slots=True)
class LightGBMThreads:
    """The threads the fits in scope use, and why."""

    requested: int
    threads: int
    reason: str


def model_trees(model_text: str) -> str:
    """Remove the thread-count parameter section from LightGBM model text.

    The remaining text preserves the trees and is loadable by LightGBM.

    Args:
        model_text: Full serialized LightGBM model.

    Returns:
        Model text without its parameter section.

    """
    return _PARAMETER_SECTION.sub("\n", model_text, count=1)


def sequential_l2(predictions: npt.NDArray[np.float64], dataset: Any) -> tuple[str, float, bool]:
    """Compute validation l2 in LightGBM's one-thread accumulation order.

    Squared differences between raw scores and float32 labels are accumulated
    in row order, then divided by the row count.

    Args:
        predictions: Raw validation scores.
        dataset: LightGBM dataset supplying validation labels.

    Returns:
        Metric name, l2 value, and the lower-is-better flag.

    """
    labels = np.asarray(dataset.get_label(), dtype=np.float64)
    difference = np.asarray(predictions, dtype=np.float64) - labels
    return "l2", float(np.cumsum(difference * difference)[-1] / float(difference.shape[0])), False


def lightgbm_fit_threads() -> int:
    """Return the thread count for this fit context, or one outside its scope."""
    return _FIT_THREADS.get()


def _canary_values(rows: int, features: int, offset: int) -> npt.NDArray[np.float64]:
    # Integer mixing, then an exact scaling: the same values on every machine.
    index = np.arange(rows * features, dtype=np.uint64) + np.uint64(offset)
    state = index * np.uint64(0x9E3779B97F4A7C15)
    state ^= state >> np.uint64(31)
    state *= np.uint64(0xBF58476D1CE4E5B9)
    state ^= state >> np.uint64(29)
    values = (state >> np.uint64(11)).astype(np.float64) / float(2**53) - 0.5
    return values.reshape(rows, features)


def _canary_target(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    noise = x[:, -1] * 0.8
    return x[:, 0] * 0.5 - x[:, 1] * x[:, 2] + 0.25 * x[:, 3] + noise


def canary_digest(threads: int) -> str:
    """Grow and early-stop the canary at `threads`; the digest of what it wrote."""
    lightgbm = import_module("lightgbm")
    train_x = _canary_values(30_000, 24, 0)
    valid_x = _canary_values(10_000, 24, 30_000 * 24)
    dataset_params = {
        "min_data_in_leaf": 20,
        "seed": 1729,
        "data_random_seed": 1729,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": threads,
        "verbosity": -1,
    }
    train = lightgbm.Dataset(train_x, label=_canary_target(train_x), params=dataset_params)
    valid = lightgbm.Dataset(valid_x, label=_canary_target(valid_x), reference=train)
    params = {
        **dataset_params,
        "objective": "l2",
        "metric": "None",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "max_depth": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "bagging_seed": 1729,
        "feature_fraction_seed": 1729,
    }
    curve: dict[str, dict[str, list[float]]] = {}
    booster = lightgbm.train(
        params,
        train,
        num_boost_round=300,
        valid_sets=[valid],
        feval=sequential_l2,
        callbacks=[
            lightgbm.record_evaluation(curve),
            lightgbm.early_stopping(20, first_metric_only=True, verbose=False),
        ],
    )
    iterations = int(booster.best_iteration)
    digest = hashlib.sha256()
    digest.update(model_trees(str(booster.model_to_string(num_iteration=iterations))).encode())
    digest.update(np.asarray(curve["valid_0"]["l2"], dtype=np.float64).tobytes())
    predictions = booster.predict(valid_x, num_iteration=iterations)
    digest.update(np.asarray(predictions, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _inherited_canary(installed: str) -> str | None:
    """The sealed canary an unsealed LightGBM grows on one thread, or None when it grows none."""
    with _PROVING:
        if installed not in _INHERITED:
            digest = canary_digest(1)
            _INHERITED[installed] = digest if digest in LIGHTGBM_THREAD_CANARIES.values() else None
        return _INHERITED[installed]


@contextmanager
def lightgbm_threads(requested: int) -> Iterator[LightGBMThreads]:
    """Scope LightGBM fits to a thread count admitted by the sealed canary.

    An unsealed LightGBM version that grows a sealed version's canary on one thread is
    proven against it; one that grows none uses one thread. A mismatched canary refuses
    the requested count before the fit begins.

    Args:
        requested: Desired thread count, clamped to at least one.

    Yields:
        The admitted count and the reason for that choice.

    Raises:
        LightGBMThreadCanaryMismatch: The sealed canary differs at this count.

    """
    requested = max(1, int(requested))
    installed = version("lightgbm")
    sealed = LIGHTGBM_THREAD_CANARIES.get(installed)
    if requested != 1 and sealed is None:
        sealed = _inherited_canary(installed)
    if requested == 1:
        chosen = LightGBMThreads(requested, 1, "one thread")
    elif sealed is None:
        chosen = LightGBMThreads(
            requested, 1, f"one thread: no sealed canary for lightgbm {installed}"
        )
    else:
        with _PROVING:
            if requested not in _PROVEN:
                if canary_digest(requested) != sealed:
                    raise LightGBMThreadCanaryMismatch(requested)
                _PROVEN.add(requested)
        chosen = LightGBMThreads(
            requested, requested, f"{requested} threads; the canary is proven at that count"
        )
    token = _FIT_THREADS.set(chosen.threads)
    try:
        yield chosen
    finally:
        _FIT_THREADS.reset(token)


__all__ = [
    "LIGHTGBM_THREAD_CANARIES",
    "LightGBMThreadCanaryMismatch",
    "LightGBMThreads",
    "canary_digest",
    "lightgbm_fit_threads",
    "lightgbm_threads",
    "model_trees",
    "sequential_l2",
]
