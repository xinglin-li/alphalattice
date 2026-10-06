"""Frozen evaluation statistics for WP60D return-model evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NoReturn

import numpy as np
import numpy.typing as npt
from pydantic import JsonValue

from alphalattice.kernel.knowledge.catalog import SystemResourceCatalog
from alphalattice.kernel.knowledge.contracts import PackageArgSpecRef
from alphalattice.kernel.quant.return_model_contracts import ReturnModelPackageCall
from alphalattice.kernel.quant.return_models import audit_return_model_package_call
from alphalattice.kernel.validation.errors import ValidationProtocolError

FloatArray = npt.NDArray[np.float64]
SEED = 1729
BOOTSTRAP_RESAMPLES = 2_000


@dataclass(frozen=True, slots=True)
class DMResult:
    """Diebold-Mariano comparison with audited numerical package calls."""

    statistic: float
    p_value: float
    lag: int
    package_calls: tuple[ReturnModelPackageCall, ...]


def _error(message: str, cause: BaseException | None = None) -> NoReturn:
    chain = () if cause is None else (f"{type(cause).__name__}: {cause}",)
    raise ValidationProtocolError(
        message,
        code="validation.return_model_incomplete",
        cause_chain=chain,
    ) from cause


def automatic_newey_west_lag(periods: int) -> int:
    """Choose the deterministic HAC lag for a period count.

    Args:
        periods: Number of loss-differential observations.

    Returns:
        Nonnegative Newey-West lag.

    Raises:
        ValidationProtocolError: If no observations are available.

    """
    if periods < 1:
        _error("return-model HAC lag requires observations")
    lag = math.floor(4.0 * (periods / 100.0) ** (2.0 / 9.0))
    assert isinstance(lag, int)
    return lag


def diebold_mariano_hac(
    model_losses: npt.ArrayLike,
    benchmark_losses: npt.ArrayLike,
    *,
    package_catalog: SystemResourceCatalog | None = None,
    package_arg_refs: tuple[PackageArgSpecRef, ...] = (),
    package_calls: tuple[ReturnModelPackageCall, ...] = (),
) -> DMResult:
    """Compare aligned loss paths with an audited HAC calculation.

    Args:
        model_losses: Losses from the candidate return model.
        benchmark_losses: Aligned benchmark losses.
        package_catalog: Catalog used when package calls must be audited.
        package_arg_refs: Accepted argument references for that audit.
        package_calls: Previously audited calls, when available.

    Returns:
        Comparison statistic, probability, lag and package call evidence.

    Raises:
        ValidationProtocolError: If losses, package calls or numerical work fail.

    """
    try:
        model = np.asarray(model_losses, dtype=np.float64)
        benchmark = np.asarray(benchmark_losses, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        _error("return-model loss evidence is invalid", exc)
    if (
        model.ndim != 1
        or benchmark.ndim != 1
        or model.shape != benchmark.shape
        or len(model) < 3
        or not np.isfinite(model).all()
        or not np.isfinite(benchmark).all()
    ):
        _error("return-model loss evidence must be finite and aligned")
    differential = model - benchmark
    lag = automatic_newey_west_lag(len(differential))
    expected_arguments: dict[str, dict[str, JsonValue]] = {
        "package.statsmodels.cov-hac": {
            "nlags": lag,
            "use_correction": True,
        },
        "package.statsmodels.ols": {
            "hasconst": True,
            "missing": "raise",
        },
    }
    if package_calls:
        calls = tuple(sorted(package_calls, key=lambda item: item.stable_id))
        if tuple(item.stable_id for item in calls) != tuple(sorted(expected_arguments)) or any(
            tuple((argument.name, argument.value) for argument in call.arguments)
            != tuple(sorted(expected_arguments[call.stable_id].items()))
            for call in calls
        ):
            _error("return-model DM/HAC package calls differ from their exact arguments")
    else:
        calls = tuple(
            audit_return_model_package_call(
                package_catalog,
                package_arg_refs,
                stable_id,
                arguments,
            )
            for stable_id, arguments in sorted(expected_arguments.items())
        )
    try:
        import statsmodels.api as sm
        from statsmodels.stats.sandwich_covariance import cov_hac

        result = sm.OLS(
            differential,
            np.ones((len(differential), 1), dtype=np.float64),
            missing="raise",
            hasconst=True,
        ).fit()
        covariance = np.asarray(cov_hac(result, nlags=lag, use_correction=True))
        standard_error = math.sqrt(float(covariance[0, 0]))
    except ImportError as exc:
        _error("statsmodels is unavailable for DM/HAC evidence", exc)
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
        _error("return-model DM/HAC evidence failed", exc)
    if standard_error <= 0.0 or not math.isfinite(standard_error):
        return DMResult(
            statistic=0.0,
            p_value=1.0,
            lag=lag,
            package_calls=calls,
        )
    statistic = float(np.mean(differential)) / standard_error
    periods = len(differential)
    hln = math.sqrt(max(0.0, (periods + 1.0 - 2.0) / periods))
    statistic *= hln
    p_value = math.erfc(abs(statistic) / math.sqrt(2.0))
    if not math.isfinite(statistic) or not math.isfinite(p_value):
        _error("return-model DM/HAC result is non-finite")
    return DMResult(
        statistic=statistic,
        p_value=p_value,
        lag=lag,
        package_calls=calls,
    )


def holm_adjust(p_values: tuple[tuple[str, float], ...]) -> dict[str, float]:
    """Apply Holm adjustment to identified hypothesis probabilities.

    Args:
        p_values: Hypothesis identifiers paired with raw probabilities.

    Returns:
        Adjusted probability by identifier, or an empty mapping for no inputs.

    Raises:
        ValidationProtocolError: If a probability lies outside zero to one.

    """
    if not p_values:
        return {}
    if any(not 0.0 <= value <= 1.0 for _, value in p_values):
        _error("return-model Holm inputs must be probabilities")
    ordered = sorted(p_values, key=lambda item: (item[1], item[0]))
    count = len(ordered)
    result: dict[str, float] = {}
    running = 0.0
    for index, (key, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * value))
        result[key] = running
    return result


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "DMResult",
    "automatic_newey_west_lag",
    "diebold_mariano_hac",
    "holm_adjust",
]
