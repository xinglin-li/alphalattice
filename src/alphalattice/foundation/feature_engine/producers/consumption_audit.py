"""Measure what every installed Formula actually consumes, by perturbing its source.

A clock declaration is a claim about causality, and comparing two Panels can only
show that values moved -- never that the new ones read the right rows. This module
answers the question directly: it perturbs one ordered source session at a time
and records which Formula outputs move. The set that moves *is* the consumed set,
measured from the executable owner rather than transcribed from a recipe.

Four invariants come out of it, and they are deliberately one-sided where the
recipe is:

``LATEST_CONSUMED_EXACT``   perturbing the session the clock names as newest
                            changes the value. For a contiguous window that is
                            ``t - formula_skip_sessions``; for a calendar
                            selection it is the final close of the month twelve
                            back, which no session offset expresses.
``NOTHING_LATER``           perturbing anything after it -- including sessions
                            after the observation itself -- changes nothing. This
                            is the no-look-ahead claim, and it is the reason the
                            probe extends past ``t``.
``NOTHING_EARLIER``         perturbing anything before the declared minimum
                            history changes nothing. A bound, not an equality:
                            ``market_drawdown_x_momentum`` needs 63 rows for its
                            Market state child while its own stock leg reads only
                            ``t`` and ``t-20``, so demanding that the oldest
                            required row matter would be false of a correct
                            Formula.
``FIRST_FINITE_MECHANICAL`` the first finite output lands at
                            ``minimum_history_rows - 1``. Asserted only where the
                            recipe says the row minimum is mechanical, which is
                            every Formula except the calendar-selected one.
``FIRST_FINITE_EXISTS``     a finite value appears at all, and nothing more.
                            This is what the calendar selection gets, and the
                            weaker name is deliberate. ``seasonality_12m``
                            declares 253 rows; on the probe's synthetic axis its
                            first finite value lands at row 280, because thirteen
                            calendar months of *business days* is not 253 rows and
                            no fixed row count expresses it. Asserting a bound
                            here -- even "no earlier than" -- would be asserting
                            something this audit has not measured and the recipe
                            cannot know.

Each Formula is probed on a frame just long enough for it, so the cost is a few
recomputations per Formula rather than a full cross product.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Final

import numpy as np
import pandas as pd

from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog
from alphalattice.foundation.feature_engine.catalog.observation_clock import (
    FeatureObservationClock,
    formula_skip_sessions,
    observation_clock_for,
)
from alphalattice.foundation.feature_engine.producers.base_materializer import (
    BaseFeatureMaterializer,
)
from alphalattice.foundation.feature_engine.producers.factors.catalog import (
    default_extension_kernel_registry,
    extension_factor_specs,
    installed_formula_specs,
)
from alphalattice.foundation.feature_engine.producers.factors.registry import FeatureKernelRegistry
from alphalattice.kernel.quant.factor_contracts import FactorSpec

FUTURE_PROBE_SESSIONS: Final = 3
"""Ordered sessions kept after the observation, so look-ahead has somewhere to hide."""

PERTURBATION_UP: Final[Mapping[str, float]] = {
    "open_raw": 1.011,
    "high_raw": 1.021,
    "low_raw": 0.983,
    "close_raw": 1.370,
    "volume_raw": 1.290,
    "open_split_adjusted": 1.011,
    "high_split_adjusted": 1.021,
    "low_split_adjusted": 0.983,
    "close_split_adjusted": 1.370,
    "provider_adjusted_close": 1.370,
    "market_provider_adjusted_close": 1.290,
    "market_return_log": 1.310,
    "sector_return_log": 1.270,
    "rev_5": 1.230,
}
"""One bump per column, deliberately unequal.

A single common factor is invisible to every Formula built from a ratio of two
prices in the same session -- Parkinson, Garman-Klass, Rogers-Satchell, the
intraday leg, Chaikin money flow -- so a scale-invariant probe reports that they
consume nothing and calls it a pass. Unequal factors break that symmetry, and the
close moves far enough that a log return changes sign.
"""

PERTURBATION_DOWN: Final[Mapping[str, float]] = {
    key: 1.0 / value for key, value in PERTURBATION_UP.items()
}
"""The mirror bump, run beside the first.

Sign statistics -- ``information_discreteness_252``, ``momentum_consistency_252``
-- move only when a return actually crosses zero. One bump forces the perturbed
session's return strongly positive and the other strongly negative, so whatever
the original sign was, exactly one of the pair changes it.
"""

PROBE_LISTING_COUNT: Final = 12
"""Enough members for the cross-sectional Formulas; the audit reads listing zero."""

_PROBED_COLUMNS: Final = (
    "open_raw",
    "high_raw",
    "low_raw",
    "close_raw",
    "volume_raw",
    "open_split_adjusted",
    "high_split_adjusted",
    "low_split_adjusted",
    "close_split_adjusted",
    "provider_adjusted_close",
    "market_provider_adjusted_close",
    "market_return_log",
    "sector_return_log",
    "rev_5",
)


@dataclass(frozen=True)
class FormulaConsumptionProbe:
    """One Formula's measured clock, beside the clock its recipe declares."""

    factor_id: str
    origin: str
    source_interval_kind: str
    declared_skip_sessions: int | None
    declared_minimum_history_rows: int
    minimum_history_is_mechanical: bool
    probe_rows: int
    observation_position: int
    first_finite_position: int | None
    declared_latest_consumed_position: int | None
    measured_latest_consumed_position: int | None
    earliest_consumed_probed_position: int | None
    """Earliest *probed* session that moved the value, not a full dependency scan.

    The probe touches only the sessions the four invariants need. A Formula whose
    true window is shallower than its declared history -- ``directional_strength_14``
    reaches about twenty sessions at a mature observation, and its 28-row minimum
    comes from a warm-up chain -- reports its own observation here, and that is a
    fact about the probe set rather than about the Formula.
    """

    probed_positions: tuple[int, ...]
    consumed_after_declared_latest: tuple[int, ...]
    consumed_after_observation: tuple[int, ...]
    consumed_before_declared_history: tuple[int, ...]
    checks_passed: tuple[str, ...]
    failures: tuple[str, ...]

    @property
    def verdict(self) -> str:
        """Report whether observed source use matches the declared Formula clock."""
        return "CONSUMPTION_MATCHES_DECLARED_CLOCK" if not self.failures else "CLOCK_MISMATCH"

    def as_payload(self) -> dict[str, object]:
        """Serialize the probe facts together with their clock verdict."""
        return {**asdict(self), "verdict": self.verdict}


def _probe_frame(rows: int, *, listings: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A deterministic multi-listing source block plus its Market reference.

    Written here rather than drawn from a workspace because a probe needs to
    perturb one session and recompute; the measurement is about which rows an
    implementation reads, and that is a property of the code rather than of any
    particular price history.
    """
    sessions = pd.bdate_range("2015-01-05", periods=rows)
    index: np.ndarray = np.arange(rows, dtype=float)

    def bars(seed: float, level: float) -> dict[str, np.ndarray]:
        """One block whose ratios and volume genuinely vary session to session.

        A fixture with a constant intraday range or a constant volume makes
        several Formulas degenerate -- a zero-variance window is *missing*, not
        consumed -- so a probe over it measures the fixture instead of the code.
        """
        # Two components on purpose. The slow one gives the long windows something
        # to see; the fast one makes the path change direction session to session,
        # without which the directional-movement legs are one-sided, ``DX``
        # saturates at 100 and a probe measures a constant instead of a Formula.
        close = level * np.exp(
            0.0003 * index
            + 0.06 * np.sin(index / 9.0 + seed)
            + 0.013 * np.sin(index / 2.3 + 1.7 * seed)
        )
        upper = 1.004 + 0.006 * np.abs(np.sin(index / 5.0 + seed))
        lower = 0.996 - 0.006 * np.abs(np.cos(index / 6.0 + seed))
        open_ = close * (1.0 + 0.004 * np.sin(index / 4.0 + seed))
        volume = 1_000_000.0 * level * (1.05 + 0.45 * np.sin(index / 7.0 + seed) ** 2)
        return {
            "open_raw": open_,
            "high_raw": close * upper,
            "low_raw": close * lower,
            "close_raw": close,
            "volume_raw": volume,
            "open_split_adjusted": open_,
            "high_split_adjusted": close * upper,
            "low_split_adjusted": close * lower,
            "close_split_adjusted": close,
            "provider_adjusted_close": close,
        }

    market_bars = bars(0.11, 1.0)
    market_close = market_bars["close_raw"]
    market = pd.DataFrame({"session_date": sessions, **market_bars})
    blocks: list[pd.DataFrame] = []
    for member in range(listings):
        block = bars(0.37 * (member + 1), 1.0 + 0.1 * member)
        blocks.append(
            pd.DataFrame(
                {
                    "listing_id": [f"L{member:02d}"] * rows,
                    "session_date": sessions,
                    **block,
                    "sector_membership_asof": ["S"] * rows,
                    "market_provider_adjusted_close": market_close,
                }
            )
        )
    source = pd.concat(blocks, ignore_index=True)
    market_return = np.full(rows, np.nan)
    market_return[1:] = np.log(market_close[1:] / market_close[:-1])
    source["market_return_log"] = np.tile(market_return, listings)
    stock = source["provider_adjusted_close"].to_numpy(dtype=float)
    returns = np.full(len(source), np.nan)
    grouped = source.groupby("listing_id", sort=False).cumcount().to_numpy()
    returns[grouped > 0] = np.log(stock[grouped > 0] / stock[np.flatnonzero(grouped > 0) - 1])
    source["sector_return_log"] = 0.4 * np.nan_to_num(returns)
    # ``market_vol_ratio_x_reversal`` reads a verified base-Panel child rather
    # than a price, so the probe has to supply one that is perturbable too.
    source["rev_5"] = np.where(grouped >= 5, -np.nan_to_num(returns) * 5.0, np.nan)
    return source, market


def _perturbed(
    source: pd.DataFrame,
    market: pd.DataFrame,
    position: int,
    factors: Mapping[str, float],
    *,
    listing_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bump one session of one listing, and the Market reference beside it.

    Only the audited listing moves. Perturbing every member equally is invisible
    to a cross-sectional Formula -- ``sector_leader_lag_5`` subtracts its own
    return from a leader mean, and a common shift cancels exactly -- so it would
    report that a leader/lag method consumes nothing.
    """
    session = market["session_date"].iloc[position]
    bumped_source = source.copy()
    own = (bumped_source["session_date"] == session) & (bumped_source["listing_id"] == listing_id)
    for column in _PROBED_COLUMNS:
        if column in bumped_source:
            bumped_source.loc[own, column] = pd.to_numeric(
                bumped_source.loc[own, column], errors="coerce"
            ) * factors.get(column, 1.0)
    # The Market child is one series shared by every member, so its session moves
    # for all of them; that is what a Market observation actually is.
    shared = bumped_source["session_date"] == session
    for column in ("market_provider_adjusted_close", "market_return_log"):
        if column in bumped_source:
            bumped_source.loc[shared, column] = pd.to_numeric(
                bumped_source.loc[shared, column], errors="coerce"
            ) * factors.get(column, 1.0)
    bumped_market = market.copy()
    market_rows = bumped_market["session_date"] == session
    for column in _PROBED_COLUMNS:
        if column in bumped_market:
            bumped_market.loc[market_rows, column] = pd.to_numeric(
                bumped_market.loc[market_rows, column], errors="coerce"
            ) * factors.get(column, 1.0)
    return bumped_source, bumped_market


def _changed(left: float, right: float) -> bool:
    if not np.isfinite(left) and not np.isfinite(right):
        return False
    if np.isfinite(left) != np.isfinite(right):
        return True
    return not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


_PROBE_DIRECTIONS: Final = ("UP", "DOWN")
"""Both bumps are run; a session counts as consumed when either one moves the value."""


def _factors(direction: str) -> Mapping[str, float]:
    return PERTURBATION_UP if direction == "UP" else PERTURBATION_DOWN


class _CoreEvaluator:
    """Recompute the shipped bundle for listing zero, once per probed session."""

    def __init__(self, catalog: FeatureCatalog, rows: int) -> None:
        self._materializer = BaseFeatureMaterializer(catalog)
        self._source, self._market = _probe_frame(rows, listings=1)
        self._cache: dict[tuple[int, str] | None, pd.DataFrame] = {}

    def values(self, probe: tuple[int, str] | None) -> pd.DataFrame:
        if probe not in self._cache:
            source, market = (
                (self._source, self._market)
                if probe is None
                else _perturbed(
                    self._source, self._market, probe[0], _factors(probe[1]), listing_id="L00"
                )
            )
            block = self._materializer.materialize_listing(
                listing_id="L00",
                projected_bars=source.drop(columns=["listing_id"], errors="ignore"),
                market_bars=market,
            )
            self._cache[probe] = block.values
        return self._cache[probe]


class _ExtensionEvaluator:
    """Recompute one extension kernel, once per probed session."""

    def __init__(self, registry: FeatureKernelRegistry, recipe: FactorSpec, rows: int) -> None:
        self._registry = registry
        self._recipe = recipe
        self._source, self._market = _probe_frame(rows, listings=PROBE_LISTING_COUNT)
        self._selected = self._source["listing_id"].to_numpy() == "L00"
        self._cache: dict[tuple[int, str] | None, np.ndarray] = {}

    def values(self, probe: tuple[int, str] | None) -> np.ndarray:
        if probe not in self._cache:
            source = (
                self._source
                if probe is None
                else _perturbed(
                    self._source, self._market, probe[0], _factors(probe[1]), listing_id="L00"
                )[0]
            )
            computed = self._registry.compute(source, self._recipe).to_numpy(dtype=float)
            self._cache[probe] = computed[self._selected]
        return self._cache[probe]


def _calendar_source_position(rows: int, observation: int) -> int | None:
    sessions = pd.bdate_range("2015-01-05", periods=rows)
    periods = sessions.to_period("M")
    selected = periods[observation] - 12
    members = [index for index, value in enumerate(periods) if value == selected]
    return members[-1] if members else None


def _calendar_earliest_position(rows: int, observation: int) -> int | None:
    sessions = pd.bdate_range("2015-01-05", periods=rows)
    periods = sessions.to_period("M")
    selected = periods[observation] - 13
    members = [index for index, value in enumerate(periods) if value == selected]
    return members[-1] if members else None


def _probe_positions(
    *, rows: int, observation: int, declared_latest: int | None, history_rows: int
) -> tuple[int, ...]:
    """Every ordered session this Formula's four invariants need touched."""
    positions: set[int] = set()
    if declared_latest is not None:
        positions.add(declared_latest)
        positions.update(range(declared_latest + 1, observation + 1))
    positions.update(range(observation + 1, rows))
    earliest_allowed = observation - (history_rows - 1)
    positions.add(earliest_allowed)
    positions.add(earliest_allowed - 1)
    positions.add(0)
    return tuple(sorted(value for value in positions if 0 <= value < rows))


def _evaluate(
    *,
    factor_id: str,
    origin: str,
    clock: FeatureObservationClock,
    baseline: np.ndarray,
    probe: Mapping[tuple[int, str], float],
    rows: int,
    observation: int,
    declared_latest: int | None,
    calendar_earliest: int | None,
) -> FormulaConsumptionProbe:
    interval = clock.source_interval
    history_rows = interval.minimum_history_rows
    finite = np.flatnonzero(np.isfinite(np.asarray(baseline, dtype=float)))
    first_finite = int(finite[0]) if len(finite) else None
    observed = float(baseline[observation])
    consumed = sorted(
        {position for (position, _direction), value in probe.items() if _changed(observed, value)}
    )
    after_latest = tuple(
        position
        for position in consumed
        if declared_latest is not None and position > declared_latest
    )
    after_observation = tuple(position for position in consumed if position > observation)
    history_floor = (
        calendar_earliest
        if interval.kind == "CALENDAR_MONTH_SELECTION"
        else observation - (history_rows - 1)
    )
    before_history = tuple(
        position for position in consumed if history_floor is not None and position < history_floor
    )

    passed: list[str] = []
    failures: list[str] = []
    if declared_latest is None:
        failures.append("DECLARED_LATEST_UNRESOLVED")
    elif declared_latest in consumed:
        passed.append("LATEST_CONSUMED_EXACT")
    else:
        failures.append("LATEST_CONSUMED_EXACT")
    if after_latest:
        failures.append("NOTHING_LATER")
    else:
        passed.append("NOTHING_LATER")
    if after_observation:
        failures.append("NO_FUTURE_SOURCE")
    else:
        passed.append("NO_FUTURE_SOURCE")
    if before_history:
        failures.append("NOTHING_EARLIER")
    else:
        passed.append("NOTHING_EARLIER")
    if interval.minimum_history_is_mechanical:
        if first_finite == history_rows - 1:
            passed.append("FIRST_FINITE_MECHANICAL")
        else:
            failures.append("FIRST_FINITE_MECHANICAL")
    elif first_finite is None:
        failures.append("FIRST_FINITE_PRESENT")
    else:
        # Existence only. Named apart from the mechanical check so a reader
        # cannot mistake it for the same kind of claim: one is an equality
        # against a declared row count, this one is "a value appeared".
        passed.append("FIRST_FINITE_EXISTS")

    return FormulaConsumptionProbe(
        factor_id=factor_id,
        origin=origin,
        source_interval_kind=interval.kind,
        declared_skip_sessions=interval.latest_consumed_offset_sessions,
        declared_minimum_history_rows=history_rows,
        minimum_history_is_mechanical=interval.minimum_history_is_mechanical,
        probe_rows=rows,
        observation_position=observation,
        first_finite_position=first_finite,
        declared_latest_consumed_position=declared_latest,
        measured_latest_consumed_position=max(consumed) if consumed else None,
        earliest_consumed_probed_position=min(consumed) if consumed else None,
        probed_positions=tuple(sorted({position for position, _ in probe})),
        consumed_after_declared_latest=after_latest,
        consumed_after_observation=after_observation,
        consumed_before_declared_history=before_history,
        checks_passed=tuple(passed),
        failures=tuple(failures),
    )


def audit_installed_formula_consumption(
    *,
    catalog: FeatureCatalog | None = None,
    registry: FeatureKernelRegistry | None = None,
) -> tuple[FormulaConsumptionProbe, ...]:
    """Probe every installed Formula, shipped and extension, on its own frame."""
    installed = catalog if catalog is not None else FeatureCatalog.load()
    kernels = registry if registry is not None else default_extension_kernel_registry()
    clocks = installed.clocks_by_factor
    probes: list[FormulaConsumptionProbe] = []

    core_rows = max(int(spec.minimum_observations) for spec in installed.factors)
    calendar_ids = installed.calendar_factor_ids
    # A calendar Formula waits on thirteen calendar months, which is more ordered
    # sessions than any row count in the catalog names.
    core_rows = max(core_rows, 300 if calendar_ids else 0) + FUTURE_PROBE_SESSIONS + 1
    core = _CoreEvaluator(installed, core_rows)
    core_observation = core_rows - 1 - FUTURE_PROBE_SESSIONS
    baseline_core = core.values(None)

    for spec in installed.factors:
        clock = clocks[spec.factor_id]
        calendar = clock.source_interval.kind == "CALENDAR_MONTH_SELECTION"
        declared_latest = (
            _calendar_source_position(core_rows, core_observation)
            if calendar
            else core_observation - formula_skip_sessions(spec)
        )
        calendar_earliest = (
            _calendar_earliest_position(core_rows, core_observation) if calendar else None
        )
        positions = _probe_positions(
            rows=core_rows,
            observation=core_observation,
            declared_latest=declared_latest,
            history_rows=clock.source_interval.minimum_history_rows,
        )
        baseline = baseline_core[spec.factor_id].to_numpy(dtype=float)
        probe = {
            (position, direction): float(
                core.values((position, direction))[spec.factor_id].to_numpy(dtype=float)[
                    core_observation
                ]
            )
            for position in positions
            for direction in _PROBE_DIRECTIONS
        }
        probes.append(
            _evaluate(
                factor_id=spec.factor_id,
                origin="DESKTOP_CATALOG",
                clock=clock,
                baseline=baseline,
                probe=probe,
                rows=core_rows,
                observation=core_observation,
                declared_latest=declared_latest,
                calendar_earliest=calendar_earliest,
            )
        )

    for recipe in extension_factor_specs():
        clock = observation_clock_for(recipe)
        history_rows = clock.source_interval.minimum_history_rows
        rows = history_rows + FUTURE_PROBE_SESSIONS + 2
        evaluator = _ExtensionEvaluator(kernels, recipe, rows)
        observation = rows - 1 - FUTURE_PROBE_SESSIONS
        declared_latest = observation - formula_skip_sessions(recipe)
        positions = _probe_positions(
            rows=rows,
            observation=observation,
            declared_latest=declared_latest,
            history_rows=history_rows,
        )
        baseline = evaluator.values(None)
        probe = {
            (position, direction): float(evaluator.values((position, direction))[observation])
            for position in positions
            for direction in _PROBE_DIRECTIONS
        }
        probes.append(
            _evaluate(
                factor_id=recipe.factor_id,
                origin="INSTALLED_EXTENSION_RECIPE",
                clock=clock,
                baseline=baseline,
                probe=probe,
                rows=rows,
                observation=observation,
                declared_latest=declared_latest,
                calendar_earliest=None,
            )
        )
    result = tuple(sorted(probes, key=lambda item: item.factor_id))
    if tuple(item.factor_id for item in result) != tuple(
        item.factor_id for item in installed_formula_specs(installed)
    ):
        raise RuntimeError("feature_engine.installed_formula_registry_consumption_incomplete")
    return result


__all__ = [
    "FUTURE_PROBE_SESSIONS",
    "PERTURBATION_DOWN",
    "PERTURBATION_UP",
    "PROBE_LISTING_COUNT",
    "FormulaConsumptionProbe",
    "audit_installed_formula_consumption",
]
