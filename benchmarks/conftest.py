"""Deterministic inputs shared by the CodSpeed benchmarks.

The benchmarks live outside ``tests/`` so the test-shape and size rules that govern the
public test suite do not apply to them, and so ``tests/conftest.py`` is never loaded on a
benchmark run. Every input is seeded and built in a fixture, outside the measured call.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def price_frame(*, periods: int, seed: int, start: str = "2022-01-03") -> pd.DataFrame:
    """One listing's projected daily bars: every column the core materializer requires."""
    dates = pd.bdate_range(start, periods=periods)
    rng = np.random.default_rng(seed)
    close = 80.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, periods)))
    spread = 0.008 + rng.uniform(0.0, 0.005, periods)
    open_ = close * np.exp(rng.normal(0.0, 0.003, periods))
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    volume = rng.integers(500_000, 4_000_000, periods).astype(float)
    return pd.DataFrame(
        {
            "session_date": dates,
            "open_raw": open_,
            "high_raw": high,
            "low_raw": low,
            "close_raw": close,
            "volume_raw": volume,
            "open_split_adjusted": open_,
            "high_split_adjusted": high,
            "low_split_adjusted": low,
            "close_split_adjusted": close,
            "provider_adjusted_close": close,
            "close_total_return_adjusted": close,
        }
    )


@pytest.fixture(scope="session")
def asset_bars() -> pd.DataFrame:
    """Two years of one listing's bars."""
    return price_frame(periods=560, seed=7)


@pytest.fixture(scope="session")
def market_bars() -> pd.DataFrame:
    """The market reference over the same sessions."""
    return price_frame(periods=560, seed=11)


@pytest.fixture(scope="session")
def cross_section() -> np.ndarray:
    """A session x listing factor matrix with a few missing values and fat tails."""
    rng = np.random.default_rng(1729)
    values = rng.standard_t(df=3, size=(252, 500))
    values[rng.random(values.shape) < 0.02] = np.nan
    return values


@pytest.fixture(scope="session")
def sector_positions() -> tuple[np.ndarray, ...]:
    """Eleven sectors over the 500 listings of ``cross_section``."""
    rng = np.random.default_rng(42)
    labels = rng.integers(0, 11, 500)
    return tuple(np.flatnonzero(labels == sector) for sector in range(11))


@pytest.fixture(scope="session")
def formula_panel() -> pd.DataFrame:
    """Shuffled long-format rows for 20 listings with late starts and missing closes."""
    rng = np.random.default_rng(7)
    rows = 320
    parts = []
    for listing in range(20):
        start = int(rng.integers(0, 60))
        count = rows - start
        close = 50.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, count)))
        opening = close * np.exp(rng.normal(0.0, 0.01, count))
        frame = pd.DataFrame(
            {
                "listing_id": f"L{listing:02d}",
                "session_date": pd.bdate_range("2020-01-01", periods=rows)[start:].date,
                "close_split_adjusted": close,
                "open_split_adjusted": opening,
                "high_split_adjusted": np.maximum(opening, close) * 1.01,
                "low_split_adjusted": np.minimum(opening, close) * 0.99,
                "provider_adjusted_close": close * 0.97,
                "volume_raw": rng.uniform(1e5, 2e6, count),
            }
        )
        frame.loc[rng.random(count) < 0.02, "close_split_adjusted"] = np.nan
        parts.append(frame)
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=1)
