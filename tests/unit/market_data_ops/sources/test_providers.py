from datetime import date
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from curl_cffi.requests.exceptions import Timeout as CurlTimeout

from alphalattice.foundation.market_data_ops.sources.providers import (
    RATE_LIMIT_BACKOFF_CAP_SECONDS,
    ProviderFetchError,
    YFinanceMarketDataProvider,
)
from alphalattice.kernel.egress import EgressRefusal

TEST_SOURCE = "src/alphalattice/foundation/market_data_ops/sources/providers.py"


@pytest.mark.proves("market_data_ops.a_rate_limited_retry_backs_off")
def test_a_rate_limited_retry_waits_longer_each_time_up_to_its_cap():
    """A retry after a rate limit waits, doubling up to its cap, where another retryable failure,
    or a last attempt, waits for nothing."""
    waits: list[float] = []
    limited = ProviderFetchError("data.rate_limited", "429", retryable=True)
    for attempt in (1, 2, 10, 11):
        assert limited.backed_off(attempt, 11, sleep=waits.append) is limited
    timeout = ProviderFetchError("data.provider_timeout", "timeout", retryable=True)
    timeout.backed_off(1, 11, sleep=waits.append)
    assert waits == [2.0, 4.0, RATE_LIMIT_BACKOFF_CAP_SECONDS]  # the last attempt waits for none


def _actions_of(monkeypatch, answer):
    """The adapter's action read on a stood-in yfinance whose Ticker answers `answer`: an actions
    frame, or the exception reading it raises."""
    monkeypatch.setenv("ALPHALATTICE_NETWORK_DISABLED", "0")  # the stand-in never connects

    class Ticker:
        def __init__(self, _symbol, session=None):
            self._data = SimpleNamespace(get=lambda *_args, **_kwargs: None)

        @property
        def actions(self):
            if isinstance(answer, BaseException):
                raise answer
            return answer

    provider = YFinanceMarketDataProvider(Path("unused"))
    provider._runtime_module = SimpleNamespace(
        Ticker=Ticker, data=SimpleNamespace(lru_cache_freezeargs=lambda get: get)
    )
    return partial(provider.fetch_action_history, listing_id="listing-aapl", provider_symbol="AAPL")


@pytest.mark.proves("market_data_ops.provider_exceptions_map_to_failure_codes")
@pytest.mark.parametrize(
    ("raised", "code", "retryable"),
    [
        (EgressRefusal("egress.host_not_allowed", "yahoo"), "egress.host_not_allowed", False),
        (TimeoutError("read"), "data.provider_timeout", True),
        (RuntimeError("HTTP 429 Too Many Requests"), "data.rate_limited", True),
        (RuntimeError("curl: session already in use"), "data.provider_session_unstable", True),
        (RuntimeError("database is locked"), "data.provider_session_unstable", True),
        # libcurl's own timeout names no "timeout", so it falls through: a defect still open.
        (CurlTimeout("curl: (28) Operation timed out", 28), "data.provider_fetch_failed", False),
        (ValueError("unexpected payload"), "data.provider_fetch_failed", False),
    ],
    ids=("egress", "timeout", "rate-limit", "curl-session", "locked", "curl-timeout", "other"),
)
def test_each_provider_exception_maps_to_its_failure_code(monkeypatch, raised, code, retryable):
    """A provider exception becomes its stable failure code and retry flag: an egress refusal keeps
    its own, timeouts, rate limits and session instability retry, and anything else does not."""
    with pytest.raises(ProviderFetchError) as caught:
        _actions_of(monkeypatch, raised)(start=date(2026, 2, 9), end=date(2026, 2, 9))
    assert (caught.value.code, caught.value.retryable) == (code, retryable)


@pytest.mark.proves("market_data_ops.provider_frame_actions_inside_the_window")
def test_a_frames_actions_are_its_window_splits_dividends_and_gains(monkeypatch):
    """A history frame's split, dividend and capital gain inside the window become actions with
    their ratio and amounts, and a negative dividend is refused by name."""
    frame = pd.DataFrame(
        {
            "Stock Splits": [0.0, 0.0, 3.0, 0.0],
            "Dividends": [0.5, 0.25, 0.0, 0.0],
            "Capital Gains": [0.0, 0.0, 0.0, 0.1],
        },
        index=pd.to_datetime(["2026-02-09", "2026-02-10", "2026-03-16", "2026-03-17"]),
    )
    actions = _actions_of(monkeypatch, frame)(start=date(2026, 2, 10), end=date(2026, 3, 17))
    assert [
        (a.effective_date, a.action_kind, a.cash_amount, a.new_shares_per_old_share)
        for a in actions
    ] == [
        (date(2026, 2, 10), "CASH_DIVIDEND", 0.25, None),
        (date(2026, 3, 16), "SPLIT", None, 3.0),
        (date(2026, 3, 17), "CAPITAL_GAIN", 0.1, None),
    ]
    refuse = _actions_of(monkeypatch, frame.assign(Dividends=-0.25))
    with pytest.raises(ProviderFetchError, match="negative dividend") as refused:
        refuse(start=date(2026, 2, 9), end=date(2026, 2, 9))
    assert refused.value.code == "data.invalid_action_payload"
