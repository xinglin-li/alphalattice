import pytest

from alphalattice.foundation.market_data_ops.sources.providers import (
    RATE_LIMIT_BACKOFF_CAP_SECONDS,
    ProviderFetchError,
)

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
