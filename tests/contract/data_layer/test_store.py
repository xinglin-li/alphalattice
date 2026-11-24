"""Every registered store keeps the same market research laws."""

from datetime import date
from functools import partial

import pytest

from alphalattice.foundation.data_layer.store import StorePort
from tests._support.market_research_laws import (
    RESEARCH_CASES,
    assert_provider_research_laws,
    assert_research_laws,
    store_readbacks,
)

TEST_SOURCE = "src/alphalattice/foundation/data_layer/store.py"
STORES = {"ducklake": StorePort.open}


@pytest.fixture(params=STORES, scope="module")
def readback(request, tmp_path_factory):
    return store_readbacks(
        tmp_path_factory.mktemp("store-contract") / "data",
        STORES[request.param],
        {"session_date": "2026-01-05", "close": 15.0},
    )


@pytest.mark.proves("data_layer.store_research_laws")
def test_every_store_keeps_the_research_laws(readback):
    assert_research_laws(readback)


_CUTOFF = date(2026, 1, 5)
_PROVIDER_BARS = (
    {"session_date": "2026-01-05", "close": 15.0},
    {"session_date": "2026-01-06", "close": 16.0},
)


def _mutated_store(mutate, readback):
    assert_research_laws(mutate(readback))


def _mutated_provider(held, replay, _readback):
    assert_provider_research_laws(_PROVIDER_BARS, _CUTOFF, held, replay)


_MUTATION_CHECKS = [
    *(
        pytest.param(name, partial(_mutated_store, mutate), id=name)
        for name, _, mutate in RESEARCH_CASES
    ),
    pytest.param(
        "provider-session-cutoff",
        partial(_mutated_provider, _PROVIDER_BARS, _PROVIDER_BARS[:1]),
        id="provider-returns-a-later-session",
    ),
    pytest.param(
        "provider-replay-all-equal",
        partial(
            _mutated_provider,
            _PROVIDER_BARS[:1],
            [{"session_date": "2026-01-05", "close": 17.0}],
        ),
        id="provider-replay-changes-a-close",
    ),
]


@pytest.mark.proves("data_layer.store_research_laws")
@pytest.mark.parametrize("name, probe", _MUTATION_CHECKS, ids=[row.id for row in _MUTATION_CHECKS])
def test_each_research_law_refuses_a_bad_readback(readback, name, probe):
    with pytest.raises(AssertionError, match=name):
        probe(readback)
