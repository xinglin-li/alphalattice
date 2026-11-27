import pytest

from alphalattice.investment.portfolio_strategy_lab.publication.finalization_ledger import (
    FinalizationLedgerStoreError,
    PortfolioFinalizationStore,
)
from tests._support.identity_fixtures import fixture_digest

TEST_SOURCE = (
    "src/alphalattice/investment/portfolio_strategy_lab/publication/finalization_ledger.py"
)


@pytest.mark.proves("portfolio_strategy_lab.finalization_requires_a_registered_candidate")
def test_a_candidate_that_was_never_committed_cannot_be_loaded(tmp_path):
    store = PortfolioFinalizationStore(tmp_path)
    candidate_hash = fixture_digest("unregistered-candidate")
    assert store.open_candidate(candidate_hash) is None
    with pytest.raises(
        FinalizationLedgerStoreError, match=r"portfolio_finalization\.candidate_not_registered"
    ):
        store.load_candidate(candidate_hash)
