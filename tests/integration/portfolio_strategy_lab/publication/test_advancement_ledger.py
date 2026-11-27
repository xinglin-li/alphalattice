import json

import pytest

from alphalattice.investment.portfolio_strategy_lab.publication.advancement_ledger import (
    AdvancementLedgerStore,
    AdvancementLedgerStoreError,
)
from alphalattice.kernel.identity import blake3_filename
from tests._support.identity_fixtures import fixture_digest

TEST_SOURCE = "src/alphalattice/investment/portfolio_strategy_lab/publication/advancement_ledger.py"


@pytest.mark.proves("portfolio_strategy_lab.advancement_refuses_an_index_for_another_program")
def test_a_program_receipt_index_cannot_claim_another_program(tmp_path):
    store = AdvancementLedgerStore(tmp_path)
    requested = fixture_digest("a")
    assert store.find_receipt_for(requested) is None
    index = store.root / "index" / "by-program" / f"{blake3_filename(requested)}.json"
    index.parent.mkdir(parents=True)
    index.write_text(
        json.dumps({"program_hash": fixture_digest("b"), "receipt_hash": fixture_digest("c")}),
        encoding="utf-8",
    )
    with pytest.raises(
        AdvancementLedgerStoreError, match=r"portfolio_advancement\.receipt_index_tampered"
    ):
        store.find_receipt_for(requested)
