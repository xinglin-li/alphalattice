import json

import pytest

from alphalattice.investment.portfolio_strategy_lab.publication.advancement_ledger import (
    AdvancementLedgerStore,
    AdvancementLedgerStoreError,
)

TEST_SOURCE = "src/alphalattice/investment/portfolio_strategy_lab/publication/advancement_ledger.py"


@pytest.mark.proves("portfolio_strategy_lab.advancement_refuses_an_index_for_another_program")
def test_a_program_receipt_index_cannot_claim_another_program(tmp_path):
    store = AdvancementLedgerStore(tmp_path)
    requested = "a" * 64
    assert store.find_receipt_for(requested) is None
    index = store.root / "index" / "by-program" / f"{requested}.json"
    index.parent.mkdir(parents=True)
    index.write_text(
        json.dumps({"program_hash": "b" * 64, "receipt_hash": "c" * 64}), encoding="utf-8"
    )
    with pytest.raises(
        AdvancementLedgerStoreError, match=r"portfolio_advancement\.receipt_index_tampered"
    ):
        store.find_receipt_for(requested)
