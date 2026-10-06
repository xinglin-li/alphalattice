"""The Portfolio evidence reclassification receipt, kept so old records read back."""

from __future__ import annotations

from pathlib import Path

import pytest

from alphalattice.investment.portfolio_strategy_lab.contracts import (
    seal_contract,
)
from alphalattice.investment.portfolio_strategy_lab.regularization.contracts import (
    PortfolioEvidenceReclassificationReceipt,
)

PLAYPEN_ROOT = Path(__file__).resolve().parents[2]
PLAYPEN_SRC = PLAYPEN_ROOT / "src"

_HASHES = tuple(f"{value:064x}" for value in range(1, 40))


def test_reclassification_is_typed_and_content_addressed() -> None:
    receipt = seal_contract(
        PortfolioEvidenceReclassificationReceipt,
        "receipt_hash",
        original_development_mandate_hash=_HASHES[0],
        consumed_oos_slate_hash=_HASHES[1],
        consumed_oos_result_hash=_HASHES[2],
        consumed_oos_evidence_hashes=(_HASHES[3],),
        typed_authority=("USER_AUTHORIZED_RECLASSIFICATION_AND_CONDITIONAL_POLICY_HOLDOUT"),
    )
    assert receipt.independent_oos_claim_retired
    with pytest.raises(ValueError, match="regularization_identity_invalid"):
        receipt.model_copy(update={"consumed_oos_result_hash": _HASHES[4]}).model_validate(
            receipt.model_copy(update={"consumed_oos_result_hash": _HASHES[4]}).model_dump()
        )
