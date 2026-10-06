"""Array firewall tests for factor order, missingness, and sealed isolation."""

from __future__ import annotations

from dataclasses import replace

import pytest

from alphalattice.investment.alpha_research.inputs.folds import AlphaArrayBoundaryError
from tests.alpha_research.fixtures import FIXTURE_FACTOR_IDS, synthetic_prepared_arrays


def test_fold_contract_rejects_duplicate_factor_authority() -> None:
    fold = synthetic_prepared_arrays().folds[0]
    with pytest.raises(AlphaArrayBoundaryError, match="FACTOR_AUTHORITY"):
        replace(fold, ordered_factor_ids=(*FIXTURE_FACTOR_IDS[:-1], FIXTURE_FACTOR_IDS[0]))
