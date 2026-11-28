"""The Risk estimator seam's conformance suite: every declared capability keeps its contract.

One suite per seam: the test runs once per capability the product declares in the seam's entry
point group, so a new estimator inherits the suite and cannot land without passing it.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from alphalattice.investment.risk_research.estimators.catalog import (
    RISK_ESTIMATORS,
    RiskEstimatorCatalog,
)
from alphalattice.kernel.identity import is_blake3

TEST_SOURCE = "src/alphalattice/investment/risk_research/estimators/catalog.py"
SEAM = "alphalattice.risk_estimators"
"""The entry point group whose capabilities this suite collects, the seam it names."""

_PROJECT = tomllib.loads(
    (Path(__file__).resolve().parents[4] / "pyproject.toml").read_text("utf-8")
)
IMPLEMENTATIONS = tuple(_PROJECT["project"]["entry-points"][SEAM])


@pytest.mark.proves("risk_research.every_declared_estimator_keeps_the_capability_contract")
@pytest.mark.parametrize("adapter_id", IMPLEMENTATIONS, ids=IMPLEMENTATIONS)
def test_every_declared_estimator_keeps_the_capability_contract(adapter_id: str) -> None:
    """A declared capability's adapter (its shape and id are the loader's refusals) has a BLAKE3
    binding, and a catalog admits the capability's route."""
    capability = RISK_ESTIMATORS.load(adapter_id)
    assert is_blake3(capability.adapter.describe_numerical_binding().numerical_binding_hash)
    catalog = RiskEstimatorCatalog((capability.adapter,), capabilities=(capability,))
    assert catalog.adapter_ids == (adapter_id,)
