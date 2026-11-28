"""The Sector forecast seam's conformance suite: every declared method keeps the adapter contract.

One suite per seam: the test runs once per method the product declares in the seam's entry point
group, so a new method inherits the suite and cannot land without passing it.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from alphalattice.investment.sector_research.models.catalog import (
    SECTOR_FORECASTS,
    SectorForecastCatalog,
)
from alphalattice.kernel.identity import is_blake3

TEST_SOURCE = "src/alphalattice/investment/sector_research/models/catalog.py"
SEAM = "alphalattice.sector_forecasts"
"""The entry point group whose methods this suite collects, the seam it names."""

_PROJECT = tomllib.loads(
    (Path(__file__).resolve().parents[4] / "pyproject.toml").read_text("utf-8")
)
IMPLEMENTATIONS = tuple(_PROJECT["project"]["entry-points"][SEAM])


@pytest.mark.proves("sector_research.every_declared_method_keeps_the_adapter_contract")
@pytest.mark.parametrize("method_id", IMPLEMENTATIONS, ids=IMPLEMENTATIONS)
def test_every_declared_method_keeps_the_adapter_contract(method_id: str) -> None:
    """A declared method's loaded adapter (its shape and id are the loader's refusals) has a
    BLAKE3 binding, and a catalog seals its one frozen recipe."""
    adapter = SECTOR_FORECASTS.load(method_id)
    assert is_blake3(adapter.describe_numerical_binding().numerical_binding_hash)
    assert SectorForecastCatalog((adapter,)).seal_recipe(method_id=method_id).method_id == method_id
