"""The Alpha model seam's conformance suite: every declared model keeps the adapter contract.

One suite per seam: the test runs once per model the product declares in the seam's entry point
group, so a new model inherits the suite and cannot land without passing it.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from alphalattice.capabilities.alpha_modeling.catalog import AlphaModelCatalog
from alphalattice.capabilities.alpha_modeling.extension import ALPHA_MODELS
from alphalattice.kernel.identity import is_blake3

TEST_SOURCE = "src/alphalattice/capabilities/alpha_modeling/catalog.py"
SEAM = "alphalattice.alpha_models"
"""The entry point group whose models this suite collects, the seam it names."""

_PROJECT = tomllib.loads(
    (Path(__file__).resolve().parents[3] / "pyproject.toml").read_text("utf-8")
)
IMPLEMENTATIONS = tuple(_PROJECT["project"]["entry-points"][SEAM])


@pytest.mark.proves("alpha_modeling.every_declared_model_keeps_the_adapter_contract")
@pytest.mark.parametrize("model_id", IMPLEMENTATIONS, ids=IMPLEMENTATIONS)
def test_every_declared_model_keeps_the_adapter_contract(model_id: str) -> None:
    """A declared model's loaded adapter (its shape and id are the loader's refusals) has a
    BLAKE3 numerical binding and a catalog installs it."""
    adapter = ALPHA_MODELS.load(model_id)
    assert is_blake3(adapter.describe_numerical_binding().numerical_binding_hash)
    assert AlphaModelCatalog((adapter,)).adapter_ids == (model_id,)
