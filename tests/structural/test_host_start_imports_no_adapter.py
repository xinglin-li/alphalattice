"""The Host's start imports no adapter: each family's modules load at their first use."""

from __future__ import annotations

import json
import sys
from importlib import metadata

import pytest

from alphalattice.capabilities.domain_contracts import spawn

TEST_SCOPE = "whole-tree"

GROUPS = (
    "alphalattice.alpha_models",
    "alphalattice.risk_estimators",
    "alphalattice.sector_forecasts",
)
START = (
    "alphalattice.control.product_host.composition.entry",
    "alphalattice.control.product_host.composition.local_web_session",
)
PROBE = "import importlib, json, sys\nfor name in sys.argv[1:]: importlib.import_module(name)\n"
PROBE += "print(json.dumps(sorted(sys.modules)))\n"


@pytest.mark.proves("product_host.the_hosts_start_imports_no_adapter_module")
def test_the_hosts_start_imports_no_declared_adapter_module() -> None:
    """A fresh interpreter importing the Host's entry and its web session holds none of the
    modules the adapter families' entry points name."""
    declared = metadata.distribution("alphalattice").entry_points
    adapters = {point.module for group in GROUPS for point in declared.select(group=group)}
    assert len(adapters) >= 14, adapters
    started = spawn.run(
        [sys.executable, "-c", PROBE, *START], capture_output=True, text=True, check=True
    )
    assert not adapters & set(json.loads(started.stdout)), sorted(
        adapters & set(json.loads(started.stdout))
    )
