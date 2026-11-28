"""Source identity binds an adapter kind's installed routes: the rule closure of a module that
declares a kind holds every module its routes select, and the route table moves its digest."""

from __future__ import annotations

from pathlib import Path

import pytest

from alphalattice.capabilities.domain_contracts.adapter_families import declared
from alphalattice.capabilities.domain_contracts.source_identity import (
    canonical_closure,
    number_deciding_rule,
    source_rule_closure_hash,
)
from alphalattice.investment.sector_research.models.catalog import (
    SECTOR_FORECASTS,
    SectorForecastCatalog,
)

TEST_SOURCE = "src/alphalattice/capabilities/domain_contracts/source_identity.py"
SECTOR_FORECASTS_MODULE = SectorForecastCatalog.__module__


@pytest.mark.proves("domain_contracts.a_kinds_installed_routes_enter_its_rule_closure")
def test_a_kinds_installed_routes_enter_the_rule_closure_that_reaches_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The closure of a module declaring a kind holds every module its routes select; two ids
    trading targets moves that module's digest and a warm role's hash, though no byte changed."""
    root, entry = Path(__file__).resolve().parents[3], SECTOR_FORECASTS_MODULE
    rule = number_deciding_rule(root)
    installed = canonical_closure((entry,), root=root, rule=rule)
    tracked = ("src/" + entry.replace(".", "/") + ".py",)
    role = {"semantic_owner": "sector_research", "numerical_role": "route-case"}
    held = source_rule_closure_hash(root=root, tracked_paths=tracked, **role)
    assert source_rule_closure_hash(root=root, tracked_paths=tracked, **role) == held
    selected = {point.module for point in declared(SECTOR_FORECASTS.group).values()}
    assert selected <= installed.keys()
    routes = {name: point.value for name, point in declared(SECTOR_FORECASTS.group).items()}
    routes["ZERO_SECTOR_FORECAST"], routes["EWMA_SECTOR_MEAN"] = (
        routes["EWMA_SECTOR_MEAN"],
        routes["ZERO_SECTOR_FORECAST"],
    )
    info = tmp_path / "alphalattice-0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: alphalattice\nVersion: 0\n")
    lines = [f"[{SECTOR_FORECASTS.group}]", *(f"{k} = {v}" for k, v in routes.items())]
    (info / "entry_points.txt").write_text("\n".join(lines) + "\n", "utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    traded = canonical_closure((entry,), root=root, rule=rule)
    assert (
        source_rule_closure_hash(root=root, tracked_paths=tracked, **role) != held
    )  # a warm role reads it
    assert traded[entry] != installed[entry]
    assert {module: traded[module] for module in selected} == {
        module: installed[module] for module in selected
    }
