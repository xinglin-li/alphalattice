"""Adapter families: catalogs and entry points name the same built-ins; others refuse by name."""

from __future__ import annotations

import typing
from pathlib import Path

import pytest

from alphalattice.capabilities.alpha_modeling.catalog import CAMPAIGN_MODELS, PRODUCTION_MODELS
from alphalattice.capabilities.alpha_modeling.contracts import AlphaModelAdapter
from alphalattice.capabilities.alpha_modeling.extension import ALPHA_MODELS
from alphalattice.capabilities.domain_contracts.adapter_families import AdapterKind
from alphalattice.investment.alpha_research.experiments.panel_methodology_models import (
    PANEL_MODELS,
)
from alphalattice.investment.risk_research.estimators import catalog as risk
from alphalattice.investment.risk_research.estimators.capability import RiskCapability
from alphalattice.investment.sector_research.contracts import SectorResearchError
from alphalattice.investment.sector_research.models import catalog as sector
from alphalattice.investment.sector_research.models.contracts import SectorForecastAdapter

TEST_SOURCE = "src/alphalattice/capabilities/domain_contracts/adapter_families.py"

KINDS: dict[str, tuple[AdapterKind[typing.Any], type[Exception], str, type, str]] = {
    "alpha": (
        ALPHA_MODELS,
        ValueError,
        "model_extension.not_found:{id}",
        AlphaModelAdapter,
        "adapter_id",
    ),
    "risk": (
        risk.RISK_ESTIMATORS,
        ValueError,
        "RISK_ESTIMATOR_ADAPTER_NOT_INSTALLED",
        RiskCapability,
        "capability_handle",
    ),
    "sector": (
        sector.SECTOR_FORECASTS,
        SectorResearchError,
        "sector_research.method_not_installed",
        SectorForecastAdapter,
        "method_id",
    ),
}
CATALOGS: dict[str, tuple[str, ...]] = {
    "alpha": (*PRODUCTION_MODELS, *CAMPAIGN_MODELS, *PANEL_MODELS),
    "risk": risk.INSTALLED_ESTIMATORS,
    "sector": tuple(sector.FROZEN_SINGLETON_PARAMETERS),
}


@pytest.mark.proves("domain_contracts.the_catalogs_and_entry_points_name_the_same_built_ins")
@pytest.mark.parametrize(
    "kind_name",
    [
        pytest.param("alpha", id="alpha"),
        pytest.param("risk", id="risk"),
        pytest.param("sector", id="sector"),
    ],
)
def test_the_catalogs_and_the_entry_points_name_the_same_built_ins(kind_name: str) -> None:
    """A catalog id no entry point declares, and a declared built-in no catalog lists, are
    named, so a catalog's ordered ids and the one list of modules never drift."""
    listed, declared = set(CATALOGS[kind_name]), KINDS[kind_name][0].ids()
    assert not listed - declared, (
        f"catalog ids no entry point declares: {sorted(listed - declared)}"
    )
    assert not declared - listed, (
        f"declared built-ins no catalog lists: {sorted(declared - listed)}"
    )


def _installed(root: Path, kind_name: str, target: str) -> None:
    """A distribution named like the product, first on the path, declaring `x` as `target`.

    `elsewhere_<kind>:Adapter` has every member of the kind's protocol and answers to another id.
    """
    kind, _, _, protocol, answers = KINDS[kind_name]
    members = sorted({*typing.get_protocol_members(protocol)} - {answers, "adapter"})
    lines = ["import types", "class Adapter:", f"    {answers} = 'elsewhere'"]
    lines += [f"    {name} = None" for name in members]
    if kind_name == "risk":  # a capability answers by the adapter it owns
        lines.append("    adapter = types.SimpleNamespace(adapter_id='elsewhere')")
    (root / f"elsewhere_{kind_name}.py").write_text("\n".join(lines) + "\n", "utf-8")
    info = root / "alphalattice-0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: alphalattice\nVersion: 0\n")
    target = target.replace("{kind}", kind_name)
    (info / "entry_points.txt").write_text(f"[{kind.group}]\nx = {target}\n", "utf-8")


@pytest.mark.proves("domain_contracts.an_adapter_family_refuses_by_name_an_id_it_cannot_serve")
@pytest.mark.parametrize(
    ("kind_name", "target"),
    [
        pytest.param("alpha", None, id="alpha-undeclared"),
        pytest.param("alpha", "no_such_family:Adapter", id="alpha-unimportable"),
        pytest.param("alpha", "builtins:dict", id="alpha-not-its-kind"),
        pytest.param("alpha", "builtins:range", id="alpha-needs-arguments"),
        pytest.param("alpha", "elsewhere_{kind}:Adapter", id="alpha-another-id"),
        pytest.param("risk", None, id="risk-undeclared"),
        pytest.param("risk", "no_such_family:Adapter", id="risk-unimportable"),
        pytest.param("risk", "builtins:dict", id="risk-not-its-kind"),
        pytest.param("risk", "builtins:range", id="risk-needs-arguments"),
        pytest.param("risk", "elsewhere_{kind}:Adapter", id="risk-another-id"),
        pytest.param("sector", None, id="sector-undeclared"),
        pytest.param("sector", "no_such_family:Adapter", id="sector-unimportable"),
        pytest.param("sector", "builtins:dict", id="sector-not-its-kind"),
        pytest.param("sector", "builtins:range", id="sector-needs-arguments"),
        pytest.param("sector", "elsewhere_{kind}:Adapter", id="sector-another-id"),
    ],
)
def test_an_id_its_family_cannot_serve_is_refused_by_the_kinds_name(
    kind_name: str, target: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No entry point for the id, a target that fails to import, a class that cannot be built
    bare, an object that is not the kind's adapter, and an adapter answering to another id:
    each the kind's own refusal."""
    kind, refusal, message, _, _ = KINDS[kind_name]
    if target is not None:
        _installed(tmp_path, kind_name, target)
        monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(refusal) as refused:
        kind.load("x")
    assert str(refused.value) == message.format(id="x")
