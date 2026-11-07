"""Runtime locations are discovered at their owners, or honestly refused."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alphalattice.control.product_host.research_authoring.panel_methodology_context import (
    PANEL_METHODOLOGY_LOCATION_IDS,
    PanelMethodologyContextError,
    discover_panel_methodology_context,
    load_panel_methodology_context_manifest,
    render_panel_methodology_context,
)
from alphalattice.foundation.feature_engine.panels.development_overlay import (
    DEVELOPMENT_METHODOLOGY_SURFACE_CATEGORY,
)
from alphalattice.investment.sector_research.inputs.storage import SectorContextStore


def _workspace(root: Path, *stores: str) -> Path:
    for store in stores:
        (root / "artifacts" / store).mkdir(parents=True, exist_ok=True)
    return root


def _statuses(**keywords: object) -> dict[str, str]:
    return {
        value.location_id: value.status
        for value in discover_panel_methodology_context(**keywords)  # type: ignore[arg-type]
    }


def test_a_search_root_is_required() -> None:
    """requirement: discovery never guesses a location out of nothing."""

    with pytest.raises(PanelMethodologyContextError, match="context_search_root_required"):
        discover_panel_methodology_context(search_roots=())


def test_owner_rejection_is_not_reported_as_absence(tmp_path: Path) -> None:
    """Owner rejection is not reported as absence."""

    workspace = _workspace(tmp_path / "ws", "feature-panel", "factor-research", "alpha-research")
    probes = {
        value.location_id: value
        for value in discover_panel_methodology_context(search_roots=(workspace,))
    }
    assert set(probes) == set(PANEL_METHODOLOGY_LOCATION_IDS)
    assert probes["legacy_panel_artifact_root"].status == "RESOLVED"
    # present but refused by the owner, versus nothing of the kind in scope
    assert probes["panel_artifact_root"].status == "REJECTED_BY_OWNER"
    assert "resolvable" in probes["panel_artifact_root"].evidence
    assert probes["failed_baseline_workspace"].status == "NOT_FOUND_IN_SEARCH_SCOPE"

    report = render_panel_methodology_context(tuple(probes.values()), searched_roots=(workspace,))
    assert report["searched_roots"] == [str(workspace)]
    assert report["manifest"] is None
    assert report["numerical_calls"] == 0
    refusal = report["refusal"]
    assert isinstance(refusal, str)
    assert refusal.startswith("research_authoring.context_not_resolved:")
    assert "panel_artifact_root=REJECTED_BY_OWNER" in refusal
    assert "failed_baseline_workspace=NOT_FOUND_IN_SEARCH_SCOPE" in refusal


def test_the_two_factor_roots_are_probed_by_their_own_readers(tmp_path: Path) -> None:
    """The two factor roots are probed by their own readers."""

    workspace = _workspace(tmp_path / "ws", "factor-research")
    artifacts = workspace / "artifacts"

    before = _statuses(search_roots=(workspace,))
    assert before["feature_artifact_root"] == "REJECTED_BY_OWNER"
    assert before["sector_context_artifact_root"] == "NOT_FOUND_IN_SEARCH_SCOPE"

    (artifacts / DEVELOPMENT_METHODOLOGY_SURFACE_CATEGORY / ("0" * 64)).mkdir(parents=True)
    manifests = SectorContextStore(artifacts).root / "manifests"
    manifests.mkdir(parents=True)
    after_store = _statuses(search_roots=(workspace,))
    assert after_store["feature_artifact_root"] == "RESOLVED"
    # the store exists, so its emptiness is a refusal and not an absence
    assert after_store["sector_context_artifact_root"] == "REJECTED_BY_OWNER"

    (manifests / "surface.json").write_text("{}", encoding="utf-8")
    assert _statuses(search_roots=(workspace,))["sector_context_artifact_root"] == "RESOLVED"


def test_execution_outcomes_have_an_independent_owner_root(tmp_path: Path) -> None:
    """regression: target artifacts cannot stand in for execution outcomes."""

    artifacts = _workspace(tmp_path / "ws") / "artifacts"
    manifests = artifacts / "data-operations" / "execution-outcomes" / "manifests"
    manifests.mkdir(parents=True)

    before = _statuses(named_locations={"execution_outcome_artifact_root": artifacts})
    assert before["execution_outcome_artifact_root"] == "REJECTED_BY_OWNER"

    (manifests / "outcome.json").write_text("{}", encoding="utf-8")
    after = _statuses(named_locations={"execution_outcome_artifact_root": artifacts})
    assert after["execution_outcome_artifact_root"] == "RESOLVED"


def test_a_named_location_is_still_decided_at_its_owner(tmp_path: Path) -> None:
    """A named location is still decided at its owner."""

    named = _workspace(tmp_path / "elsewhere", "factor-research") / "artifacts"

    # no search root at all: a named location is a complete request on its own
    probes = _statuses(named_locations={"feature_artifact_root": named})
    assert probes["feature_artifact_root"] == "REJECTED_BY_OWNER"
    assert probes["repository_root"] == "NOT_FOUND_IN_SEARCH_SCOPE"

    (named / DEVELOPMENT_METHODOLOGY_SURFACE_CATEGORY / ("0" * 64)).mkdir(parents=True)
    assert (
        _statuses(named_locations={"feature_artifact_root": named})["feature_artifact_root"]
        == "RESOLVED"
    )

    with pytest.raises(PanelMethodologyContextError, match="field_not_a_location:panel_snapshot"):
        discover_panel_methodology_context(named_locations={"panel_snapshot": named})


def test_a_manifest_routes_and_never_carries_identity(tmp_path: Path) -> None:
    """requirement: a routing file states locations and claims no authority."""

    workspace = _workspace(tmp_path / "ws", "factor-research")
    entries = {value: str(workspace) for value in PANEL_METHODOLOGY_LOCATION_IDS}
    path = tmp_path / "context.json"

    path.write_text(json.dumps({"manifest": entries}), encoding="utf-8")
    assert set(load_panel_methodology_context_manifest(path)) == set(PANEL_METHODOLOGY_LOCATION_IDS)

    path.write_text(
        json.dumps({"manifest": {**entries, "panel_snapshot_hash": "0" * 40}}), encoding="utf-8"
    )
    with pytest.raises(PanelMethodologyContextError, match="field_not_a_location"):
        load_panel_methodology_context_manifest(path)

    partial = dict(entries)
    partial.pop("panel_artifact_root")
    path.write_text(json.dumps({"manifest": partial}), encoding="utf-8")
    with pytest.raises(PanelMethodologyContextError, match="incomplete:panel_artifact_root"):
        load_panel_methodology_context_manifest(path)

    path.write_text(
        json.dumps({"manifest": {**entries, "panel_artifact_root": str(tmp_path / "absent")}}),
        encoding="utf-8",
    )
    with pytest.raises(PanelMethodologyContextError, match="location_absent:panel_artifact_root"):
        load_panel_methodology_context_manifest(path)
