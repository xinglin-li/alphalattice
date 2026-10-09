"""What the Risk estimator campaign left behind: the Host's estimate firewall.

An adapter that misdescribes its own estimate is refused before anything terminal is written.
"""

from __future__ import annotations

import dataclasses
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from alphalattice.investment.risk_research.estimators.catalog import (
    RiskEstimatorCatalog,
    build_installed_risk_estimator_catalog,
)
from alphalattice.investment.risk_research.estimators.contracts import (
    RiskEstimatorAdapter,
    RiskEstimatorNumericalBinding,
    RiskEstimatorRecipeEnvelope,
)
from alphalattice.investment.risk_research.experiments.execution import (
    RiskDevelopmentExecutor,
)
from alphalattice.investment.risk_research.experiments.window import (
    resolve_development_input_binding,
)
from alphalattice.investment.risk_research.surfaces.artifacts import RiskArtifactStore
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
)
from tests.researcher_methodology_surface.real_workspace import RealRiskWorkspace
from tests.researcher_methodology_surface.risk_development_program_support import (
    _admission,
)
from tests.researcher_methodology_surface.risk_development_support import (
    _binding,
    _bounded_authority,
)


class _LyingAdapter:
    """The real adapter, with one field of its own diagnostics falsified."""

    def __init__(self, inner: RiskEstimatorAdapter, *, field: str, value: object) -> None:
        self._inner = inner
        self._field = field
        self._value = value
        self.adapter_id = inner.adapter_id
        self.recipe_schema_id = inner.recipe_schema_id

    def describe_numerical_binding(self) -> RiskEstimatorNumericalBinding:
        # Delegated, so this adapter passes every identity check and the only
        # thing left to catch it is the Host's look at what it returned.
        return self._inner.describe_numerical_binding()

    def validate_recipe(self, recipe: RiskEstimatorRecipeEnvelope) -> object:
        return self._inner.validate_recipe(recipe)

    def estimate(
        self,
        *,
        recipe: RiskEstimatorRecipeEnvelope,
        inputs: object,
    ) -> object:
        estimate = self._inner.estimate(recipe=recipe, inputs=inputs)  # type: ignore[arg-type]
        return dataclasses.replace(
            estimate,
            diagnostics=estimate.diagnostics.model_copy(update={self._field: self._value}),
        )


class _LyingCatalog(RiskEstimatorCatalog):
    def __init__(self, *, field: str, value: object) -> None:
        # The installed adapter set, so this catalog's identity matches the
        # Program's and the run reaches the estimate firewall rather than the
        # governance guard above it.
        super().__init__(build_installed_risk_estimator_catalog().adapters)
        self._field = field
        self._value = value

    def resolve(self, recipe: RiskEstimatorRecipeEnvelope) -> RiskEstimatorAdapter:
        return _LyingAdapter(  # type: ignore[return-value]
            super().resolve(recipe), field=self._field, value=self._value
        )


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("matrix_hash", "f" * 64, "estimate_matrix_hash_mismatch"),
        ("formation_session", date(1999, 1, 4), "estimate_formation_session_mismatch"),
        # The environment an estimate names is its provenance, not checked (LAWS.md ID6).
    ],
)
def test_an_adapter_that_misdescribes_its_own_estimate_is_refused(
    real_risk_workspace: RealRiskWorkspace,
    tmp_path: Path,
    field: str,
    value: object,
    code: str,
) -> None:
    """requirement: the Host checks the identity of what an adapter returned.

    Deliberately narrow. The Host does not re-derive the mathematics -- no second
    eigendecomposition, no PD tolerance, no conditioning rule -- because that
    would put a copy of every method's science in a writer that is supposed to
    know none of it.

    What it cannot delegate is the adapter's claim about *which* numbers these
    are. ``matrix_hash`` is what the chunk index records and what replay compares,
    the formation session decides where the estimate lands on the axis, and the
    environment decides which capability it is attributed to. An adapter that
    misstates any of the three publishes a graph that verifies perfectly against
    numbers nobody computed.
    """

    authority, _requested = _bounded_authority(real_risk_workspace, count=3)
    input_binding, bounded = resolve_development_input_binding(
        authority=authority,
        return_surface=real_risk_workspace.return_surface,
        return_reader=real_risk_workspace.return_reader,
        freshness_probe=real_risk_workspace.freshness_probe,
    )
    output = tmp_path / "lying"
    with pytest.raises(AuthoringError, match=code):
        RiskDevelopmentExecutor(estimators=_LyingCatalog(field=field, value=value)).execute(
            binding=_binding(),
            input_binding=input_binding,
            admission=_admission(),
            return_surface=real_risk_workspace.return_surface,
            return_reader=real_risk_workspace.return_reader,
            bounded_sessions=bounded,
            output_workspace=output,
            sector_by_listing_id=real_risk_workspace.sector_by_listing_id,
        )

    # No terminal artifact carrying the false identity exists. The input binding
    # is deliberately absent from this list: it is written before the build, as a
    # commitment made in advance rather than a description of what happened.
    store = RiskArtifactStore(output)
    for category in ("development/covariance-surfaces", "development/covariance-diagnostics"):
        assert not (store.root / category).exists() or not any(
            (store.root / category).glob("*.json")
        ), category
    chunks = store.root / "covariance" / "chunks"
    assert not chunks.exists() or not any(chunks.glob("*.bin"))


def _publish_feature_lineage(
    artifact_root: Path,
    *,
    sector_by_listing_id: dict[str, str],
    manifest_revision: str,
    catalog_hash: str,
) -> tuple[str, Any, Any]:
    """Publish one honest Feature Panel lineage through the Feature Desk's owners.

    Four artifacts, because four is what the Sector authority walks: the panel
    manifest, its composition, the Feature-owned recovery binding that names the
    exact sector map, and the map itself. A real workspace holds a great deal
    more, none of which this authority reads, so none of it is built here.
    """

    import json

    from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
    from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
        PanelClosureArtifactStore,
    )
    from alphalattice.foundation.feature_engine.panels.closure_contracts import (
        SectorRevisionEntry,
    )
    from alphalattice.foundation.feature_engine.panels.feature_closure_contracts import (
        PanelRecoveryClosureBinding,
    )
    from alphalattice.foundation.feature_engine.panels.feature_closure_ledger import (
        FeatureClosureLedger,
        identified,
    )
    from alphalattice.kernel.shared_kernel.identity import canonical_hash

    listing_ids = tuple(sorted(sector_by_listing_id))
    entries = tuple(
        SectorRevisionEntry(
            listing_id=listing,
            provider="YAHOO",
            provider_symbol=f"SYM{index:04d}",
            sector_name=sector_by_listing_id[listing],
            payload_hash=canonical_hash({"payload": listing}),
            evidence_hash=canonical_hash({"evidence": listing}),
        )
        for index, listing in enumerate(listing_ids)
    )
    sector_map = _sector_map(entries, manifest_revision=manifest_revision)

    resolver = ArtifactResolver(artifact_root)
    ledger = FeatureClosureLedger(PanelClosureArtifactStore(resolver))
    ledger.publish_sector_map(sector_map)

    panel_binding_hash = canonical_hash({"panel-binding": manifest_revision})
    panel_content_hash = canonical_hash({"panel-content": manifest_revision})
    identity: dict[str, object] = {
        "kind": "PanelArtifactComposition",
        "binding": {
            "manifest_revision": manifest_revision,
            "sector_revision": sector_map.sector_revision,
            "catalog_hash": catalog_hash,
            "policy_hash": canonical_hash({"policy": manifest_revision}),
            "panel_binding_hash": panel_binding_hash,
            "history_start": "2020-01-02",
            "as_of_session": "2024-01-02",
            "factor_ids": ["factor.case-study"],
        },
        "content": {
            "panel_content_hash": panel_content_hash,
            "row_count": len(listing_ids),
            "availability_count": len(listing_ids),
            "history_start": "2020-01-02",
            "as_of_session": "2024-01-02",
        },
        "schema_hash": canonical_hash({"schema": "case-study"}),
        "chunks": [],
    }
    staging = artifact_root / "feature-panel" / "composition-staging" / panel_binding_hash
    staging.mkdir(parents=True, exist_ok=True)
    (staging / f"{panel_content_hash}.json").write_text(
        json.dumps(
            {**identity, "composition_hash": canonical_hash(identity)},
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    manifest: dict[str, object] = {
        "kind": "FeaturePanelManifest",
        "panel_binding_hash": panel_binding_hash,
        "panel_content_hash": panel_content_hash,
        "listing_set_hash": canonical_hash(tuple(sorted(listing_ids))),
        "active_listing_count": len(listing_ids),
    }
    snapshot_hash = canonical_hash(manifest)
    manifests = artifact_root / "feature-panel" / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / f"{snapshot_hash}.json").write_text(
        json.dumps(
            {**manifest, "snapshot_hash": snapshot_hash}, sort_keys=True, separators=(",", ":")
        ),
        encoding="utf-8",
    )

    recovery = identified(
        PanelRecoveryClosureBinding,
        {
            "snapshot_hash": snapshot_hash,
            "panel_content_hash": panel_content_hash,
            "panel_binding_hash": panel_binding_hash,
            "catalog_hash": catalog_hash,
            "closure_head_hash": canonical_hash({"head": manifest_revision}),
            "closure_transition_cursor": 0,
            "sector_map_hash": sector_map.map_hash,
            "listing_ids": listing_ids,
            "sessions": (date(2024, 1, 2),),
            "factor_ids": ("factor.case-study",),
            "panel_source_state_hash": canonical_hash({"source": manifest_revision}),
        },
        "binding_hash",
    )
    ledger.publish_panel_binding(recovery)
    return snapshot_hash, sector_map, recovery


def _sector_map(
    entries: tuple[Any, ...], *, manifest_revision: str, revision: str | None = None
) -> Any:
    """Seal one sector map, optionally keeping a revision its entries disown."""

    from alphalattice.foundation.feature_engine.panels.closure_contracts import SectorRevisionMap
    from alphalattice.foundation.feature_engine.panels.feature_closure_ledger import identified
    from alphalattice.foundation.market_data_ops.returns.sector_revision_identity import (
        sector_revision_hash,
    )

    honest = sector_revision_hash(
        manifest_revision=manifest_revision,
        observations=[entry.model_dump(mode="json") for entry in entries],
    )
    return identified(
        SectorRevisionMap,
        {
            "manifest_revision": manifest_revision,
            "sector_revision": honest if revision is None else revision,
            "entries": entries,
        },
        "map_hash",
    )
