"""Host projection of complete Factor formulas and independent admission receipts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from alphalattice.control.product_host.research_authoring.feature_activations import (
    feature_catalog_for,
)
from alphalattice.foundation.factor_research.experiments.authoring import (
    factor_inventory_from_panel_manifest,
)
from alphalattice.foundation.feature_engine.catalog.contracts import (
    FeatureCatalog,
    desktop_core_feature_bundle,
)
from alphalattice.foundation.feature_engine.producers.factors.catalog import (
    default_extension_kernel_registry,
    installed_formula_specs,
)
from alphalattice.foundation.feature_engine.producers.factors.registry import (
    factor_methodology_hash,
)


def describe_feature_input(
    *, input_binding_hash: str, panel: dict[str, Any], workspace: Path | None = None
) -> dict[str, object]:
    """Recorded columns and current definitions, without reading or computing values.

    An installed formula is not a materialized column. A same-named formula in
    this build is not its historical definition unless the method identities
    agree. Keep both facts visible; never fill an old artifact with today's spec.
    """
    inventory = {entry.factor_id: entry for entry in factor_inventory_from_panel_manifest(panel)}
    # The definitions of the catalog the Panel was built under, a workspace's activations
    # included (EX); the shipped ones when it names none this workspace kept.
    recorded = str(panel.get("safe_summary", {}).get("lineage", {}).get("catalog_hash", ""))
    catalog = (
        None if workspace is None else feature_catalog_for(workspace, recorded)
    ) or FeatureCatalog.load()
    specs = {spec.factor_id: spec for spec in installed_formula_specs(catalog)}
    registry = default_extension_kernel_registry()
    core = desktop_core_feature_bundle()
    columns = []
    for factor_id in sorted(inventory.keys() | specs.keys()):
        recorded = inventory.get(factor_id)
        spec = specs.get(factor_id)
        current_method = (
            factor_methodology_hash(
                spec, implementation_hash=registry.implementation_hash(spec, core_bundle=core)
            )
            if spec is not None
            else None
        )
        status = (
            "NOT_IN_SELECTED_INPUT"
            if recorded is None
            else "RESEARCH_LOCAL_PREPARED"
            if panel.get("kind") == "DevelopmentFeatureInput"
            and panel["safe_summary"]["factor_catalog_summary"][factor_id].get("development_only")
            else "NOT_REGISTERED_IN_THIS_BUILD"
            if spec is None
            else "MATCHES_RECORDED_INPUT"
            if current_method == recorded.methodology_hash
            else "DIFFERS_FROM_RECORDED_INPUT"
        )
        columns.append(
            {
                "factor_id": factor_id,
                "in_selected_input": recorded is not None,
                "definition_status": status,
                "recorded_methodology_hash": recorded.methodology_hash if recorded else None,
                "recorded_implementation_hash": recorded.implementation_hash if recorded else None,
                "current_methodology_hash": current_method,
                "current_definition": spec.model_dump(mode="json") if spec else None,
            }
        )
    return {
        "input_binding_hash": input_binding_hash,
        "panel_snapshot_hash": panel["snapshot_hash"],
        "input_factor_count": len(inventory),
        "registered_formula_count": len(specs),
        "registered_not_in_input_count": len(specs.keys() - inventory.keys()),
        "columns": columns,
        "claim": "RECORDED_INPUT_COLUMNS_AND_CURRENT_DEFINITIONS_NOT_ROW_AVAILABILITY",
        "limits": [
            "Registration does not build a column or admit it to this research input.",
            "Current definitions describe this build, not a differently bound historical column.",
            "Factor screening, curation and model inputs are separate selections.",
        ],
    }


__all__ = [
    "describe_feature_input",
]
