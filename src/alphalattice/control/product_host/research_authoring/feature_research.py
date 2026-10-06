"""Resolve immutable research inputs for local Feature-definition planning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from alphalattice.control.product_host.research_authoring.factor_inputs import (
    factor_input_paths,
    read_factor_bundle,
)
from alphalattice.control.product_host.research_authoring.feature_activations import (
    feature_catalog_for,
)
from alphalattice.control.product_host.storage.inventory import require_storage_capacity
from alphalattice.control.product_host.storage.retention import require_no_pending_cleanup
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog
from alphalattice.foundation.feature_engine.catalog.crud_contracts import FeatureCatalogRevision
from alphalattice.foundation.feature_engine.catalog.research import (
    ResearchFeatureChange,
    ResearchFeaturePlan,
    plan_research_feature_change,
    research_feature_controls,
)
from alphalattice.foundation.feature_engine.catalog.service import FeatureCatalogCrudPlanner
from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
    PanelClosureArtifactStore,
)
from alphalattice.foundation.feature_engine.producers.factors.catalog import (
    default_extension_kernel_registry,
)

CATEGORY = "research-feature-plans"


class ResearchFeatureDefinitions:
    """Small definition artifacts in the existing Feature store, never a Panel writer."""

    def __init__(self, workspace: Path) -> None:
        """Wire retained definitions and the live workspace storage admission.

        Args:
            workspace: Caller-owned admitted workspace.
        """
        self.workspace = workspace.resolve()
        self.store = PanelClosureArtifactStore(ArtifactResolver(self.workspace / "artifacts"))

    def read(self, plan_hash: str) -> ResearchFeaturePlan:
        """Reopen and validate the exact retained feature definition plan.

        Args:
            plan_hash: Exact plan content identity.

        Returns:
            Validated retained feature plan.

        Raises:
            ValueError: Plan identity differs from the requested reference.
        """
        value: ResearchFeaturePlan = ResearchFeaturePlan.model_validate_json(
            json.dumps(self.store.load_json(category=CATEGORY, content_hash=plan_hash))
        )
        if value.plan_hash != plan_hash:
            raise ValueError("feature_research.plan_reference_mismatch")
        return value

    def _base(
        self, input_binding_hash: str, parent_plan_hash: str | None
    ) -> tuple[FeatureCatalog, FeatureCatalogRevision, dict[str, Any]]:
        # Definition planning reads sealed metadata, not every numerical file.
        # A build must independently admit and verify its actual inputs.
        bundle = read_factor_bundle(self.workspace, input_binding_hash, verify=False)
        _source, artifacts = factor_input_paths(self.workspace, input_binding_hash)
        resolver = ArtifactResolver(artifacts)
        panel = resolver.load_feature_panel_manifest(
            resolver.feature_panel_manifest_uri(bundle.panel_snapshot_hash)
        )
        if parent_plan_hash is not None:
            parent = self.read(parent_plan_hash)
            if (
                parent.request.input_binding_hash != input_binding_hash
                or parent.source_panel_snapshot_hash != bundle.panel_snapshot_hash
            ):
                raise ValueError("feature_research.parent_input_mismatch")
            return FeatureCatalog.from_payload(parent.candidate_payload), parent.candidate, panel
        # The catalog the Panel was built under: the shipped one, or one this workspace's
        # activations made (EX).
        resolved = feature_catalog_for(
            self.workspace, str(panel["safe_summary"]["lineage"]["catalog_hash"])
        )
        if resolved is None:
            raise ValueError("feature_research.recorded_catalog_definition_unavailable")
        catalog = resolved
        revision = FeatureCatalogCrudPlanner(default_extension_kernel_registry()).revision(catalog)
        return catalog, revision, panel

    def controls(
        self, input_binding_hash: str, parent_plan_hash: str | None = None
    ) -> dict[str, Any]:
        """Project base definition metadata and an explicit local raw/prepared-value edit template.

        Args:
            input_binding_hash: Exact admitted input revision.
            parent_plan_hash: Optional exact retained parent definition plan.

        Returns:
            Installed controls, source counts/period, local definitions and edit template; metadata
            reads grant no build admission.
        """
        catalog, revision, panel = self._base(input_binding_hash, parent_plan_hash)
        return {
            **research_feature_controls(),
            "input_binding_hash": input_binding_hash,
            "source_panel_snapshot_hash": panel["snapshot_hash"],
            "base_revision": revision.model_dump(mode="json"),
            "source_period": {"start": panel["history_start"], "end": panel["as_of_session"]},
            "source_listing_count": panel["active_listing_count"],
            "source_input_factor_count": len(panel["safe_summary"]["factor_catalog_summary"]),
            "local_definition_factor_count": len(catalog.factor_ids),
            "defined_but_not_in_source": sorted(
                set(catalog.factor_ids) - set(panel["safe_summary"]["factor_catalog_summary"])
            ),
            "template": {
                "input_binding_hash": input_binding_hash,
                "base_revision_hash": revision.revision_hash,
                "parent_plan_hash": parent_plan_hash,
                "edits": [],
                "reason": "",
            },
            "numerical_materialization_available": True,
            "materialization_scope": (
                "RESEARCH_LOCAL_RAW_AND_PREPROCESSED_VALUES_NOT_GLOBAL_INSTALLATION"
            ),
            "build_outputs": ["RAW_VALUES", "PREPROCESSED_VALUES"],
            "source_verification": "METADATA_ONLY_NOT_BUILD_ADMISSION",
            "definitions": [v.model_dump(mode="json") for v in catalog.factors],
        }

    def plan(self, document: dict[str, Any]) -> ResearchFeaturePlan:
        """Seal and publish an explicit local definition delta under write capacity ownership.

        Args:
            document: Explicit feature change declaration.

        Returns:
            Exact plan after publication and equal readback.

        Raises:
            ValueError: Cleanup, declaration/source, capacity or exact publication/readback is
                invalid.
        """
        require_no_pending_cleanup(self.workspace)
        request = ResearchFeatureChange.model_validate_json(json.dumps(document))
        catalog, base, panel = self._base(request.input_binding_hash, request.parent_plan_hash)
        plan = plan_research_feature_change(
            catalog=catalog,
            base=base,
            source_panel_snapshot_hash=str(panel["snapshot_hash"]),
            request=request,
            preprocessing_recipes=(
                {}
                if request.parent_plan_hash is None
                else self.read(request.parent_plan_hash).preprocessing_recipes
            ),
        )
        payload = plan.model_dump(mode="json")
        path = self.store.root / CATEGORY / f"{plan.plan_hash}.json"
        if not path.exists():
            require_storage_capacity(
                self.workspace,
                additional_bytes=len(
                    json.dumps(
                        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    ).encode()
                ),
            )
        self.store.publish_json(category=CATEGORY, content_hash=plan.plan_hash, payload=payload)
        if self.read(plan.plan_hash) != plan:
            raise ValueError("feature_research.plan_readback_mismatch")
        return plan

    def readback(self, plan_hash: str) -> dict[str, Any]:
        """Read an exact definition delta and its legal build/read/revise requests.

        Args:
            plan_hash: Exact retained definition plan.

        Returns:
            Candidate revision, delta/work, authored document/YAML and zero numerical calls;
            materialization and downstream admission are not proved.
        """
        plan = self.read(plan_hash)
        document = plan.request.model_dump(mode="json")
        return {
            "status": "DEFINITION_PLANNED",
            "plan_hash": plan.plan_hash,
            "scope": plan.scope,
            "input_binding_hash": plan.request.input_binding_hash,
            "source_panel_snapshot_hash": plan.source_panel_snapshot_hash,
            "base_revision_hash": plan.base.revision_hash,
            "candidate_revision": plan.candidate.model_dump(mode="json"),
            "delta": plan.delta.model_dump(mode="json"),
            "work": plan.work.model_dump(mode="json"),
            "work_scope": "DEFINITION_DELTA_NOT_PROOF_OF_PARENT_MATERIALIZATION",
            "required_fields": list(plan.required_fields),
            "numerical_call_count": 0,
            "document": document,
            # The declaration itself, as `feature plan --file` reads it and the controls'
            # template is, never a whole request around it (V559).
            "yaml": yaml.safe_dump(document, allow_unicode=True, sort_keys=False),
            "materialization": "NOT_CHECKED_USE_EXACT_BUILD_TASK_NO_FACTOR_OR_ALPHA_ADMISSION",
            "next_requests": {
                "build": {
                    "operation": "FEATURE_CATALOG_BUILD",
                    "feature_plan_hash": plan.plan_hash,
                    "feature_output": "PREPROCESSED_VALUES",
                },
                "readback": {
                    "operation": "FEATURE_CATALOG_READBACK",
                    "feature_plan_hash": plan.plan_hash,
                },
                "revise": {
                    "operation": "FEATURE_CATALOG_CONTROLS",
                    "input_binding_hash": plan.request.input_binding_hash,
                    "feature_plan_hash": plan.plan_hash,
                },
            },
        }
