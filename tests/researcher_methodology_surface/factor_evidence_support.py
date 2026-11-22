"""Produce one real Factor development checkpoint for Alpha to be authorized by.

Alpha's development foundation binds a Factor checkpoint by hash, so an Alpha
case needs one to exist. It is produced by running the real Factor path -- the
canonical Host executor over the real workspace -- rather than by writing a
plausible JSON file, because a hand-written checkpoint would let the Alpha cases
pass while the two Desks disagreed about what a checkpoint contains.

The evidence is written to a root of its own, separate from both the source
workspace and any Alpha output workspace. That separation is asserted in the
product; here it is simply how the fixture is arranged.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from alphalattice.control.product_host.composition.research_authoring import (
    build_research_program_workflow,
    host_resolved_factor_inventory,
)
from alphalattice.control.product_host.research_authoring.authority import (
    WorkspaceResearchAuthorityResolver,
)
from alphalattice.control.product_host.research_authoring.execution import (
    build_installed_desk_executors,
)
from alphalattice.control.product_host.research_authoring.factor_inputs import logical_panel_owner
from alphalattice.control.research_program.authoring.document import load_authoring_document
from alphalattice.control.research_program.authoring.workflow import ResearchProgramWorkflow
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.factor_research.evaluation.oos_evidence import (
    build_factor_evidence_policy,
)
from alphalattice.foundation.factor_research.evaluation.redundancy import (
    build_factor_redundancy_policy,
)
from alphalattice.foundation.factor_research.experiments.authoring import (
    FactorInventoryEntry,
    factor_inventory_from_panel_manifest,
)
from alphalattice.foundation.factor_research.experiments.development_evidence import (
    factor_development_receipt_handle,
)
from alphalattice.foundation.feature_engine.panels.development_input import (
    ResolvedDevelopmentFeatureInput,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.actor_execution.contracts import ActorKind
from alphalattice.protocols.research_authoring.contracts import ResearchExperimentEnvelope
from tests.researcher_methodology_surface.real_workspace import (
    FACTOR_SCALE,
    MARKET_PROFILE_ID,
    RealRiskWorkspace,
)

CASE_ROOT = Path(__file__).resolve().parent
_FIXTURE = CASE_ROOT / "fixtures" / "factor_screening_development.yaml"

# Scale, not science. The production defaults require a hundred common listings
# per redundancy pair and a hundred cross-sectional observations; this workspace
# has ten listings by design, because its purpose is to be a real workspace that
# builds in ninety seconds. The thresholds in the product are untouched.
DEVELOPMENT_EVIDENCE_POLICY = build_factor_evidence_policy(minimum_cross_section_observations=10)
DEVELOPMENT_REDUNDANCY_POLICY = build_factor_redundancy_policy(
    minimum_common_listings=10, minimum_formal_periods=3
)


def short_lookback_factor_input(workspace: RealRiskWorkspace) -> ResolvedDevelopmentFeatureInput:
    """Select the short-lookback axis of a real, verified development Panel."""
    resolver = ArtifactResolver(workspace.artifact_root)
    panel = dict(resolver.load_feature_panel_manifest(workspace.panel_manifest_ref))
    marker = logical_panel_owner(workspace.workspace).verify_snapshot(workspace.panel_snapshot_hash)
    # The two-year calendar has three formal folds before the holdout.
    # Short-lookback recipes have observations in that window on the admitted Panel.
    factor_ids = tuple(
        spec.factor_id for spec in workspace.feature_catalog.factors if spec.window_sessions <= 21
    )
    binding = canonical_hash(
        {"manifest": workspace.panel_snapshot_hash, "inherited_factor_ids": factor_ids}
    )
    return ResolvedDevelopmentFeatureInput.compose(
        source_handle=f"research-features@{binding}",
        input_binding_hash=binding,
        base_manifest=panel,
        base_artifact_root=workspace.artifact_root,
        base_logical_panel_hash=marker.logical_panel_hash,
        base_logical_semantic_index_hash=marker.logical_semantic_index_hash,
        inherited_factor_ids=factor_ids,
        overlay=None,
        overlay_root=workspace.artifact_root,
    )


def factor_inventory(
    workspace: RealRiskWorkspace,
    feature_input: ResolvedDevelopmentFeatureInput | None,
    *,
    market_profile_id: str = MARKET_PROFILE_ID,
) -> tuple[FactorInventoryEntry, ...]:
    if feature_input is not None:
        return factor_inventory_from_panel_manifest(feature_input.panel_manifest)
    return host_resolved_factor_inventory(
        workspace=workspace.workspace, market_profile_id=market_profile_id
    )


def factor_workflow(
    workspace: RealRiskWorkspace,
    root: Path,
    document: dict[str, Any],
    inventory: tuple[FactorInventoryEntry, ...],
    feature_input: ResolvedDevelopmentFeatureInput | None,
    *,
    sufficient: bool = False,
) -> tuple[ResearchProgramWorkflow, WorkspaceResearchAuthorityResolver]:
    if feature_input is not None:
        document["experiment"]["data_snapshot_handle"] = feature_input.source_handle
    envelope = ResearchExperimentEnvelope.create(**document["experiment"])
    authority = WorkspaceResearchAuthorityResolver(
        workspace=workspace.workspace,
        feature_catalog=workspace.feature_catalog,
        feature_kernels=workspace.feature_kernels,
        feature_input=feature_input,
    )
    short_policy = feature_input is None and not sufficient
    executors = build_installed_desk_executors(
        envelope=envelope,
        workspace=workspace.workspace,
        workspace_root=root,
        document=document,
        feature_catalog=workspace.feature_catalog,
        feature_kernels=workspace.feature_kernels,
        feature_input=feature_input,
        factor_evidence_policy=DEVELOPMENT_EVIDENCE_POLICY if short_policy else None,
        factor_redundancy_policy=DEVELOPMENT_REDUNDANCY_POLICY if short_policy else None,
    )
    if sufficient:
        (factor,) = (each for each in executors if hasattr(each, "minimum_required_listings"))
        assert factor.minimum_required_listings == FACTOR_SCALE.minimum_factor_listings == 100
    return build_research_program_workflow(
        workspace=workspace.workspace,
        workspace_root=root,
        executors=executors,
        factor_inventory=inventory,
        authority=authority,
    ), authority


def factor_document(
    inventory: tuple[Any, ...],
    *,
    output_workspace: str | None = None,
    selected: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Replace the compiler fixture's synthetic Factor axis with the published axis."""

    draft = copy.deepcopy(dict(load_authoring_document(_FIXTURE.read_text(encoding="utf-8"))))
    section = dict(draft["factor"])
    section["factor_ids"] = list(
        selected if selected is not None else [entry.factor_id for entry in inventory[:4]]
    )
    draft["factor"] = section
    if output_workspace is not None:
        experiment = dict(draft["experiment"])
        experiment["output_workspace"] = output_workspace
        experiment["baseline_workspace"] = f"{output_workspace}-baseline"
        draft["experiment"] = experiment
    return draft


FACTOR_EVIDENCE_WORKSPACE = "factor-evidence"
"""The output workspace the Factor run writes to, and therefore the evidence root.

The workflow resolves ``workspace_root / output_workspace`` as a run's output
workspace, and the Factor writer publishes its category directly beneath that.
So the root an Alpha run is told to read is that resolved directory, not the
tree above it -- naming it once here is what keeps the writer and the reader
pointing at the same place.
"""


def factor_development_checkpoint(
    workspace: RealRiskWorkspace,
    *,
    root: Path,
    selected: tuple[str, ...] | None = None,
    output_workspace: str = FACTOR_EVIDENCE_WORKSPACE,
    feature_input: ResolvedDevelopmentFeatureInput | None = None,
) -> tuple[str, tuple[str, ...], Path]:
    """Produce the receipt that authorizes Alpha, its selected axis and its exact root."""
    inventory = factor_inventory(workspace, feature_input)
    document = factor_document(inventory, output_workspace=output_workspace, selected=selected)
    workflow, _ = factor_workflow(workspace, root, document, inventory, feature_input)
    evidence, _binding = workflow.run(document, actor_kind=ActorKind.HUMAN, actor_id="researcher")
    handle = factor_development_receipt_handle(evidence.artifact_uris[0])
    return (
        handle,
        tuple(str(value) for value in document["factor"]["factor_ids"]),
        root / output_workspace,
    )


__all__ = [
    "DEVELOPMENT_EVIDENCE_POLICY",
    "DEVELOPMENT_REDUNDANCY_POLICY",
    "FACTOR_EVIDENCE_WORKSPACE",
    "factor_development_checkpoint",
    "factor_document",
]
