"""The Alpha development authoring document, its store root and the research
workflow over the real workspace.

The execution suite and the canonical development chain both drive them;
the chain used to import them from the suite.
Test support beside its owner; nothing here is product authority or evidence.
"""

from __future__ import annotations

import copy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from alphalattice.control.product_host.composition.research_authoring import (
    build_research_program_workflow,
)
from alphalattice.control.product_host.research_authoring.authority import (
    WorkspaceResearchAuthorityResolver,
)
from alphalattice.control.product_host.research_authoring.execution import (
    build_installed_desk_executors,
)
from alphalattice.control.research_program.authoring.document import load_authoring_document
from alphalattice.foundation.feature_engine.panels.development_input import (
    ResolvedDevelopmentFeatureInput,
)
from alphalattice.investment.alpha_research.experiments.policies import (
    load_alpha_split_policy,
)
from alphalattice.protocols.research_authoring.contracts import ResearchExperimentEnvelope
from tests.researcher_methodology_surface.factor_evidence_support import (
    factor_development_checkpoint,
    factor_inventory,
)
from tests.researcher_methodology_surface.real_workspace import (
    RealRiskWorkspace,
    publish_causal_outcomes,
)

CASE_ROOT = Path(__file__).resolve().parent

_FIXTURE = CASE_ROOT / "fixtures" / "alpha_model_development.yaml"

_FROZEN_AT = datetime(2026, 8, 2, tzinfo=UTC)


def _ALPHA_STORE_ROOT(workspace_root: Path) -> Path:
    """Where the executor put its store, spelled once.

    The document names ``workspaces/research/alpha-development`` as its output
    workspace and the executor writes its store beneath that. A reader told a
    different root finds nothing and reports "no receipt" rather than "wrong
    root", which is the failure this constant exists to avoid.
    """

    return workspace_root / "workspaces" / "research" / "alpha-development" / "alpha-development"


def _document(*, handle: str, feature_ids: tuple[str, ...], **overrides: object) -> dict[str, Any]:
    draft = copy.deepcopy(dict(load_authoring_document(_FIXTURE.read_text(encoding="utf-8"))))
    section = dict(draft["alpha"])
    section["factor_evidence_handle"] = handle
    section["ordered_feature_ids"] = list(feature_ids)
    section.update(overrides)
    draft["alpha"] = section
    return draft


def alpha_authority_for(
    workspace: RealRiskWorkspace, root: Path
) -> tuple[str, tuple[str, ...], Path]:
    """Publish outcomes and a Factor checkpoint for direct callers and pytest fixtures."""

    publish_causal_outcomes(workspace, at=_FROZEN_AT)
    return factor_development_checkpoint(workspace, root=root)


def _workflow(
    workspace: RealRiskWorkspace,
    document: dict[str, Any],
    *,
    evidence_root: Path,
    workspace_root: Path,
    feature_input: ResolvedDevelopmentFeatureInput | None = None,
) -> Any:
    if feature_input is not None:
        document["experiment"]["data_snapshot_handle"] = feature_input.source_handle
    envelope = ResearchExperimentEnvelope.create(**document["experiment"])
    executors = build_installed_desk_executors(
        envelope=envelope,
        workspace=workspace.workspace,
        workspace_root=workspace_root,
        document=document,
        factor_evidence_root=evidence_root,
        feature_input=feature_input,
        alpha_split_policy=load_alpha_split_policy(),
    )
    return build_research_program_workflow(
        workspace=workspace.workspace,
        workspace_root=workspace_root,
        executors=executors,
        factor_inventory=factor_inventory(
            workspace, feature_input, market_profile_id=envelope.universe_handle
        ),
        authority=WorkspaceResearchAuthorityResolver(
            workspace=workspace.workspace, feature_input=feature_input
        ),
        alpha_compiler=executors[0].compiler,
    )
