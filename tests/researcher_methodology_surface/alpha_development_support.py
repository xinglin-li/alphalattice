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
    host_resolved_factor_inventory,
)
from alphalattice.control.product_host.research_authoring.execution import (
    build_installed_desk_executors,
)
from alphalattice.control.research_program.authoring.document import load_authoring_document
from alphalattice.investment.alpha_research.experiments.policies import (
    load_alpha_split_policy,
)
from tests.researcher_methodology_surface.factor_evidence_support import (
    factor_development_checkpoint,
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
    """Publish outcomes and one Factor checkpoint, as a plain callable.

    Extracted from the execution suite's fixture so a case in another module
    can obtain the same authority without importing a pytest fixture, which
    is not a function.
    """

    publish_causal_outcomes(workspace, at=_FROZEN_AT)
    return factor_development_checkpoint(workspace, root=root)


def _workflow(
    workspace: RealRiskWorkspace,
    document: dict[str, Any],
    *,
    evidence_root: Path,
    workspace_root: Path,
) -> Any:
    envelope_section = dict(document["experiment"])
    from alphalattice.protocols.research_authoring.contracts import ResearchExperimentEnvelope

    envelope = ResearchExperimentEnvelope.create(**envelope_section)
    executors = build_installed_desk_executors(
        envelope=envelope,
        workspace=workspace.workspace,
        workspace_root=workspace_root,
        document=document,
        factor_evidence_root=evidence_root,
        alpha_split_policy=load_alpha_split_policy(),
    )
    return build_research_program_workflow(
        workspace=workspace.workspace,
        workspace_root=workspace_root,
        executors=executors,
        factor_inventory=host_resolved_factor_inventory(
            workspace=workspace.workspace,
            market_profile_id=envelope.universe_handle,
        ),
        alpha_compiler=executors[0].compiler,
    )
