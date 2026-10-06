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
from alphalattice.control.product_host.research_authoring.execution import (
    build_installed_desk_executors,
)
from alphalattice.control.research_program.authoring.document import load_authoring_document
from alphalattice.foundation.factor_research.evaluation.oos_evidence import (
    build_factor_evidence_policy,
)
from alphalattice.foundation.factor_research.evaluation.redundancy import (
    build_factor_redundancy_policy,
)
from alphalattice.foundation.factor_research.experiments.development_evidence import (
    factor_development_receipt_handle,
)
from alphalattice.protocols.actor_execution.contracts import ActorKind
from alphalattice.protocols.research_authoring.contracts import ResearchExperimentEnvelope
from tests.researcher_methodology_surface.real_workspace import RealRiskWorkspace

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


def factor_document(
    inventory: tuple[Any, ...],
    *,
    output_workspace: str | None = None,
    selected: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """The authored Factor document, selecting Factors the real panel publishes.

    The fixture names a synthetic inventory, which is right for the compiler
    cases that prove admission and never execute. A run has to select Factors
    that exist in the published panel.
    """

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
) -> tuple[str, tuple[str, ...], Path]:
    """Run the canonical Factor path once and return its receipt handle.

    Returns the handle, the **selected** axis the receipt reports on, and the
    evidence root it was written to. All three together, because an Alpha
    document must declare a subset of that selected axis and be told that root:
    pairing a handle with someone else's axis or root is exactly the mistake
    returning them separately invites.

    The handle is the *receipt*, not the deterministic child. The child says what
    was computed; the receipt says which authored Program computed it and over
    which axes, and only the second can authorize a downstream run.
    """

    inventory = host_resolved_factor_inventory(
        workspace=workspace.workspace, market_profile_id="us-current-index-research"
    )
    document = factor_document(inventory, output_workspace=output_workspace, selected=selected)
    envelope = ResearchExperimentEnvelope.create(**document["experiment"])
    executors = build_installed_desk_executors(
        envelope=envelope,
        workspace=workspace.workspace,
        workspace_root=root,
        document=document,
        # Development-scale policies, supplied at the composition site that knows
        # the workspace is ten listings rather than lowered in the product.
        factor_evidence_policy=DEVELOPMENT_EVIDENCE_POLICY,
        factor_redundancy_policy=DEVELOPMENT_REDUNDANCY_POLICY,
    )
    workflow = build_research_program_workflow(
        workspace=workspace.workspace,
        workspace_root=root,
        executors=executors,
        factor_inventory=inventory,
    )
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
