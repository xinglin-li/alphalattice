"""Whether a recorded Risk Program names the method the installed code seals (V314).

A Risk Program binds the selected estimator's numerical binding and the installed catalog's hash,
and both move with the estimator's code, so an edit that moved no number refused every sealed Risk
study's replay (`risk_research.execution_plan_changed`, two in U0 at `40235563`). Such an edit is
recorded as a successor of each (`risk_research.estimator.<adapter>`,
`risk_research.estimator_catalog`), which the executor already follows; a replay follows them as
well, as the Alpha and Portfolio plans follow theirs (LAWS.md ID1).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from alphalattice.investment.risk_research.estimators.catalog import (
    ESTIMATOR_CATALOG_ROLE,
    numerical_binding_role,
)
from alphalattice.investment.risk_research.experiments.compiler import RiskExperimentCompiler
from alphalattice.investment.risk_research.experiments.contracts import (
    RiskDevelopmentProgramBinding,
)
from alphalattice.kernel.shared_kernel.identity_successors import is_current, predecessors
from alphalattice.protocols.research_authoring.contracts import (
    ResearchExperimentEnvelope,
    ResolvedResearchAuthority,
    SealedResearchProgram,
)

_MEANING = ("kind", "envelope_hash", "resolved_sessions", "parameter_domain_hash", "authority_hash")
"""What no estimator edit moves: a Program that differs in any of these is another study."""


def program_follows_recorded_moves(
    recorded: SealedResearchProgram,
    installed: SealedResearchProgram,
    binding: RiskDevelopmentProgramBinding,
    *,
    root: Path | None = None,
) -> bool:
    """Whether ``recorded`` names the method ``installed`` seals, through recorded moves.

    It does when the two are equal, or when every field no estimator edit moves is equal, the
    recorded catalog follows recorded moves to the installed one, and the recorded method binding
    is ``binding`` rebuilt with a recorded predecessor of its selected estimator's numerical
    binding. The Program's own hashes follow from those; how they are derived is the plan's
    implementation, checked through its successors beside this.

    Args:
        recorded: The plan's sealed Program.
        installed: The Program the installed code seals for the same document.
        binding: The installed development binding behind ``installed``.
        root: The checkout whose successor records are read.

    Returns:
        Whether a replay may read the evidence sealed under ``recorded`` as current.
    """

    if recorded == installed:
        return True
    if any(getattr(recorded, name) != getattr(installed, name) for name in _MEANING):
        return False
    if installed.method_binding_hash != binding.development_binding_hash:
        return False
    if not is_current(
        ESTIMATOR_CATALOG_ROLE, recorded.catalog_hash, installed.catalog_hash, root=root
    ):
        return False
    role = numerical_binding_role(binding.selected_adapter_id)
    return any(
        RiskDevelopmentProgramBinding.create(
            catalog_hash=recorded.catalog_hash,
            selected_adapter_id=binding.selected_adapter_id,
            selected_numerical_binding_hash=value,
            recipe_hash=binding.recipe_hash,
            parameter_domain_hash=binding.parameter_domain_hash,
        ).development_binding_hash
        == recorded.method_binding_hash
        for value in predecessors(role, binding.selected_numerical_binding_hash, root=root)
    )


def replay_follows_recorded_moves(
    recorded: SealedResearchProgram,
    installed: SealedResearchProgram,
    *,
    envelope: ResearchExperimentEnvelope,
    document: Mapping[str, Any],
    authority: ResolvedResearchAuthority,
) -> bool:
    """``program_follows_recorded_moves`` with the binding the installed code compiles.

    Args:
        recorded: The plan's sealed Program.
        installed: The Program the installed code seals for the same document.
        envelope: The document's envelope, as the installed code sealed it.
        document: The plan's document.
        authority: The authority the installed code resolved for it.

    Returns:
        Whether a replay may read the evidence sealed under ``recorded`` as current.
    """

    if recorded == installed:
        return True
    binding = (
        RiskExperimentCompiler()
        .compile_development_program(envelope=envelope, document=document, authority=authority)
        .binding
    )
    return program_follows_recorded_moves(recorded, installed, binding)


__all__ = ["program_follows_recorded_moves", "replay_follows_recorded_moves"]
