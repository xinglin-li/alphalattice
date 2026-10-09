"""Compile one authored Risk experiment into a typed development Program.

The Desk owns its methodology contract. The common envelope only says which
snapshot, universe, dates, budget, determinism, and workspace were requested;
everything about *which estimator with which parameters* is validated here
against the Desk's installed catalog and its declared parameter domain.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import Field

from alphalattice.investment.risk_research.estimators.capability import (
    RiskCapabilityError,
    RiskRecipeAdmission,
)
from alphalattice.investment.risk_research.estimators.catalog import (
    RiskEstimatorCatalog,
    build_installed_risk_estimator_catalog,
)
from alphalattice.investment.risk_research.experiments.contracts import (
    RISK_EXPERIMENT_KIND,
    RiskDevelopmentProgramBinding,
)
from alphalattice.investment.risk_research.experiments.policy import enforce_execution_policy
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    DeskProgramCompilation,
    DeskSection,
    ResearchExperimentEnvelope,
    ResolvedResearchAuthority,
    refuse_unknown_section_keys,
)


class RiskEstimatorSection(DeskSection):
    """The estimator a Risk study runs: an installed capability and its parameters."""

    capability: str
    """The installed covariance capability's handle, as `study controls` lists it."""
    parameters: dict[str, Any] = Field(default_factory=dict)
    """The capability's parameters, each on an axis its parameter domain declares."""


class RiskSection(DeskSection):
    """A Risk covariance study's section."""

    estimator: RiskEstimatorSection
    """The estimator the study runs."""


@dataclass(frozen=True, slots=True)
class RiskCompiledMethod:
    """What compilation decided, including the recipe that must actually run.

    The admission is returned rather than recomputed by the executor. The
    executor used to receive only identity hashes, so the numerical builder was
    free to construct its own recipe -- and did. Handing the sealed object along
    is what makes the authored parameters reach the estimator.

    ``admission.sealed_recipe`` is typed ``object`` all the way through this
    module: the compiler admitted a schema it does not name, and narrowing it is
    the job of whichever executor knows how to run it.
    """

    binding: RiskDevelopmentProgramBinding
    desk_program_hash: str
    admission: RiskRecipeAdmission


class RiskExperimentCompiler:
    """Desk-owned compiler; produces identity, never numerical results."""

    kind = RISK_EXPERIMENT_KIND

    def __init__(
        self,
        estimators: RiskEstimatorCatalog | None = None,
        *,
        playpen_root: Path | None = None,
    ) -> None:
        self._estimators = estimators or build_installed_risk_estimator_catalog()
        self._playpen_root = playpen_root or resolve_playpen_root(Path(__file__))

    def compile_desk_program(
        self,
        *,
        envelope: ResearchExperimentEnvelope,
        document: Mapping[str, Any],
        authority: ResolvedResearchAuthority,
    ) -> DeskProgramCompilation:
        compiled = self.compile_development_program(
            envelope=envelope,
            document=document,
            authority=authority,
        )
        binding = compiled.binding
        return DeskProgramCompilation(
            desk_program_hash=compiled.desk_program_hash,
            catalog_hash=binding.catalog_hash,
            # The Desk's own implementation identity, carried by name so the
            # sealed Program consumes it rather than merely referencing a
            # catalog that could change underneath it.
            method_binding_hash=binding.development_binding_hash,
            parameter_domain_hash=binding.parameter_domain_hash,
        )

    def compile_development_program(
        self,
        *,
        envelope: ResearchExperimentEnvelope,
        document: Mapping[str, Any],
        authority: ResolvedResearchAuthority,
    ) -> RiskCompiledMethod:
        """Validate the authored method and return its development identity."""

        section = document.get("risk")
        if not isinstance(section, dict):
            raise AuthoringError("risk_research.authoring_section_missing")
        refuse_unknown_section_keys(section, RiskSection, place="risk")
        estimator = section.get("estimator")
        if not isinstance(estimator, dict):
            raise AuthoringError("risk_research.authoring_estimator_missing")
        refuse_unknown_section_keys(estimator, RiskEstimatorSection, place="risk.estimator")
        capability = estimator.get("capability")
        if not isinstance(capability, str) or not capability:
            raise AuthoringError("risk_research.authoring_capability_invalid")
        parameters = estimator.get("parameters", {})
        if not isinstance(parameters, dict):
            raise AuthoringError("risk_research.authoring_parameters_invalid")

        # The capability is the only route from an authored handle to a sealed
        # recipe. This compiler no longer knows which schema it just admitted:
        # it names no recipe contract, no adapter, and no family, so a method is
        # added by installing a capability rather than by editing here.
        try:
            admission = self._estimators.admit_recipe(
                capability_handle=capability,
                parameters=parameters,
                # The envelope's declared seed, checked against what the selected
                # capability can actually consume. It is not passed to any
                # estimator: a deterministic method that accepted a seed would
                # give the same numbers two Program identities.
                seed=envelope.determinism.seed,
            )
        except RiskCapabilityError as error:
            raise AuthoringError(str(error)) from error

        catalog_binding = self._estimators.binding
        binding = RiskDevelopmentProgramBinding.create(
            # Governance: what the Host installed.
            catalog_hash=catalog_binding.catalog_hash,
            # Methodology: what is about to compute. Only the selected
            # capability. Folding every installed capability in here made
            # "the Host installed something else" indistinguishable from
            # "these numbers would come out differently".
            selected_adapter_id=admission.adapter_id,
            selected_numerical_binding_hash=admission.selected_numerical_binding_hash,
            recipe_hash=admission.recipe_hash,
            parameter_domain_hash=admission.parameter_domain_hash,
            # The code that runs is bound by the study plan's implementation hash
            # (binding plan, P), not by the Program; the environment is provenance,
            # recorded beside each estimate (LAWS.md ID6).
        )
        desk_program_hash = str(
            canonical_hash(
                {
                    "kind": self.kind,
                    # The envelope is bound once, by the sealed Program itself (B6).
                    "adapter_id": admission.adapter_id,
                    "development_binding_hash": binding.development_binding_hash,
                    # The resolved authority, not the requested range: the
                    # Program is bound to the Panel and universe that actually
                    # exist, so it cannot name a session the data never had.
                    "authority_hash": authority.authority_hash,
                }
            )
        )
        # The declared budget and threads, checked where every command's seal compiles, so a
        # PLAN that passes is never refused the same rule at RUN.
        enforce_execution_policy(
            envelope=envelope,
            planned_numerical_calls=len(authority.sessions),
            adapter_bindings=tuple(
                adapter.describe_numerical_binding() for adapter in self._estimators.adapters
            ),
        )
        return RiskCompiledMethod(
            binding=binding,
            desk_program_hash=desk_program_hash,
            admission=admission,
        )


__all__ = [
    "RISK_EXPERIMENT_KIND",
    "RiskCompiledMethod",
    "RiskEstimatorSection",
    "RiskExperimentCompiler",
    "RiskSection",
]
