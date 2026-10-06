"""Method-neutral development artifacts, shared by every Risk capability.

``HistoricalCovarianceSurface.recipe`` is typed ``CovarianceRecipe``, so it can
only ever describe one recipe schema. Pointing a second method at it would mean
either writing the wrong recipe into that field or giving the new method an
artifact of its own -- and the second choice is worse, because then every
subsequent method adds another branch to whatever decides which artifact to
write.

So there is exactly one development surface contract and both capabilities use
it. What it binds instead of a typed recipe is the ``RiskEstimatorRecipeEnvelope``
the adapter actually received, which is schema-neutral by construction: adapter
id, schema id, parameters, and the schema's own recipe hash.

Current and frozen artifact schemas are untouched. These live in their own
categories with their own identities, and nothing here is ever written to a
current or admitted pointer.
"""

from __future__ import annotations

from datetime import date
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.investment.risk_research.contracts import (
    HistoricalCovarianceChunk,
    RiskFormationEvaluation,
)
from alphalattice.investment.risk_research.estimators.contracts import (
    RiskEstimatorRecipeEnvelope,
)
from alphalattice.investment.risk_research.experiments.contracts import (
    RiskDevelopmentProgramBinding,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

_HASH = r"^[0-9a-f]{64}$"

DEVELOPMENT_SURFACE_CATEGORY = "development/covariance-surfaces"
DEVELOPMENT_DIAGNOSTICS_CATEGORY = "development/covariance-diagnostics"


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class RiskDevelopmentDiagnostics(_Contract):
    """Per-formation evaluations for one development build, any method.

    ``RiskFormationEvaluation`` is already method-neutral -- eigenvalues, trace,
    condition number, realized-versus-predicted -- so a second method needs no
    new diagnostic shape, only its own values.
    """

    kind: Literal["RiskDevelopmentDiagnostics"] = "RiskDevelopmentDiagnostics"
    input_binding_hash: str = Field(pattern=_HASH)
    recipe_hash: str = Field(pattern=_HASH)
    evaluations: tuple[RiskFormationEvaluation, ...] = Field(min_length=1)
    component_shrinkages: tuple[tuple[str, tuple[float, ...]], ...] = ()
    """Per-formation intensity of each named component of a blended method.

    ``RiskFormationEvaluation.shrinkage`` is one number, which is the whole
    story for a method that fits one estimator and only a summary for a method
    that fits several. It is inside the frozen execution closure and cannot
    grow a field, and the parts are not recoverable afterwards without
    refitting -- so they are recorded here, in the development diagnostics,
    which no published identity depends on.

    Empty for a single-component method. The series is per formation and in
    formation order, so a Campaign can aggregate it any way it needs without
    the writer having chosen a statistic on its behalf.
    """

    diagnostics_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(
        cls,
        *,
        input_binding_hash: str,
        recipe_hash: str,
        evaluations: tuple[RiskFormationEvaluation, ...],
        component_shrinkages: tuple[tuple[str, tuple[float, ...]], ...] = (),
    ) -> Self:
        values = {
            "kind": "RiskDevelopmentDiagnostics",
            "input_binding_hash": input_binding_hash,
            "recipe_hash": recipe_hash,
            "evaluations": [value.model_dump(mode="json") for value in evaluations],
            "component_shrinkages": [[name, list(series)] for name, series in component_shrinkages],
        }
        return cls(
            input_binding_hash=input_binding_hash,
            recipe_hash=recipe_hash,
            evaluations=evaluations,
            component_shrinkages=component_shrinkages,
            diagnostics_hash=str(canonical_hash(values)),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        expected = canonical_hash(self.model_dump(mode="json", exclude={"diagnostics_hash"}))
        if self.diagnostics_hash != expected:
            raise ValueError("risk_research.development_diagnostics_identity_invalid")
        return self


class RiskDevelopmentCheckpoint(_Contract):
    """A mutable working prefix; terminal evidence remains content-addressed."""

    execution_key: str = Field(pattern=_HASH)
    expected_formation_count: int = Field(ge=1)
    chunks: tuple[HistoricalCovarianceChunk, ...]
    evaluations: tuple[RiskFormationEvaluation, ...]
    component_shrinkages: tuple[tuple[str, tuple[float, ...]], ...] = ()
    checkpoint_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        draft = cls.model_construct(**values, checkpoint_hash="")
        return cls(
            **values,
            checkpoint_hash=str(
                canonical_hash(draft.model_dump(mode="json", exclude={"checkpoint_hash"}))
            ),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_prefix(self) -> Self:
        sessions = tuple(s for chunk in self.chunks for s in chunk.formation_sessions)
        hashes = tuple(h for chunk in self.chunks for h in chunk.matrix_hashes)
        if (
            len(sessions) != len(self.evaluations)
            or len(sessions) > self.expected_formation_count
            or sessions != tuple(sorted(set(sessions)))
            or sessions != tuple(e.formation_session for e in self.evaluations)
            or hashes != tuple(e.matrix_hash for e in self.evaluations)
            or any(len(series) != len(sessions) for _, series in self.component_shrinkages)
            or self.checkpoint_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"checkpoint_hash"}))
        ):
            raise ValueError("risk_research.development_checkpoint_invalid")
        return self


class RiskDevelopmentCovarianceSurface(_Contract):
    """One bounded development build, produced by any installed capability.

    The method is described by ``recipe_envelope`` and
    ``selected_numerical_binding_hash`` rather than by a typed recipe field, so
    this contract does not have to change when a capability with a different
    recipe schema is installed.
    """

    kind: Literal["RiskDevelopmentCovarianceSurface"] = "RiskDevelopmentCovarianceSurface"
    identity_class: Literal["DEVELOPMENT_EVIDENCE_ONLY_NEVER_PUBLISHED"] = (
        "DEVELOPMENT_EVIDENCE_ONLY_NEVER_PUBLISHED"
    )
    input_binding_hash: str = Field(pattern=_HASH)
    capability_handle: str = Field(min_length=1, max_length=128)
    recipe_envelope: RiskEstimatorRecipeEnvelope
    recipe_hash: str = Field(pattern=_HASH)
    """The schema's *own* recipe identity, which is what the Program binds.

    Distinct from ``recipe_envelope.recipe_hash``, which covers the envelope --
    adapter id, schema id and parameters. Both are real identities of different
    things, and conflating them would silently compare unlike values.
    """
    program_binding: RiskDevelopmentProgramBinding
    """The whole Program binding, not a copy of four of its hashes.

    Carrying the hashes flat made the surface unable to answer for itself. Its
    method fields were free-standing strings, so a surface could name a
    ``development_binding_hash`` that no combination of its other fields would
    ever produce, and replay had nothing to check that against.

    Embedded, the binding revalidates on read: parsing the artifact recomputes
    ``selected_method_binding_hash`` and ``development_binding_hash`` from the
    fields underneath them, so an edited method field fails to parse rather than
    verifying against itself. What replay then has to establish is only the part
    the artifact cannot know -- that this binding is the Program being replayed.
    """

    ordered_listing_ids: tuple[str, ...] = Field(min_length=1)
    formation_sessions: tuple[date, ...] = Field(min_length=1)
    chunks: tuple[HistoricalCovarianceChunk, ...] = Field(min_length=1)
    diagnostics_hash: str = Field(pattern=_HASH)
    limitations: tuple[str, ...] = Field(min_length=1)
    surface_hash: str = Field(pattern=_HASH)

    @property
    def selected_numerical_binding_hash(self) -> str:
        return self.program_binding.selected_numerical_binding_hash

    @property
    def selected_method_binding_hash(self) -> str:
        return self.program_binding.selected_method_binding_hash

    @property
    def development_binding_hash(self) -> str:
        return self.program_binding.development_binding_hash

    @property
    def numerical_environment_hash(self) -> str | None:
        """The environment a Program sealed before E0 declared; None since (LAWS.md ID6)."""
        return self.program_binding.numerical_environment_hash

    @classmethod
    def create(cls, **values: object) -> Self:
        draft = dict(values)
        draft.pop("surface_hash", None)
        provisional = cls.model_construct(**draft, surface_hash="0" * 64)
        identity = provisional.model_dump(mode="json", exclude={"surface_hash"})
        return cls(**draft, surface_hash=str(canonical_hash(identity)))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        # The surface's own method fields must agree with the Program binding it
        # carries. Without this the artifact could describe one method in its
        # envelope and a different one in its binding, and every hash on it would
        # still be internally valid.
        if self.recipe_hash != self.program_binding.recipe_hash:
            raise ValueError("risk_research.development_surface_recipe_not_its_program")
        if self.capability_handle != self.recipe_envelope.recipe_schema_id:
            raise ValueError("risk_research.development_surface_capability_route_invalid")
        if self.recipe_envelope.adapter_id != self.program_binding.selected_adapter_id:
            raise ValueError("risk_research.development_surface_adapter_route_invalid")
        if sum(chunk.matrix_count for chunk in self.chunks) != len(self.formation_sessions):
            raise ValueError("risk_research.development_surface_count_invalid")
        if any(chunk.asset_count != len(self.ordered_listing_ids) for chunk in self.chunks):
            raise ValueError("risk_research.development_surface_axis_invalid")
        computed = tuple(session for chunk in self.chunks for session in chunk.formation_sessions)
        # Order, not membership: the covariance axis is positional.
        if computed != tuple(self.formation_sessions):
            raise ValueError("risk_research.development_surface_session_axis_invalid")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"surface_hash"}))
        if self.surface_hash != expected:
            raise ValueError("risk_research.development_surface_identity_invalid")
        return self


__all__ = [
    "DEVELOPMENT_DIAGNOSTICS_CATEGORY",
    "DEVELOPMENT_SURFACE_CATEGORY",
    "RiskDevelopmentCovarianceSurface",
    "RiskDevelopmentDiagnostics",
]
