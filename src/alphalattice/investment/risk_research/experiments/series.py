"""A compact immutable index over date-scoped Risk development runs; no estimator."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.investment.risk_research.experiments.development_artifacts import (
    DEVELOPMENT_DIAGNOSTICS_CATEGORY,
    DEVELOPMENT_SURFACE_CATEGORY,
    RiskDevelopmentCovarianceSurface,
    RiskDevelopmentDiagnostics,
)
from alphalattice.investment.risk_research.experiments.window import (
    RISK_INPUT_BINDING_CATEGORY,
    RiskDevelopmentInputBinding,
)
from alphalattice.investment.risk_research.surfaces.artifacts import RiskArtifactStore
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.research_authoring.contracts import ResolvedResearchAuthority

SERIES_CATEGORY = "development/covariance-series"
_HASH = r"^[0-9a-f]{64}$"


class RiskSeriesMember(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)
    input_binding_hash: str = Field(pattern=_HASH)
    surface_hash: str = Field(pattern=_HASH)
    diagnostics_hash: str = Field(pattern=_HASH)


class RiskDevelopmentSeries(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: str = "RiskDevelopmentSeries"
    program_hash: str = Field(pattern=_HASH)
    authority: ResolvedResearchAuthority
    members: tuple[RiskSeriesMember, ...] = Field(min_length=2)
    diagnostics_hash: str = Field(pattern=_HASH)
    series_hash: str = Field(pattern=_HASH)

    @property
    def input_binding_hash(self) -> str:
        return str(
            canonical_hash(
                {
                    "kind": "RiskDevelopmentInputSeries",
                    "authority": self.authority.authority_hash,
                    "bindings": tuple(member.input_binding_hash for member in self.members),
                }
            )
        )

    @classmethod
    def create(cls, **values: object) -> Self:
        draft = cls.model_construct(**values, series_hash="")
        return cls(
            **values,
            series_hash=canonical_hash(draft.model_dump(mode="json", exclude={"series_hash"})),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def identity(self) -> Self:
        if self.kind != "RiskDevelopmentSeries" or len(self.members) != len(
            self.authority.listing_scopes
        ):
            raise ValueError("risk_research.series_scope_invalid")
        if self.series_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"series_hash"})
        ):
            raise ValueError("risk_research.series_identity_invalid")
        return self


def combined_diagnostics(
    *, input_binding_hash: str, children: tuple[RiskDevelopmentDiagnostics, ...]
) -> RiskDevelopmentDiagnostics:
    if not children or len({child.recipe_hash for child in children}) != 1:
        raise ValueError("risk_research.series_method_mismatch")
    names = tuple(name for name, _ in children[0].component_shrinkages)
    if any(tuple(name for name, _ in child.component_shrinkages) != names for child in children):
        raise ValueError("risk_research.series_component_mismatch")
    return RiskDevelopmentDiagnostics.create(
        input_binding_hash=input_binding_hash,
        recipe_hash=children[0].recipe_hash,
        evaluations=tuple(row for child in children for row in child.evaluations),
        component_shrinkages=tuple(
            (
                name,
                tuple(
                    value for child in children for value in dict(child.component_shrinkages)[name]
                ),
            )
            for name in names
        ),
    )


def load_series_member(
    store: RiskArtifactStore, member: RiskSeriesMember
) -> tuple[
    RiskDevelopmentInputBinding, RiskDevelopmentCovarianceSurface, RiskDevelopmentDiagnostics
]:
    return (
        RiskDevelopmentInputBinding.model_validate(
            store.load_json(
                category=RISK_INPUT_BINDING_CATEGORY,
                uri=store.uri(RISK_INPUT_BINDING_CATEGORY, member.input_binding_hash),
                identity_field="input_binding_hash",
            )
        ),
        RiskDevelopmentCovarianceSurface.model_validate(
            store.load_json(
                category=DEVELOPMENT_SURFACE_CATEGORY,
                uri=store.uri(DEVELOPMENT_SURFACE_CATEGORY, member.surface_hash),
                identity_field="surface_hash",
            )
        ),
        RiskDevelopmentDiagnostics.model_validate(
            store.load_json(
                category=DEVELOPMENT_DIAGNOSTICS_CATEGORY,
                uri=store.uri(DEVELOPMENT_DIAGNOSTICS_CATEGORY, member.diagnostics_hash),
                identity_field="diagnostics_hash",
            )
        ),
    )
