"""Lightweight public contracts for deterministic return-model fitting."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, JsonValue, model_validator

from alphalattice.kernel.knowledge.contracts import (
    PackageArgumentAudit,
)
from alphalattice.kernel.knowledge.enums import ResolutionStatus
from alphalattice.kernel.shared_kernel.domain.base import (
    DomainModel,
    ShortString,
)

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(ge=1)]


class ReturnModelParameter(DomainModel):
    """Named scalar parameter in a return-model configuration."""

    name: ShortString
    value: JsonValue

    @model_validator(mode="after")
    def validate_scalar(self) -> ReturnModelParameter:
        """Reject list and mapping values from the parameter grid."""
        if isinstance(self.value, dict | list):
            raise ValueError("return-model parameter values must be JSON scalars")
        return self


class ReturnModelPackageCall(DomainModel):
    """Accepted package call with its audited normalized arguments."""

    stable_id: ShortString
    arguments: tuple[ReturnModelParameter, ...]
    audit: PackageArgumentAudit

    @model_validator(mode="after")
    def validate_call(self) -> ReturnModelPackageCall:
        """Require sorted arguments matching a resolved package audit."""
        names = tuple(item.name for item in self.arguments)
        if names != tuple(sorted(set(names))):
            raise ValueError("return-model package arguments must be sorted and unique")
        if (
            self.audit.status is not ResolutionStatus.RESOLVED
            or self.audit.reference.stable_id != self.stable_id
            or self.audit.submitted_argument_names != names
            or self.audit.normalized_arguments
            != tuple((item.name, item.value) for item in self.arguments)
        ):
            raise ValueError("return-model package call differs from its accepted audit")
        return self


__all__ = [
    "ReturnModelPackageCall",
    "ReturnModelParameter",
]
