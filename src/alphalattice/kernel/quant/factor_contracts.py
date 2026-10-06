"""The factor specification contract and the fields that decide a factor's numbers."""

from __future__ import annotations

import math
from typing import Annotated, Final

from pydantic import Field, field_validator

from alphalattice.kernel.quant.factor_enums import FactorFamily, FactorTrack
from alphalattice.kernel.shared_kernel.domain.base import (
    DomainModel,
    NonEmptyString,
    ShortString,
)

NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(ge=1)]
FiniteNonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]


NUMERICAL_SPEC_FIELDS: Final = (
    "formula",
    "formula_ref",
    "lag_sessions",
    "minimum_observations",
    "required_fields",
    "return_convention",
    "window_sessions",
)
"""Every ``FactorSpec`` field that can move a factor's numbers or its availability.

The contract names it, beside the spec, so that a reader asking whether a factor is still the
same method (the Feature Desk's control check, Alpha's simple-signal provenance) imports the
contract and not the producers that compute factors (UC, LAWS.md ID8). A field that changes the
numbers and is absent from this tuple is a factor that can be rewritten invisibly.
"""


class FactorSpec(DomainModel):
    """Governed factor formula, input fields, history, and tolerance."""

    factor_id: ShortString
    family: FactorFamily
    formula_ref: ShortString
    formula: NonEmptyString
    window_sessions: PositiveInt
    lag_sessions: NonNegativeInt
    return_convention: ShortString
    required_fields: tuple[ShortString, ...] = Field(min_length=1)
    literature_sources: tuple[NonEmptyString, ...] = Field(min_length=1)
    minimum_observations: PositiveInt
    absolute_tolerance: FiniteNonNegative
    relative_tolerance: FiniteNonNegative
    track: FactorTrack
    core_anchor: bool = False

    # Each rule refuses with its code at its field, so a refusal names the field and the rule
    # and never a value (V453).
    @field_validator("required_fields", "literature_sources")
    @classmethod
    def sorted_unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Require evidence fields listed sorted and unique."""
        if values != tuple(sorted(set(values))):
            raise ValueError("factor_spec.values_sorted_unique")
        return values

    @field_validator("formula_ref")
    @classmethod
    def factor_namespace(cls, value: str) -> str:
        """Require a formula reference in the factor namespace."""
        if not value.startswith("factor."):
            raise ValueError("factor_spec.formula_ref_factor_namespace")
        return value


def finite_factor_value(value: float) -> float:
    """Require a finite factor value and canonicalize signed zero."""
    if not math.isfinite(value):
        raise ValueError("factor value must be finite")
    return 0.0 if value == 0.0 else value


__all__ = [
    "NUMERICAL_SPEC_FIELDS",
    "FactorSpec",
    "finite_factor_value",
]
