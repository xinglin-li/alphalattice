"""Common-axis evidence and HAR trigger for Alpha cross-sectional scale methods."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.sealing import seal_model


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class CrossSectionalScaleMethodEvidence(_Contract):
    """Seal one dispersion method accuracy and scale-ratio summary on common observations."""

    recipe_id: str
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    forecast_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    common_observation_count: int = Field(ge=1)
    mean_absolute_error: float = Field(ge=0.0, allow_inf_nan=False)
    normalized_mean_absolute_error: float = Field(ge=0.0, allow_inf_nan=False)
    mean_forecast_to_realized_ratio: float = Field(gt=0.0, allow_inf_nan=False)
    evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal the declared common-observation scaling accuracy evidence.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical evidence_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        return seal_model(cls, values, field="evidence_hash")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require the exact canonical dispersion method-evidence identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: evidence_hash differs from the complete accuracy/scale-ratio payload.
        """
        if self.evidence_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"evidence_hash"})
        ):
            raise ValueError("ALPHA_SCALE_METHOD_EVIDENCE_INVALID")
        return self


class CrossSectionalScaleComparisonEvidence(_Contract):
    """Seal primary scale comparisons, a selected recipe and the HAR trigger disposition."""

    kind: Literal["CrossSectionalScaleComparisonEvidence"] = "CrossSectionalScaleComparisonEvidence"
    target_evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    horizon_sessions: Literal[1, 5]
    primary_methods: tuple[CrossSectionalScaleMethodEvidence, ...] = Field(min_length=3)
    selected_primary_recipe_id: str
    selected_primary_recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    har_disposition: Literal["TRIGGERED", "NOT_TRIGGERED"]
    har_trigger_reason: str
    evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require a compared primary recipe and exact comparison identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Selected method/hash is absent from primary_methods or evidence_hash is
                inconsistent.
        """
        if (self.selected_primary_recipe_id, self.selected_primary_recipe_hash) not in {
            (value.recipe_id, value.recipe_hash) for value in self.primary_methods
        } or self.evidence_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"evidence_hash"})
        ):
            raise ValueError("ALPHA_SCALE_COMPARISON_EVIDENCE_INVALID")
        return self


__all__ = ["CrossSectionalScaleComparisonEvidence", "CrossSectionalScaleMethodEvidence"]
