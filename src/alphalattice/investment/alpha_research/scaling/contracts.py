"""Contracts for the cross-sectional dispersion forecast.

The stock model predicts a standardized residual ``z``. Portfolio needs a return,
so something has to supply the scale that converts one into the other. This owner
supplies it, and the contracts here exist to make the conversion checkable rather
than plausible.

The scale is *not* recomputed here. It is read from the canonical target evidence,
because the only scale that inverts a particular ``z`` is the one that divided it.
A dispersion computed independently -- from the same returns, in the same units,
by an equally reasonable formula -- would be a different number, and multiplying
by it would produce a quantity that looks like a return and is not one. So these
contracts bind the exact target evidence they read, and refuse a forecast whose
scale cannot be traced to it.
"""

from __future__ import annotations

from datetime import date
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash

LAGGED_XS_DISPERSION_RECIPE_ID = "LAGGED_XS_DISPERSION"
EWMA_XS_DISPERSION_RECIPE_ID = "EWMA_XS_DISPERSION"
ASYMMETRIC_EWMA_XS_DISPERSION_RECIPE_ID = "ASYMMETRIC_EWMA_XS_DISPERSION"
HAR_XS_DISPERSION_RECIPE_ID = "HAR_XS_DISPERSION"


class CrossSectionalScalingError(ValueError):
    """Stable failure raised before any dispersion forecast is used."""


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class ScalingImplementationBinding(_Contract):
    """Executable identity of one installed scale method.

    Alpha-owned rather than shared with the Feature or Risk equivalents: the same
    idea, but a scale method and a covariance estimator have different closures,
    and a single cross-Desk binding would make an edit to one rotate the identity
    of the other.
    """

    kind: Literal["ScalingImplementationBinding"] = "ScalingImplementationBinding"
    implementation_id: str = Field(min_length=1, max_length=128)
    implementation_owners: tuple[str, ...] = Field(min_length=1)
    implementation_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    numerical_environment_hash: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$", exclude_if=lambda v: v is None
    )
    """Only on a binding sealed before E0: the environment is provenance (LAWS.md ID6)."""
    implementation_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        implementation_id: str,
        implementation_owners: tuple[str, ...],
        implementation_content_hash: str,
    ) -> Self:
        """Seal dispersion implementation owners and switched source-content identity.

        Args:
            implementation_id: Installed result-deciding implementation identifier.
            implementation_owners: Unique declared implementation owners in supplied order.
            implementation_content_hash: Exact switched implementation-content identity.

        Returns:
            Validated implementation binding with its canonical self identity.

        Raises:
            pydantic.ValidationError: Implementation fields or owner/identity consistency violate
                the model.
        """
        values: dict[str, object] = {
            "kind": "ScalingImplementationBinding",
            "implementation_id": implementation_id,
            "implementation_owners": list(implementation_owners),
            "implementation_content_hash": implementation_content_hash,
        }
        return cls(**values, implementation_binding_hash=canonical_hash(values))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require unique scaling implementation owners and exact binding identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            CrossSectionalScalingError: An owner repeats or implementation_binding_hash differs from
                the complete declared payload.
        """
        if len(set(self.implementation_owners)) != len(self.implementation_owners):
            raise CrossSectionalScalingError("SCALING_IMPLEMENTATION_OWNERS_DUPLICATED")
        if self.implementation_binding_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"implementation_binding_hash"})
        ):
            raise CrossSectionalScalingError("SCALING_IMPLEMENTATION_IDENTITY_INVALID")
        return self


class LaggedXsDispersionRecipe(_Contract):
    """The naive causal control: the most recent scale that had already matured.

    ``source_offset_sessions`` is the whole method. At formation close ``T`` the
    outcome attached to formation ``T - L`` finished at ``T``, so its realized
    dispersion is the newest one an observer at ``T`` could know. Taking anything
    newer is look-ahead; taking anything older is a different method.

    The offset is copied from the outcome recipe's own maturity lag rather than
    chosen, which is why a one-session lane binds ``2`` and a five-session lane
    binds ``6`` without this contract knowing what a horizon is.

    When the scale at exactly ``T - L`` is unavailable the forecast is
    unavailable. Reaching further back for the newest available value would be a
    reasonable method and is not this one; doing it silently would make the
    horizon of the scale unknowable from the evidence.
    """

    kind: Literal["LaggedXsDispersionRecipe"] = "LaggedXsDispersionRecipe"
    recipe_id: Literal["LAGGED_XS_DISPERSION"] = "LAGGED_XS_DISPERSION"
    source_lane: Literal["CANONICAL_REDEMEANED_RESIDUAL_CROSS_SECTIONAL_STD"] = (
        "CANONICAL_REDEMEANED_RESIDUAL_CROSS_SECTIONAL_STD"
    )
    """Names the exact lane: the ``ddof=1`` scale of the re-demeaned bounded
    residual, which is the denominator of the target ``z``. Not the raw residual
    and not a robust scale -- either would fail to invert the standardization."""

    source_offset_sessions: int = Field(ge=2)
    unavailable_source_policy: Literal["FAIL_CLOSED"] = "FAIL_CLOSED"
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, *, source_offset_sessions: int) -> Self:
        """Seal lagged scaling on the canonical re-demeaned residual dispersion lane.

        Args:
            source_offset_sessions: Declared causal source offset admitted by the recipe model.

        Returns:
            Validated lagged recipe and its canonical identity.

        Raises:
            pydantic.ValidationError: The source offset or declared recipe fields violate the model.
        """
        values: dict[str, object] = {
            "kind": "LaggedXsDispersionRecipe",
            "recipe_id": LAGGED_XS_DISPERSION_RECIPE_ID,
            "source_lane": "CANONICAL_REDEMEANED_RESIDUAL_CROSS_SECTIONAL_STD",
            "source_offset_sessions": int(source_offset_sessions),
            "unavailable_source_policy": "FAIL_CLOSED",
        }
        return cls(**values, recipe_hash=canonical_hash(values))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require the exact canonical lagged-dispersion recipe identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            CrossSectionalScalingError: recipe_hash differs from the complete declared lagged
                recipe.
        """
        if self.recipe_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"recipe_hash"})
        ):
            raise CrossSectionalScalingError("SCALING_RECIPE_IDENTITY_INVALID")
        return self


class RecursiveXsDispersionRecipe(_Contract):
    """Causal recursive scale recipe over already-mature target dispersion."""

    kind: Literal["RecursiveXsDispersionRecipe"] = "RecursiveXsDispersionRecipe"
    recipe_id: Literal[
        "EWMA_XS_DISPERSION",
        "ASYMMETRIC_EWMA_XS_DISPERSION",
        "HAR_XS_DISPERSION",
    ]
    source_lane: Literal["CANONICAL_REDEMEANED_RESIDUAL_CROSS_SECTIONAL_STD"] = (
        "CANONICAL_REDEMEANED_RESIDUAL_CROSS_SECTIONAL_STD"
    )
    source_offset_sessions: int = Field(ge=2)
    initialization_policy: Literal["FIRST_CAUSALLY_MATURE_FINITE_VALUE"] = (
        "FIRST_CAUSALLY_MATURE_FINITE_VALUE"
    )
    missing_update_policy: Literal["CARRY_PREVIOUS_STATE"] = "CARRY_PREVIOUS_STATE"
    symmetric_half_life_sessions: Literal[10, 21, 42] | None = None
    rise_half_life_sessions: Literal[10] | None = None
    decay_half_life_sessions: Literal[42] | None = None
    har_daily_sessions: Literal[1] | None = None
    har_weekly_sessions: Literal[5] | None = None
    har_monthly_sessions: Literal[21] | None = None
    har_minimum_fit_rows: Literal[63] | None = None
    har_ridge_ratio: float | None = Field(default=None, gt=0.0, le=1e-6)
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        recipe_id: Literal[
            "EWMA_XS_DISPERSION",
            "ASYMMETRIC_EWMA_XS_DISPERSION",
            "HAR_XS_DISPERSION",
        ],
        source_offset_sessions: int,
        symmetric_half_life_sessions: Literal[10, 21, 42] | None = None,
    ) -> Self:
        """Seal installed EWMA, asymmetric EWMA or HAR scaling parameters.

        Symmetric EWMA admits half-lives 10, 21 or 42. Asymmetric EWMA fixes rise/decay at 10/42;
        HAR fixes daily/weekly/monthly windows 1/5/21, 63 fit rows and ridge ratio 1e-6.

        Args:
            recipe_id: Exact installed recursive dispersion method.
            source_offset_sessions: Declared causal source lag.
            symmetric_half_life_sessions: Required admitted half-life for symmetric EWMA; otherwise
                optional.

        Returns:
            Validated recursive recipe with fixed method-specific parameters and canonical
            recipe_hash.

        Raises:
            pydantic.ValidationError: Method-specific parameters or recipe consistency violate the
                model.
        """
        values: dict[str, object] = {
            "kind": "RecursiveXsDispersionRecipe",
            "recipe_id": recipe_id,
            "source_lane": "CANONICAL_REDEMEANED_RESIDUAL_CROSS_SECTIONAL_STD",
            "source_offset_sessions": source_offset_sessions,
            "initialization_policy": "FIRST_CAUSALLY_MATURE_FINITE_VALUE",
            "missing_update_policy": "CARRY_PREVIOUS_STATE",
            "symmetric_half_life_sessions": symmetric_half_life_sessions,
            "rise_half_life_sessions": 10
            if recipe_id == ASYMMETRIC_EWMA_XS_DISPERSION_RECIPE_ID
            else None,
            "decay_half_life_sessions": 42
            if recipe_id == ASYMMETRIC_EWMA_XS_DISPERSION_RECIPE_ID
            else None,
            "har_daily_sessions": 1 if recipe_id == HAR_XS_DISPERSION_RECIPE_ID else None,
            "har_weekly_sessions": 5 if recipe_id == HAR_XS_DISPERSION_RECIPE_ID else None,
            "har_monthly_sessions": 21 if recipe_id == HAR_XS_DISPERSION_RECIPE_ID else None,
            "har_minimum_fit_rows": 63 if recipe_id == HAR_XS_DISPERSION_RECIPE_ID else None,
            "har_ridge_ratio": 1e-6 if recipe_id == HAR_XS_DISPERSION_RECIPE_ID else None,
        }
        return cls(**values, recipe_hash=canonical_hash(values))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_recipe(self) -> Self:
        """Require exact method-specific half-life/HAR parameters and recipe identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            CrossSectionalScalingError: Declared method parameters are outside the admitted values
                or recipe_hash is inconsistent.
        """
        if self.recipe_id == EWMA_XS_DISPERSION_RECIPE_ID:
            valid = self.symmetric_half_life_sessions in {10, 21, 42}
        elif self.recipe_id == ASYMMETRIC_EWMA_XS_DISPERSION_RECIPE_ID:
            valid = self.rise_half_life_sessions == 10 and self.decay_half_life_sessions == 42
        else:
            valid = (
                self.har_daily_sessions == 1
                and self.har_weekly_sessions == 5
                and self.har_monthly_sessions == 21
                and self.har_minimum_fit_rows == 63
                and self.har_ridge_ratio == 1e-6
            )
        if not valid or self.recipe_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"recipe_hash"})
        ):
            raise CrossSectionalScalingError("SCALING_RECURSIVE_RECIPE_INVALID")
        return self


type XsDispersionRecipe = LaggedXsDispersionRecipe | RecursiveXsDispersionRecipe


class CrossSectionalDispersionForecast(_Contract):
    """One forecast surface: a scale per formation, and where each came from.

    ``source_sessions`` is carried beside the values rather than left implicit.
    A same-shaped surface shifted by one session is the failure this owner exists
    to prevent, and it is only detectable when each forecast names the formation
    whose realized scale it used.
    """

    kind: Literal["CrossSectionalDispersionForecast"] = "CrossSectionalDispersionForecast"
    recipe_id: str = Field(min_length=1, max_length=96)
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    implementation_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    implementation: ScalingImplementationBinding
    """The implementation binding itself, not just its digest.

    Same reason as the Feature side: a hash that no document expands is an
    opaque leaf, and a reader cannot audit a closure it cannot see. Embedded
    because it is small, immutable, and meaningless apart from the forecast that
    cites it."""

    target_evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_recipe_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    outcome_method_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    maturity_lag_sessions: int = Field(ge=2)
    ordered_listing_ids_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_dispersion_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    formation_sessions: tuple[date, ...] = Field(min_length=1)
    forecast_values: tuple[float | None, ...] = Field(min_length=1)
    source_sessions: tuple[date | None, ...] = Field(min_length=1)
    forecast_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require canonical forecast axes, paired causal sources and exact implementation lineage.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            CrossSectionalScalingError: Axes differ/are unordered, value/source availability is
                partial, a scale is nonpositive, source is not earlier than formation or
                implementation/forecast identities disagree.
        """
        if not (
            len(self.formation_sessions) == len(self.forecast_values) == len(self.source_sessions)
        ):
            raise CrossSectionalScalingError("SCALING_FORECAST_AXIS_MISMATCH")
        if tuple(sorted(set(self.formation_sessions))) != self.formation_sessions:
            raise CrossSectionalScalingError("SCALING_FORECAST_AXIS_UNORDERED")
        for formation, value, source in zip(
            self.formation_sessions, self.forecast_values, self.source_sessions, strict=True
        ):
            if (value is None) != (source is None):
                raise CrossSectionalScalingError("SCALING_FORECAST_SOURCE_INCOMPLETE")
            if value is not None and not (value > 0.0):
                raise CrossSectionalScalingError("SCALING_FORECAST_VALUE_INVALID")
            # Causality, re-checked at the contract rather than trusted from the
            # executor: a scale may only be built from a strictly earlier
            # formation, whatever produced this object.
            if source is not None and source >= formation:
                raise CrossSectionalScalingError("SCALING_FORECAST_SOURCE_NOT_CAUSAL")
        if self.implementation.implementation_binding_hash != self.implementation_binding_hash:
            raise CrossSectionalScalingError("SCALING_IMPLEMENTATION_CHILD_MISMATCH")
        if self.forecast_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"forecast_hash"})
        ):
            raise CrossSectionalScalingError("SCALING_FORECAST_IDENTITY_INVALID")
        return self

    def available_by_session(self) -> dict[date, float]:
        """Only the formations that carry a scale, so a missing one cannot be used."""
        return {
            session: float(value)
            for session, value in zip(self.formation_sessions, self.forecast_values, strict=True)
            if value is not None
        }


__all__ = [
    "ASYMMETRIC_EWMA_XS_DISPERSION_RECIPE_ID",
    "EWMA_XS_DISPERSION_RECIPE_ID",
    "HAR_XS_DISPERSION_RECIPE_ID",
    "LAGGED_XS_DISPERSION_RECIPE_ID",
    "CrossSectionalDispersionForecast",
    "CrossSectionalScalingError",
    "LaggedXsDispersionRecipe",
    "RecursiveXsDispersionRecipe",
    "ScalingImplementationBinding",
    "XsDispersionRecipe",
]
