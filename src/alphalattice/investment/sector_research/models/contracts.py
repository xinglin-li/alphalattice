"""Small typed boundary for deterministic Sector forecast methods.

The Host keeps causality, ordered axes, formation time, evidence identity and
publication. An adapter owns only its recipe schema, its numerical behaviour and
its warmup semantics. The Protocol is deliberately four members: Sector has no
``research_authoring`` consumer, so there is no ``SealedResearchProgram``, no
execution-evidence contract and no compiler/executor split here -- reusing those
classes from Risk would be a framework with nobody on the other end.

The causal training boundary lives in ``BoundSectorForecastInput`` itself.
Every training label carries its availability clock, and constructing an input
whose labels were not matured at the forecast formation raises before any
adapter can be called -- the refusal is structural, not a convention each model
is trusted to honour.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, Protocol, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from threadpoolctl import threadpool_info, threadpool_limits  # type: ignore[import-untyped]

from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.source_identity import (
    switched_source_identity,
)

from ..contracts import FloatArray, SectorResearchError

INSUFFICIENT_MATURED_HISTORY = "INSUFFICIENT_MATURED_HISTORY"
"""The one typed warmup outcome: below its minimum matured history a method
emits no forecast for a sector. No zero fill, no backfill, no seeded fallback."""

type SectorForecastUnavailableReason = Literal["INSUFFICIENT_MATURED_HISTORY"]


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class SectorForecastRecipe(_Contract):
    """One sealed method recipe: the id, its schema, and every moving constant.

    Parameters are session counts and nothing else, so the field is typed
    ``dict[str, int]`` rather than an open mapping: a recipe smuggling a float
    knob or a string mode through an ``Any`` bag would widen the method space
    without widening its declared identity.
    """

    kind: Literal["SectorForecastRecipe"] = "SectorForecastRecipe"
    method_id: str = Field(min_length=1, max_length=96)
    recipe_schema_id: str = Field(min_length=1, max_length=128)
    parameters: dict[str, int]
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        method_id: str,
        recipe_schema_id: str,
        parameters: Mapping[str, int],
    ) -> Self:
        """Seal explicit method, schema and integer parameters without catalog admission.

        Args:
            method_id: Forecast method identifier.
            recipe_schema_id: Exact method recipe-schema identifier.
            parameters: Integer parameter mapping copied into the canonical recipe.

        Returns:
            Validated recipe with recipe_hash; the installed catalog separately admits its
            singleton.

        Raises:
            pydantic.ValidationError: Values violate the forecast recipe contract.
        """
        values: dict[str, object] = {
            "kind": "SectorForecastRecipe",
            "method_id": method_id,
            "recipe_schema_id": recipe_schema_id,
            "parameters": dict(parameters),
        }
        return cls(**values, recipe_hash=canonical_hash(values))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require the exact canonical forecast-recipe identity.

        Returns:
            This contract after checking its declared consistency rules.

        Raises:
            SectorResearchError: recipe_hash differs from the complete recipe payload excluding that
                hash.
        """
        if self.recipe_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"recipe_hash"})
        ):
            raise SectorResearchError("sector_research.forecast_recipe_identity_invalid")
        return self


SECTOR_PURE_NUMPY_STACK = "SINGLE_THREAD_PURE_NUMPY"
"""Adapters whose arithmetic is numpy reductions only, and never reaches BLAS."""

SECTOR_SCIKIT_LEARN_STACK = "SINGLE_THREAD_NUMPY_SCIKIT_LEARN"
"""Adapters that fit through scikit-learn, whose bits depend on the thread count."""

SECTOR_NUMERICAL_THREAD_LIMIT = 1
"""The one thread every Sector fit runs under, applied and verified, not assumed."""


def sector_numerical_environment_hash(stack: str = SECTOR_PURE_NUMPY_STACK) -> str:
    """The numerical stack one adapter declares: its threading, not the installed versions.

    Keyed by the stack the adapter actually uses. For the scikit-learn stack the
    thread limit is part of the identity rather than an assumption about the host:
    coordinate descent and the prediction matmul reach BLAS, whose reduction order
    moves the last bits with the thread count, so the limit that decides those bits
    is declared here and enforced by ``sector_numerical_thread_policy`` around the
    fit. The interpreter and the libraries' versions are the environment,
    provenance and never this identity (LAWS.md ID6).
    """
    values: dict[str, object] = {"threading": stack}
    if stack == SECTOR_SCIKIT_LEARN_STACK:
        values["thread_limit"] = SECTOR_NUMERICAL_THREAD_LIMIT
    return str(canonical_hash(values))


@contextmanager
def sector_numerical_thread_policy() -> Iterator[None]:
    """Run one Sector fit under its declared thread limit, and prove it held.

    Applying the limit is not enough on its own: a pool that ignored the request
    would produce numbers the recorded environment does not describe, so the
    observed pools are checked inside the scope and a violation refuses before
    any forecast is returned.
    """
    with threadpool_limits(limits=SECTOR_NUMERICAL_THREAD_LIMIT):
        if any(
            int(pool["num_threads"]) > SECTOR_NUMERICAL_THREAD_LIMIT
            for pool in threadpool_info()
            if pool.get("num_threads") is not None
        ):
            raise SectorResearchError("sector_research.numerical_thread_policy_failed")
        yield


class SectorNumericalBinding(_Contract):
    """Content identity of one adapter's deterministic numerical behaviour.

    ``implementation_content_hash`` uses the installed source-rule closure: Python
    syntax and its result-deciding imports, keyed by semantic component id. The
    W9b switch preserves the recorded value while that rule is unchanged; prose
    changes are neutral, and a numerical implementation change moves the rule.
    """

    method_id: str = Field(min_length=1, max_length=96)
    forecast_content_format_id: str = Field(min_length=1, max_length=128)
    implementation_component_ids: tuple[str, ...] = Field(min_length=1)
    implementation_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    numerical_environment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    deterministic_policy: dict[str, str]
    numerical_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        method_id: str,
        forecast_content_format_id: str,
        implementation_sources: Mapping[str, Path],
        deterministic_policy: Mapping[str, str],
        numerical_stack: str = SECTOR_PURE_NUMPY_STACK,
    ) -> Self:
        """Seal the installed source-rule closure, numerical stack and deterministic policy.

        Args:
            method_id: Forecast method owning this numerical behavior.
            forecast_content_format_id: Exact forecast-content format identifier.
            implementation_sources: Source closure entries keyed by stable semantic component
                identifier.
            deterministic_policy: Declared estimator/arithmetic and warmup rules.
            numerical_stack: Declared NumPy or scikit-learn threading stack.

        Returns:
            Validated numerical binding using the switched source-rule identity.

        Raises:
            ValueError: The source inventory is empty or its files cannot be bound to a checkout.
            pydantic.ValidationError: Binding values violate the contract.
        """
        values: dict[str, object] = {
            "method_id": method_id,
            "forecast_content_format_id": forecast_content_format_id,
            "implementation_component_ids": tuple(sorted(implementation_sources)),
            "implementation_content_hash": switched_source_identity(
                dict(implementation_sources),
                semantic_owner="sector_research",
                numerical_role="SECTOR_FORECAST",
            ),
            "numerical_environment_hash": sector_numerical_environment_hash(numerical_stack),
            "deterministic_policy": dict(deterministic_policy),
        }
        return cls(**values, numerical_binding_hash=canonical_hash(values))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require sorted unique implementation components and exact numerical binding.

        Returns:
            This contract after checking its declared consistency rules.

        Raises:
            SectorResearchError: Component identifiers are unordered/duplicated or
                numerical_binding_hash is inconsistent.
        """
        if self.implementation_component_ids != tuple(
            sorted(set(self.implementation_component_ids))
        ) or self.numerical_binding_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"numerical_binding_hash"})
        ):
            raise SectorResearchError("sector_research.numerical_binding_invalid")
        return self


@dataclass(frozen=True, slots=True)
class BoundSectorForecastInput:
    """Read-only matured Sector training history admitted for one formation.

    Construction is the causal boundary: every training label's availability
    clock must satisfy ``target_available_at <= forecast_formation_at``, by
    date comparison rather than index arithmetic, and a violation raises here --
    before any adapter is called, whichever caller assembled the input.
    """

    target_evidence_hash: str
    ordered_sectors: tuple[str, ...]
    forecast_formation_at: date
    training_formation_sessions: tuple[date, ...]
    training_target_available_sessions: tuple[date, ...]
    training_values: FloatArray

    @classmethod
    def create(
        cls,
        *,
        target_evidence_hash: str,
        ordered_sectors: tuple[str, ...],
        forecast_formation_at: date,
        training_formation_sessions: tuple[date, ...],
        training_target_available_sessions: tuple[date, ...],
        training_values: FloatArray,
    ) -> Self:
        """Construct a read-only float64 view and enforce causal training availability.

        Args:
            target_evidence_hash: Exact admitted target-evidence identity.
            ordered_sectors: Nonempty sorted unique sector axis.
            forecast_formation_at: Forecast formation date bounding available training labels.
            training_formation_sessions: Sorted unique training formation dates.
            training_target_available_sessions: Availability date of each corresponding training
                label.
            training_values: Session-by-sector observations; NaN may represent missing history.

        Returns:
            Bound input holding a non-writeable float64 view with verified axes and maturity clocks.

        Raises:
            SectorResearchError: Shape, axes, infinity, writeability or label availability violates
                admission.
        """
        values = np.asarray(training_values, dtype=np.float64)
        readonly = values.view()
        readonly.setflags(write=False)
        return cls(
            target_evidence_hash=target_evidence_hash,
            ordered_sectors=ordered_sectors,
            forecast_formation_at=forecast_formation_at,
            training_formation_sessions=training_formation_sessions,
            training_target_available_sessions=training_target_available_sessions,
            training_values=readonly,
        )

    def __post_init__(self) -> None:
        """Require aligned read-only training axes and labels matured by forecast formation.

        Raises:
            SectorResearchError: Identity length, sector/session axes, tensor shape, infinity or
                label clocks are invalid.
        """
        if (
            len(self.target_evidence_hash) != 64
            or not self.ordered_sectors
            or self.ordered_sectors != tuple(sorted(set(self.ordered_sectors)))
            or self.training_values.ndim != 2
            or self.training_values.shape
            != (len(self.training_formation_sessions), len(self.ordered_sectors))
            or len(self.training_target_available_sessions) != len(self.training_formation_sessions)
            or self.training_values.flags.writeable
            or bool(np.any(np.isinf(self.training_values)))
        ):
            raise SectorResearchError("sector_research.bound_forecast_input_invalid")
        if self.training_formation_sessions != tuple(sorted(set(self.training_formation_sessions))):
            raise SectorResearchError("sector_research.bound_forecast_input_axis_unordered")
        for formation, available_at in zip(
            self.training_formation_sessions,
            self.training_target_available_sessions,
            strict=True,
        ):
            if available_at < formation:
                raise SectorResearchError("sector_research.bound_forecast_input_clock_invalid")
            if available_at > self.forecast_formation_at:
                raise SectorResearchError("sector_research.training_label_immature")


@dataclass(frozen=True, slots=True)
class SectorForecastValues:
    """One formation's forecast per sector: a value or a typed absence, never both."""

    method_id: str
    forecast_formation_at: date
    ordered_sectors: tuple[str, ...]
    values: tuple[float | None, ...]
    unavailable_reasons: tuple[str | None, ...]

    def __post_init__(self) -> None:
        """Require aligned forecast cells with finite values or explicit unavailability.

        Raises:
            SectorResearchError: Sector/value/reason lengths differ or a cell is
                inconsistent/nonfinite.
        """
        if (
            not self.ordered_sectors
            or len(self.values) != len(self.ordered_sectors)
            or len(self.unavailable_reasons) != len(self.ordered_sectors)
        ):
            raise SectorResearchError("sector_research.forecast_values_axis_invalid")
        for value, reason in zip(self.values, self.unavailable_reasons, strict=True):
            if (value is None) == (reason is None):
                raise SectorResearchError("sector_research.forecast_values_cell_invalid")
            if value is not None and not np.isfinite(value):
                raise SectorResearchError("sector_research.forecast_values_nonfinite")


class SectorForecastAdapter(Protocol):
    """Desk-specific deterministic forecast seam; the Host retains authority.

    Axis validity, causal admissibility, evidence identity and publication stay
    with the Host and its store. An adapter that returns an inadmissible
    forecast fails there, not silently.
    """

    method_id: str
    recipe_schema_id: str

    def describe_numerical_binding(self) -> SectorNumericalBinding:
        """Describe the implementation and deterministic policy deciding this method output.

        Returns:
            Exact source-rule and numerical-policy binding for the installed method.
        """
        ...

    def validate_recipe(self, recipe: SectorForecastRecipe) -> None:
        """Validate this adapter method/schema route and admitted parameter shape.

        Args:
            recipe: Sealed forecast recipe to check before numerical work.

        Raises:
            SectorResearchError: The recipe route or parameters are incompatible with this method.
        """
        ...

    def forecast(
        self,
        *,
        bound_input: BoundSectorForecastInput,
        recipe: SectorForecastRecipe,
    ) -> SectorForecastValues:
        """Forecast one Sector formation from already admitted matured observations.

        Args:
            bound_input: Host-admitted causal training history and exact ordered sector axis.
            recipe: Installed method recipe admitted before execution.

        Returns:
            Finite forecast values or explicit warmup-unavailable cells on the bound sector axis.

        Raises:
            SectorResearchError: Recipe or numerical output fails the adapter contract.
        """
        ...


class SectorForecastCapabilityIdentity(_Contract):
    """Bind an installed method to its recipe schema, numerical owner and admitted recipe.

    The numerical binding names result-deciding implementation/policy; admitted_recipe_hash names
    the frozen singleton admitted by the catalog. This declaration does not execute a forecast.
    """

    method_id: str = Field(min_length=1, max_length=96)
    recipe_schema_id: str = Field(min_length=1, max_length=128)
    numerical_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    admitted_recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    """The frozen singleton recipe, inside catalog identity rather than beside it.

    Without this the catalog stated only that an *implementation* was installed,
    and a re-sealed, internally self-consistent surface carrying any parameters
    at all -- a 999-session half-life, say -- verified as that implementation's
    work. "No search domain" is a claim about parameters, so the one admissible
    parameter point is part of what "installed" means, and the verifier compares
    every surface's recipe hash against it.
    """


class SectorForecastCatalogBinding(_Contract):
    """Content identity of the explicitly installed Sector forecast capabilities.

    A contract rather than a property of the catalog class, so the verifier can
    hold and compare it without importing the catalog module -- which imports
    the adapters, and the verifier must not be able to reach a model through
    its own import graph.
    """

    ordered_capabilities: tuple[SectorForecastCapabilityIdentity, ...] = Field(min_length=1)
    catalog_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require unique installed capability methods and exact catalog identity.

        Returns:
            This contract after checking its declared consistency rules.

        Raises:
            SectorResearchError: A method repeats or catalog_hash differs from the complete
                capability payload.
        """
        keys = tuple(value.method_id for value in self.ordered_capabilities)
        if len(set(keys)) != len(keys):
            raise SectorResearchError("sector_research.catalog_capability_duplicated")
        if self.catalog_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"catalog_hash"})
        ):
            raise SectorResearchError("sector_research.catalog_identity_invalid")
        return self


__all__ = [
    "INSUFFICIENT_MATURED_HISTORY",
    "SECTOR_NUMERICAL_THREAD_LIMIT",
    "SECTOR_PURE_NUMPY_STACK",
    "SECTOR_SCIKIT_LEARN_STACK",
    "BoundSectorForecastInput",
    "SectorForecastAdapter",
    "SectorForecastCapabilityIdentity",
    "SectorForecastCatalogBinding",
    "SectorForecastRecipe",
    "SectorForecastValues",
    "SectorNumericalBinding",
    "sector_numerical_environment_hash",
    "sector_numerical_thread_policy",
]
