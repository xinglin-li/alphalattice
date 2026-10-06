"""Small typed boundary for deterministic Risk estimator implementations.

The Host keeps causality, ordered axes, formation time, evidence identity, and
publication. An adapter owns only its recipe schema, numerical behavior, and
diagnostics. This module must stay importable without the numerical stack.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.investment.risk_research.contracts import CovarianceDiagnostics
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.source_identity import switched_source_identity

type FloatArray = npt.NDArray[np.float64]

SINGLE_THREAD_NUMERICAL_CAPABILITY = "risk-estimator:single-thread-numerical-policy"


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class RiskEstimatorRecipeEnvelope(_Contract):
    """Host-sealed recipe routed to one explicitly installed estimator adapter."""

    adapter_id: str = Field(min_length=1, max_length=96)
    recipe_schema_id: str = Field(min_length=1, max_length=128)
    parameters: dict[str, Any]
    recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        adapter_id: str,
        recipe_schema_id: str,
        parameters: Mapping[str, object],
    ) -> Self:
        """Seal copied Risk parameters against an explicit adapter and recipe schema.

        Args:
            adapter_id: Numerical adapter identifier.
            recipe_schema_id: Exact recipe-schema identifier.
            parameters: Explicit parameter mapping copied into the envelope.

        Returns:
            Validated envelope with its canonical recipe_hash.

        Raises:
            pydantic.ValidationError: Envelope values violate the declared contract.
        """
        values = {
            "adapter_id": adapter_id,
            "recipe_schema_id": recipe_schema_id,
            "parameters": dict(parameters),
        }
        return cls(**values, recipe_hash=canonical_hash(values))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require the exact canonical recipe-envelope identity.

        Returns:
            This envelope after verifying recipe_hash.

        Raises:
            ValueError: recipe_hash differs from the complete envelope payload excluding that hash.
        """
        if self.recipe_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"recipe_hash"})
        ):
            raise ValueError("RISK_ESTIMATOR_RECIPE_IDENTITY_INVALID")
        return self


def implementation_content_hash(*sources: Path) -> str:
    """The identity of the modules that will actually run, by their rule (LAWS.md ID3, V95).

    ``implementation_owners`` is a tuple of module *names*, and a name is not an
    identity: the same string denotes whatever that module currently contains.
    An adapter could be rewritten completely while its declared owners, its
    adapter id and therefore its numerical binding all stood still -- which is
    exactly the "same identifier, changed code" case the catalog is supposed to
    make impossible.

    Each adapter passes its own implementation files, so a capability installed
    from outside this package states its content the same way the built-in one
    does. The files are the rule closure's entries, hashed as syntax with what they
    import inside the number-deciding packages, and the identity keeps the byte
    value it had when the rule replaced the bytes (``switched_source_identity``).
    Third-party owners are deliberately not hashed here: their versions are the
    environment, recorded beside each estimate and never identity (LAWS.md ID6).
    """
    if not sources:
        raise ValueError("RISK_ESTIMATOR_IMPLEMENTATION_SOURCES_EMPTY")
    paths = {Path(source).name: Path(source) for source in sources}
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("RISK_ESTIMATOR_IMPLEMENTATION_SOURCE_MISSING")
    if len(paths) != len(sources):
        raise ValueError("RISK_ESTIMATOR_IMPLEMENTATION_SOURCES_AMBIGUOUS")
    return switched_source_identity(
        paths,
        semantic_owner="risk_research.estimators",
        numerical_role="RISK_ESTIMATOR_IMPLEMENTATION",
    )


class RiskEstimatorNumericalBinding(_Contract):
    """Content identity of one adapter's deterministic numerical behavior."""

    adapter_id: str = Field(min_length=1, max_length=96)
    estimate_content_format_id: str = Field(min_length=1, max_length=128)
    implementation_owners: tuple[str, ...] = Field(min_length=1)
    implementation_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    """Bytes of the first-party modules named in ``implementation_owners``.

    Without this the binding was a statement about naming, not about code.
    """

    numerical_environment_hash: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$", exclude_if=lambda v: v is None
    )
    """Only on a binding sealed before E0, which folded the environment in.

    The environment is provenance, recorded beside each estimate and never part of
    this identity (LAWS.md ID6, the user's decision of 2026-09-26): a dependency
    upgrade moves no Risk method, and whether it moves a number is answered by U0.
    """

    deterministic_policy: dict[str, Any]
    required_runtime_capabilities: tuple[str, ...]
    numerical_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        adapter_id: str,
        estimate_content_format_id: str,
        implementation_owners: tuple[str, ...],
        implementation_content_hash: str,
        deterministic_policy: Mapping[str, object],
        required_runtime_capabilities: tuple[str, ...],
    ) -> Self:
        """Seal declared implementation, output format and deterministic execution policy.

        Args:
            adapter_id: Numerical adapter identifier.
            estimate_content_format_id: Exact numerical-output content format.
            implementation_owners: Unique implementation owner names in supplied order.
            implementation_content_hash: Identity of the switched implementation closure.
            deterministic_policy: Declared result-deciding arithmetic and execution rules.
            required_runtime_capabilities: Unique required capabilities in supplied order.

        Returns:
            Validated numerical binding with its canonical identity.

        Raises:
            pydantic.ValidationError: Binding fields or declared consistency violate the contract.
        """
        values = {
            "adapter_id": adapter_id,
            "estimate_content_format_id": estimate_content_format_id,
            "implementation_owners": implementation_owners,
            "implementation_content_hash": implementation_content_hash,
            "deterministic_policy": dict(deterministic_policy),
            "required_runtime_capabilities": required_runtime_capabilities,
        }
        return cls(**values, numerical_binding_hash=canonical_hash(values))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require unique owner/capability axes and the exact numerical binding identity.

        Returns:
            This binding after verifying both supplied axes and numerical_binding_hash.

        Raises:
            ValueError: An owner/capability repeats or numerical_binding_hash differs from the
                complete payload.
        """
        if (
            self.implementation_owners != tuple(dict.fromkeys(self.implementation_owners))
            or self.required_runtime_capabilities
            != tuple(dict.fromkeys(self.required_runtime_capabilities))
            or self.numerical_binding_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"numerical_binding_hash"}))
        ):
            raise ValueError("RISK_ESTIMATOR_NUMERICAL_BINDING_INVALID")
        return self


@dataclass(frozen=True, slots=True)
class BoundRiskReturnInput:
    """Read-only causal return window admitted by the Risk Desk Host.

    The exact ordered listing axis, the originating return surface, and the
    formation session travel with the values so an adapter cannot be handed a
    same-shaped window from another axis or another session.
    """

    return_surface_hash: str
    ordered_listing_ids: tuple[str, ...]
    formation_session: date
    returns: FloatArray

    @classmethod
    def create(
        cls,
        *,
        return_surface_hash: str,
        ordered_listing_ids: tuple[str, ...],
        formation_session: date,
        returns: FloatArray,
    ) -> Self:
        """Construct an axis-validated read-only float64 view of admitted Risk returns.

        Args:
            return_surface_hash: Exact source-return identity.
            ordered_listing_ids: Nonempty unique listing axis in supplied order.
            formation_session: Formation associated with this numerical input.
            returns: Finite session-by-listing values; the returned view can share storage.

        Returns:
            Bound input retaining a non-writeable float64 view and declared source/formation
            lineage.

        Raises:
            ValueError: Source identity length, axis, shape or numerical values violate the
                bound-input contract.
        """
        values = np.asarray(returns, dtype=np.float64)
        readonly = values.view()
        readonly.setflags(write=False)
        return cls(
            return_surface_hash=return_surface_hash,
            ordered_listing_ids=ordered_listing_ids,
            formation_session=formation_session,
            returns=readonly,
        )

    def __post_init__(self) -> None:
        """Require a finite two-dimensional read-only matrix on the unique listing axis.

        Raises:
            ValueError: Source identity length, listing uniqueness, matrix shape, writeability or
                finiteness is invalid.
        """
        if (
            len(self.return_surface_hash) != 64
            or not self.ordered_listing_ids
            or self.ordered_listing_ids != tuple(dict.fromkeys(self.ordered_listing_ids))
            or self.returns.ndim != 2
            or self.returns.shape[1] != len(self.ordered_listing_ids)
            or self.returns.flags.writeable
            or not np.isfinite(self.returns).all()
        ):
            raise ValueError("RISK_BOUND_RETURN_INPUT_INVALID")


@dataclass(frozen=True, slots=True)
class EstimatedCovariance:
    """One formation-session risk estimate produced by an installed adapter."""

    matrix: FloatArray
    forecast_volatility: FloatArray
    eigenvalues: FloatArray
    eigenvectors: FloatArray
    diagnostics: CovarianceDiagnostics
    component_shrinkages: tuple[tuple[str, float], ...] = ()
    """Named intensities of a method built from more than one shrunk component.

    ``CovarianceDiagnostics.shrinkage`` is one float, which is the whole story
    for a method that fits one estimator and a summary for a method that fits
    several. A blended method reporting only the summary would hide the
    component that actually carries its risk, so the parts travel with the
    estimate under the names the method gives them.

    Empty for a single-component method, and deliberately not part of any
    artifact identity: this is a diagnostic the Campaign aggregates, not a
    second claim about which matrix was produced.
    """


class RiskEstimatorAdapter(Protocol):
    """Desk-specific deterministic estimator seam; the Host retains authority.

    Positive-definiteness, symmetry, finite-value, and artifact identity
    validation stay with the Risk Desk Host and its publication owners. An
    adapter that returns an inadmissible estimate fails there, not silently.
    """

    adapter_id: str
    recipe_schema_id: str

    def describe_numerical_binding(self) -> RiskEstimatorNumericalBinding:
        """Describe the implementation closure and policy deciding this adapter output.

        Returns:
            Exact numerical owner binding for the installed adapter.
        """
        ...

    def validate_recipe(self, recipe: RiskEstimatorRecipeEnvelope) -> object:
        """Validate the adapter route, schema and concrete Risk recipe parameters.

        Args:
            recipe: Sealed envelope to admit before numerical execution.

        Returns:
            Concrete typed recipe admitted by this numerical adapter.

        Raises:
            ValueError: Adapter/schema routing or concrete recipe admission fails.
        """
        ...

    def estimate(
        self,
        *,
        recipe: RiskEstimatorRecipeEnvelope,
        inputs: BoundRiskReturnInput,
    ) -> EstimatedCovariance:
        """Estimate formation covariance from an admitted recipe and bound causal returns.

        Args:
            recipe: Sealed recipe for this installed adapter.
            inputs: Bound return history and exact ordered listing/formation axis.

        Returns:
            Estimated covariance with its numerical diagnostics.

        Raises:
            ValueError: Recipe or numerical input violates this adapter contract.
        """
        ...


__all__ = [
    "SINGLE_THREAD_NUMERICAL_CAPABILITY",
    "BoundRiskReturnInput",
    "EstimatedCovariance",
    "RiskEstimatorAdapter",
    "RiskEstimatorNumericalBinding",
    "RiskEstimatorRecipeEnvelope",
    "implementation_content_hash",
]
