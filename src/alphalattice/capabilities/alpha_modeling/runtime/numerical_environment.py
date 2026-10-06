"""Package and thread authority for deterministic Alpha model execution."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from threadpoolctl import threadpool_info, threadpool_limits  # type: ignore[import-untyped]

from alphalattice.kernel.shared_kernel.environment import package_versions
from alphalattice.kernel.shared_kernel.identity import canonical_hash

from ..contracts import NESTED_FIT_RUNTIME_CAPABILITY, AlphaModelNumericalBinding

SINGLE_THREAD_RUNTIME_CAPABILITY = "single-thread"


class AlphaNumericalEnvironmentError(RuntimeError):
    """Operational model environment failure, never scientific evidence."""

    failure_class = "OPERATIONAL_FAILURE"

    def __init__(self, code: str) -> None:
        """Record the stable operational failure code."""
        self.code = code
        super().__init__(code)


class AlphaModelNumericalEnvironment(BaseModel):  # type: ignore[misc]
    """Content identity of packages and execution limits admitted for one adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["AlphaModelNumericalEnvironment"] = "AlphaModelNumericalEnvironment"
    numerical_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    package_versions: tuple[tuple[str, str], ...]
    configured_thread_count: int | None = Field(default=None, ge=1)
    environment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        numerical_binding_hash: str,
        package_versions: tuple[tuple[str, str], ...],
        configured_thread_count: int | None,
    ) -> Self:
        """Bind the numerical settings to their canonical content hash.

        Args:
            numerical_binding_hash: Identity of the admitted adapter binding.
            package_versions: Installed package names and versions in sorted order.
            configured_thread_count: Admitted library thread limit, if any.

        Returns:
            The validated numerical environment record.

        """
        values = {
            "kind": "AlphaModelNumericalEnvironment",
            "numerical_binding_hash": numerical_binding_hash,
            "package_versions": package_versions,
            "configured_thread_count": configured_thread_count,
        }
        return cls(**values, environment_hash=canonical_hash(values))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Reject an unsorted package list or mismatched environment hash.

        Returns:
            This environment when its content identity is valid.

        Raises:
            ValueError: Package order or the recorded hash is invalid.

        """
        if self.package_versions != tuple(
            sorted(set(self.package_versions), key=lambda value: value[0])
        ) or self.environment_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"environment_hash"})
        ):
            raise ValueError("ALPHA_MODEL_NUMERICAL_ENVIRONMENT_IDENTITY_INVALID")
        return self


def _package_requirement(capability: str) -> tuple[str, str | None] | None:
    if capability in {
        NESTED_FIT_RUNTIME_CAPABILITY,
        SINGLE_THREAD_RUNTIME_CAPABILITY,
    }:
        return None
    package, marker, expected = capability.partition("==")
    if package not in {"numpy", "scikit-learn", "lightgbm"}:
        raise AlphaNumericalEnvironmentError("ALPHA_MODEL_RUNTIME_CAPABILITY_UNSUPPORTED")
    return package, expected if marker else None


def resolve_alpha_model_numerical_environment(
    binding: AlphaModelNumericalBinding,
) -> AlphaModelNumericalEnvironment:
    """Record the installed packages a fit runs on, before any adapter fit.

    The environment is provenance, recorded beside the fit (LAWS.md ID6): a package the
    binding needs must be installed, and its installed version is recorded; a version the
    binding declares is the one it was validated with, never a gate.
    """
    packages: dict[str, str] = {}
    for capability in binding.required_runtime_capabilities:
        requirement = _package_requirement(capability)
        if requirement is None:
            continue
        package, _declared = requirement
        installed = dict(package_versions((package,)))[package]
        if installed == "absent":
            raise AlphaNumericalEnvironmentError("ALPHA_MODEL_RUNTIME_PACKAGE_MISSING")
        packages[package] = installed
    configured_threads: int | None = None
    if SINGLE_THREAD_RUNTIME_CAPABILITY in binding.required_runtime_capabilities:
        configured_threads = 1
    return AlphaModelNumericalEnvironment.create(
        numerical_binding_hash=binding.numerical_binding_hash,
        package_versions=tuple(sorted(packages.items())),
        configured_thread_count=configured_threads,
    )


def _validate_effective_thread_boundary(
    environment: AlphaModelNumericalEnvironment,
) -> None:
    if environment.configured_thread_count is None:
        return
    observed = tuple(
        int(value["num_threads"])
        for value in threadpool_info()
        if isinstance(value.get("num_threads"), int)
    )
    if observed and max(observed) > environment.configured_thread_count:
        raise AlphaNumericalEnvironmentError(
            "ALPHA_MODEL_RUNTIME_EFFECTIVE_THREAD_BOUNDARY_MISMATCH"
        )


@contextmanager
def alpha_model_numerical_scope(
    binding: AlphaModelNumericalBinding,
) -> Iterator[AlphaModelNumericalEnvironment]:
    """Apply and verify the binding's execution limits around one fit/predict call."""
    environment = resolve_alpha_model_numerical_environment(binding)
    library_thread_limit = environment.configured_thread_count
    limiter = (
        threadpool_limits(limits=library_thread_limit)
        if library_thread_limit is not None
        else nullcontext()
    )
    with limiter:
        yield environment
        if library_thread_limit is not None:
            observed = environment.model_copy(
                update={"configured_thread_count": library_thread_limit}
            )
            _validate_effective_thread_boundary(observed)


__all__ = [
    "SINGLE_THREAD_RUNTIME_CAPABILITY",
    "AlphaModelNumericalEnvironment",
    "AlphaNumericalEnvironmentError",
    "alpha_model_numerical_scope",
    "resolve_alpha_model_numerical_environment",
]
