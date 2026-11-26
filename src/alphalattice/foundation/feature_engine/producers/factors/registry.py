"""The stable kernel protocol and the mechanics every extension value passes through.

Split out of ``catalog`` because the two were one module and had to stop being
one. A method family's implementation identity binds this module -- the registry
resolves a kernel, validates its inputs, checks the returned shape and coerces
the numeric domain, so its rule syntax shapes every extension value. It must
*not* bind the installed composition, and that is exactly what a single module
forced: registering a second family edited the same file, so an unchanged family
rotated its implementation identity because somebody installed something else.

That is the opposite of what the extension surface promises. "Add a method
without invalidating unrelated methods" is the whole reason a per-family closure
exists, and a closure that contains the list of installed families cannot keep
it.

So the boundary here is: mechanics that decide what a kernel's output *is* live
in this module and are measured; the decision about *which* kernels a build
installs lives in ``catalog`` and is measured by
``FeatureKernelRegistry.installed_capability_hash`` instead -- an identity that
moves when the installed set changes and reaches no factor's own identity.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from alphalattice.foundation.feature_engine.catalog.contracts import DesktopCoreFeatureBundle
from alphalattice.foundation.feature_engine.producers.arithmetic_identity import (
    method_family_content_hash,
)
from alphalattice.foundation.feature_engine.producers.factors.core_bundle import (
    control_implementation_hash,
    is_maintained_control,
    numerical_spec_hash,
)
from alphalattice.kernel.quant.factor_contracts import FactorSpec
from alphalattice.kernel.shared_kernel.identity import canonical_hash

FeatureKernelFunction = Callable[[pd.DataFrame, FactorSpec], pd.Series]


@dataclass(frozen=True)
class RegisteredFeatureKernel:
    """Declare one extension kernel and the measured family which owns its callable.

    Attributes:
        implementation_id: Stable handle resolved by a factor recipe.
        method_family: Family whose arithmetic implements this kernel.
        method_family_owners: Sorted unique module paths in its measured rule closure.
        required_fields: Source fields required by the numerical callable.
        implementation_hash: Declared implementation summary; measured content is bound separately.
        compute: Code-owned numerical callable accepting a detached source and recipe.
    """

    implementation_id: str
    method_family: str
    method_family_owners: tuple[str, ...]
    required_fields: tuple[str, ...]
    implementation_hash: str
    compute: FeatureKernelFunction

    def __post_init__(self) -> None:
        """Verify that the kernel declares the measured family which owns its callable.

        Raises:
            ValueError: Family or owners are absent, owners are not sorted and unique,
                or the compute callable's module is outside the declared family.
        """
        if not self.method_family or not self.method_family_owners:
            raise ValueError("Feature kernel must declare the family that computes it")
        if self.method_family_owners != tuple(sorted(set(self.method_family_owners))):
            raise ValueError("Feature kernel method-family owners must be sorted and unique")
        compute_owner = getattr(self.compute, "__module__", None)
        if compute_owner not in self.method_family_owners:
            raise ValueError(
                "Feature kernel compute owner is absent from its measured method-family closure"
            )


class FeatureKernelRegistry:
    """Resolve only implementations registered by code, never model-supplied Python."""

    def __init__(self, kernels: tuple[RegisteredFeatureKernel, ...] = ()) -> None:
        """Install an explicit set of code-owned extension kernels.

        Args:
            kernels: Implementations to index by their stable handles.

        Raises:
            ValueError: More than one kernel names the same implementation ID.
        """
        if len({item.implementation_id for item in kernels}) != len(kernels):
            raise ValueError("Feature kernel registry contains duplicate implementation IDs")
        self._kernels = {item.implementation_id: item for item in kernels}

    @property
    def installed_capability_hash(self) -> str:
        """Which kernels this build installs, as one identity.

        This is where "a second family was installed" belongs, and it is
        deliberately not folded into any factor's implementation or methodology
        identity. A consumer asking "is this the same build" reads this; a
        consumer asking "is this the same code behind this factor" reads the
        factor's own identity, and installing something unrelated must not move
        that one.

        Ordered by implementation id rather than by registration order: two
        builds that installed the same kernels are the same installed capability
        however their composition function happened to list them.
        """
        return str(
            canonical_hash(
                {
                    "kind": "InstalledFeatureKernelCapability",
                    "kernels": [
                        {
                            "implementation_id": item.implementation_id,
                            "method_family": item.method_family,
                            "method_family_owners": list(item.method_family_owners),
                            "required_fields": list(item.required_fields),
                            "implementation_hash": item.implementation_hash,
                            # The declaration says what the implementation is;
                            # this binds the bytes that actually implement it.
                            # Without both, the installed build identity would
                            # stand still through an unaccompanied code edit.
                            "method_family_content_hash": method_family_content_hash(
                                item.method_family, item.method_family_owners
                            ),
                        }
                        for item in sorted(
                            self._kernels.values(), key=lambda value: value.implementation_id
                        )
                    ],
                }
            )
        )

    def resolve(self, implementation_id: str) -> RegisteredFeatureKernel:
        """Resolve an installed code-owned kernel by its stable implementation handle.

        Args:
            implementation_id: Handle named by the factor recipe.

        Returns:
            The explicitly registered numerical kernel.

        Raises:
            ValueError: No installed kernel owns the requested implementation ID.
        """
        try:
            return self._kernels[implementation_id]
        except KeyError as exc:
            raise ValueError(
                f"Feature implementation is not code-owned: {implementation_id}"
            ) from exc

    def implementation_hash(
        self,
        specification: FactorSpec,
        *,
        core_bundle: DesktopCoreFeatureBundle,
    ) -> str:
        """Identify the rule syntax which computes one Factor under its qualified recipe.

        Args:
            specification: Factor recipe identifying the maintained control or extension kernel.
            core_bundle: Qualified maintained bundle used to resolve the control route.

        Returns:
            Maintained-control identity or the extension's declared summary and measured
            method-family content identity. Installed composition is bound separately.

        Raises:
            ValueError: The extension is unregistered or its required fields differ from
                the recipe's ordered field declaration.
        """
        if is_maintained_control(specification, core_bundle=core_bundle):
            return control_implementation_hash(specification, core_bundle=core_bundle)
        kernel = self.resolve(specification.formula_ref)
        if specification.required_fields != tuple(sorted(kernel.required_fields)):
            raise ValueError("Feature specification required fields differ from its kernel")
        return str(
            canonical_hash(
                {
                    "declared": kernel.implementation_hash,
                    "method_family": kernel.method_family,
                    "source_content": method_family_content_hash(
                        kernel.method_family, kernel.method_family_owners
                    ),
                }
            )
        )

    def compute(self, source: pd.DataFrame, specification: FactorSpec) -> pd.Series:
        """Run the resolved kernel on a detached source frame and coerce its numeric result.

        Args:
            source: Listing/session rows with every field declared by the installed kernel.
            specification: Qualified recipe naming the implementation to execute.

        Returns:
            Floating results on the kernel's returned index. Nonnumeric entries become
            missing; row count must equal the source frame's row count.

        Raises:
            ValueError: The implementation is unregistered, source fields are absent,
                or the kernel returns a different number of rows.
        """
        kernel = self.resolve(specification.formula_ref)
        required = {"listing_id", "session_date", *kernel.required_fields}
        missing = sorted(required - set(source.columns))
        if missing:
            raise ValueError(f"Feature kernel input is missing columns: {missing}")
        result = kernel.compute(source.copy(), specification)
        if len(result) != len(source):
            raise ValueError("Feature kernel returned the wrong row count")
        return pd.to_numeric(result, errors="coerce").astype(float)


def factor_methodology_hash(
    specification: FactorSpec,
    *,
    implementation_hash: str,
) -> str:
    """The identity of one factor's method: its own recipe plus its own code.

    Deliberately **local**. It binds only what belongs to this factor -- its id,
    its numerical recipe, and the identity of the kernel that computes it -- and
    not the catalog revision it happens to be installed in. Folding the revision
    in here would mean installing an unrelated factor invalidated the identity of
    every existing one, which is the opposite of what an identity is for. The
    revision and the ordered axis are bound separately, one layer up, where they
    are genuinely properties of the catalog rather than of a factor.

    Distinct from ``implementation_hash``, which answers only "which code". Two
    factors can share a kernel and differ in window or lag; they are different
    methods, and until now the Factor Desk could not tell them apart.
    """
    return str(
        canonical_hash(
            {
                "kind": "FactorMethodology",
                "factor_id": specification.factor_id,
                "implementation_hash": implementation_hash,
                "numerical_spec_hash": numerical_spec_hash(specification),
            }
        )
    )


__all__ = [
    "FeatureKernelFunction",
    "FeatureKernelRegistry",
    "RegisteredFeatureKernel",
    "factor_methodology_hash",
]
