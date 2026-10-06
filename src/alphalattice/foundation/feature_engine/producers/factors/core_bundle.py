"""Bind the current Base Panel activation manifest to its arithmetic.

Narrow on purpose. The control formulas are a separately governed source that
this milestone does not promote and must not fork, so nothing here reimplements
one: the bundle's declared content still comes from the shipped catalog resource
and the numbers still come from ``producers.base_materializer``. What this module
adds is the *binding* between them -- the identity a catalog publishes for a
control Factor, and the test that decides whether a recipe is still one of the
maintained controls at all.

It exists as its own module because the Factor namespace beside it must be able
to ask both questions without importing an extension kernel registry, and
because a reader looking for where base activations attach should find one file rather
than a helper buried in the registry that also owns installation.
"""

from __future__ import annotations

from alphalattice.foundation.feature_engine.catalog.contracts import (
    DesktopCoreFeatureBundle,
    desktop_core_feature_bundle,
)
from alphalattice.foundation.feature_engine.producers.arithmetic_identity import (
    control_arithmetic_content_hash,
)
from alphalattice.kernel.quant.factor_contracts import NUMERICAL_SPEC_FIELDS, FactorSpec
from alphalattice.kernel.shared_kernel.identity import canonical_hash


def numerical_spec_hash(specification: FactorSpec) -> str:
    """Hash only the fields of a recipe that can change what it produces."""
    payload = specification.model_dump(mode="json")
    return str(canonical_hash({key: payload[key] for key in NUMERICAL_SPEC_FIELDS}))


def is_maintained_control(
    specification: FactorSpec, *, core_bundle: DesktopCoreFeatureBundle
) -> bool:
    """True when this recipe is still exactly the control the bundle maintains.

    Matched on the numerical fields rather than on the id. A recipe that borrowed
    a control's id and changed its window is not that control, and resolving it
    as one would publish the control's implementation identity over numbers the
    control never produced.
    """
    declared = dict(core_bundle.numerical_spec_hashes)
    return declared.get(specification.factor_id) == numerical_spec_hash(specification)


def control_implementation_hash(
    specification: FactorSpec, *, core_bundle: DesktopCoreFeatureBundle
) -> str:
    """Identify the code that computes one maintained control, by its bytes.

    Three of the four components are *declarations*: the bundle hash, the factor
    id and the formula reference all survive an arbitrary rewrite of the
    arithmetic. An identity built from declarations alone cannot answer the one
    question a consumer asks of it -- did the code that produced these values
    change -- and a Panel identity that leaned on it once licensed backfilling
    provenance nobody could prove.

    So the measured control closure is folded in beside them. It is scoped to the
    shared owners and no longer to every extension kernel in the build: an
    unshipped experimental Factor changing under a researcher's hands is not a
    reason to rotate the identity of unchanged base activations.
    """
    return str(
        canonical_hash(
            {
                "owner": core_bundle.bundle_hash,
                "factor_id": specification.factor_id,
                "formula_ref": specification.formula_ref,
                "source_content": control_arithmetic_content_hash(),
            }
        )
    )


__all__ = [
    "NUMERICAL_SPEC_FIELDS",
    "DesktopCoreFeatureBundle",
    "control_implementation_hash",
    "desktop_core_feature_bundle",
    "is_maintained_control",
    "numerical_spec_hash",
]
