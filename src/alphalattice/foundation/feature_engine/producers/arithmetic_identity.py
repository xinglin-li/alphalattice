"""Identify the code that computes Factor values, by its rule (LAWS.md ID3).

The Feature capability once had four identities that read like implementation identity and
contained none: a hand-written description of the engine, the core bundle's declared fields, a
core Factor's ``{bundle_hash, factor_id, formula_ref}`` and an extension kernel's own summary.
Each survived a rewrite of the arithmetic. This module is the measured side, apart from both the
catalog contracts and the kernel registry because both need it and depend on each other.

There are two kinds of closure, not one. The controls bind the shared owners that compute them,
and each extension method family binds its own modules plus those shared owners, because the
materializer post-processes every extension series it materializes. So an extension edit cannot
reach a control, and a shared-owner edit reaches both.

Each closure is a rule closure: the owners and every module they import inside the
number-deciding packages, hashed as syntax, so a comment or a docstring moves nothing and a module
the arithmetic runs is never left out. The walk skips what decides none of the values a
closure covers: this module, the installed composition (which kernels a build installs has its own
identity, ``FeatureKernelRegistry.installed_capability_hash``), the catalog contracts (governance:
editing an admission rule moves catalog identity, not the arithmetic) and, for the controls, the
registry mechanics every extension value passes through.

The switch from byte closures kept every identity. ``config/identity-switch.json`` records each
component's rule value and the byte value it had when the rule replaced the bytes; while its rule
value is the recorded one, a component keeps the byte value every Panel and catalog binding was
sealed under (``switched_identity``). A changed rule is a new identity; a comment never is.

A Panel and a catalog binding compare these values by equality, so the value they bind is held at
its recorded origin (LAWS.md ID1): a change recorded in ``config/identity-successors.json`` under
the component's role as keeping every value it computes leaves the bound value where it was,
and an unrecorded change is a new identity. The roles read the value before that hold
(``control_arithmetic_rule_identity``, ``method_family_rule_identity``), so the readout names
every change the hold absorbs.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache, lru_cache
from importlib.util import find_spec
from pathlib import Path
from typing import Final

from alphalattice.kernel.shared_kernel import source_identity
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.identity_successors import recorded_origin
from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root

SHARED_FACTOR_ARITHMETIC_OWNERS: Final = (
    "alphalattice.foundation.feature_engine.producers.base_materializer",
    "alphalattice.kernel.quant.factor_formulas",
)
"""The modules whose bytes decide what *any* Factor value is.

``base_materializer`` computes the shipped desktop series, and it also
materializes every extension the catalog names: it projects the required source
columns, calls the registry, and passes the result through the same finiteness
guard the core series get. So it is shared in fact and not merely by location.
``quant.factor_formulas`` contributes the annualization constant and the
ineligibility reasons the materializer scales and labels with -- a small surface,
but one that moves numbers, so it sits inside the closure rather than in a
footnote outside it.

Catalog and contract modules are deliberately absent. Editing an admission rule
should move catalog identity, not claim the arithmetic changed. That is the same
scoping the Panel preprocessing identity already uses.
"""

EXTENSION_KERNEL_OWNERS: Final = (
    "alphalattice.foundation.feature_engine.producers.factors.registry",
)
"""The registry mechanics every extension value passes through on its way out.

Present in the method-family closures and absent from the control closure, which
is exactly the asymmetry the registry has in fact: it resolves, shape-checks and
coerces the dtype of extension results and never touches a core series.

The *installed composition* is deliberately not here, and that correction is the
point. While mechanics and composition shared one module, installing a second
family edited that module and rotated the first family's implementation identity
-- so the extension surface broke its own promise that adding a method does not
invalidate unrelated methods. Which kernels a build installs is a real identity
and it has its own home: ``FeatureKernelRegistry.installed_capability_hash``.
"""

FACTOR_METHOD_FAMILY_OWNERS: Final[Mapping[str, tuple[str, ...]]] = {
    "OPEN_INTRADAY": ("alphalattice.foundation.feature_engine.producers.factors.open_intraday",),
    "ABSOLUTE_STATE": ("alphalattice.foundation.feature_engine.producers.factors.absolute_state",),
    "DOWNSIDE_TAIL": (
        "alphalattice.foundation.feature_engine.producers.factors.downside_tail",
        "alphalattice.foundation.feature_engine.producers.factors.series_math",
    ),
    "RESIDUAL_REVERSAL": (
        "alphalattice.foundation.feature_engine.producers.factors.residual_reversal",
        "alphalattice.foundation.feature_engine.producers.factors.series_math",
    ),
    "STATE_INTERACTION": (
        "alphalattice.foundation.feature_engine.producers.factors.interactions",
        "alphalattice.foundation.feature_engine.producers.factors.series_math",
    ),
    "SECTOR_LEADER_LAG": ("alphalattice.foundation.feature_engine.producers.factors.leader_lag",),
    "SESSION_OBSERVATION": (
        "alphalattice.foundation.feature_engine.producers.factors.session_observation",
    ),
    "SESSION_LIQUIDITY": (
        "alphalattice.foundation.feature_engine.producers.factors.session_liquidity",
    ),
    "FACTOR_FORMULA": (
        "alphalattice.foundation.feature_engine.producers.factors.formula",
        "alphalattice.foundation.feature_engine.producers.factors.formula_language",
    ),
}
"""Every method family this *product* installs, and the modules that compute it.

Explicit and closed. A product family is added by naming it here and writing its
module, never by scanning a directory: a closure discovered from the filesystem
would make a Factor's identity depend on what happened to be lying next to it.

Deliberately not the only source of a family closure. An external consumer can
register a kernel whose formulas live outside this package entirely, and such a
method still deserves a *measured* implementation identity rather than a declared
one -- otherwise the first thing anyone integrating against this surface loses is
the property the surface exists to provide. So a registered kernel carries its
own owner modules and ``method_family_content_hash`` takes them as an argument;
this mapping is what the product's own composition passes in.
"""


_ROOT: Final = resolve_playpen_root(Path(__file__))
"""The checkout the rule walks: its ``src`` tree and its ``config`` tables."""

FACTOR_VALUE_ROLE: Final = "feature_engine.factor_value"
"""The prefix of each Factor value's role in ``config/identity-roles.json``: ``.controls`` for
the controls, ``.<FAMILY>`` for a method family. Its recorded moves hold the bound value."""

WALK_EXCLUDED: Final = frozenset(
    {
        "alphalattice.foundation.feature_engine.producers.arithmetic_identity",
        "alphalattice.foundation.feature_engine.producers.factors.catalog",
        "alphalattice.foundation.feature_engine.catalog.contracts",
    }
)
"""What no Factor closure walks: this module, the installed composition, the catalog contracts."""

PREPROCESSING_WALK_EXCLUDED: Final = WALK_EXCLUDED | frozenset(
    {
        "alphalattice.foundation.feature_engine.producers.preprocessing.catalog",
        "alphalattice.foundation.feature_engine.producers.preprocessing.contracts",
    }
)
"""What no preprocessing closure walks: those, and the preprocessing catalog and contracts,
which are governance: an admission edit moves catalog identity, not the transformation."""


def feature_component_identity(
    component: str,
    *,
    owners: tuple[str, ...],
    excluded: frozenset[str],
    semantic_owner: str,
    numerical_role: str,
) -> str:
    """The identity of one Feature component: its rule closure, kept at its byte value.

    Owners inside ``alphalattice`` are walked by the rule; an owner outside it, a method an
    external consumer registers, is hashed as its own syntax, found through the import system.
    While the rule value is the one the switch recorded, the component keeps the byte value it
    had then, so what was sealed before the switch stays current.

    Args:
        component: The switch table's key for this component.
        owners: The modules that compute its values.
        excluded: The modules its walk skips.
        semantic_owner: Who owns the identity.
        numerical_role: What kind of number it identifies.

    Returns:
        The identity.

    Raises:
        ValueError: If an owner cannot be found or the switch table cannot be read.
    """
    inside = tuple(owner for owner in owners if owner.startswith("alphalattice."))
    outside: dict[str, str] = {}
    for owner in owners:
        if owner in inside:
            continue
        spec = find_spec(owner)
        if spec is None or spec.origin is None:
            raise ValueError("feature_engine.factor_arithmetic_owner_unresolved")
        outside[owner] = source_identity.source_syntax_sha256(Path(spec.origin))
    rule = str(
        canonical_hash(
            {
                "semantic_owner": semantic_owner,
                "numerical_role": numerical_role,
                "modules": source_identity.number_deciding_closure(
                    inside,
                    root=_ROOT,
                    rule=source_identity.number_deciding_rule(_ROOT),
                    excluded=excluded,
                ),
                "outside": outside,
            }
        )
    )
    return source_identity.switched_identity(component, rule, root=_ROOT)


def control_arithmetic_rule_identity() -> str:
    """The controls' identity as the rule measures this tree, before recorded moves hold it.

    Returns:
        The switched rule value: what the controls' role reads.
    """
    return feature_component_identity(
        "FACTOR_VALUE:controls",
        owners=SHARED_FACTOR_ARITHMETIC_OWNERS,
        excluded=WALK_EXCLUDED | frozenset(EXTENSION_KERNEL_OWNERS),
        semantic_owner="feature_engine.producers",
        numerical_role="FACTOR_VALUE",
    )


@lru_cache(maxsize=1)
def control_arithmetic_content_hash() -> str:
    """The identity of the code that computes the activated base controls.

    Module-scoped rather than per-control on purpose: the controls share one materializer and
    its helpers, so a per-control claim would be precision the code does not have. The walk
    also skips the registry mechanics, which only extension values pass through. Held at its
    recorded origin, so a move recorded as keeping every control's value keeps every Panel.

    Returns:
        The controls' identity.
    """
    return recorded_origin(
        f"{FACTOR_VALUE_ROLE}.controls", control_arithmetic_rule_identity(), root=_ROOT
    )


@cache
def method_family_rule_identity(method_family: str, owners: tuple[str, ...]) -> str:
    """One method family's identity as the rule measures this tree, before recorded moves hold it.

    Args:
        method_family: The family's name.
        owners: The modules that hold its formulas.

    Returns:
        The switched rule value: what the family's role reads.

    Raises:
        ValueError: If the family names no owner: an empty closure would give an external method
            the identity of the arithmetic it merely runs on.
    """
    if not owners:
        raise ValueError("feature_engine.factor_method_family_owners_required")
    return feature_component_identity(
        f"FACTOR_VALUE:{method_family}",
        owners=(*SHARED_FACTOR_ARITHMETIC_OWNERS, *EXTENSION_KERNEL_OWNERS, *owners),
        excluded=WALK_EXCLUDED,
        semantic_owner="feature_engine.producers.factors",
        numerical_role="FACTOR_VALUE",
    )


@cache
def method_family_content_hash(method_family: str, owners: tuple[str, ...]) -> str:
    """The identity of the code that computes one extension method family's values.

    The family's own modules, the registry mechanics its values pass through and the shared
    owners underneath, together: a closure naming only the family would stand still through a
    rewrite of the materializer that post-processes its output. ``owners`` are import paths,
    so a family defined outside this package measures the same way a product family does.
    Held at its recorded origin, as the controls are.

    Args:
        method_family: The family's name.
        owners: The modules that hold its formulas.

    Returns:
        The family's identity.

    Raises:
        ValueError: If the family names no owner: an empty closure would give an external method
            the identity of the arithmetic it merely runs on.
    """
    return recorded_origin(
        f"{FACTOR_VALUE_ROLE}.{method_family}",
        method_family_rule_identity(method_family, owners),
        root=_ROOT,
    )


def installed_method_family_owners(method_family: str) -> tuple[str, ...]:
    """The owner modules of one family this product installs.

    An unregistered family is refused rather than answered with an empty tuple: a
    caller asking the *product* about a family nobody installed has a bug, and an
    empty answer would hide it behind a well-formed hash one layer down.

    Args:
        method_family: Family name registered by the installed product composition.

    Returns:
        Declared modules that compute that family's values.

    Raises:
        ValueError: The product has not installed the requested method family.
    """
    owners = FACTOR_METHOD_FAMILY_OWNERS.get(method_family)
    if owners is None:
        raise ValueError("feature_engine.factor_method_family_not_installed")
    return owners


__all__ = [
    "EXTENSION_KERNEL_OWNERS",
    "FACTOR_METHOD_FAMILY_OWNERS",
    "FACTOR_VALUE_ROLE",
    "PREPROCESSING_WALK_EXCLUDED",
    "SHARED_FACTOR_ARITHMETIC_OWNERS",
    "WALK_EXCLUDED",
    "control_arithmetic_content_hash",
    "control_arithmetic_rule_identity",
    "feature_component_identity",
    "installed_method_family_owners",
    "method_family_content_hash",
    "method_family_rule_identity",
]
