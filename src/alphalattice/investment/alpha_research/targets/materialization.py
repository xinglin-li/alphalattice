"""The result of rebuilding one published canonical target surface from source.

A contract rather than a service, and here rather than beside the service that
produces it, for the reason the rest of this domain is already split that way:
``targets.canonical`` owns what a canonical surface *is*, and the development
service in ``experiments`` owns resolving, publishing and verifying one. A
consumer of a rebuilt surface -- the return-unit calibration is the first --
needs the type and not the service, and importing the service to get the type
puts a store, a resolver and an outcome reader in its import graph.
"""

from __future__ import annotations

from dataclasses import dataclass

from .canonical import (
    CanonicalAlphaTargetEvidence,
    CanonicalAlphaTargetRecipeBinding,
    CanonicalAlphaTargetSurface,
)


@dataclass(frozen=True, slots=True)
class CanonicalTargetMaterialization:
    """One published surface rebuilt from source, and the reconciliation proving it.

    Development readback, never a second publication. The values are recompiled
    from the read-only causal outcome authority through the same compiler the
    original run used, and are admitted only when every lane identity the
    published evidence carries -- and the lane digest covering the ones it does
    not -- reproduce exactly.

    That last part is the point. ``simple_economic_return`` has no field of its
    own on the evidence, so a consumer wanting the raw economic return had no
    durable value table to read and no way to prove a rebuilt one was the same
    lane. It is covered by ``lane_identity_hash``, so reproducing that hash is a
    statement about this lane rather than only about the six stored beside it.
    """

    evidence: CanonicalAlphaTargetEvidence
    recipe_binding: CanonicalAlphaTargetRecipeBinding
    surface: CanonicalAlphaTargetSurface
    outcome_method_binding_hash: str
    maturity_lag_sessions: int
    simple_economic_return_identity: str
    """The rebuilt lane's own byte identity, derived on readback and published nowhere."""


__all__ = ["CanonicalTargetMaterialization"]
