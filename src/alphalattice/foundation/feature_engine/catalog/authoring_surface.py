"""The Factor authoring surface: each complete installed formula and its development admission.

What a researcher or an agent authors against: every installed Factor formula with its
implementation and specification identities, whether its development admission admitted or
refused it, and the next action it needs; and the development axis a study reads, derived from
its controls and the admitted candidates.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from alphalattice.foundation.feature_engine.producers.factors.catalog import extension_factor_specs
from alphalattice.foundation.feature_engine.producers.factors.specifications import (
    FactorDevelopmentAdmissionReceipt,
    FactorFormulaSpecificationCatalog,
    admit_factor_development_capabilities,
    build_installed_factor_development_capabilities,
    build_installed_factor_formula_specifications,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash


class FactorAuthoringEntry(BaseModel):  # type: ignore[misc]
    """One installed formula with its development admission and the next action it needs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["FactorAuthoringEntry"] = "FactorAuthoringEntry"
    factor_id: str
    installed_formula_ref: str
    implementation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    specification_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    formula_status: Literal["COMPLETE"] = "COMPLETE"
    development_disposition: Literal["ADMITTED", "REFUSED"]
    development_reason: str
    admitted_to_catalog: bool
    preprocessing_role: str
    settled_method_mechanics: tuple[str, ...]
    required_next_action: str


class FactorAuthoringSurface(BaseModel):  # type: ignore[misc]
    """Every installed formula's entry, the admitted and refused ids, and the surface's hash."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["FactorAuthoringSurface"] = "FactorAuthoringSurface"
    entries: tuple[FactorAuthoringEntry, ...]
    admitted_factor_ids: tuple[str, ...]
    refused_factor_ids: tuple[str, ...]
    surface_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


def describe_factor_authoring_surface(
    *,
    catalog: FactorFormulaSpecificationCatalog | None = None,
    admission_receipts: tuple[FactorDevelopmentAdmissionReceipt, ...] | None = None,
) -> FactorAuthoringSurface:
    """Describe every installed Factor formula as an author meets it.

    Args:
        catalog: The formula specifications; the installed ones by default.
        admission_receipts: The development admissions recorded; none by default.

    Returns:
        The surface, each entry with its admission and next action.
    """
    specifications = catalog or build_installed_factor_formula_specifications()
    receipts = admission_receipts or admit_factor_development_capabilities()
    decisions = {item.factor_id: item for item in receipts}
    capabilities = {
        item.factor_id: item for item in build_installed_factor_development_capabilities()
    }
    entries = []
    for recipe in extension_factor_specs():
        specification = specifications.resolve(recipe.factor_id)
        decision = decisions[recipe.factor_id]
        capability = capabilities[recipe.factor_id]
        entries.append(
            FactorAuthoringEntry(
                factor_id=recipe.factor_id,
                installed_formula_ref=recipe.formula_ref,
                implementation_hash=capability.implementation_hash,
                specification_hash=specification.specification_hash,
                development_disposition=decision.disposition,
                development_reason=decision.reason,
                admitted_to_catalog=decision.disposition == "ADMITTED",
                preprocessing_role=capability.preprocessing_role,
                settled_method_mechanics=(
                    f"PRICE_BASIS: {specification.price_basis}",
                    f"FORMULA: {specification.formula}",
                    f"FORMATION_CUTOFF: {specification.formation_cutoff}",
                    f"SOURCE_INTERVAL: {specification.clock.source_interval}",
                    f"MINIMUM_ORDERED_SOURCE_ROWS: {specification.minimum_ordered_source_rows}",
                    f"FINITE_POLICY: {specification.finite_policy}",
                ),
                required_next_action=(
                    "NONE -- selectable for development evidence."
                    if decision.disposition == "ADMITTED"
                    else f"RESOLVE_AUTHORITY -- {decision.reason}"
                ),
            )
        )
    ordered = tuple(sorted(entries, key=lambda item: item.factor_id))
    payload = {
        "kind": "FactorAuthoringSurface",
        "entries": [item.model_dump(mode="json") for item in ordered],
    }
    return FactorAuthoringSurface(
        entries=ordered,
        admitted_factor_ids=tuple(
            item.factor_id for item in ordered if item.development_disposition == "ADMITTED"
        ),
        refused_factor_ids=tuple(
            item.factor_id for item in ordered if item.development_disposition == "REFUSED"
        ),
        surface_hash=str(canonical_hash(payload)),
    )
