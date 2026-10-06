"""Source-bound local Feature edits; never activate the daily catalog."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog
from alphalattice.foundation.feature_engine.catalog.crud_contracts import (
    FeatureCatalogDelta,
    FeatureCatalogRevision,
    FeatureCatalogUpdatePlan,
)
from alphalattice.foundation.feature_engine.catalog.service import (
    FeatureCatalogCrudError,
    FeatureCatalogCrudPlanner,
    FeatureCatalogEditor,
)
from alphalattice.foundation.feature_engine.producers.factors.catalog import (
    default_extension_kernel_registry,
    installed_feature_kernels,
    installed_formula_specs,
)
from alphalattice.foundation.feature_engine.producers.factors.formula import (
    FORMULA_IDS,
    formula_controls,
    formula_preprocessing,
    formula_specification,
)
from alphalattice.foundation.feature_engine.producers.factors.formula_language import (
    FormulaError,
)
from alphalattice.kernel.quant.factor_contracts import FactorSpec
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.sealing import seal_model

_HASH = r"^[0-9a-f]{64}$"


class ResearchFeatureEdit(BaseModel):  # type: ignore[misc]
    """Describe one proposed research Formula edit."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    operation: Literal["CREATE", "UPDATE", "RETIRE", "RENAME"]
    factor_id: str = Field(min_length=1, max_length=160)
    specification: FactorSpec | None = None
    preprocessing_recipe: str | None = Field(default=None, exclude_if=lambda value: value is None)
    """For a formula factor, the Panel preprocessing recipe its declaration chooses (EX): required
    for one, refused for any other factor."""

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def check_shape(self) -> Self:
        """Verify the edit operation and specification shape."""
        if (self.operation == "RETIRE") != (self.specification is None):
            raise ValueError("feature_research.specification_required_except_retire")
        return self


class ResearchFeatureChange(BaseModel):  # type: ignore[misc]
    """Describe the governed impact of a research edit."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    input_binding_hash: str = Field(pattern=_HASH)
    base_revision_hash: str = Field(pattern=_HASH)
    parent_plan_hash: str | None = Field(default=None, pattern=_HASH)
    edits: tuple[ResearchFeatureEdit, ...] = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=1, max_length=2000)


class ResearchFeaturePlan(BaseModel):  # type: ignore[misc]
    """An immutable definition plan, not proof that any column has been built."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["ResearchFeaturePlan"] = "ResearchFeaturePlan"
    scope: Literal["RESEARCH_LOCAL_NO_GLOBAL_ACTIVATION"] = "RESEARCH_LOCAL_NO_GLOBAL_ACTIVATION"
    request: ResearchFeatureChange
    source_panel_snapshot_hash: str = Field(pattern=_HASH)
    base: FeatureCatalogRevision
    candidate: FeatureCatalogRevision
    candidate_payload: dict[str, Any]
    delta: FeatureCatalogDelta
    work: FeatureCatalogUpdatePlan
    required_fields: tuple[str, ...]
    preprocessing_recipes: dict[str, str] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    """Each formula factor of the candidate and the preprocessing recipe its declaration chose."""
    plan_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: Any) -> Self:
        """Create a content-addressed research Feature plan."""
        return seal_model(cls, values, field="plan_hash")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def verify(self) -> Self:
        """Verify the plan identity and edit scope."""
        factors = self.candidate_payload.get("factors")
        if not isinstance(factors, list) or set(self.candidate_payload) != {
            "stable_id",
            "observation_clock",
            "factors",
            "maintenance_policy",
        }:
            raise ValueError("feature_research.plan_binding_invalid")
        needed = set(self.work.base_compute_factor_ids)
        required = tuple(
            sorted(
                {
                    field
                    for item in self.candidate.features
                    if item.factor_id in needed
                    for field in item.specification.required_fields
                }
            )
        )
        if (
            self.plan_hash != canonical_hash(self.model_dump(mode="json", exclude={"plan_hash"}))
            or self.request.base_revision_hash != self.base.revision_hash
            or self.candidate.parent_revision_hash != self.base.revision_hash
            or self.delta.candidate_revision_hash != self.candidate.revision_hash
            or self.delta.base_revision_hash != self.base.revision_hash
            or self.work.delta_hash != self.delta.delta_hash
            or self.work.base_revision_hash != self.base.revision_hash
            or self.work.candidate_revision_hash != self.candidate.revision_hash
            or self.required_fields != required
            or self.candidate.maintenance_policy != self.candidate_payload.get("maintenance_policy")
            or self.candidate.stable_id != self.candidate_payload.get("stable_id")
            or tuple(FactorSpec.model_validate_json(json.dumps(v)) for v in factors)
            != tuple(v.specification for v in self.candidate.features)
            or set(self.preprocessing_recipes)
            != {
                v.factor_id
                for v in self.candidate.features
                if v.specification.formula_ref in FORMULA_IDS
            }
        ):
            raise ValueError("feature_research.plan_binding_invalid")
        return self


def research_feature_controls() -> dict[str, object]:
    """Discover code-owned recipes/kernels, not a second installation registry."""
    registry = default_extension_kernel_registry()
    return {
        "request_schema": ResearchFeatureChange.model_json_schema(),
        "registered_formulas": [v.model_dump(mode="json") for v in installed_formula_specs()],
        "registered_kernels": [
            {
                "formula_ref": v.implementation_id,
                "method_family": v.method_family,
                "required_fields": list(v.required_fields),
            }
            for v in sorted(installed_feature_kernels(), key=lambda v: v.implementation_id)
        ],
        "formula_language": formula_controls(),
        "installed_capability_hash": registry.installed_capability_hash,
        "scope": "RESEARCH_LOCAL_NO_GLOBAL_ACTIVATION",
        "claim": "REGISTERED_ARITHMETIC_NOT_MATERIALIZED_OR_ADMITTED_TO_FACTOR_RESEARCH",
    }


def research_feature_execution_spec(plan: ResearchFeaturePlan) -> dict[str, object]:
    """Read the complete executed definition independently of its authored declaration.

    Args:
        plan: Verified research feature plan.

    Returns:
        Input revision, source Panel, candidate definitions, compute/reconciliation scope,
        required fields and every preprocessing recipe. Declaration reasons and plan handles
        do not execute; the candidate revision already binds its base and definitions.
    """
    return {
        "input_binding_hash": plan.request.input_binding_hash,
        "source_panel_snapshot_hash": plan.source_panel_snapshot_hash,
        "candidate_revision_hash": plan.candidate.revision_hash,
        "work": plan.work.model_dump(mode="json", exclude={"plan_hash", "delta_hash", "kind"}),
        "required_fields": plan.required_fields,
        "preprocessing_recipes": plan.preprocessing_recipes,
    }


def _kept(specification: FactorSpec | None) -> FactorSpec | None:
    """A formula factor's spec as its kernel keeps it (EX, V88); any other spec as written."""
    if specification is None or specification.formula_ref not in FORMULA_IDS:
        return specification
    try:
        return formula_specification(specification)
    except FormulaError as error:
        raise FeatureCatalogCrudError(
            str(error),
            "The formula is outside the language; `feature controls` states its leaves, "
            "operators and rules under formula_language.",
        ) from error


def plan_research_feature_change(
    *,
    catalog: FeatureCatalog,
    base: FeatureCatalogRevision,
    source_panel_snapshot_hash: str,
    request: ResearchFeatureChange,
    preprocessing_recipes: Mapping[str, str] | None = None,
) -> ResearchFeaturePlan:
    """Pure edits plus the existing change classifier; never run a kernel here.

    `preprocessing_recipes` are the base's formula factors and their recipes (a parent plan's);
    each edit of a formula factor names its own.
    """
    if request.base_revision_hash != base.revision_hash or tuple(catalog.factors) != tuple(
        v.specification for v in base.features
    ):
        raise FeatureCatalogCrudError(
            "feature_research.base_revision_mismatch",
            "Use the exact definition revision supplied by discovery/readback.",
        )
    candidate = catalog
    renames = []
    recipes = dict(preprocessing_recipes or {})
    for edit in request.edits:
        specification = _kept(edit.specification)
        formula = specification is not None and specification.formula_ref in FORMULA_IDS
        if edit.preprocessing_recipe is not None and not formula:
            raise FeatureCatalogCrudError(
                "feature_research.preprocessing_recipe_formula_only",
                "Only a formula factor names its preprocessing recipe; an installed kernel's "
                "factor takes its installed role.",
            )
        recipes.pop(edit.factor_id, None)
        if formula:
            assert specification is not None
            try:
                recipes[specification.factor_id] = formula_preprocessing(edit.preprocessing_recipe)
            except FormulaError as error:
                raise FeatureCatalogCrudError(
                    str(error),
                    "A formula factor names `preprocessing_recipe`; `feature controls` lists the "
                    "choices under formula_language.",
                ) from error
        editor = FeatureCatalogEditor(candidate)
        prior = next((v for v in candidate.factors if v.factor_id == edit.factor_id), None)
        if (
            edit.operation in {"UPDATE", "RENAME"}
            and prior is not None
            and specification is not None
            and (
                prior.absolute_tolerance != specification.absolute_tolerance
                or prior.relative_tolerance != specification.relative_tolerance
            )
        ):
            raise FeatureCatalogCrudError(
                "feature_research.validation_policy_change_not_admitted",
                "Validation tolerances require a separate scientific-policy decision, "
                "not a metadata edit.",
            )
        match edit.operation:
            case "CREATE":
                assert specification is not None
                if edit.factor_id != specification.factor_id:
                    raise ValueError("feature_research.create_factor_id_mismatch")
                candidate = editor.create(specification)
            case "UPDATE":
                candidate = editor.update(edit.factor_id, specification)
            case "RETIRE":
                candidate = editor.delete(edit.factor_id)
            case "RENAME":
                assert specification is not None
                candidate = editor.rename(edit.factor_id, specification)
                renames.append((edit.factor_id, specification.factor_id))
    planner = FeatureCatalogCrudPlanner(default_extension_kernel_registry())
    revision = planner.revision(candidate, parent_revision_hash=base.revision_hash)
    change = planner.prepare(base=base, candidate=revision, rename_pairs=tuple(renames))
    needed = set(change.plan.base_compute_factor_ids)
    fields = tuple(
        sorted(
            {
                f
                for v in revision.features
                if v.factor_id in needed
                for f in v.specification.required_fields
            }
        )
    )
    return ResearchFeaturePlan.create(
        request=request,
        source_panel_snapshot_hash=source_panel_snapshot_hash,
        base=base,
        candidate=revision,
        candidate_payload=candidate.to_payload(),
        delta=change.delta,
        work=change.plan,
        required_fields=fields,
        preprocessing_recipes=recipes,
    )
