"""Typed contracts for governed Feature Catalog CRUD transitions."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.quant.factor_contracts import FactorSpec
from alphalattice.kernel.shared_kernel.identity import canonical_hash


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class FeatureRevisionStatus(StrEnum):
    """Name the active or retired state of a Feature revision."""

    ACTIVE = "ACTIVE"
    RETIRED = "RETIRED"


class FeatureCatalogOperation(StrEnum):
    """Name a governed catalog edit operation."""

    CREATE = "CREATE"
    METADATA_UPDATE = "METADATA_UPDATE"
    RESEARCH_CLASSIFICATION_UPDATE = "RESEARCH_CLASSIFICATION_UPDATE"
    NUMERICAL_UPDATE = "NUMERICAL_UPDATE"
    DELETE = "DELETE"
    RENAME = "RENAME"


class FeatureRevision(_Contract):
    """Bind a Formula specification and implementation to one revision hash."""

    factor_id: str
    specification: FactorSpec
    implementation_id: str
    implementation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    maintenance_cost: str
    maximum_invalidation_sessions: int = Field(ge=1)
    status: FeatureRevisionStatus = FeatureRevisionStatus.ACTIVE
    revision_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_revision(self) -> FeatureRevision:
        """Verify the Formula identity and revision hash."""
        if self.factor_id != self.specification.factor_id:
            raise ValueError("Feature revision factor ID does not match its specification")
        if self.implementation_id != self.specification.formula_ref:
            raise ValueError("Feature revision implementation must match formula_ref")
        if self.revision_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"revision_hash"})
        ):
            raise ValueError("Feature revision hash is invalid")
        return self

    @classmethod
    def create(
        cls,
        *,
        specification: FactorSpec,
        implementation_hash: str,
        maintenance_cost: str,
        maximum_invalidation_sessions: int,
        status: FeatureRevisionStatus = FeatureRevisionStatus.ACTIVE,
    ) -> FeatureRevision:
        """Create a content-addressed Feature revision."""
        values = {
            "factor_id": specification.factor_id,
            "specification": specification,
            "implementation_id": specification.formula_ref,
            "implementation_hash": implementation_hash,
            "maintenance_cost": maintenance_cost,
            "maximum_invalidation_sessions": maximum_invalidation_sessions,
            "status": status,
        }
        return cls(**values, revision_hash=canonical_hash(_jsonable(values)))


class FeatureCatalogRevision(_Contract):
    """Represent one immutable active catalog revision."""

    kind: Literal["FeatureCatalogRevision"] = "FeatureCatalogRevision"
    stable_id: str
    parent_revision_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    core_bundle_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    features: tuple[FeatureRevision, ...]
    maintenance_policy: dict[str, object]
    maintenance_policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    revision_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_catalog_revision(self) -> FeatureCatalogRevision:
        """Verify the active axis and revision hash."""
        factor_ids = tuple(item.factor_id for item in self.features)
        if not factor_ids or factor_ids != tuple(sorted(set(factor_ids))):
            raise ValueError("Feature Catalog revision axis must be sorted and unique")
        if any(item.status != FeatureRevisionStatus.ACTIVE for item in self.features):
            raise ValueError("active Feature Catalog revision cannot contain retired entries")
        if self.revision_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"revision_hash"})
        ):
            raise ValueError("Feature Catalog revision hash is invalid")
        return self

    @property
    def factor_ids(self) -> tuple[str, ...]:
        """Return active Formula IDs in revision order."""
        return tuple(item.factor_id for item in self.features)

    @classmethod
    def create(
        cls,
        *,
        stable_id: str,
        parent_revision_hash: str | None,
        core_bundle_hash: str,
        features: tuple[FeatureRevision, ...],
        maintenance_policy: dict[str, object],
        maintenance_policy_hash: str,
    ) -> FeatureCatalogRevision:
        """Create a catalog revision with a sorted Feature axis."""
        ordered = tuple(sorted(features, key=lambda item: item.factor_id))
        values = {
            "kind": "FeatureCatalogRevision",
            "stable_id": stable_id,
            "parent_revision_hash": parent_revision_hash,
            "core_bundle_hash": core_bundle_hash,
            "features": ordered,
            "maintenance_policy": maintenance_policy,
            "maintenance_policy_hash": maintenance_policy_hash,
        }
        return cls(**values, revision_hash=canonical_hash(_jsonable(values)))


class FeatureCatalogDelta(_Contract):
    """Record the typed operations and affected Formula IDs of an edit."""

    kind: Literal["FeatureCatalogDelta"] = "FeatureCatalogDelta"
    base_revision_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_revision_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    operations: tuple[FeatureCatalogOperation, ...]
    created_factor_ids: tuple[str, ...] = ()
    retired_factor_ids: tuple[str, ...] = ()
    metadata_updated_factor_ids: tuple[str, ...] = ()
    research_classification_updated_factor_ids: tuple[str, ...] = ()
    numerical_updated_factor_ids: tuple[str, ...] = ()
    rename_pairs: tuple[tuple[str, str], ...] = ()
    delta_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_delta(self) -> FeatureCatalogDelta:
        """Verify sorted edit scopes and the delta hash."""
        scopes = (
            self.created_factor_ids,
            self.retired_factor_ids,
            self.metadata_updated_factor_ids,
            self.research_classification_updated_factor_ids,
            self.numerical_updated_factor_ids,
        )
        if any(scope != tuple(sorted(set(scope))) for scope in scopes):
            raise ValueError("Feature Catalog delta scopes must be sorted and unique")
        if self.delta_hash != canonical_hash(self.model_dump(mode="json", exclude={"delta_hash"})):
            raise ValueError("Feature Catalog delta hash is invalid")
        return self


class FactorReconciliationScope(_Contract):
    """Identify the diagnostics affected by a catalog transition."""

    diagnostic_factor_ids: tuple[str, ...]
    family_fdr_ids: tuple[str, ...]
    redundancy_factor_ids: tuple[str, ...]
    decision_surface_changed: bool
    scope_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_scope(self) -> FactorReconciliationScope:
        """Verify the reconciliation scope hash."""
        if self.scope_hash != canonical_hash(self.model_dump(mode="json", exclude={"scope_hash"})):
            raise ValueError("Factor reconciliation scope hash is invalid")
        return self


class FeatureCatalogUpdatePlan(_Contract):
    """Describe the numerical and Panel work required by an edit."""

    kind: Literal["FeatureCatalogUpdatePlan"] = "FeatureCatalogUpdatePlan"
    base_revision_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_revision_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    delta_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_compute_factor_ids: tuple[str, ...]
    panel_compute_factor_ids: tuple[str, ...]
    panel_drop_factor_ids: tuple[str, ...]
    numerical_work_required: bool
    factor_reconciliation: FactorReconciliationScope
    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_plan(self) -> FeatureCatalogUpdatePlan:
        """Verify numerical work disposition and plan hash."""
        if self.numerical_work_required != bool(self.base_compute_factor_ids):
            raise ValueError("Feature Catalog numerical-work disposition is invalid")
        if self.plan_hash != canonical_hash(self.model_dump(mode="json", exclude={"plan_hash"})):
            raise ValueError("Feature Catalog update plan hash is invalid")
        return self


def _jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, StrEnum):
        return value.value
    return value


__all__ = [
    "FactorReconciliationScope",
    "FeatureCatalogDelta",
    "FeatureCatalogOperation",
    "FeatureCatalogRevision",
    "FeatureCatalogUpdatePlan",
    "FeatureRevision",
    "FeatureRevisionStatus",
]
