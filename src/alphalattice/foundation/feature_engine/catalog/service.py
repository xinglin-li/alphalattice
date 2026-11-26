"""Pure Feature Catalog editing and impact planning for the research Task owners."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import cast

from pydantic import BaseModel, TypeAdapter

from alphalattice.foundation.feature_engine.catalog.contracts import (
    FeatureCatalog,
    desktop_core_feature_bundle,
)
from alphalattice.foundation.feature_engine.catalog.crud_contracts import (
    FactorReconciliationScope,
    FeatureCatalogDelta,
    FeatureCatalogOperation,
    FeatureCatalogRevision,
    FeatureCatalogUpdatePlan,
    FeatureRevision,
)
from alphalattice.foundation.feature_engine.producers.factors.core_bundle import (
    NUMERICAL_SPEC_FIELDS,
)
from alphalattice.foundation.feature_engine.producers.factors.registry import FeatureKernelRegistry
from alphalattice.kernel.quant.factor_contracts import FactorSpec
from alphalattice.kernel.shared_kernel.identity import canonical_hash

_NUMERICAL_FIELDS = frozenset(NUMERICAL_SPEC_FIELDS)
_RESEARCH_CLASSIFICATION_FIELDS = frozenset({"core_anchor", "family", "track"})


class FeatureCatalogCrudError(RuntimeError):
    """Report a typed Feature Catalog edit refusal."""

    def __init__(self, failure_code: str, detail: str) -> None:
        """Store the refusal code and human-readable detail."""
        super().__init__(detail)
        self.failure_code = failure_code


@dataclass(frozen=True)
class FeatureCatalogPreparedChange:
    """Hold candidate revisions, delta, and update plan before activation."""

    base: FeatureCatalogRevision
    candidate: FeatureCatalogRevision
    delta: FeatureCatalogDelta
    plan: FeatureCatalogUpdatePlan


class FeatureCatalogCrudPlanner:
    """Plan a catalog edit and its reconciliation work without executing it."""

    def __init__(self, registry: FeatureKernelRegistry) -> None:
        """Use the installed Formula implementation registry for planning."""
        self.registry = registry
        self.core_bundle = desktop_core_feature_bundle()

    def revision(
        self,
        catalog: FeatureCatalog,
        *,
        parent_revision_hash: str | None = None,
    ) -> FeatureCatalogRevision:
        """Build a content-addressed revision from a Feature Catalog."""
        features = tuple(
            FeatureRevision.create(
                specification=specification,
                implementation_hash=self.registry.implementation_hash(
                    specification, core_bundle=self.core_bundle
                ),
                maintenance_cost=catalog.contracts_by_factor[
                    specification.factor_id
                ].maintenance_cost,
                maximum_invalidation_sessions=catalog.contracts_by_factor[
                    specification.factor_id
                ].maximum_invalidation_sessions,
            )
            for specification in catalog.factors
        )
        return FeatureCatalogRevision.create(
            stable_id=catalog.binding.stable_id,
            parent_revision_hash=parent_revision_hash,
            core_bundle_hash=self.core_bundle.bundle_hash,
            features=features,
            maintenance_policy=cast(dict[str, object], catalog.maintenance_policy),
            maintenance_policy_hash=catalog.binding.materializer_policy_hash,
        )

    def prepare(
        self,
        *,
        base: FeatureCatalogRevision,
        candidate: FeatureCatalogRevision,
        rename_pairs: tuple[tuple[str, str], ...] = (),
    ) -> FeatureCatalogPreparedChange:
        """Compare revisions and plan the affected numerical work."""
        if candidate.parent_revision_hash != base.revision_hash:
            raise FeatureCatalogCrudError(
                "feature_catalog.stale_base", "Candidate catalog does not bind the active revision."
            )
        old = {item.factor_id: item for item in base.features}
        new = {item.factor_id: item for item in candidate.features}
        created = tuple(sorted(set(new) - set(old)))
        retired = tuple(sorted(set(old) - set(new)))
        normalized_renames = tuple(sorted(set(rename_pairs)))
        if any(
            source not in retired or target not in created for source, target in normalized_renames
        ):
            raise FeatureCatalogCrudError(
                "feature_catalog.rename_invalid", "Rename must be an explicit Create plus Retire."
            )
        metadata: list[str] = []
        classification: list[str] = []
        numerical: list[str] = []
        for factor_id in sorted(set(old) & set(new)):
            before = old[factor_id].specification.model_dump(mode="json")
            after = new[factor_id].specification.model_dump(mode="json")
            changed = {key for key in before if before[key] != after[key]}
            if (
                old[factor_id].implementation_hash != new[factor_id].implementation_hash
                or changed & _NUMERICAL_FIELDS
            ):
                numerical.append(factor_id)
            elif changed & _RESEARCH_CLASSIFICATION_FIELDS:
                classification.append(factor_id)
            elif changed:
                metadata.append(factor_id)
        operations = set()
        if created:
            operations.add(FeatureCatalogOperation.CREATE)
        if retired:
            operations.add(FeatureCatalogOperation.DELETE)
        if metadata:
            operations.add(FeatureCatalogOperation.METADATA_UPDATE)
        if classification:
            operations.add(FeatureCatalogOperation.RESEARCH_CLASSIFICATION_UPDATE)
        if numerical:
            operations.add(FeatureCatalogOperation.NUMERICAL_UPDATE)
        if normalized_renames:
            operations.add(FeatureCatalogOperation.RENAME)
        if not operations:
            raise FeatureCatalogCrudError(
                "feature_catalog.empty_delta", "Candidate catalog does not change active semantics."
            )
        delta_values = {
            "kind": "FeatureCatalogDelta",
            "base_revision_hash": base.revision_hash,
            "candidate_revision_hash": candidate.revision_hash,
            "operations": tuple(sorted(operations, key=lambda item: item.value)),
            "created_factor_ids": created,
            "retired_factor_ids": retired,
            "metadata_updated_factor_ids": tuple(metadata),
            "research_classification_updated_factor_ids": tuple(classification),
            "numerical_updated_factor_ids": tuple(numerical),
            "rename_pairs": normalized_renames,
        }
        delta = FeatureCatalogDelta(
            **delta_values, delta_hash=canonical_hash(_jsonable(delta_values))
        )
        computed = tuple(sorted({*created, *numerical}))
        affected_factor_ids = tuple(sorted({*computed, *retired, *classification}))
        families = set()
        for factor_id in affected_factor_ids:
            if factor_id in old:
                families.add(str(old[factor_id].specification.family.value))
            if factor_id in new:
                families.add(str(new[factor_id].specification.family.value))
        scope_values = {
            "diagnostic_factor_ids": computed,
            "family_fdr_ids": tuple(sorted(families)),
            "redundancy_factor_ids": tuple(sorted({*computed, *retired})),
            "decision_surface_changed": bool(affected_factor_ids),
        }
        scope = FactorReconciliationScope(**scope_values, scope_hash=canonical_hash(scope_values))
        plan_values = {
            "kind": "FeatureCatalogUpdatePlan",
            "base_revision_hash": base.revision_hash,
            "candidate_revision_hash": candidate.revision_hash,
            "delta_hash": delta.delta_hash,
            "base_compute_factor_ids": computed,
            "panel_compute_factor_ids": computed,
            "panel_drop_factor_ids": retired,
            "numerical_work_required": bool(computed),
            "factor_reconciliation": scope,
        }
        plan = FeatureCatalogUpdatePlan(
            **plan_values, plan_hash=canonical_hash(_jsonable(plan_values))
        )
        return FeatureCatalogPreparedChange(base=base, candidate=candidate, delta=delta, plan=plan)


class FeatureCatalogEditor:
    """Pure catalog edits; the existing research Task owners execute approved plans."""

    def __init__(self, catalog: FeatureCatalog) -> None:
        """Copy a catalog payload for pure edits."""
        self._payload = catalog.to_payload()

    def create(self, specification: object) -> FeatureCatalog:
        """Add a new Formula to a candidate catalog."""
        spec = FactorSpec.model_validate(specification)
        factors = self._factor_payloads()
        if spec.factor_id in {str(item["factor_id"]) for item in factors}:
            raise FeatureCatalogCrudError(
                "feature_catalog.factor_already_exists",
                f"Feature already exists: {spec.factor_id}",
            )
        factors.append(spec.model_dump(mode="json"))
        return self._catalog(factors)

    def update(self, factor_id: str, specification: object) -> FeatureCatalog:
        """Replace an active Formula without changing its ID."""
        spec = FactorSpec.model_validate(specification)
        if spec.factor_id != factor_id:
            raise FeatureCatalogCrudError(
                "feature_catalog.update_changes_factor_id",
                "Use rename when the Feature ID changes.",
            )
        factors = self._factor_payloads()
        positions = [index for index, item in enumerate(factors) if item["factor_id"] == factor_id]
        if len(positions) != 1:
            raise FeatureCatalogCrudError(
                "feature_catalog.factor_not_found", f"Feature is not active: {factor_id}"
            )
        factors[positions[0]] = spec.model_dump(mode="json")
        return self._catalog(factors)

    def delete(self, factor_id: str) -> FeatureCatalog:
        """Remove an active Formula from a candidate catalog."""
        factors = self._factor_payloads()
        retained = [item for item in factors if item["factor_id"] != factor_id]
        if len(retained) == len(factors):
            raise FeatureCatalogCrudError(
                "feature_catalog.factor_not_found", f"Feature is not active: {factor_id}"
            )
        self._drop_policy_reference(factor_id)
        return self._catalog(retained)

    def rename(self, factor_id: str, specification: object) -> FeatureCatalog:
        """Replace an active Formula under a distinct ID."""
        spec = FactorSpec.model_validate(specification)
        if spec.factor_id == factor_id:
            raise FeatureCatalogCrudError(
                "feature_catalog.rename_preserves_factor_id",
                "Rename requires a distinct target Feature ID.",
            )
        factors = self._factor_payloads()
        if factor_id not in {str(item["factor_id"]) for item in factors}:
            raise FeatureCatalogCrudError(
                "feature_catalog.factor_not_found", f"Feature is not active: {factor_id}"
            )
        if spec.factor_id in {str(item["factor_id"]) for item in factors}:
            raise FeatureCatalogCrudError(
                "feature_catalog.factor_already_exists",
                f"Feature already exists: {spec.factor_id}",
            )
        self._rename_policy_reference(factor_id, spec.factor_id)
        retained = [item for item in factors if item["factor_id"] != factor_id]
        retained.append(spec.model_dump(mode="json"))
        return self._catalog(retained)

    def _factor_payloads(self) -> list[dict[str, object]]:
        return [dict(item) for item in cast(list[dict[str, object]], self._payload["factors"])]

    def _catalog(self, factors: list[dict[str, object]]) -> FeatureCatalog:
        payload = dict(self._payload)
        payload["factors"] = sorted(factors, key=lambda item: str(item["factor_id"]))
        return FeatureCatalog.from_payload(payload)

    def _drop_policy_reference(self, factor_id: str) -> None:
        policy = cast(dict[str, object], self._payload["maintenance_policy"])
        for key in (
            "market_dependent_factor_ids",
            "finite_calendar_factor_ids",
            "window_local_cumulative_factor_ids",
        ):
            policy[key] = [value for value in cast(list[str], policy[key]) if value != factor_id]
        overrides = cast(dict[str, int], policy["maximum_invalidation_overrides"])
        overrides.pop(factor_id, None)

    def _rename_policy_reference(self, source: str, target: str) -> None:
        policy = cast(dict[str, object], self._payload["maintenance_policy"])
        for key in (
            "market_dependent_factor_ids",
            "finite_calendar_factor_ids",
            "window_local_cumulative_factor_ids",
        ):
            values = cast(list[str], policy[key])
            policy[key] = sorted(target if value == source else value for value in values)
        overrides = cast(dict[str, int], policy["maximum_invalidation_overrides"])
        if source in overrides:
            overrides[target] = overrides.pop(source)


def _jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, FeatureCatalogOperation):
        return value.value
    if isinstance(value, datetime):
        return TypeAdapter(datetime).dump_python(value, mode="json")
    return value


__all__ = [
    "FeatureCatalogCrudError",
    "FeatureCatalogCrudPlanner",
    "FeatureCatalogEditor",
    "FeatureCatalogPreparedChange",
]
