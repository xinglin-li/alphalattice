"""Prepare a local Feature overlay from exact raw values and the sealed input's context."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date

from alphalattice.control.product_host.research_authoring.execution import _sector_by_listing_id
from alphalattice.control.product_host.research_authoring.factor_inputs import (
    factor_input_paths,
    logical_panel_owner,
    read_factor_bundle,
)
from alphalattice.control.product_host.research_authoring.feature_research import (
    ResearchFeatureDefinitions,
)
from alphalattice.control.product_host.storage.inventory import require_storage_capacity
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.feature_engine.catalog.research_values import (
    PREPARATION_CATEGORY,
    ResearchFeatureMaterialization,
    ResearchFeaturePreparation,
    ResearchFormulaValues,
)
from alphalattice.foundation.feature_engine.contracts import (
    FeaturePanelBinding,
    PanelSourceExclusion,
)
from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
    PanelClosureArtifactStore,
)
from alphalattice.foundation.feature_engine.panels.development_input import (
    DevelopmentFeatureOverlayManifest,
    ResolvedDevelopmentFeatureInput,
)
from alphalattice.foundation.feature_engine.panels.development_overlay import (
    DevelopmentFeatureOverlayService,
    load_development_feature_overlay_manifest,
    prepared_overlay_source_identity,
    read_prepared_feature_overlay_admission,
)
from alphalattice.foundation.feature_engine.panels.observation_clock_authority import (
    FeaturePanelObservationClockVerifier,
)
from alphalattice.foundation.feature_engine.panels.reader import FeaturePanelReader
from alphalattice.foundation.feature_engine.producers.factors.core_bundle import numerical_spec_hash
from alphalattice.foundation.feature_engine.producers.factors.specifications import (
    build_research_formula_specification,
)
from alphalattice.foundation.feature_engine.producers.preprocessing.catalog import (
    build_installed_panel_preprocessing_catalog,
)
from alphalattice.foundation.market_data_ops.sources.membership import membership_identity
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.kernel.shared_kernel.identity import canonical_hash


@dataclass(frozen=True)
class FeaturePreprocessingContext:
    """Retain exact base Panel, session/listing, sector, membership and exclusion axes."""

    input_binding_hash: str
    base_panel_snapshot_hash: str
    binding: FeaturePanelBinding
    sessions: tuple[date, ...]
    listing_ids: tuple[str, ...]
    sectors: Mapping[str, str]
    members: dict[date, tuple[str, ...]]
    exclusions: dict[date, tuple[str, ...]]

    def identity(self, raw: ResearchFeatureMaterialization) -> str:
        """Bind exact raw columns to this preprocessing source context.

        Args:
            raw: Explicit retained raw feature materialization.

        Returns:
            Deterministic prepared-overlay source identity.
        """
        return prepared_overlay_source_identity(
            input_binding_hash=self.input_binding_hash,
            base_panel_snapshot_hash=self.base_panel_snapshot_hash,
            raw_column_references=raw.columns,
            sessions=self.sessions,
            sector_by_listing_id=self.sectors,
            members_by_session=self.members,
            source_exclusions_by_session=self.exclusions,
        )


def _context(
    definitions: ResearchFeatureDefinitions,
    raw: ResearchFeatureMaterialization,
    *,
    current_policy: bool,
) -> FeaturePreprocessingContext:
    workspace = definitions.workspace
    plan = definitions.read(raw.definition_plan_hash)
    bundle = read_factor_bundle(workspace, raw.input_binding_hash)
    source, artifacts = factor_input_paths(workspace, bundle.binding_hash)
    resolver = ArtifactResolver(artifacts)
    if bundle.panel_snapshot_hash != plan.source_panel_snapshot_hash:
        raise ValueError("feature_research.preprocessing_source_mismatch")
    if current_policy:
        clock = FeaturePanelObservationClockVerifier(resolver=resolver).verify(
            bundle.panel_snapshot_hash
        )
        if not clock.verified:
            raise ValueError(clock.failure_code or "feature_research.source_clock_unverified")
    panel = resolver.load_feature_panel_manifest(
        resolver.feature_panel_manifest_uri(bundle.panel_snapshot_hash)
    )
    lineage = panel["safe_summary"]["lineage"]
    binding = FeaturePanelBinding.create(
        **{
            key: str(lineage[key])
            for key in (
                "manifest_revision",
                "sector_revision",
                "catalog_hash",
                "spy_revision",
                "policy_hash",
            )
        }
    )
    if binding.panel_binding_hash != panel["panel_binding_hash"]:
        raise ValueError("feature_research.panel_binding_mismatch")
    if raw.columns:
        first = ResearchFormulaValues(definitions.store).read(raw.columns[0][1])
        listings = first.listing_ids
        if first.sessions != bundle.sessions:
            raise ValueError("feature_research.preprocessing_axis_mismatch")
    else:
        listings = FeaturePanelReader(resolver).listing_axis(
            resolver.feature_panel_manifest_uri(bundle.panel_snapshot_hash)
        )
    if canonical_hash(listings) != panel["listing_set_hash"]:
        raise ValueError("feature_research.preprocessing_axis_mismatch")
    market = MarketDataRepository(source)
    universe = market.load_universe_manifest_revision(binding.manifest_revision)
    sectors, _ = _sector_by_listing_id(
        workspace=source,
        market_profile_id=universe.profile.market_profile_id,
        panel_manifest=panel,
        listing_ids=listings,
    )
    schedule = market.membership_schedule(
        universe.profile.market_profile_id,
        sessions=bundle.sessions,
        fallback_listing_ids=tuple(v.listing_id for v in universe.listings),
    )
    members = schedule.members_by_session(bundle.sessions)
    recorded = panel["safe_summary"].get("membership")
    if recorded is not None:
        for day, names in members.items():
            epochs = [
                v
                for v in recorded["epochs"]
                if date.fromisoformat(v["first_session"])
                <= day
                <= date.fromisoformat(v["last_session"])
            ]
            if len(epochs) != 1 or epochs[0]["membership_hash"] != membership_identity(names):
                raise ValueError("feature_research.recorded_membership_mismatch")
        exclusions = tuple(
            PanelSourceExclusion.from_payload(v) for v in recorded.get("source_exclusions", ())
        )
    else:
        if any(tuple(names) != listings for names in members.values()):
            raise ValueError("feature_research.legacy_membership_mismatch")
        exclusions = ()
    excluded = {
        day: tuple(
            sorted(
                {
                    v.listing_id
                    for v in exclusions
                    if v.first_session <= day <= v.last_session and v.listing_id in members[day]
                }
            )
        )
        for day in bundle.sessions
    }
    return FeaturePreprocessingContext(
        bundle.binding_hash,
        bundle.panel_snapshot_hash,
        binding,
        bundle.sessions,
        listings,
        sectors,
        members,
        excluded,
    )


def prepare_feature_overlay(
    *,
    definitions: ResearchFeatureDefinitions,
    raw: ResearchFeatureMaterialization,
    implementation_hash: str,
    cancelled: Callable[[], bool],
) -> ResearchFeaturePreparation:
    """Materialize and publish an exact research overlay under admitted capacity and cancellation.

    Args:
        definitions: Exact workspace definition owner with a write capacity bound.
        raw: Verified raw feature materialization.
        implementation_hash: Exact installed preparation implementation.
        cancelled: Explicit cancellation callback.

    Returns:
        Sealed preparation receipt with optional overlay and actual preprocessing call count.

    Raises:
        ValueError: Capacity is unbound or exact source/preparation is inadmissible.
    """
    workspace = definitions.workspace
    context = _context(definitions, raw, current_policy=True)
    plan = definitions.read(raw.definition_plan_hash)
    recipes = {v.factor_id: v.specification for v in plan.candidate.features}

    def capacity(size: int) -> None:
        require_storage_capacity(workspace, additional_bytes=size)

    overlay_hash, calls = None, 0
    if raw.columns:
        overlay, calls = DevelopmentFeatureOverlayService().materialize_prepared(
            output_root=workspace / "artifacts",
            input_binding_hash=context.input_binding_hash,
            base_panel_snapshot_hash=context.base_panel_snapshot_hash,
            panel_binding=context.binding,
            definitions=tuple(recipes[name] for name, _ in raw.columns),
            raw_column_references=raw.columns,
            sessions=context.sessions,
            listing_ids=context.listing_ids,
            sector_by_listing_id=context.sectors,
            members_by_session=context.members,
            source_exclusions_by_session=context.exclusions,
            capacity=capacity,
            cancelled=cancelled,
            preprocessing_recipes=plan.preprocessing_recipes,
        )
        overlay_hash = overlay.overlay_hash
    receipt = ResearchFeaturePreparation.create(
        definition_plan_hash=plan.plan_hash,
        input_binding_hash=context.input_binding_hash,
        raw_materialization_hash=raw.content_hash,
        overlay_hash=overlay_hash,
        implementation_hash=implementation_hash,
        preprocessing_calls_in_this_attempt=calls,
    )
    PanelClosureArtifactStore(
        ArtifactResolver(workspace / "artifacts"), capacity=capacity
    ).publish_json(
        category=PREPARATION_CATEGORY,
        content_hash=receipt.content_hash,
        payload=receipt.model_dump(mode="json"),
    )
    return receipt


def verify_feature_preparation(
    *,
    definitions: ResearchFeatureDefinitions,
    raw: ResearchFeatureMaterialization,
    receipt: ResearchFeaturePreparation,
    implementation_hash: str,
    current_policy: bool = False,
) -> tuple[FeaturePreprocessingContext, DevelopmentFeatureOverlayManifest | None]:
    """Verify exact preparation receipt, source, overlay and required policy.

    Verify receipt, source axes and exact admitted overlay definitions and optional current policy.

    Args:
        definitions: Exact workspace feature definition owner.
        raw: Exact retained raw materialization.
        receipt: Exact retained preparation receipt.
        implementation_hash: Required installed preparation implementation.
        current_policy: Whether to require current installed formula/preprocessing specifications.

    Returns:
        Verified preprocessing context and optional exact overlay.

    Raises:
        ValueError: Receipt/source/axis/definition or required installed preprocessing policy
            differs.
    """
    if (
        receipt.definition_plan_hash != raw.definition_plan_hash
        or receipt.input_binding_hash != raw.input_binding_hash
        or receipt.raw_materialization_hash != raw.content_hash
        or receipt.implementation_hash != implementation_hash
    ):
        raise ValueError("feature_research.preprocessing_receipt_mismatch")
    context = _context(definitions, raw, current_policy=current_policy)
    if not raw.columns:
        if receipt.overlay_hash is not None or receipt.preprocessing_calls_in_this_attempt != 0:
            raise ValueError("feature_research.unexpected_preprocessing")
        return context, None
    if receipt.overlay_hash is None:
        raise ValueError("feature_research.preprocessing_missing")
    overlay = load_development_feature_overlay_manifest(
        output_root=definitions.workspace / "artifacts", overlay_hash=receipt.overlay_hash
    )
    if (
        overlay.base_panel_snapshot_hash != context.base_panel_snapshot_hash
        or overlay.base_panel_binding_hash != context.binding.panel_binding_hash
        or overlay.source_identity != context.identity(raw)
        or overlay.raw_column_references != raw.columns
        or overlay.ordered_listing_axis != context.listing_ids
        or overlay.ordered_session_axis != tuple(str(d) for d in context.sessions)
    ):
        raise ValueError("feature_research.preprocessing_source_mismatch")
    admission = read_prepared_feature_overlay_admission(
        definitions.workspace / "artifacts", overlay
    )
    plan = definitions.read(raw.definition_plan_hash)
    requested = {v.factor_id: v.specification for v in plan.candidate.features}
    if admission.numerical_definitions != tuple(
        (name, numerical_spec_hash(requested[name])) for name, _ in raw.columns
    ):
        raise ValueError("feature_research.preprocessing_definition_mismatch")
    if current_policy:
        expected = tuple(
            build_research_formula_specification(
                requested[name],
                source_session_count=len(context.sessions),
                preprocessing_recipe=plan.preprocessing_recipes.get(name),
            )
            for name, _ in raw.columns
        )
        catalog = build_installed_panel_preprocessing_catalog()
        if expected != admission.specifications or not catalog.catalog_current(
            overlay.preprocessing_catalog_hash
        ):
            raise ValueError("feature_research.preprocessing_policy_mismatch")
        for column in overlay.columns:
            spec = next(v for v in expected if v.factor_id == column.factor_id)
            recipe, _ = catalog.resolve_executable(spec.preprocessing_role)
            if (
                column.preprocessing_recipe_id != spec.preprocessing_role
                or column.preprocessing_recipe_hash != recipe.recipe_hash
                or not catalog.implementation_current(
                    spec.preprocessing_role, column.preprocessing_implementation_hash
                )
            ):
                raise ValueError("feature_research.preprocessing_policy_mismatch")
    return context, overlay


def resolve_prepared_feature_input(
    *,
    definitions: ResearchFeatureDefinitions,
    raw: ResearchFeatureMaterialization,
    receipt: ResearchFeaturePreparation,
    implementation_hash: str,
    current_policy: bool,
) -> ResolvedDevelopmentFeatureInput:
    """Resolve one source for all research consumers, with no new files or registration."""
    context, overlay = verify_feature_preparation(
        definitions=definitions,
        raw=raw,
        receipt=receipt,
        implementation_hash=implementation_hash,
        current_policy=current_policy,
    )
    source, artifacts = factor_input_paths(definitions.workspace, raw.input_binding_hash)
    marker = logical_panel_owner(source).verify_snapshot(context.base_panel_snapshot_hash)
    resolver = ArtifactResolver(artifacts)
    panel = resolver.load_feature_panel_manifest(
        resolver.feature_panel_manifest_uri(context.base_panel_snapshot_hash)
    )
    return ResolvedDevelopmentFeatureInput.compose(
        source_handle=f"research-features@{receipt.content_hash}",
        input_binding_hash=raw.input_binding_hash,
        base_manifest=dict(panel),
        base_artifact_root=artifacts,
        base_logical_panel_hash=marker.logical_panel_hash,
        base_logical_semantic_index_hash=marker.logical_semantic_index_hash,
        inherited_factor_ids=raw.inherited_input_factor_ids,
        overlay=overlay,
        overlay_root=definitions.workspace / "artifacts",
        definition_plan_hash=raw.definition_plan_hash,
    )
