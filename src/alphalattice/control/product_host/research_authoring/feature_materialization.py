"""Build only local Formula columns over an exact, immutable research input."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from alphalattice.control.product_host.research_authoring.factor_inputs import (
    factor_input_paths,
    read_factor_bundle,
)
from alphalattice.control.product_host.research_authoring.feature_research import (
    ResearchFeatureDefinitions,
)
from alphalattice.control.product_host.storage.inventory import require_storage_capacity
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.control.workspace_runtime.database import retain_workspace_database
from alphalattice.foundation.feature_engine.catalog.contracts import desktop_core_feature_bundle
from alphalattice.foundation.feature_engine.catalog.crud_contracts import FeatureRevision
from alphalattice.foundation.feature_engine.catalog.research import ResearchFeaturePlan
from alphalattice.foundation.feature_engine.catalog.research_values import (
    RECEIPT_CATEGORY,
    ResearchFeatureMaterialization,
    ResearchFormulaValues,
    column_binding_hash,
)
from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
    PanelClosureArtifactStore,
)
from alphalattice.foundation.feature_engine.panels.materialization_identity import (
    align_feature_source_sessions,
)
from alphalattice.foundation.feature_engine.panels.reader import FeaturePanelReader
from alphalattice.foundation.feature_engine.producers.base_materializer import _REQUIRED_COLUMNS
from alphalattice.foundation.feature_engine.producers.factors.catalog import (
    default_extension_kernel_registry,
)
from alphalattice.foundation.feature_engine.producers.factors.core_bundle import numerical_spec_hash
from alphalattice.foundation.feature_engine.producers.factors.formula import (
    FORMULA_POINT_IN_TIME_FIELDS,
    FORMULA_SECTOR_FIELD,
)
from alphalattice.foundation.feature_engine.producers.sector_aggregates import sector_return_log
from alphalattice.foundation.feature_engine.storage.repositories import FeatureStateRepository
from alphalattice.foundation.market_data_ops.sources.manifest import UniverseManifest
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.kernel.quant.sector_history import SectorHistory
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.identity_successors import is_current, predecessors
from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root
from alphalattice.kernel.shared_kernel.source_identity import source_rule_closure_hash

MAXIMUM_NEW_COLUMN_BUFFER_BYTES = 512 * 1024 * 1024


SOURCE_PROJECTION_ROLE = "product_host.research_feature_source_projection"
"""The identity role of the raw-formula source projection (`config/identity-roles.json`): a
column or a Task recorded under a value recorded moves lead from is current."""


def feature_source_projection_hash() -> str:
    """The code that projects a factor's raw formula sources, by the rule.

    The numerics it runs on are the environment, provenance and never this identity
    (LAWS.md ID6); the Host code that locates the sealed input bundle decides no value, the
    bundle being bound by its content hash (UC, LAWS.md ID8).
    """
    return source_rule_closure_hash(
        root=resolve_playpen_root(Path(__file__)),
        semantic_owner="feature_engine",
        numerical_role="RESEARCH_LOCAL_RAW_FORMULA_SOURCE",
        tracked_paths=tuple(
            "src/alphalattice/" + p
            for p in (
                "control/product_host/research_authoring/feature_materialization.py",
                "foundation/feature_engine/storage/repositories.py",
                "foundation/feature_engine/panels/materialization_identity.py",
                "foundation/market_data_ops/publication/projection.py",
                "foundation/market_data_ops/sources/contracts.py",
                "foundation/market_data_ops/storage/duckdb.py",
            )
        ),
    )


def feature_value_sources(
    definitions: ResearchFeatureDefinitions, plan: ResearchFeaturePlan
) -> tuple[tuple[str, ...], tuple[FeatureRevision, ...]]:
    """Inherited input columns versus local values owed, independent of a delta's label."""
    source_plan, seen = plan, set()
    while source_plan.request.parent_plan_hash is not None:
        if source_plan.plan_hash in seen:
            raise ValueError("feature_research.definition_cycle")
        seen.add(source_plan.plan_hash)
        source_plan = definitions.read(source_plan.request.parent_plan_hash)
        if source_plan.request.input_binding_hash != plan.request.input_binding_hash:
            raise ValueError("feature_research.parent_input_mismatch")
    base = {v.factor_id: v for v in source_plan.base.features}
    inherited: list[str] = []
    owned: list[FeatureRevision] = []
    for entry in plan.candidate.features:
        previous = base.get(entry.factor_id)
        if (
            previous is not None
            and previous.implementation_hash == entry.implementation_hash
            and numerical_spec_hash(previous.specification)
            == numerical_spec_hash(entry.specification)
        ):
            inherited.append(entry.factor_id)
        else:
            owned.append(entry)
    return tuple(sorted(inherited)), tuple(owned)


def _sector_returns(
    *,
    market: MarketDataRepository,
    feature: FeatureStateRepository,
    universe: UniverseManifest,
    lineage: Mapping[str, Any],
    listings: tuple[str, ...],
    sessions: tuple[date, ...],
    closes: Sequence[pd.Series],
) -> npt.NDArray[np.float64]:
    """Each listing's Sector return at each session, as the Panel's sessions read the Sector.

    The history is the Panel's own (its lineage's reclassifications over the store's current
    classification of its revision), each session's members the Universe's for that session,
    and the closes the listings' adjusted closes on the session axis (V359).

    Raises:
        ValueError: `feature_research.sector_leaf_revision_moved` when the store's Sector
            revision is no longer the Panel's.
    """
    store_history = feature.sector_history(universe, listing_ids=listings)
    if store_history is None or store_history.current_revision != str(lineage["sector_revision"]):
        raise ValueError("feature_research.sector_leaf_revision_moved")
    history = SectorHistory.of_panel(lineage, dict(store_history.subset(listings)))
    schedule = market.membership_schedule(
        universe.profile.market_profile_id,
        sessions=sessions,
        fallback_listing_ids=tuple(value.listing_id for value in universe.listings),
    )
    members_by_session = schedule.members_by_session(sessions)
    position = {listing: index for index, listing in enumerate(listings)}
    members: npt.NDArray[np.bool_] = np.zeros((len(sessions), len(listings)), dtype=np.bool_)
    for row, session in enumerate(sessions):
        for listing in members_by_session[session]:
            if listing in position:
                members[row, position[listing]] = True
    values: npt.NDArray[np.float64] = sector_return_log(
        np.column_stack(
            [pd.to_numeric(close, errors="coerce").to_numpy(dtype=np.float64) for close in closes]
        ),
        sessions=sessions,
        listing_ids=listings,
        sectors=history,
        members=members,
    )
    return values


def materialize_feature_columns(
    *,
    definitions: ResearchFeatureDefinitions,
    plan: ResearchFeaturePlan,
    implementation_hash: str,
    source_projection_hash: str,
    cancelled: Callable[[], bool] = lambda: False,
) -> ResearchFeatureMaterialization:
    """No Panel activation, fit or source writes; unchanged input columns stay referenced."""
    workspace = definitions.workspace
    if cancelled():
        raise ValueError("feature_research.cancelled_at_safe_checkpoint")
    store = PanelClosureArtifactStore(
        ArtifactResolver(workspace / "artifacts"),
        capacity=lambda n: require_storage_capacity(workspace, additional_bytes=n),
    )
    values_owner = ResearchFormulaValues(store)
    inherited, owned = feature_value_sources(definitions, plan)
    bundle = read_factor_bundle(workspace, plan.request.input_binding_hash, verify=False)
    if bundle.panel_snapshot_hash != plan.source_panel_snapshot_hash:
        raise ValueError("feature_research.input_snapshot_mismatch")
    source, artifacts = factor_input_paths(workspace, bundle.binding_hash)
    resolver = ArtifactResolver(artifacts)
    manifest = resolver.load_feature_panel_manifest(
        resolver.feature_panel_manifest_uri(bundle.panel_snapshot_hash)
    )
    registry = default_extension_kernel_registry()
    core = desktop_core_feature_bundle()
    columns, reused, pending, bindings = [], [], [], {}
    pending_keys: dict[str, str] = {}
    aliases: list[tuple[str, str]] = []
    for entry in owned:
        spec = entry.specification
        if registry.implementation_hash(spec, core_bundle=core) != entry.implementation_hash:
            raise ValueError("feature_research.implementation_changed_replan")
        missing = set(spec.required_fields) - {
            *_REQUIRED_COLUMNS,
            FORMULA_SECTOR_FIELD,
            *FORMULA_POINT_IN_TIME_FIELDS,
        }
        if missing:
            raise ValueError(
                "feature_research.source_dependency_not_prepared:" + ",".join(sorted(missing))
            )
        binding = dict(
            input_binding_hash=bundle.binding_hash,
            numerical_spec_hash=numerical_spec_hash(spec),
            implementation_hash=entry.implementation_hash,
            source_projection_hash=source_projection_hash,
        )
        key = column_binding_hash(**binding)
        bindings[entry.factor_id] = {**binding, "binding_hash": key}
        # A column sealed under a projection recorded moves lead from is this one's.
        found = None
        for projection in predecessors(SOURCE_PROJECTION_ROLE, source_projection_hash):
            found = values_owner.lookup(
                column_binding_hash(**{**binding, "source_projection_hash": projection})
            )
            if found is not None:
                break
        if found is not None:
            if (
                found.sessions != bundle.sessions
                or canonical_hash(found.listing_ids) != manifest["listing_set_hash"]
            ):
                raise ValueError("feature_research.cached_axis_mismatch")
            columns.append((entry.factor_id, found.content_hash))
            reused.append(entry.factor_id)
        else:
            if key in pending_keys:
                aliases.append((entry.factor_id, pending_keys[key]))
            else:
                pending_keys[key] = entry.factor_id
                pending.append(entry)
    calls = 0
    if pending:
        if not is_current(
            SOURCE_PROJECTION_ROLE, source_projection_hash, feature_source_projection_hash()
        ):
            raise ValueError("feature_research.source_projection_changed")
        read_factor_bundle(
            workspace, bundle.binding_hash
        )  # Full bytes only when actually building.
        listings = FeaturePanelReader(resolver).listing_axis(
            resolver.feature_panel_manifest_uri(bundle.panel_snapshot_hash)
        )
        shape = (len(bundle.sessions), len(listings))
        if len(pending) * shape[0] * shape[1] * 8 > MAXIMUM_NEW_COLUMN_BUFFER_BYTES:
            raise ValueError("feature_research.new_column_buffer_budget_exceeded")
        arrays: dict[str, npt.NDArray[np.float64]] = {
            entry.factor_id: np.full(shape, np.nan, dtype=np.float64) for entry in pending
        }
        market = MarketDataRepository(source)
        feature = FeatureStateRepository(source, market_data=market)
        universe = market.load_universe_manifest_revision(
            manifest["safe_summary"]["lineage"]["manifest_revision"]
        )
        sector_leaf = any(
            FORMULA_SECTOR_FIELD in entry.specification.required_fields for entry in pending
        )
        # The as-traded fields only when a formula reads a point-in-time leaf (V345).
        as_traded = any(
            set(FORMULA_POINT_IN_TIME_FIELDS) & set(entry.specification.required_fields)
            for entry in pending
        )
        with retain_workspace_database(source / "market-data.duckdb", read_only=True):
            sources: list[tuple[pd.DataFrame, set[date]]] = []
            for listing in listings:
                if cancelled():
                    raise ValueError("feature_research.cancelled_at_safe_checkpoint")
                raw, _raw_hash, _actions_hash = feature.projected_feature_frame(
                    universe,
                    listing_id=listing,
                    start=bundle.sessions[0],
                    through=bundle.sessions[-1],
                    allow_missing_adjusted=True,
                    as_traded=as_traded,
                )
                sources.append(
                    (
                        align_feature_source_sessions(pd.DataFrame(raw), bundle.sessions),
                        {row["session_date"] for row in raw},
                    )
                )
            sector_returns = (
                _sector_returns(
                    market=market,
                    feature=feature,
                    universe=universe,
                    lineage=manifest["safe_summary"]["lineage"],
                    listings=listings,
                    sessions=bundle.sessions,
                    closes=[frame["provider_adjusted_close"] for frame, _ in sources],
                )
                if sector_leaf
                else None
            )
            for column, (listing, (aligned, observed)) in enumerate(
                zip(listings, sources, strict=True)
            ):
                if cancelled():
                    raise ValueError("feature_research.cancelled_at_safe_checkpoint")
                frame = aligned.assign(listing_id=listing)
                if sector_returns is not None:
                    frame = frame.assign(**{FORMULA_SECTOR_FIELD: sector_returns[:, column]})
                count: npt.NDArray[np.int64] = np.cumsum(
                    [day in observed for day in bundle.sessions], dtype=np.int64
                )
                for entry in pending:
                    computed = registry.compute(frame, entry.specification).to_numpy(
                        dtype=np.float64
                    )
                    calls += 1
                    valid = np.isfinite(computed) & (
                        count >= entry.specification.minimum_observations
                    )
                    arrays[entry.factor_id][:, column] = np.where(valid, computed, np.nan)
        for entry in pending:
            if cancelled():
                raise ValueError("feature_research.cancelled_at_safe_checkpoint")
            value = values_owner.publish(
                values=arrays.pop(entry.factor_id),
                **bindings[entry.factor_id],
                sessions=bundle.sessions,
                listing_ids=listings,
            )
            columns.append((entry.factor_id, value.content_hash))
        completed = dict(columns)
        for alias, original in aliases:
            columns.append((alias, completed[original]))
            reused.append(alias)
    if cancelled():
        raise ValueError("feature_research.cancelled_at_safe_checkpoint")
    receipt = ResearchFeatureMaterialization.create(
        definition_plan_hash=plan.plan_hash,
        input_binding_hash=bundle.binding_hash,
        implementation_hash=implementation_hash,
        columns=tuple(sorted(columns)),
        inherited_input_factor_ids=tuple(sorted(inherited)),
        computed_factor_ids=tuple(sorted(v.factor_id for v in pending)),
        reused_factor_ids=tuple(sorted(reused)),
        kernel_calls_in_this_attempt=calls,
    )
    store.publish_json(
        category=RECEIPT_CATEGORY,
        content_hash=receipt.content_hash,
        payload=receipt.model_dump(mode="json"),
    )
    return receipt
