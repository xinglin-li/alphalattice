"""Bounded model renewal over the existing Alpha numerical and artifact owners.

The source is a sealed, local observation/label snapshot. No calendar, Provider,
workspace pointer, portfolio state or task runtime is owned by this module.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import date
from functools import partial
from hashlib import sha256
from io import BytesIO
from itertools import chain
from pathlib import Path
from typing import Literal, Self, cast

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

from alphalattice.capabilities.alpha_modeling.adapters.lightgbm_dynamic_panel import (
    DynamicPanelLightGBMAdapter,
    build_dynamic_panel_lightgbm_recipe,
    build_dynamic_panel_lightgbm_search_domain,
)
from alphalattice.capabilities.alpha_modeling.catalog import AlphaModelCatalog
from alphalattice.capabilities.alpha_modeling.contracts import (
    AlphaEstimatorContent,
    BoundAlphaModelFitInput,
    BoundAlphaTrainingInput,
    alpha_model_array_content_hash,
)
from alphalattice.capabilities.alpha_modeling.runtime.service import AlphaModelRuntimeService
from alphalattice.control.workspace_runtime.content_store import (
    verified_array_read_scope,
    verified_npz_arrays,
    verified_request_proof,
    verified_source_value,
    verify_source_checks,
)
from alphalattice.investment.alpha_research.experiments.development_artifacts import (
    AlphaDevelopmentArtifactStore,
)
from alphalattice.investment.alpha_research.inputs.frozen_price_volume import (
    FrozenPriceVolumeInputs,
    apply_frozen_price_volume_scale,
    prepare_frozen_price_volume_history,
)
from alphalattice.investment.alpha_research.inputs.panel_feature_materialization import (
    _fit_temporal_scale,
    preflight_panel_feature_plan,
)
from alphalattice.investment.alpha_research.inputs.panel_feature_views import (
    PanelFeatureSourceArrays,
)
from alphalattice.investment.alpha_research.publication.artifacts import AlphaCurrentArtifactStore
from alphalattice.investment.alpha_research.scores.frozen_inference import (
    AdmittedFrozenInference,
    ComponentFeatureHistory,
    FrozenMarketScale,
    admit_frozen_inference,
    component_feature_surfaces,
)
from alphalattice.investment.alpha_research.scores.heterogeneous_product import (
    INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY,
    HeterogeneousAlphaComponentRecipe,
    component_recipe_is_current,
    is_installed_component,
)
from alphalattice.investment.alpha_research.scores.heterogeneous_replay import (
    AlphaRuntimeHeterogeneousPredictionOwner,
)
from alphalattice.investment.alpha_research.scores.product_lifecycle import (
    AlphaLifecycleError,
    AlphaModelLifecycleRecipe,
    ResolvedAlphaRefitPlan,
    _LifecycleContract,
    resolve_alpha_refit_plan,
)
from alphalattice.investment.alpha_research.scores.product_replay import (
    AlphaProductScoreProjection,
    HeterogeneousFormationScoreInput,
    HeterogeneousLiveModel,
    HeterogeneousVintageFeatureSurface,
    _heterogeneous_candidate,
    score_heterogeneous_component,
)
from alphalattice.investment.alpha_research.targets.component_training import (
    compile_frozen_component_training_targets,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

type FloatArray = npt.NDArray[np.float64]

_VERIFIED: ContextVar[dict[tuple[str, ...], object] | None] = ContextVar(
    "verified_lifecycle_admissions", default=None
)
"""Operation-local reuse of admission and model-set verifications; absent unless a scope opens it.

A ``ContextVar`` rather than a module global so the scope is explicit and a Task
worker thread carries its own. Outside a scope every call verifies, exactly as
it did before this existed.
"""


@contextmanager
def verified_lifecycle_admissions() -> Iterator[None]:
    """Verify each lifecycle admission and model set once per operation, then release.

    ``admit_component_inference`` proves an admission from its bytes: the
    training observations and every prepared refit's arrays are read and their
    content hashes re-derived, the initial children re-read. One Host operation
    -- a readback, an exact-reuse RUN, one Task stage -- asks that of the same
    admission a dozen times, because every score verification re-admits the
    authority the score captured, and each answer re-reads the same hundreds of
    megabytes. Inside this scope the proof is performed once per (store, path,
    expected hash) and the verified contract reused for the rest of the
    operation; a fresh inference wrapper is still built per call, so the
    per-formation fit and prediction counters a score seals start at zero as
    before. ``verify_model_set`` reuses its proof the same way, per (store,
    publication hash). The scope is the operation that read the bytes: it is
    entered by the Host at one request or one Task stage and released with it,
    never kept across requests.
    """
    with verified_array_read_scope():
        # A Host request may already own this scope around its full projection. Nested
        # owners join that request-local memo rather than hiding it or clearing it.
        if _VERIFIED.get() is not None:
            yield
            return
        token = _VERIFIED.set({})
        try:
            yield
        finally:
            scope = _VERIFIED.get()
            if scope is not None:
                scope.clear()
            _VERIFIED.reset(token)


def _array_file(store: AlphaDevelopmentArtifactStore, **arrays: npt.NDArray[np.generic]) -> str:
    stream = BytesIO()
    np.savez(stream, **arrays)
    return str(
        store._publish_packed_bytes(category="current/lifecycle-arrays", payload=stream.getvalue())
    )


def _arrays(store: AlphaDevelopmentArtifactStore, identity: str) -> dict[str, FloatArray]:
    values, _ = store._frozen_arrays_with_identity("lifecycle-arrays", identity)
    return values


def _array_values(
    store: AlphaDevelopmentArtifactStore,
    identity: str,
    payload: bytes,
    file_identity: tuple[object, ...],
) -> dict[str, FloatArray]:
    values = verified_npz_arrays(
        payload,
        identity=(
            "alpha-development",
            str(store.root),
            "lifecycle-arrays",
            identity,
            *file_identity,
        ),
    )
    return values


def _verify_arrays(store: AlphaDevelopmentArtifactStore, identity: str) -> None:
    """Retain an admission's complete structural proof, without packed or decoded bytes."""

    def check() -> None:
        payload, file_identity = store._frozen_payload_with_identity("lifecycle-arrays", identity)

        def decode() -> None:
            with np.load(BytesIO(payload), allow_pickle=False) as archive:
                for name in archive.files:
                    _ = archive[name]  # Decode every member, including CRC and no-pickle checks.

        verified_request_proof(
            ("lifecycle-npz-proof-v1", str(store.root), identity, *file_identity), decode
        )

    verified_source_value(
        ("lifecycle-npz-proof", str(store.root), identity),
        (store._path("current/lifecycle-arrays", identity, "bin"),),
        check,
        nbytes=0,
    )


def read_training_prices(
    store: AlphaDevelopmentArtifactStore, observation_hash: str
) -> FrozenPriceVolumeInputs:
    """Resolve immutable observation values without exposing current publication operations."""
    return AlphaCurrentArtifactStore(store.root.parent).load_frozen_observations(observation_hash)


def copy_training_observations(
    source: AlphaDevelopmentArtifactStore,
    target: AlphaDevelopmentArtifactStore,
    observations: AlphaTrainingObservations,
) -> FrozenPriceVolumeInputs:
    """Copy retained observation/label payloads and metadata into another lifecycle store.

    Args:
        source: Existing development artifact store with the sealed observation and lifecycle
            arrays.
        target: Caller-owned destination store for exact imported records/payloads.
        observations: Declared lifecycle training-observation record to import.

    Returns:
        Model-validated original price/volume inputs read back before copying.
    """
    current = AlphaCurrentArtifactStore(source.root.parent)
    snapshot = current.load_frozen_observation_snapshot(observations.observation_hash)
    original = current.load_frozen_observations(observations.observation_hash)
    target._publish("frozen-observation-snapshots", snapshot, "snapshot_hash")
    for category, identity in (
        ("frozen-observation-arrays", snapshot.array_content_hash),
        ("lifecycle-arrays", observations.array_file_hash),
    ):
        target.import_packed(source, category=category, content_hash=identity)
    target._publish("lifecycle-training-observations", observations, "content_hash")
    return original


class AlphaTrainingObservations(_LifecycleContract):
    """A target owner's labels, availability axis and independent observation seal."""

    purpose: Literal["POST_OBSERVED_MODEL_RENEWAL_QA"] = "POST_OBSERVED_MODEL_RENEWAL_QA"
    observation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_method_id: str
    target_authority_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    label_available_sessions: tuple[date | None, ...]
    array_file_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_support_hash: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$", exclude_if=lambda v: v is None
    )


def publish_training_observations(
    store: AlphaCurrentArtifactStore,
    *,
    observation_hash: str,
    target_method_id: str,
    target_authority_hash: str,
    label_available_sessions: tuple[date | None, ...],
    targets: FloatArray,
    eligible: npt.NDArray[np.bool_],
    training_support_hash: str | None = None,
) -> AlphaTrainingObservations:
    """Publish dated targets and eligibility with explicit label-availability authority.

    Every available label matures after its source formation. Unavailable dates cannot carry finite
    targets, and requested eligibility cannot leave a declared reference population.

    Args:
        store: Current artifact publication/readback owner.
        observation_hash: Exact frozen observation selecting source axes.
        target_method_id: Declared lifecycle target method.
        target_authority_hash: Exact source target authority identity.
        label_available_sessions: Availability date or explicit absence for each formation.
        targets: Float64 formation-by-listing targets.
        eligible: Boolean target/support eligibility on the same axis.
        training_support_hash: Optional admitted training-support identity.

    Returns:
        Sealed training observations selecting the persisted target/eligibility arrays.

    Raises:
        AlphaLifecycleError: Target/eligibility axes or dtypes, availability clocks, unavailable
            labels or reference-population support are invalid.
    """
    source = store.load_frozen_observations(observation_hash)
    shape = (len(source.formation_sessions), len(source.ordered_listing_ids))
    if (
        targets.shape != shape
        or targets.dtype != np.float64
        or eligible.shape != shape
        or eligible.dtype != np.bool_
        or len(label_available_sessions) != len(source.formation_sessions)
        or any(
            end is not None and end <= start
            for start, end in zip(source.formation_sessions, label_available_sessions, strict=True)
        )
    ):
        raise AlphaLifecycleError("alpha_research.training_observations_axis_invalid")
    if any(
        end is None and np.isfinite(targets[i]).any()
        for i, end in enumerate(label_available_sessions)
    ):
        raise AlphaLifecycleError("alpha_research.training_observations_unavailable_labels")
    if source.reference_eligible is not None and np.any(eligible & ~source.reference_eligible):
        raise AlphaLifecycleError("alpha_research.training_rows_outside_reference")
    value = AlphaTrainingObservations.create(
        observation_hash=observation_hash,
        target_method_id=target_method_id,
        target_authority_hash=target_authority_hash,
        label_available_sessions=label_available_sessions,
        array_file_hash=_array_file(store, targets=targets, eligible=eligible),
        training_support_hash=training_support_hash,
    )
    store._publish("lifecycle-training-observations", value, "content_hash")
    return value


def publish_component_training_observations(
    store: AlphaCurrentArtifactStore,
    *,
    component: HeterogeneousAlphaComponentRecipe,
    source: FrozenPriceVolumeInputs,
    training: PanelFeatureSourceArrays,
) -> AlphaTrainingObservations:
    """Prepare the frozen training support, not just the selected Feature mask.

    The research source's common sparse-view mask is a separate obligation from
    the component's 18/20 inputs. Dropping it admits extra historical rows even
    when every selected model column is finite.
    """
    n = len(training.formation_sessions)
    if (
        source.formation_sessions[:n] != training.formation_sessions
        or source.ordered_listing_ids != training.ordered_listing_ids
        or dict(source.sector_by_listing_id) != dict(training.sector_by_listing_id)
        or getattr(source.sector_by_listing_id, "reclassifications", ())
        != getattr(training.sector_by_listing_id, "reclassifications", ())
        or (source.reference_eligible is None) != (training.reference_eligible is None)
        or (
            source.reference_eligible is not None
            and not np.array_equal(source.reference_eligible[:n], training.reference_eligible)
        )
    ):
        raise AlphaLifecycleError("alpha_research.training_source_axes_disagree")
    support = preflight_panel_feature_plan(
        source=training,
        selected_method_ids=("SPARSE_SESSION_AMPLITUDE",),
        maximum_aggregation_span=1,
    )
    values, ends = compile_frozen_component_training_targets(
        source=source,
        formation_sessions=training.formation_sessions,
        holding_end_sessions=training.holding_end_sessions,
        raw_log_returns=training.raw_log_execution_returns,
        simple_returns=training.raw_simple_execution_returns,
        target_method_id=component.target_recipe,
        reference_eligible=(
            source.reference_eligible[:n] if source.reference_eligible is not None else None
        ),
        nominal_member_count=(
            source.nominal_member_count[:n] if source.nominal_member_count is not None else None
        ),
    )
    shape = (len(source.formation_sessions), len(source.ordered_listing_ids))
    targets: FloatArray = np.full(shape, np.nan, dtype=np.float64)
    eligible: npt.NDArray[np.bool_] = np.zeros(shape, dtype=np.bool_)
    targets[:n], eligible[:n] = values, support.common_row_mask
    snapshot = store.publish_frozen_observations(source, disposition="RECORDED_INPUT_QA")
    store._publish("lifecycle-training-support", support.preflight, "preflight_hash")
    return publish_training_observations(
        store,
        observation_hash=snapshot.snapshot_hash,
        target_method_id=component.target_recipe,
        target_authority_hash=canonical_hash(
            {"source": dict(training.source_identity_hashes), "method": component.target_recipe}
        ),
        label_available_sessions=(*ends, *((None,) * (shape[0] - n))),
        targets=targets,
        eligible=eligible,
        training_support_hash=support.preflight.preflight_hash,
    )


class AlphaPreparedRefit(_LifecycleContract):
    """Bind one resolved causal refit plan to prepared row/value authority and market scaling."""

    plan: ResolvedAlphaRefitPlan
    observations_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    array_file_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_axis_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_count: int = Field(ge=1)
    market_scale: FrozenMarketScale

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_preparation(self) -> Self:
        """Require exact observation, vintage and training-session preparation lineage.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            AlphaLifecycleError: Plan/scaler source bindings, scaler vintage or scaler training
                sessions differ from the prepared plan.
        """
        if (
            self.plan.source_binding_hash != self.observations_hash
            or self.market_scale.source_binding_hash != self.observations_hash
            or self.market_scale.vintage != self.plan.vintage
            or self.market_scale.training_sessions != self.plan.training_sessions
        ):
            raise AlphaLifecycleError("alpha_research.refit_preparation_binding_invalid")
        return self

    def child_operation(self, *, recipe_hash: str, numerical_binding_hash: str) -> str:
        """Hash one prepared-input, recipe and numerical-owner fit operation.

        Args:
            recipe_hash: Exact child model recipe.
            numerical_binding_hash: Installed result-deciding numerical owner identity.

        Returns:
            Canonical ordered prepared-content/recipe/numerical operation identity.
        """
        return str(canonical_hash([self.content_hash, recipe_hash, numerical_binding_hash]))


def prepare_alpha_refit(
    store: AlphaDevelopmentArtifactStore,
    *,
    plan: ResolvedAlphaRefitPlan,
    observations: AlphaTrainingObservations,
    component: HeterogeneousAlphaComponentRecipe,
) -> AlphaPreparedRefit:
    """Prepare causal admitted component training rows and persist their exact fit binding.

    The resolved plan is rederived from retained observations. Labels must mature strictly before
    first formation; features reuse inference arithmetic and training-only market scaling. Candidate
    support intersects eligibility, finite targets and finite features before row/value identities
    are sealed.

    Args:
        store: Development artifact publication/readback owner.
        plan: Exact resolved lifecycle refit plan.
        observations: Sealed target/eligibility observations with availability dates.
        component: Installed component recipe deciding feature/target/candidate semantics.

    Returns:
        Persisted prepared refit with plan, row/value authority and market scale.

    Raises:
        AlphaLifecycleError: Plan/source/target binding, label maturity/shape or supported
            training-row count violates admission.
    """
    source = read_training_prices(store, observations.observation_hash)
    expected = resolve_alpha_refit_plan(
        lifecycle=plan.lifecycle,
        vintage=plan.vintage,
        sessions=source.formation_sessions,
        component_recipe_hash=component.recipe_hash,
        source_binding_hash=observations.content_hash,
        ordered_listing_ids=source.ordered_listing_ids,
        ordered_feature_ids=component.ordered_feature_ids,
    )
    if expected != plan or observations.target_method_id != component.target_recipe:
        raise AlphaLifecycleError("alpha_research.refit_input_binding_invalid")
    positions: npt.NDArray[np.int64] = np.asarray(
        [source.formation_sessions.index(day) for day in plan.training_sessions], dtype=np.int64
    )
    if len(observations.label_available_sessions) != len(source.formation_sessions) or any(
        (end := observations.label_available_sessions[int(position)]) is None
        or end >= plan.first_formation
        for position in positions
    ):
        raise AlphaLifecycleError("alpha_research.refit_labels_not_mature")
    labels = _arrays(store, observations.array_file_hash)
    shape = (len(source.formation_sessions), len(source.ordered_listing_ids))
    if (
        labels["targets"].shape != shape
        or labels["eligible"].shape != shape
        or labels["eligible"].dtype != np.bool_
    ):
        raise AlphaLifecycleError("alpha_research.refit_label_axis_invalid")
    # Historical rows use exactly the one-day inference arithmetic; calculate
    # the common history once rather than once per training date or seed.
    history = prepare_frozen_price_volume_history(
        source,
        ordered_feature_ids=plan.ordered_feature_ids,
        through=plan.training_sessions[-1],
    )
    market = np.array(source.market_context_values, copy=True)
    market[1:, 0] = market[:-1, 0]
    market[0, 0] = np.nan
    _, params, _ = _fit_temporal_scale(market, positions)
    scale = FrozenMarketScale(
        vintage=plan.vintage,
        training_sessions=plan.training_sessions,
        source_binding_hash=observations.content_hash,
        center=params["center"],
        scale=params["scale"],
    )
    features = apply_frozen_price_volume_scale(
        history[positions].reshape(-1, len(plan.ordered_feature_ids)),
        ordered_feature_ids=plan.ordered_feature_ids,
        market_center=scale.center,
        market_scale=scale.scale,
    )
    targets = labels["targets"][positions].reshape(-1)
    momentum = source.formula_values.get("mom_252_21")
    candidate = np.stack(
        [
            _heterogeneous_candidate(
                component.component_id,
                eligible=(
                    labels["eligible"][position]
                    if source.reference_eligible is None
                    else labels["eligible"][position] & source.reference_eligible[position]
                ),
                momentum=None if momentum is None else momentum[position],
            )
            for position in positions
        ]
    )
    supported = candidate.reshape(-1) & np.isfinite(targets) & np.isfinite(features).all(axis=1)
    features = np.ascontiguousarray(features[supported])
    targets = np.ascontiguousarray(targets[supported])
    if len(targets) <= len(plan.ordered_feature_ids):
        raise AlphaLifecycleError("alpha_research.refit_training_support_insufficient")
    rows: npt.NDArray[np.int64] = np.flatnonzero(supported).astype("<i8")
    row_hash = canonical_hash(
        {
            "sessions": plan.training_sessions,
            "listings": plan.ordered_listing_ids,
            "positions": sha256(rows.tobytes()).hexdigest(),
        }
    )
    binding = canonical_hash(
        {
            "plan": plan.content_hash,
            "observations": observations.content_hash,
            "features": alpha_model_array_content_hash(features),
            "targets": alpha_model_array_content_hash(targets),
            "rows": row_hash,
            "weight_rule": "EQUAL_ROWS",
            "scale": scale.model_dump(mode="json"),
        }
    )
    value = AlphaPreparedRefit.create(
        plan=plan,
        observations_hash=observations.content_hash,
        array_file_hash=_array_file(store, features=features, targets=targets),
        training_binding_hash=binding,
        row_axis_hash=row_hash,
        row_count=len(targets),
        market_scale=scale,
    )
    store._publish("lifecycle-prepared-refits", value, "content_hash")
    return value


class AlphaRenewedChild(_LifecycleContract):
    """Retain one fitted vintage/seed child and its operation/content/provenance evidence."""

    prepared_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    lifecycle_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    vintage: str
    seed: int
    operation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    estimator_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    environment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class AlphaImportedChild(_LifecycleContract):
    """Explicit historical payload admission, never a claim that this runtime fitted it."""

    prepared_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    lifecycle_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    vintage: str
    seed: int
    estimator_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_index_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    environment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


def read_lifecycle_child(
    store: AlphaDevelopmentArtifactStore, child: AlphaRenewedChild | AlphaImportedChild
) -> tuple[AlphaPreparedRefit, AlphaEstimatorContent]:
    """Read a retained imported or renewed model against its exact prepared fit authority.

    Args:
        store: Development artifact readback owner.
        child: Declared imported or renewed vintage/seed model record.

    Returns:
        Validated preparation and estimator content with matching feature/vintage/lifecycle lineage.

    Raises:
        AlphaLifecycleError: Installed component, fit receipt/operation or
            model/feature/vintage/lifecycle binding disagrees.
    """
    prepared = store._load(
        "lifecycle-prepared-refits", child.prepared_hash, "content_hash", AlphaPreparedRefit
    )
    if isinstance(child, AlphaImportedChild):
        payload = store._frozen_payload("imported-models", child.payload_hash)
        model = AlphaEstimatorContent.model_validate_json(payload)
    else:
        _, model, receipt = store.load_model_fit_sidecar(child.operation_hash)
        component = next(
            (
                c
                for c in INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.components
                if component_recipe_is_current(
                    c.component_id, prepared.plan.component_recipe_hash, c.recipe_hash
                )
            ),
            None,
        )
        if component is None:
            raise AlphaLifecycleError("alpha_research.lifecycle_component_not_installed")
        recipe = build_dynamic_panel_lightgbm_recipe(
            component.estimator_point.resolve(seed=child.seed)
        )
        numerical = DynamicPanelLightGBMAdapter().describe_numerical_binding()
        expected_operation = prepared.child_operation(
            recipe_hash=recipe.recipe_hash, numerical_binding_hash=numerical.numerical_binding_hash
        )
        if (
            receipt.provenance_hash != child.provenance_hash
            or receipt.training_binding_hash != prepared.training_binding_hash
            or receipt.numerical_environment_hash != child.environment_hash
            or receipt.recipe_hash != recipe.recipe_hash
            or child.operation_hash != expected_operation
            or receipt.package_identity_hash != prepared.plan.lifecycle.content_hash
            or receipt.fit_plan_hash != canonical_hash([expected_operation, "FIXED_ITERATION"])
        ):
            raise AlphaLifecycleError("alpha_research.renewed_child_receipt_invalid")
    if (
        model.content_hash != child.estimator_hash
        or model.ordered_feature_ids != prepared.plan.ordered_feature_ids
        or prepared.plan.vintage != child.vintage
        or prepared.plan.lifecycle.content_hash != child.lifecycle_hash
    ):
        raise AlphaLifecycleError("alpha_research.renewed_child_binding_invalid")
    return prepared, model


@dataclass
class AlphaRefitBudget:
    """Caller admission, separate from scientific policy; charge before each fit."""

    maximum_fit_attempts: int
    attempts: int = 0
    store: AlphaCurrentArtifactStore | None = None
    admission_hash: str | None = None
    calls: int = 0
    numerical_calls: int = 0

    def require(self, count: int) -> None:
        """Check requested fit count against the remaining admitted attempt budget.

        When durable store/admission authority is present, spent attempts are counted from retained
        attempt records instead of trusting the in-memory counter.

        Args:
            count: Number of additional fits required before proceeding.

        Raises:
            AlphaLifecycleError: Requested fits exceed remaining admitted attempts.
        """
        spent = self.attempts
        if self.store is not None and self.admission_hash is not None:
            spent = len(
                tuple(
                    (self.store.root / "current/lifecycle-fit-attempts" / self.admission_hash).glob(
                        "*.json"
                    )
                )
            )
        if count > self.maximum_fit_attempts - spent:
            raise AlphaLifecycleError("alpha_research.refit_budget_insufficient")

    def charge(self) -> None:
        """Charge one attempt before fitting and retain its durable attempt record when configured.

        A durable budget requires an exact admission identity. Attempts and call count increment
        before the record write; this method does not execute the numerical fit.

        Raises:
            AlphaLifecycleError: Durable admission identity is absent/invalid or the admitted budget
                is exhausted.
        """
        if self.store is not None:
            if self.admission_hash is None or len(self.admission_hash) != 64:
                raise AlphaLifecycleError("alpha_research.refit_budget_admission_invalid")
            root = self.store.root / "current/lifecycle-fit-attempts" / self.admission_hash
            self.attempts = len(tuple(root.glob("*.json")))
        if self.maximum_fit_attempts < 0 or self.attempts >= self.maximum_fit_attempts:
            raise AlphaLifecycleError("alpha_research.refit_budget_exhausted")
        self.attempts += 1
        self.calls += 1
        if self.store is not None:
            self.store._atomic_write(
                root / f"{self.attempts}.json",
                (str(self.attempts) + "\n").encode("ascii"),
            )


def verified_prepared_values(
    store: AlphaDevelopmentArtifactStore, prepared: AlphaPreparedRefit
) -> dict[str, FloatArray]:
    """One verified training value within a read scope; at most four/512 MiB.

    Dated model sets repeatedly name the same four live vintages. Retain only
    their immutable input values, not predictions or cross-request proof. The
    same matrix is checked again after scope exit (or bounded eviction).
    """
    scope = _VERIFIED.get()
    key = ("prepared-values", str(store.root), prepared.content_hash)
    if scope is not None and key in scope:
        value = cast(dict[str, FloatArray], scope.pop(key))
        scope[key] = value
        return value
    values = _arrays(store, prepared.array_file_hash)
    _check_prepared_values(prepared, values)
    if scope is not None:
        held = [k for k in scope if k[0] == "prepared-values"]
        limit = 512 * 1024 * 1024
        size = sum(v.nbytes for v in values.values())
        retained = sum(
            v.nbytes for k in held for v in cast(dict[str, FloatArray], scope[k]).values()
        )
        while held and (len(held) >= 4 or retained + size > limit):
            prior = cast(dict[str, FloatArray], scope.pop(held.pop(0)))
            retained -= sum(v.nbytes for v in prior.values())
        if size <= limit:
            scope[key] = values
    return values


def _check_prepared_values(prepared: AlphaPreparedRefit, values: dict[str, FloatArray]) -> None:
    plan = prepared.plan
    features, targets = values["features"], values["targets"]
    binding = canonical_hash(
        {
            "plan": plan.content_hash,
            "observations": prepared.observations_hash,
            "features": alpha_model_array_content_hash(features),
            "targets": alpha_model_array_content_hash(targets),
            "rows": prepared.row_axis_hash,
            "weight_rule": "EQUAL_ROWS",
            "scale": prepared.market_scale.model_dump(mode="json"),
        }
    )
    if binding != prepared.training_binding_hash or len(targets) != prepared.row_count:
        raise AlphaLifecycleError("alpha_research.refit_prepared_values_invalid")


def _verify_prepared_values(
    store: AlphaDevelopmentArtifactStore, prepared: AlphaPreparedRefit
) -> None:
    """Retain a complete training-binding proof while every source lease remains valid."""
    semantic_identity = (
        prepared.content_hash,
        prepared.array_file_hash,
        prepared.plan.content_hash,
        prepared.observations_hash,
        prepared.row_axis_hash,
        canonical_hash(prepared.market_scale.model_dump(mode="json")),
        prepared.training_binding_hash,
        prepared.row_count,
    )

    def check() -> None:
        payload, file_identity = store._frozen_payload_with_identity(
            "lifecycle-arrays", prepared.array_file_hash
        )

        def reconcile() -> None:
            _check_prepared_values(
                prepared, _array_values(store, prepared.array_file_hash, payload, file_identity)
            )

        verified_request_proof(
            (
                "lifecycle-prepared-values-proof-v1",
                str(store.root),
                *semantic_identity,
                *file_identity,
            ),
            reconcile,
        )

    verified_source_value(
        ("lifecycle-prepared-values-proof", str(store.root), *semantic_identity),
        (store._path("current/lifecycle-arrays", prepared.array_file_hash, "bin"),),
        check,
        nbytes=0,
    )


def fit_alpha_refit_child(
    store: AlphaCurrentArtifactStore,
    *,
    prepared: AlphaPreparedRefit,
    component: HeterogeneousAlphaComponentRecipe,
    seed: int,
    budget: AlphaRefitBudget,
) -> AlphaRenewedChild:
    """One fixed-recipe child, durable reuse through the existing fit sidecar."""
    plan = prepared.plan
    if seed not in plan.lifecycle.seeds or plan.component_recipe_hash != component.recipe_hash:
        raise AlphaLifecycleError("alpha_research.refit_child_recipe_invalid")
    values = verified_prepared_values(store, prepared)
    features, targets = values["features"], values["targets"]
    binding = prepared.training_binding_hash
    recipe = build_dynamic_panel_lightgbm_recipe(component.estimator_point.resolve(seed=seed))
    domain = build_dynamic_panel_lightgbm_search_domain()
    runtime = AlphaModelRuntimeService(AlphaModelCatalog((DynamicPanelLightGBMAdapter(),)))
    numerical = runtime.resolve_numerical_binding(recipe=recipe, domain=domain)
    operation = prepared.child_operation(
        recipe_hash=recipe.recipe_hash,
        numerical_binding_hash=numerical.numerical_binding_hash,
    )
    pointer = store.root / "current/model-fit-sidecar-by-operation" / f"{operation}.json"
    if not pointer.exists():
        # FIXED_ITERATION consumes no tuning partition. Preserve the existing
        # capability protocol without manufacturing an early-stopping experiment.
        fit_plan = BoundAlphaModelFitInput(
            fit_plan_hash=canonical_hash([operation, "FIXED_ITERATION"]),
            protocol_id="NESTED_EARLY_STOPPING_REFIT",
            parent_training_binding_hash=binding,
            ordered_feature_ids=plan.ordered_feature_ids,
            tuning_training_row_axis_hash=prepared.row_axis_hash,
            purge_row_axis_hash=canonical_hash([]),
            tuning_validation_row_axis_hash=prepared.row_axis_hash,
            tuning_training_feature_values_hash=alpha_model_array_content_hash(features),
            tuning_training_target_values_hash=alpha_model_array_content_hash(targets),
            tuning_validation_feature_values_hash=alpha_model_array_content_hash(features),
            tuning_validation_target_values_hash=alpha_model_array_content_hash(targets),
            tuning_training_features=features,
            tuning_training_targets=targets,
            tuning_validation_features=features,
            tuning_validation_targets=targets,
            selection_metric_id="fixed_iteration",
            maximum_iterations=500,
            early_stopping_rounds=50,
        )
        budget.charge()
        fitted = runtime.fit(
            recipe=recipe,
            domain=domain,
            fit_plan=fit_plan,
            training_input=BoundAlphaTrainingInput(
                training_binding_hash=binding,
                ordered_feature_ids=plan.ordered_feature_ids,
                features=features,
                targets=targets,
            ),
            package_identity_hash=plan.lifecycle.content_hash,
        )
        budget.numerical_calls += (
            fitted.provenance.fit_call_count + fitted.provenance.predict_call_count
        )
        store.publish_model_fit_sidecar(
            operation_binding_hash=operation,
            estimator_content=fitted.fit.estimator_content,
            fit_provenance=fitted.provenance,
        )
    _, estimator, provenance = store.load_model_fit_sidecar(operation)
    if (
        provenance.recipe_hash != recipe.recipe_hash
        or provenance.training_binding_hash != binding
        or provenance.package_identity_hash != plan.lifecycle.content_hash
        or estimator.ordered_feature_ids != plan.ordered_feature_ids
    ):
        raise AlphaLifecycleError("alpha_research.refit_sidecar_binding_invalid")
    child = AlphaRenewedChild.create(
        prepared_hash=prepared.content_hash,
        lifecycle_hash=plan.lifecycle.content_hash,
        vintage=plan.vintage,
        seed=seed,
        operation_hash=operation,
        estimator_hash=estimator.content_hash,
        provenance_hash=provenance.provenance_hash,
        environment_hash=provenance.numerical_environment_hash,
    )
    store._publish("lifecycle-model-children", child, "content_hash")
    return child


def reuse_refit_child(
    source: AlphaDevelopmentArtifactStore,
    destination: AlphaDevelopmentArtifactStore,
    *,
    prepared: AlphaPreparedRefit,
    component: HeterogeneousAlphaComponentRecipe,
    seed: int,
) -> bool:
    """Relocate verified fit evidence, never invent a fit in the receiving workspace."""
    recipe = build_dynamic_panel_lightgbm_recipe(component.estimator_point.resolve(seed=seed))
    numeric = DynamicPanelLightGBMAdapter().describe_numerical_binding().numerical_binding_hash
    operation = prepared.child_operation(
        recipe_hash=recipe.recipe_hash, numerical_binding_hash=numeric
    )
    if not (source.root / "current/model-fit-sidecar-by-operation" / f"{operation}.json").exists():
        return False
    _, model, receipt = source.load_model_fit_sidecar(operation)
    if (
        receipt.recipe_hash != recipe.recipe_hash
        or receipt.training_binding_hash != prepared.training_binding_hash
        or receipt.package_identity_hash != prepared.plan.lifecycle.content_hash
        or receipt.fit_plan_hash != canonical_hash([operation, "FIXED_ITERATION"])
        or model.ordered_feature_ids != prepared.plan.ordered_feature_ids
    ):
        raise AlphaLifecycleError("alpha_research.refit_reuse_binding_invalid")
    destination.publish_model_fit_sidecar(
        operation_binding_hash=operation, estimator_content=model, fit_provenance=receipt
    )
    return True


class AlphaModelSetPublication(_LifecycleContract):
    """Complete verified set, published after its children; never a current pointer."""

    component: HeterogeneousAlphaComponentRecipe
    lifecycle: AlphaModelLifecycleRecipe
    formation: date
    children: tuple[AlphaRenewedChild | AlphaImportedChild, ...]

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def complete(self) -> Self:
        """Require the exact lifecycle vintage-by-seed population and one coherent environment.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            AlphaLifecycleError: Ordered children, child lifecycle lineage or numerical environment
                coherence differs from the declared publication.
        """
        expected = tuple(
            (vintage, seed)
            for vintage in self.lifecycle.vintages(self.formation)
            for seed in self.lifecycle.seeds
        )
        if (
            tuple((c.vintage, c.seed) for c in self.children) != expected
            or any(c.lifecycle_hash != self.lifecycle.content_hash for c in self.children)
            or len({c.environment_hash for c in self.children}) != 1
        ):
            raise AlphaLifecycleError("alpha_research.renewed_model_set_incomplete")
        return self


def verify_model_set(store: AlphaDevelopmentArtifactStore, value: AlphaModelSetPublication) -> None:
    """Verify each model child and its prepared values against the declared lifecycle set.

    Within an active verification scope, exact store/content proofs may be reused. Prepared-value
    verification is shared across children using the same preparation.

    Args:
        store: Development artifact readback owner.
        value: Sealed model-set publication whose children and preparations are checked.

    Raises:
        AlphaLifecycleError: Child content/receipt, component/lifecycle/feature lineage, formation
            causality or prepared values fail verification.
    """
    scope = _VERIFIED.get()
    key = ("model-set", str(store.root), value.content_hash)
    if scope is not None and key in scope:
        return
    verified_inputs: set[str] = set()
    for child in value.children:
        prepared, model = read_lifecycle_child(store, child)
        if (
            prepared.plan.component_recipe_hash != value.component.recipe_hash
            or prepared.plan.lifecycle != value.lifecycle
            or model.ordered_feature_ids != value.component.ordered_feature_ids
            or prepared.plan.first_formation > value.formation
        ):
            raise AlphaLifecycleError("alpha_research.renewed_model_set_binding_invalid")
        if prepared.content_hash not in verified_inputs:
            proof = ("prepared-proof", str(store.root), prepared.content_hash)
            if scope is None or proof not in scope:
                _verify_prepared_values(store, prepared)
                if scope is not None:
                    scope[proof] = True
            verified_inputs.add(prepared.content_hash)
    if scope is not None:
        scope[key] = value


def publish_model_set(store: AlphaCurrentArtifactStore, value: AlphaModelSetPublication) -> str:
    """Verify a complete lifecycle model set before immutable publication.

    Args:
        store: Current model-set publication/readback owner.
        value: Sealed model-set publication to verify and retain.

    Returns:
        Published model-set URI.

    Raises:
        AlphaLifecycleError: Model-set child or prepared-value verification fails.
    """
    verify_model_set(store, value)
    return str(store._publish("lifecycle-model-sets", value, "content_hash"))


def publish_lifecycle_projection(
    store: AlphaDevelopmentArtifactStore, value: AlphaProductScoreProjection
) -> str:
    """Persist score/live arrays with the declared lifecycle projection metadata.

    Args:
        store: Development artifact owner for packed lifecycle arrays.
        value: Typed component score projection and its numerical lanes.

    Returns:
        Content identity of the persisted packed projection payload.
    """
    return _array_file(
        store,
        scores=value.scores,
        live=value.live,
        metadata=np.asarray(value.model_dump_json()),
    )


def read_lifecycle_projection(
    store: AlphaDevelopmentArtifactStore, identity: str
) -> AlphaProductScoreProjection:
    """Reconstruct a typed lifecycle score projection from retained arrays and metadata.

    Args:
        store: Development artifact packed-payload readback owner.
        identity: Exact packed projection content identity.

    Returns:
        Model-validated projection using retained metadata, scores and live mask.
    """
    values = _arrays(store, identity)
    return cast(
        AlphaProductScoreProjection,
        AlphaProductScoreProjection.model_validate(
            {
                **json.loads(str(values["metadata"].item())),
                "scores": values["scores"],
                "live": values["live"],
            }
        ),
    )


class AlphaModelLifecycleAdmission(_LifecycleContract):
    """Operator-admitted QA source, initial models and bounded training permission."""

    purpose: Literal["POST_OBSERVED_MODEL_RENEWAL_QA"] = "POST_OBSERVED_MODEL_RENEWAL_QA"
    component: HeterogeneousAlphaComponentRecipe
    lifecycle: AlphaModelLifecycleRecipe
    observations_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    prepared: tuple[AlphaPreparedRefit, ...]
    initial_children: tuple[AlphaImportedChild, ...]
    formation_start: date
    formation_end: date
    maximum_fit_attempts: int = Field(ge=0)
    environment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fit_vintages: tuple[str, ...] = Field(default=(), exclude_if=lambda v: not v)
    training_factor_ids: tuple[str, ...] = Field(default=(), exclude_if=lambda v: not v)
    previous_admission_hash: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$", exclude_if=lambda v: v is None
    )
    source_transition_hash: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$", exclude_if=lambda v: v is None
    )
    renewal_through: date | None = Field(default=None, exclude_if=lambda v: v is None)
    fit_budget_binding_hash: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$", exclude_if=lambda v: v is None
    )

    @property
    def budget_binding(self) -> str:
        """Read the explicit durable fit-budget binding or historical admission fallback.

        Returns:
            fit_budget_binding_hash when supplied, otherwise admission content_hash.
        """
        return self.fit_budget_binding_hash or self.content_hash

    @property
    def authority_hash(self) -> str:
        """Read this sealed lifecycle admission authority identity.

        Returns:
            The admission content_hash.
        """
        return self.content_hash

    @property
    def model_set(self) -> HeterogeneousAlphaComponentRecipe:
        """Read the admitted component recipe owning this lifecycle model set.

        Returns:
            The retained component recipe.
        """
        return self.component

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_admission(self) -> Self:
        """Require unique causal lifecycle preparations and coherent initial-child authority.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            AlphaLifecycleError: Vintage/factor/fit axes repeat, period/renewal bounds disagree,
                preparation lifecycle/component authority differs or initial-child
                preparation/lifecycle/environment lineage is inconsistent.
        """
        vintages = tuple(v.plan.vintage for v in self.prepared)
        if (
            len(set(vintages)) != len(vintages)
            or self.training_factor_ids != tuple(sorted(set(self.training_factor_ids)))
            or len(set(self.fit_vintages)) != len(self.fit_vintages)
            or (
                self.renewal_through is None
                and bool(self.prepared)
                and not set(self.fit_vintages) <= set(vintages)
            )
            or self.formation_start > self.formation_end
            or (self.renewal_through is not None and self.renewal_through < self.formation_end)
            or any(
                p.plan.lifecycle != self.lifecycle
                or p.plan.component_recipe_hash != self.component.recipe_hash
                for p in self.prepared
            )
            or len({(c.vintage, c.seed) for c in self.initial_children})
            != len(self.initial_children)
            or any(
                c.prepared_hash not in {p.content_hash for p in self.prepared}
                or c.lifecycle_hash != self.lifecycle.content_hash
                or c.environment_hash != self.environment_hash
                for c in self.initial_children
            )
        ):
            raise AlphaLifecycleError("alpha_research.lifecycle_admission_invalid")
        return self

    def supports(self, formation: date) -> bool:
        """Check admitted formation bounds and matured preparation for every live vintage.

        Args:
            formation: Formation whose current model epoch is queried.

        Returns:
            True when formation is within declared bounds and every required vintage has mature
            preparation.
        """
        return self.formation_start <= formation <= self.formation_end and set(
            self.lifecycle.vintages(formation)
        ) <= {p.plan.vintage for p in self.prepared if p.plan.first_formation <= formation}

    def can_prepare(self, formation: date) -> bool:
        """Check whether training authority permits preparation of every required live vintage.

        Args:
            formation: Formation within the admitted initial/renewal interval.

        Returns:
            True when training factors exist and required vintages are prepared or admitted for
            fitting.
        """
        return (
            bool(self.training_factor_ids)
            and self.formation_start <= formation <= (self.renewal_through or self.formation_end)
            and set(self.lifecycle.vintages(formation))
            <= ({p.plan.vintage for p in self.prepared} | set(self.fit_vintages))
        )

    def planned_refits(
        self,
        store: AlphaCurrentArtifactStore,
        formations: tuple[date, ...],
        *,
        allow_preparation: bool = False,
    ) -> tuple[tuple[str, int], ...]:
        """Plan missing admitted vintage/seed fits after exact existing-sidecar readback.

        Initial children and exact retained sidecars need no new fit. Optional preparation permits
        only declared fit vintages; the total missing population must fit the durable admission
        budget.

        Args:
            store: Current artifact owner providing retained fit sidecars and attempt records.
            formations: Formation dates whose union of live vintages is planned.
            allow_preparation: Permit declared fit vintages without an existing prepared record.

        Returns:
            Ordered missing (vintage, seed) operations admitted by the remaining budget.

        Raises:
            AlphaLifecycleError: Required epoch/preparation is absent, a sidecar binding differs, a
                fit period is undeclared or remaining budget is insufficient.
        """
        numeric = DynamicPanelLightGBMAdapter().describe_numerical_binding().numerical_binding_hash
        initial = {(v.vintage, v.seed) for v in self.initial_children}
        missing: list[tuple[str, int]] = []
        for vintage in sorted({v for day in formations for v in self.lifecycle.vintages(day)}):
            prepared = next((p for p in self.prepared if p.plan.vintage == vintage), None)
            if prepared is None:
                if allow_preparation and vintage in self.fit_vintages:
                    missing.extend((vintage, seed) for seed in self.lifecycle.seeds)
                    continue
                raise AlphaLifecycleError("alpha_research.model_epoch_unavailable")
            for seed in self.lifecycle.seeds:
                if (vintage, seed) in initial:
                    continue
                recipe = build_dynamic_panel_lightgbm_recipe(
                    self.component.estimator_point.resolve(seed=seed)
                )
                operation = prepared.child_operation(
                    recipe_hash=recipe.recipe_hash, numerical_binding_hash=numeric
                )
                if (
                    store.root / "current/model-fit-sidecar-by-operation" / f"{operation}.json"
                ).exists():
                    _, _, receipt = store.load_model_fit_sidecar(operation)
                    if (
                        receipt.recipe_hash != recipe.recipe_hash
                        or receipt.training_binding_hash != prepared.training_binding_hash
                    ):
                        raise AlphaLifecycleError("alpha_research.refit_sidecar_binding_invalid")
                else:
                    if vintage not in self.fit_vintages:
                        raise AlphaLifecycleError("alpha_research.refit_period_not_admitted")
                    missing.append((vintage, seed))
        AlphaRefitBudget(
            self.maximum_fit_attempts, store=store, admission_hash=self.budget_binding
        ).require(len(missing))
        return tuple(missing)

    def children_for(
        self, store: AlphaCurrentArtifactStore, formation: date, *, budget: AlphaRefitBudget
    ) -> AlphaModelSetPublication:
        """Read, reuse or fit admitted lifecycle children and publish their verified complete set.

        Args:
            store: Current artifact owner for child fits/readback and model-set publication.
            formation: Formation supported by the admitted lifecycle preparations.
            budget: Attempt budget charged by any required numerical child fits.

        Returns:
            Verified complete model-set publication for the declared live-vintage window.

        Raises:
            AlphaLifecycleError: Formation/fit admission, durable budget, child readback or
                model-set completeness fails.
        """
        if not self.supports(formation):
            raise AlphaLifecycleError("alpha_research.model_epoch_unavailable")
        budget.require(len(self.planned_refits(store, (formation,))))
        children: list[AlphaImportedChild | AlphaRenewedChild] = []
        initial = {(v.vintage, v.seed): v for v in self.initial_children}
        for vintage in self.lifecycle.vintages(formation):
            prepared = next(v for v in self.prepared if v.plan.vintage == vintage)
            for seed in self.lifecycle.seeds:
                child = initial.get((vintage, seed))
                if child is not None:
                    read_lifecycle_child(store, child)
                    children.append(child)
                else:
                    children.append(
                        fit_alpha_refit_child(
                            store,
                            prepared=prepared,
                            component=self.component,
                            seed=seed,
                            budget=budget,
                        )
                    )
        result = AlphaModelSetPublication.create(
            component=self.component,
            lifecycle=self.lifecycle,
            formation=next(
                p.plan.first_formation
                for p in self.prepared
                if p.plan.vintage == self.lifecycle.vintages(formation)[0]
            ),
            children=tuple(children),
        )
        publish_model_set(store, result)
        return result


def renew_lifecycle_admission(
    store: AlphaCurrentArtifactStore,
    *,
    previous: AlphaModelLifecycleAdmission,
    observations: AlphaTrainingObservations,
    through: date,
    source_policy: Literal["APPEND_ONLY", "REVISED_INPUTS_NEW_VINTAGES_ONLY"] = "APPEND_ONLY",
) -> AlphaModelLifecycleAdmission:
    """Seal an append-only successor, preserving old period inputs and fit budget."""
    if not previous.training_factor_ids or through > (
        previous.renewal_through or previous.formation_end
    ):
        raise AlphaLifecycleError("alpha_research.training_source_renewal_not_admitted")
    old = store._load(
        "lifecycle-training-observations",
        previous.observations_hash,
        "content_hash",
        AlphaTrainingObservations,
    )
    source = store.load_frozen_observations(observations.observation_hash)
    old_source = store.load_frozen_observations(old.observation_hash)
    if source_policy == "APPEND_ONLY":
        source.require_unchanged_prefix(old_source)
    elif (
        source.formation_sessions[: len(old_source.formation_sessions)]
        != old_source.formation_sessions
    ):
        raise AlphaLifecycleError("alpha_research.training_source_calendar_changed")
    before, after = (
        _arrays(store, old.array_file_hash),
        _arrays(store, observations.array_file_hash),
    )
    known: npt.NDArray[np.int64] = np.asarray(
        [
            i
            for i, end in enumerate(old.label_available_sessions)
            if end is not None and end <= old_source.formation_sessions[-1]
        ],
        dtype=np.int64,
    )
    if source_policy == "APPEND_ONLY" and (
        any(
            old.label_available_sessions[int(i)] != observations.label_available_sessions[int(i)]
            for i in known
        )
        or any(
            not np.array_equal(before[name][known], after[name][known], equal_nan=True)
            for name in ("targets", "eligible")
        )
    ):
        raise AlphaLifecycleError("alpha_research.training_source_correction_not_admitted")
    if through > source.formation_sessions[-1] or through < previous.formation_start:
        raise AlphaLifecycleError("alpha_research.training_source_renewal_support_invalid")
    if (
        observations.target_method_id != previous.component.target_recipe
        or observations.training_support_hash is None
    ):
        raise AlphaLifecycleError("alpha_research.training_source_renewal_binding_invalid")
    prepared = {p.plan.vintage: p for p in previous.prepared}
    needed = {
        v
        for day in source.formation_sessions
        if previous.formation_start <= day <= through
        for v in previous.lifecycle.vintages(day)
    }
    for vintage in sorted(needed - prepared.keys()):
        if vintage not in previous.fit_vintages:
            raise AlphaLifecycleError("alpha_research.refit_period_not_admitted")
        plan = resolve_alpha_refit_plan(
            lifecycle=previous.lifecycle,
            vintage=vintage,
            sessions=source.formation_sessions,
            component_recipe_hash=previous.component.recipe_hash,
            source_binding_hash=observations.content_hash,
            ordered_listing_ids=source.ordered_listing_ids,
            ordered_feature_ids=previous.component.ordered_feature_ids,
        )
        prepared[vintage] = prepare_alpha_refit(
            store, plan=plan, observations=observations, component=previous.component
        )
    value = AlphaModelLifecycleAdmission.create(
        **{
            **previous.model_dump(exclude={"content_hash"}),
            "observations_hash": observations.content_hash,
            "prepared": tuple(prepared[v] for v in sorted(prepared)),
            "formation_end": previous.renewal_through or through,
            "previous_admission_hash": previous.content_hash,
            "fit_budget_binding_hash": previous.budget_binding,
            "source_transition_hash": canonical_hash(
                [
                    previous.observations_hash,
                    observations.content_hash,
                    "REVISED_INPUTS_NEW_VINTAGES_ONLY",
                ]
            )
            if source_policy != "APPEND_ONLY"
            else None,
        }
    )
    store._publish("lifecycle-admissions", previous, "content_hash")
    store._publish("lifecycle-admissions", value, "content_hash")
    return value


def verify_lifecycle_successor(
    store: AlphaCurrentArtifactStore,
    *,
    ancestor: AlphaModelLifecycleAdmission,
    successor: AlphaModelLifecycleAdmission,
) -> None:
    """A stored descendant cannot turn a source update into a wider fit grant."""
    seen: set[str] = set()
    current = successor
    while current != ancestor:
        if current.content_hash in seen or current.previous_admission_hash is None:
            raise AlphaLifecycleError("alpha_research.lifecycle_successor_unbound")
        seen.add(current.content_hash)
        parent = (
            ancestor
            if current.previous_admission_hash == ancestor.content_hash
            else store._load(
                "lifecycle-admissions",
                current.previous_admission_hash,
                "content_hash",
                AlphaModelLifecycleAdmission,
            )
        )
        if (
            current.component != parent.component
            or current.lifecycle != parent.lifecycle
            or current.training_factor_ids != parent.training_factor_ids
            or current.budget_binding != parent.budget_binding
            or current.maximum_fit_attempts != parent.maximum_fit_attempts
            or current.fit_vintages != parent.fit_vintages
            or current.renewal_through != parent.renewal_through
            or current.initial_children != parent.initial_children
            or current.formation_start != parent.formation_start
            or current.formation_end > (parent.renewal_through or parent.formation_end)
            or not {p.content_hash for p in parent.prepared}
            <= {p.content_hash for p in current.prepared}
            or (
                current.source_transition_hash is not None
                and current.source_transition_hash
                != canonical_hash(
                    [
                        parent.observations_hash,
                        current.observations_hash,
                        "REVISED_INPUTS_NEW_VINTAGES_ONLY",
                    ]
                )
            )
        ):
            raise AlphaLifecycleError("alpha_research.lifecycle_successor_permission_changed")
        current = parent


@dataclass(frozen=True)
class _LifecycleScoringRecipe:
    component_id: str
    recipe_hash: str
    feature_axis_hash: str
    feature_count: int
    ordered_feature_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    vintage_count: int
    vintage_weights: tuple[int, ...]
    candidate_semantics: str
    score_aggregation: str


class AdmittedRenewingInference:
    """A renewable source for the same feature/scoring application operations."""

    def __init__(self, artifact_root: Path, authority: AlphaModelLifecycleAdmission):
        """Bind renewing scoring to lifecycle authority and its deterministic prediction owner.

        Args:
            artifact_root: Caller-owned artifact root used for current model storage.
            authority: Sealed lifecycle admission selecting one installed component.
        """
        self.store, self.authority = AlphaCurrentArtifactStore(artifact_root), authority
        self.prediction_owner = AlphaRuntimeHeterogeneousPredictionOwner(
            component_ids=(authority.component.component_id,),
        )
        self.model_set_publication_hash: str | None = None
        self.fit_calls = 0
        self.fit_numerical_calls = 0
        self.models = tuple(authority.initial_children)

    def features(
        self,
        source: FrozenPriceVolumeInputs,
        formation: date,
        *,
        history: ComponentFeatureHistory | None = None,
    ) -> tuple[HeterogeneousVintageFeatureSurface, ...]:
        """Prepare formation features with the exact admitted vintage market scalers.

        Args:
            source: Frozen source observations used for feature preparation.
            formation: Formation supported by admitted lifecycle preparations.
            history: Optional precomputed history on the identical source object and feature axis.

        Returns:
            Ordered admitted vintage feature surfaces bound to lifecycle authority.

        Raises:
            AlphaLifecycleError: Model epoch is unavailable or cached history source/feature
                authority differs.
        """
        if not self.authority.supports(formation):
            raise AlphaLifecycleError("alpha_research.model_epoch_unavailable")
        scales = tuple(
            next(p.market_scale for p in self.authority.prepared if p.plan.vintage == vintage)
            for vintage in self.authority.lifecycle.vintages(formation)
        )
        if history is not None:
            if (
                history.source is not source
                or history.feature_ids != self.authority.component.ordered_feature_ids
            ):
                raise AlphaLifecycleError("alpha_research.feature_history_source_mismatch")
            return history.surfaces(
                formation, scales=scales, authority_hash=self.authority.content_hash
            )
        return tuple(
            component_feature_surfaces(
                source,
                formation=formation,
                feature_ids=self.authority.component.ordered_feature_ids,
                scales=scales,
                authority_hash=self.authority.content_hash,
            )
        )

    def score(
        self,
        *,
        formation: date,
        listing_ids: tuple[str, ...],
        eligible: npt.NDArray[np.bool_],
        surfaces: tuple[HeterogeneousVintageFeatureSurface, ...],
        raw_12_1_momentum: FloatArray | None = None,
    ) -> AlphaProductScoreProjection:
        """Renew required admitted models and score the declared live-vintage feature window.

        A durable budget governs missing child fits. The complete model-set publication is read back
        before prediction; observed fit counts and publication identity are retained by this scorer.

        Args:
            formation: Supported score formation.
            listing_ids: Exact ordered listing axis.
            eligible: Decision-eligibility mask on that axis.
            surfaces: Exact lifecycle vintage-count feature surfaces on admitted feature/listing
                axes.
            raw_12_1_momentum: Optional raw candidate-selection momentum lane.

        Returns:
            Typed heterogeneous component score projection under the declared lifecycle window.

        Raises:
            AlphaLifecycleError: Epoch, score axes, fit budget or model child authority fails.
            HeterogeneousScoringError: Bound component prediction/scoring fails.
        """
        authority = self.authority
        _heterogeneous_candidate(
            authority.component.component_id, eligible=eligible, momentum=raw_12_1_momentum
        )
        if len(surfaces) != authority.lifecycle.vintage_count or any(
            surface.ordered_listing_ids != listing_ids
            or surface.ordered_feature_ids != authority.component.ordered_feature_ids
            for surface in surfaces
        ):
            raise AlphaLifecycleError("alpha_research.lifecycle_scoring_axis_invalid")
        budget = AlphaRefitBudget(
            authority.maximum_fit_attempts,
            store=self.store,
            admission_hash=authority.budget_binding,
        )
        publication = authority.children_for(
            self.store,
            formation,
            budget=budget,
        )
        self.fit_calls = budget.calls
        self.fit_numerical_calls = budget.numerical_calls
        self.model_set_publication_hash = publication.content_hash
        models = []
        for child in publication.children:
            prepared, estimator = read_lifecycle_child(self.store, child)
            recipe = build_dynamic_panel_lightgbm_recipe(
                authority.component.estimator_point.resolve(seed=child.seed)
            )
            models.append(
                HeterogeneousLiveModel(
                    vintage=child.vintage,
                    seed=child.seed,
                    recipe_hash=recipe.recipe_hash,
                    lineage_hash=child.content_hash,
                    training_binding_hash=prepared.training_binding_hash,
                    estimator=estimator,
                )
            )
        component, lifecycle = authority.component, authority.lifecycle
        view = _LifecycleScoringRecipe(
            component.component_id,
            component.recipe_hash,
            component.feature_axis_hash,
            component.feature_count,
            component.ordered_feature_ids,
            lifecycle.seeds,
            lifecycle.vintage_count,
            lifecycle.vintage_weights,
            component.candidate_semantics,
            lifecycle.score_aggregation,
        )
        return score_heterogeneous_component(
            component=view,
            inputs=HeterogeneousFormationScoreInput.create(
                formation_session=formation,
                ordered_listing_ids=listing_ids,
                decision_eligible=eligible,
                raw_12_1_momentum=raw_12_1_momentum,
                feature_surfaces=surfaces,
                models=tuple(models),
                model_set_manifest_hash=publication.content_hash,
            ),
            prediction_owner=self.prediction_owner,
            declared_vintages=lifecycle.vintages(formation),
        )


def read_lifecycle_admission(root: Path, *, expected_hash: str) -> AlphaModelLifecycleAdmission:
    """Read exact installed lifecycle admission while refusing mixed frozen authority.

    Args:
        root: Admission file or directory containing model-lifecycle-admission.json.
        expected_hash: Required sealed lifecycle admission content identity.

    Returns:
        Validated admission for the exact installed component recipe.

    Raises:
        AlphaLifecycleError: Expected identity, competing frozen authority or installed component
            binding differs.
    """
    path = root if root.is_file() else root / "model-lifecycle-admission.json"
    authority = AlphaModelLifecycleAdmission.model_validate_json(path.read_bytes())
    if authority.content_hash != expected_hash or (root / "inference-authority.json").exists():
        raise AlphaLifecycleError("alpha_research.lifecycle_admission_binding_invalid")
    if not is_installed_component(authority.component):
        raise AlphaLifecycleError("alpha_research.lifecycle_component_not_installed")
    return cast(AlphaModelLifecycleAdmission, authority)


def admit_component_inference(
    root: Path, *, store: AlphaCurrentArtifactStore, expected_hash: str
) -> AdmittedFrozenInference | AdmittedRenewingInference:
    """Admit verified renewing authority or the declared frozen inference fallback.

    A missing lifecycle admission selects the frozen-authority path. An active verification scope
    can reuse exact store/path/authority admission proof; it does not grant a new fit budget.

    Args:
        root: Caller-owned frozen/lifecycle authority file or directory.
        store: Current artifact store used to verify lifecycle retained evidence.
        expected_hash: Required authority content identity for the selected path.

    Returns:
        Admitted renewing or frozen component scorer.

    Raises:
        AlphaLifecycleError: Lifecycle admission and retained evidence cannot be verified.
        FrozenInferenceError: The selected frozen fallback authority/model closure cannot be
            verified.
    """
    path = root if root.is_file() else root / "model-lifecycle-admission.json"
    if not path.exists():
        return admit_frozen_inference(root, expected_hash=expected_hash)
    scope = _VERIFIED.get()
    key = ("admission", str(store.root), str(path), expected_hash)
    if scope is not None and key in scope:
        return AdmittedRenewingInference(
            store.root.parent, cast(AlphaModelLifecycleAdmission, scope[key])
        )
    authority = _verify_lifecycle_admission(root, store=store, expected_hash=expected_hash)
    if scope is not None:
        scope[key] = authority
    return AdmittedRenewingInference(store.root.parent, authority)


def _verify_lifecycle_admission(
    root: Path, *, store: AlphaCurrentArtifactStore, expected_hash: str
) -> AlphaModelLifecycleAdmission:
    """Prove one renewing admission from its bytes: observations, prepared refits, children."""

    authority = read_lifecycle_admission(root, expected_hash=expected_hash)
    observations = store._load(
        "lifecycle-training-observations",
        authority.observations_hash,
        "content_hash",
        AlphaTrainingObservations,
    )
    store.load_frozen_observation_snapshot(observations.observation_hash)
    frozen_hashes = dict.fromkeys((observations.observation_hash,))
    array_hashes = [observations.array_file_hash]
    verified_observations = {observations.content_hash}
    for prepared in authority.prepared:
        if (
            store._load(
                "lifecycle-prepared-refits",
                prepared.content_hash,
                "content_hash",
                AlphaPreparedRefit,
            )
            != prepared
        ):
            raise AlphaLifecycleError("alpha_research.lifecycle_prepared_binding_invalid")
        array_hashes.append(prepared.array_file_hash)
        if prepared.observations_hash not in verified_observations:
            prior_observations = store._load(
                "lifecycle-training-observations",
                prepared.observations_hash,
                "content_hash",
                AlphaTrainingObservations,
            )
            if prior_observations.observation_hash not in frozen_hashes:
                store.load_frozen_observation_snapshot(prior_observations.observation_hash)
                frozen_hashes[prior_observations.observation_hash] = None
            array_hashes.append(prior_observations.array_file_hash)
            verified_observations.add(prepared.observations_hash)
    verify_source_checks(
        chain(
            (partial(store.verify_frozen_observations, identity) for identity in frozen_hashes),
            (partial(_verify_arrays, store, identity) for identity in dict.fromkeys(array_hashes)),
        )
    )
    for child in authority.initial_children:
        read_lifecycle_child(store, child)
    return authority
