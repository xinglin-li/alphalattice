"""Pickle-safe, publication-free execution of one independent Panel Alpha fold."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from types import MappingProxyType
from typing import Any, Literal

from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from alphalattice.investment.alpha_research.experiments.panel_methodology_authoring import (
    PanelResearchMethodologyRequest,
)
from alphalattice.investment.alpha_research.experiments.panel_methodology_models import (
    PanelModelFoldResult,
    execute_panel_model_fold,
)
from alphalattice.investment.alpha_research.inputs.panel_feature_materialization import (
    PanelFeatureProjection,
    materialize_panel_feature_projection,
    preflight_panel_feature_plan,
)
from alphalattice.investment.alpha_research.inputs.panel_feature_views import (
    BoolArray,
    FloatArray,
    FoldScaleReceipt,
    InstalledPanelViewId,
    PanelFeatureSourceArrays,
)
from alphalattice.kernel.quant.sector_history import SectorHistory


class PanelAlphaFoldExecutionError(ValueError):
    """Stable failure at the independent Alpha-fold worker boundary."""


@dataclass(frozen=True, slots=True)
class PanelAlphaFold:
    fold_index: int
    training_sessions: tuple[date, ...]
    validation_sessions: tuple[date, ...]
    source_manifest_hash: str


@dataclass(frozen=True, slots=True)
class PanelFeatureSourcePayload:
    """Serializable source payload whose arrays may be process-shared by joblib."""

    formation_sessions: tuple[date, ...]
    holding_end_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]
    ordered_factor_ids: tuple[str, ...]
    absolute_state_factor_ids: tuple[str, ...]
    ordered_sector_ids: tuple[str, ...]
    sector_by_listing_id: Mapping[str, str]
    raw_formula_values: FloatArray
    total_return_target_z: FloatArray
    raw_log_execution_returns: FloatArray
    raw_simple_execution_returns: FloatArray
    sector_context_values: FloatArray
    market_context_values: FloatArray
    source_identity_hashes: dict[str, str]
    reference_eligible: BoolArray | None = None

    @classmethod
    def from_source(cls, source: PanelFeatureSourceArrays) -> PanelFeatureSourcePayload:
        return cls(
            formation_sessions=source.formation_sessions,
            holding_end_sessions=source.holding_end_sessions,
            ordered_listing_ids=source.ordered_listing_ids,
            ordered_factor_ids=source.ordered_factor_ids,
            absolute_state_factor_ids=source.absolute_state_factor_ids,
            ordered_sector_ids=source.ordered_sector_ids,
            # A history crosses as itself (V346); a plain map as a picklable dict.
            sector_by_listing_id=(
                source.sector_by_listing_id
                if isinstance(source.sector_by_listing_id, SectorHistory)
                else dict(source.sector_by_listing_id)
            ),
            raw_formula_values=source.raw_formula_values,
            total_return_target_z=source.total_return_target_z,
            raw_log_execution_returns=source.raw_log_execution_returns,
            raw_simple_execution_returns=source.raw_simple_execution_returns,
            sector_context_values=source.sector_context_values,
            market_context_values=source.market_context_values,
            source_identity_hashes=dict(source.source_identity_hashes),
            reference_eligible=source.reference_eligible,
        )

    def restore(self) -> PanelFeatureSourceArrays:
        # Small arrays may cross the process boundary by pickle rather than
        # read-only memmap. Restore the input contract in either transport.
        for values in (
            self.raw_formula_values,
            self.total_return_target_z,
            self.raw_log_execution_returns,
            self.raw_simple_execution_returns,
            self.sector_context_values,
            self.market_context_values,
            self.reference_eligible,
        ):
            if values is not None:
                values.setflags(write=False)
        return PanelFeatureSourceArrays(
            formation_sessions=self.formation_sessions,
            holding_end_sessions=self.holding_end_sessions,
            ordered_listing_ids=self.ordered_listing_ids,
            ordered_factor_ids=self.ordered_factor_ids,
            absolute_state_factor_ids=self.absolute_state_factor_ids,
            ordered_sector_ids=self.ordered_sector_ids,
            sector_by_listing_id=(
                self.sector_by_listing_id
                if isinstance(self.sector_by_listing_id, SectorHistory)
                else MappingProxyType(dict(self.sector_by_listing_id))
            ),
            raw_formula_values=self.raw_formula_values,
            total_return_target_z=self.total_return_target_z,
            raw_log_execution_returns=self.raw_log_execution_returns,
            raw_simple_execution_returns=self.raw_simple_execution_returns,
            sector_context_values=self.sector_context_values,
            market_context_values=self.market_context_values,
            source_identity_hashes=MappingProxyType(dict(self.source_identity_hashes)),
            reference_eligible=self.reference_eligible,
        )


@dataclass(frozen=True, slots=True)
class PanelAlphaFoldWorkerRequest:
    program_hash: str
    expected_feature_preflight_hash: str
    source: PanelFeatureSourcePayload
    fold: PanelAlphaFold
    methodology: PanelResearchMethodologyRequest
    maximum_aggregation_span: int
    numerical_thread_count: int = 1


@dataclass(frozen=True, slots=True)
class PanelAlphaFoldWorkerResult:
    fold: PanelModelFoldResult
    scale_receipts: tuple[FoldScaleReceipt, ...]
    feature_preflight_hash: str
    numerical_thread_count: int = 1
    covariance_open_count: int = 0
    optimizer_call_count: int = 0
    solver_call_count: int = 0


def split_panel_inner_sessions(
    training_sessions: tuple[date, ...], *, purge_sessions: int = 2
) -> tuple[tuple[date, ...], tuple[date, ...]]:
    """Own the one causal inner split consumed by sequential and worker routes."""

    validation_count = max(1, len(training_sessions) // 5)
    split = len(training_sessions) - validation_count - purge_sessions
    if split < 63:
        raise PanelAlphaFoldExecutionError("alpha_research.inner_fold_too_short")
    return training_sessions[:split], training_sessions[split + purge_sessions :]


def _pin_worker_process() -> None:
    """Pin the environment of a process that runs folds and nothing else."""

    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ["ALPHALATTICE_NETWORK_DISABLED"] = "1"


def execute_panel_alpha_fold_worker(
    request: PanelAlphaFoldWorkerRequest,
    *,
    owns_process: bool = False,
) -> PanelAlphaFoldWorkerResult:
    """Compute one fold with no artifact, covariance, optimizer, or pointer owner.

    The fold's thread limit holds around its whole computation, so one fold
    computes the same bytes inline and in a worker. Only a spawned worker owns
    its process and pins that process's environment as well. Inline, the
    process is the Host's, shared with its other Tasks, and the fold writes
    nothing process-wide: an environment written there would outlive the fold,
    turn the Host's network off and move every binding that reads it.
    """

    if request.numerical_thread_count != 1:
        raise PanelAlphaFoldExecutionError("alpha_research.worker_thread_plan_invalid")
    if owns_process:
        _pin_worker_process()
    with threadpool_limits(limits=request.numerical_thread_count):
        return _compute_fold(request)


def _compute_fold(request: PanelAlphaFoldWorkerRequest) -> PanelAlphaFoldWorkerResult:
    source = request.source.restore()
    plan = preflight_panel_feature_plan(
        source=source,
        selected_method_ids=request.methodology.feature_view_ids,
        maximum_aggregation_span=request.maximum_aggregation_span,
    )
    if plan.preflight.preflight_hash != request.expected_feature_preflight_hash:
        raise PanelAlphaFoldExecutionError("alpha_research.worker_feature_preflight_mismatch")
    inner_training, inner_validation = split_panel_inner_sessions(request.fold.training_sessions)
    receipts: dict[str, FoldScaleReceipt] = {}

    def provider(
        boundary_id: Literal["INNER", "OUTER"],
        training_sessions: tuple[date, ...],
        transform_sessions: tuple[date, ...],
    ) -> tuple[
        Callable[[InstalledPanelViewId], PanelFeatureProjection],
        Callable[[], None],
    ]:
        role_cache: dict[str, Any] = {}
        current: tuple[InstalledPanelViewId, PanelFeatureProjection] | None = None

        def resolve(method_id: InstalledPanelViewId) -> PanelFeatureProjection:
            nonlocal current
            if current is not None and current[0] == method_id:
                return current[1]
            projection = materialize_panel_feature_projection(
                plan=plan,
                method_id=method_id,
                program_hash=request.program_hash,
                fold_index=request.fold.fold_index,
                boundary_id=boundary_id,
                training_sessions=training_sessions,
                transform_sessions=transform_sessions,
                role_materialization_cache=role_cache,
            )
            receipts.update({value.receipt_hash: value for value in projection.scale_receipts})
            current = (method_id, projection)
            return projection

        def release() -> None:
            nonlocal current
            current = None
            role_cache.clear()

        return resolve, release

    inner, release_inner = provider("INNER", inner_training, inner_validation)
    outer, release_outer = provider(
        "OUTER", request.fold.training_sessions, request.fold.validation_sessions
    )
    try:
        result = execute_panel_model_fold(
            program_hash=request.program_hash,
            fold_index=request.fold.fold_index,
            request=request.methodology,
            inner_projection=inner,
            outer_projection=outer,
            holding_horizon_sessions=1,
            release_inner_projection=release_inner,
        )
    finally:
        release_inner()
        release_outer()
    return PanelAlphaFoldWorkerResult(
        fold=result,
        scale_receipts=tuple(receipts[key] for key in sorted(receipts)),
        feature_preflight_hash=plan.preflight.preflight_hash,
        numerical_thread_count=request.numerical_thread_count,
    )


__all__ = [
    "PanelAlphaFold",
    "PanelAlphaFoldExecutionError",
    "PanelAlphaFoldWorkerRequest",
    "PanelAlphaFoldWorkerResult",
    "PanelFeatureSourcePayload",
    "execute_panel_alpha_fold_worker",
    "split_panel_inner_sessions",
]
