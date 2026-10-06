"""Run one bounded development build for whichever capability was selected.

This is the generic Risk development writer. It knows exactly four things about
the method it is running:

``RiskEstimatorRecipeEnvelope``   what was sealed, schema-neutral
``RiskEstimatorAdapter``          what computes
``BoundRiskReturnInput``          what it is fed
``EstimatedCovariance``           what it returns

It does not import a recipe contract, does not narrow one, and has no branch on
adapter id, schema id or family. A second capability reaches ``estimate`` here
by being installed and selected, and by nothing else. An executor that had to
recognise a schema before running it would only be the compiler's old hardcoding
relocated, and the next method would need another branch.

Why this is not a copy of ``build_historical_covariance_surface``: that function
owns production orchestration -- checkpoint resume, cancellation, dossiers,
peak-RSS accounting, and the current artifact schema. None of that belongs to a
bounded development run, and the parts that carry the numbers -- the adapter,
``BoundRiskReturnInput``, ``evaluate_formation``, and
``publish_covariance_chunk`` -- are reused unchanged rather than reimplemented.

This module must not import ``risk_research.publication``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
from numpy.typing import NDArray

from alphalattice.investment.risk_research.contracts import (
    CausalRiskReturnSurface,
    RiskFormationEvaluation,
)
from alphalattice.investment.risk_research.estimators.contracts import (
    BoundRiskReturnInput,
    EstimatedCovariance,
    RiskEstimatorAdapter,
    RiskEstimatorRecipeEnvelope,
)
from alphalattice.investment.risk_research.estimators.matrix_identity import matrix_content_hash
from alphalattice.investment.risk_research.evaluation.formation import evaluate_formation
from alphalattice.investment.risk_research.experiments.contracts import (
    RiskDevelopmentProgramBinding,
)
from alphalattice.investment.risk_research.experiments.development_artifacts import (
    DEVELOPMENT_DIAGNOSTICS_CATEGORY,
    DEVELOPMENT_SURFACE_CATEGORY,
    RiskDevelopmentCheckpoint,
    RiskDevelopmentCovarianceSurface,
    RiskDevelopmentDiagnostics,
)
from alphalattice.investment.risk_research.experiments.window import (
    REQUIRED_LOOKBACK_SESSIONS,
    RiskDevelopmentInputBinding,
)
from alphalattice.investment.risk_research.surfaces.artifacts import (
    HistoricalCovarianceReader,
    RiskArtifactStore,
)
from alphalattice.investment.risk_research.surfaces.returns import CausalRiskReturnReader
from alphalattice.kernel.quant.sector_history import reclassification_payload, sector_slices
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.research_authoring.contracts import AuthoringError

type FloatArray = NDArray[np.float64]

DEVELOPMENT_CHUNK_FORMATIONS = 21
"""Formations per packed chunk, matching the production writer's chunking."""

DEVELOPMENT_LIMITATIONS = (
    "Development evidence only; no current or admitted Risk publication.",
    "Current-universe research only; historical membership is not point-in-time.",
)


@dataclass(frozen=True, slots=True)
class RiskDevelopmentBuild:
    """A completed bounded development build and its two sealed artifacts."""

    surface: RiskDevelopmentCovarianceSurface
    diagnostics: RiskDevelopmentDiagnostics
    estimate_calls: int


class RiskDevelopmentCancelled(Exception):
    """Cancellation after a sealed numerical chunk, not a lost partial surface."""


def _sector_balanced_weights(
    ordered_listing_ids: tuple[str, ...],
    sector_by_listing_id: Mapping[str, str],
) -> FloatArray:
    try:
        sectors = tuple(sector_by_listing_id[listing_id] for listing_id in ordered_listing_ids)
    except KeyError as error:
        raise AuthoringError("research_authoring.sector_axis_incomplete") from error
    unique = tuple(sorted(set(sectors)))
    weights: FloatArray = np.empty(len(sectors), dtype=np.float64)
    for sector in unique:
        positions = np.flatnonzero(np.asarray(sectors, dtype=object) == sector)
        weights[positions] = 1.0 / (len(unique) * len(positions))
    weights.setflags(write=False)
    return weights


def _admit_estimate(
    estimate: EstimatedCovariance,
    *,
    ordered_listing_ids: tuple[str, ...],
    formation_session: date,
) -> None:
    """Refuse an estimate whose own diagnostics do not describe it.

    Deliberately narrow. This is an *identity* firewall, not a second opinion on
    the mathematics: positive-definiteness, conditioning and shrinkage admission
    belong to the adapter that owns the model, and re-deriving them here would
    put a copy of every method's science in the generic writer -- the branch this
    whole path exists to avoid.

    What the Host cannot delegate is the adapter's claim about *which* numbers
    these are. ``diagnostics.matrix_hash`` is what the chunk index records and
    what replay compares, so an adapter that returned one matrix and named
    another would have published a graph that verifies perfectly against numbers
    nobody computed. Same for the session and the environment: those decide where
    the estimate lands on the formation axis and which capability it is
    attributed to, and neither is checkable after the artifact is sealed.

    Every check runs before the estimate reaches a chunk, so a rejected run has
    written no terminal artifact.
    """

    asset_count = len(ordered_listing_ids)
    if estimate.matrix.shape != (asset_count, asset_count):
        raise AuthoringError("research_authoring.estimate_axis_mismatch")
    if not np.isfinite(estimate.matrix).all():
        raise AuthoringError("research_authoring.estimate_not_finite")
    diagnostics = estimate.diagnostics
    # Recomputed from the returned matrix rather than trusted: this is the value
    # the chunk index and replay both key on.
    if matrix_content_hash(estimate.matrix) != diagnostics.matrix_hash:
        raise AuthoringError("research_authoring.estimate_matrix_hash_mismatch")
    if diagnostics.formation_session != formation_session:
        raise AuthoringError("research_authoring.estimate_formation_session_mismatch")
    if diagnostics.asset_count != asset_count:
        raise AuthoringError("research_authoring.estimate_asset_count_mismatch")


def build_development_covariance_surface(
    *,
    adapter: RiskEstimatorAdapter,
    recipe_envelope: RiskEstimatorRecipeEnvelope,
    recipe_identity_hash: str,
    capability_handle: str,
    binding: RiskDevelopmentProgramBinding,
    input_binding: RiskDevelopmentInputBinding,
    return_surface: CausalRiskReturnSurface,
    return_reader: CausalRiskReturnReader,
    bounded_sessions: Sequence[date],
    artifact_store: RiskArtifactStore,
    sector_by_listing_id: Mapping[str, str],
    cancel_requested: Callable[[], bool] | None = None,
    reserve_output: Callable[[int], None] | None = None,
) -> RiskDevelopmentBuild:
    """Estimate every bound formation and seal one development surface."""

    ordered_listing_ids = tuple(input_binding.ordered_listing_ids)
    formation_sessions = tuple(input_binding.formation_sessions)
    axis = tuple(bounded_sessions)
    expected = REQUIRED_LOOKBACK_SESSIONS + len(formation_sessions) + input_binding.next_sessions
    if len(axis) != expected:
        raise AuthoringError("research_authoring.bounded_window_inconsistent")

    returns = return_reader.read_sessions(return_surface, axis)
    equal_weights: FloatArray = np.full(
        len(ordered_listing_ids), 1.0 / len(ordered_listing_ids), dtype=np.float64
    )
    # Each formation's book is balanced over the Sectors in force at it (V346).
    sector_weights: list[FloatArray] = []
    for rows, mapping in sector_slices(sector_by_listing_id, formation_sessions):
        weights = _sector_balanced_weights(ordered_listing_ids, mapping)
        sector_weights.extend(weights for _ in range(rows.stop - rows.start))

    evaluations: list[RiskFormationEvaluation] = []
    chunks = []
    chunk_sessions: list[date] = []
    chunk_matrices: list[FloatArray] = []
    chunk_hashes: list[str] = []
    previous_matrix: FloatArray | None = None
    previous_maximum: float | None = None
    estimate_calls = 0
    # Ordered by first appearance, so the series stay aligned to the formation
    # axis and a method that names its components keeps that naming durable.
    component_series: dict[str, list[float]] = {}

    execution_key = str(
        canonical_hash(
            {
                "binding": binding.development_binding_hash,
                "input": input_binding.input_binding_hash,
                "sector": dict(sector_by_listing_id),
                **reclassification_payload(sector_by_listing_id),
            }
        )
    )
    checkpoint_path = (
        artifact_store.root / "development/build-checkpoints" / f"{execution_key}.json"
    )
    if checkpoint_path.exists():
        checkpoint = RiskDevelopmentCheckpoint.model_validate_json(checkpoint_path.read_bytes())
        if (
            checkpoint.execution_key != execution_key
            or checkpoint.expected_formation_count != len(formation_sessions)
            or tuple(e.formation_session for e in checkpoint.evaluations)
            != formation_sessions[: len(checkpoint.evaluations)]
        ):
            raise AuthoringError("risk_research.development_checkpoint_scope_mismatch")
        reader = HistoricalCovarianceReader(artifact_store.root.parent)
        for chunk in checkpoint.chunks:
            if chunk.asset_count != len(ordered_listing_ids):
                raise AuthoringError("risk_research.development_checkpoint_axis_mismatch")
            with reader.lease(chunk) as lease:
                for formation, expected_hash in zip(
                    chunk.formation_sessions, chunk.matrix_hashes, strict=True
                ):
                    matrix = lease.matrix(formation)
                    if matrix_content_hash(matrix) != expected_hash:
                        raise AuthoringError("risk_research.development_checkpoint_matrix_mismatch")
                    previous_matrix = matrix
        chunks.extend(checkpoint.chunks)
        evaluations.extend(checkpoint.evaluations)
        component_series.update((k, list(v)) for k, v in checkpoint.component_shrinkages)
        if evaluations:
            previous_maximum = evaluations[-1].maximum_eigenvalue

    if reserve_output is not None:
        remaining = len(formation_sessions) - len(evaluations)
        packed_width = len(ordered_listing_ids) * (len(ordered_listing_ids) + 1) // 2
        reserve_output(remaining * packed_width * 8 + len(formation_sessions) * 8192)
    for offset in range(len(evaluations), len(formation_sessions)):
        if not chunk_sessions and cancel_requested is not None and cancel_requested():
            raise RiskDevelopmentCancelled("risk_research.cancelled_at_chunk_boundary")
        formation_session = formation_sessions[offset]
        # The window ends at the formation session; the next row is the realized
        # session it is evaluated against. Positions are derived from the bound
        # axis, so the run cannot read outside what was sealed.
        end = REQUIRED_LOOKBACK_SESSIONS + offset
        window = returns[end - REQUIRED_LOOKBACK_SESSIONS : end + 1]
        estimate = adapter.estimate(
            recipe=recipe_envelope,
            inputs=BoundRiskReturnInput.create(
                return_surface_hash=return_surface.surface_hash,
                ordered_listing_ids=ordered_listing_ids,
                formation_session=formation_session,
                returns=window,
            ),
        )
        estimate_calls += 1
        # Before the estimate is allowed to reach a chunk or a diagnostic.
        _admit_estimate(
            estimate,
            ordered_listing_ids=ordered_listing_ids,
            formation_session=formation_session,
        )
        evaluations.append(
            evaluate_formation(
                estimate=estimate,
                next_session=axis[end + 1],
                next_return=returns[end + 1],
                equal_weights=equal_weights,
                sector_weights=sector_weights[offset],
                previous_matrix=previous_matrix,
                previous_maximum_eigenvalue=previous_maximum,
            )
        )
        for name, intensity in estimate.component_shrinkages:
            component_series.setdefault(name, []).append(float(intensity))
        previous_matrix = np.array(estimate.matrix, copy=True)
        previous_matrix.setflags(write=False)
        previous_maximum = estimate.diagnostics.maximum_eigenvalue
        chunk_sessions.append(formation_session)
        chunk_matrices.append(estimate.matrix)
        chunk_hashes.append(estimate.diagnostics.matrix_hash)
        if len(chunk_matrices) == DEVELOPMENT_CHUNK_FORMATIONS or offset == (
            len(formation_sessions) - 1
        ):
            chunks.append(
                artifact_store.publish_covariance_chunk(
                    formation_sessions=tuple(chunk_sessions),
                    matrices=tuple(chunk_matrices),
                    matrix_hashes=tuple(chunk_hashes),
                )
            )
            chunk_sessions.clear()
            chunk_matrices.clear()
            chunk_hashes.clear()
            checkpoint = RiskDevelopmentCheckpoint.create(
                execution_key=execution_key,
                expected_formation_count=len(formation_sessions),
                chunks=tuple(chunks),
                evaluations=tuple(evaluations),
                component_shrinkages=tuple((k, tuple(v)) for k, v in component_series.items()),
            )
            # Reuse the existing Risk atomic writer; this working cursor is
            # separate from its immutable published surface/diagnostic families.
            artifact_store._atomic_write(
                checkpoint_path, checkpoint.model_dump_json().encode("utf-8")
            )

    if any(len(series) != len(evaluations) for series in component_series.values()):
        # A method that reported components on some formations and not others
        # would publish a series that cannot be read against the formation axis.
        raise AuthoringError("research_authoring.component_shrinkage_axis_incomplete")
    diagnostics = RiskDevelopmentDiagnostics.create(
        input_binding_hash=input_binding.input_binding_hash,
        recipe_hash=recipe_identity_hash,
        evaluations=tuple(evaluations),
        component_shrinkages=tuple(
            (name, tuple(series)) for name, series in component_series.items()
        ),
    )
    artifact_store.publish_json(
        category=DEVELOPMENT_DIAGNOSTICS_CATEGORY,
        payload=diagnostics.model_dump(mode="json"),
        identity_field="diagnostics_hash",
    )
    surface = RiskDevelopmentCovarianceSurface.create(
        input_binding_hash=input_binding.input_binding_hash,
        capability_handle=capability_handle,
        recipe_envelope=recipe_envelope,
        recipe_hash=recipe_identity_hash,
        program_binding=binding,
        ordered_listing_ids=ordered_listing_ids,
        formation_sessions=formation_sessions,
        chunks=tuple(chunks),
        diagnostics_hash=diagnostics.diagnostics_hash,
        limitations=DEVELOPMENT_LIMITATIONS,
    )
    artifact_store.publish_json(
        category=DEVELOPMENT_SURFACE_CATEGORY,
        payload=surface.model_dump(mode="json"),
        identity_field="surface_hash",
    )
    return RiskDevelopmentBuild(
        surface=surface,
        diagnostics=diagnostics,
        estimate_calls=estimate_calls,
    )


__all__ = [
    "DEVELOPMENT_CHUNK_FORMATIONS",
    "DEVELOPMENT_LIMITATIONS",
    "RiskDevelopmentBuild",
    "build_development_covariance_surface",
]
