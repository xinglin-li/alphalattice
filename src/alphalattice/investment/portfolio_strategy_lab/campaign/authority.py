"""The covariance and reference-mark lanes the paired Portfolio route still consumes.

What is left of the Stage-6 campaign's input authority (the campaign itself retired with
R01, 2026-09-29). The paired panel-methodology route reads a rematerialized Risk covariance
surface a published graph already names, projects it to its decision axis as a sealed lane the
optimizer can recognise, and values its reference book against one Market-owned close-mark
surface. The lanes are opened once and travel with the binding that names them, so what a run
sealed and what it computed cannot come apart.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

from alphalattice.capabilities.portfolio_backtesting.contracts import (
    MARKED_TO_MARKET_AT_CLOSE_T,
    PortfolioStateTransitionBinding,
    RebalanceClockBinding,
)
from alphalattice.capabilities.portfolio_backtesting.reference_marks import ReferenceMarkLane
from alphalattice.foundation.market_data_ops.publication.session_marks import (
    SessionMarkArtifactStore,
    SessionMarkSurface,
)
from alphalattice.investment.portfolio_strategy_lab.inputs.development import (
    RiskInputAuthority,
)
from alphalattice.investment.risk_research.evaluation.campaign_metrics import (
    unpack_chunk_matrices,
)
from alphalattice.investment.risk_research.experiments.development_artifacts import (
    DEVELOPMENT_SURFACE_CATEGORY,
    RiskDevelopmentCovarianceSurface,
)
from alphalattice.investment.risk_research.surfaces.artifacts import (
    RiskArtifactStore,
)
from alphalattice.investment.risk_research.surfaces.decomposition import RiskAllocationProjection
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.project_layout import source_root
from alphalattice.kernel.shared_kernel.source_identity import (
    switched_source_identity,
)

from .upstream import (
    PortfolioUpstreamAuthorityError,
    load_exact_risk_input_binding,
)

type FloatArray = npt.NDArray[np.float64]


_CAMPAIGN_COVARIANCE_LANE_SEAL = object()


@dataclass(frozen=True, slots=True)
class CampaignValidatedCovarianceLane:
    """Campaign-owned proof for one projected immutable covariance lane."""

    values: FloatArray
    _seal: object = field(repr=False)

    def owns(self, covariance: FloatArray) -> bool:
        return bool(
            self._seal is _CAMPAIGN_COVARIANCE_LANE_SEAL
            and covariance.shape == self.values.shape[-2:]
            and np.shares_memory(covariance, self.values)
        )


@dataclass(frozen=True, slots=True)
class ResolvedStageSixCovariance:
    """The Stage 5 selected method, rematerialized on the Portfolio formations.

    The matrices are projected chunk by chunk rather than held. A 494-formation
    surface over 466 listings is 858 MB of float64 and the decision-axis
    projection is another 848 MB; materialising both at once is a gigabyte of
    duplicated evidence for the sake of an intermediate nobody reads.
    """

    evidence_root: Path
    surface: RiskDevelopmentCovarianceSurface
    input_binding: RiskInputAuthority
    formation_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]

    @property
    def surface_hash(self) -> str:
        """Which Risk surface this is, so a consumer can refuse a different one."""
        return str(self.surface.surface_hash)

    @property
    def input_binding_hash(self) -> str:
        """The immutable Risk input set this exact surface declares."""
        return str(self.surface.input_binding_hash)

    def project(
        self, *, sessions: tuple[date, ...], positions: npt.NDArray[np.int64]
    ) -> CampaignValidatedCovarianceLane:
        """The ordered principal submatrix at ``positions``, for each named session.

        A symmetric permutation of a principal submatrix, so the result stays
        principal and stays positive semi-definite whatever order the decision
        axis is in. Nothing is zero-filled: a session this surface does not carry
        is a refusal.
        """
        wanted = {value: index for index, value in enumerate(sessions)}
        if len(wanted) != len(sessions):
            raise PortfolioUpstreamAuthorityError(
                "portfolio_strategy_lab.stage_six_covariance_axis_duplicated"
            )
        store = RiskArtifactStore(self.evidence_root)
        projected: FloatArray = np.empty(
            (len(sessions), positions.size, positions.size), dtype=np.float64
        )
        seen = 0
        for chunk in self.surface.chunks:
            if not any(value in wanted for value in chunk.formation_sessions):
                continue
            matrices = unpack_chunk_matrices(store=store, chunk=chunk)
            for offset, session in enumerate(chunk.formation_sessions):
                index = wanted.get(session)
                if index is None:
                    continue
                projected[index] = matrices[offset][np.ix_(positions, positions)]
                seen += 1
        if seen != len(sessions):
            raise PortfolioUpstreamAuthorityError(
                "portfolio_strategy_lab.stage_six_covariance_axis_incomplete"
            )
        projected = np.ascontiguousarray(projected, dtype=np.float64)
        if not np.isfinite(projected).all() or any(
            not np.allclose(value, value.T, rtol=0.0, atol=1e-12) for value in projected
        ):
            raise PortfolioUpstreamAuthorityError(
                "portfolio_strategy_lab.stage_six_covariance_not_symmetric"
            )
        projected.setflags(write=False)
        return CampaignValidatedCovarianceLane(
            values=projected,
            _seal=_CAMPAIGN_COVARIANCE_LANE_SEAL,
        )

    def covariance_lane(
        self, *, sessions: tuple[date, ...], listing_ids: tuple[str, ...]
    ) -> CampaignValidatedCovarianceLane:
        """The covariance on ``listing_ids`` at each named session, for a policy's solver.

        The ordered principal submatrix of `project`, the listing axis named by listing rather
        than by position; a listing the surface does not carry is a refusal.
        """
        index = {listing: position for position, listing in enumerate(self.ordered_listing_ids)}
        if len(set(listing_ids)) != len(listing_ids) or any(v not in index for v in listing_ids):
            raise PortfolioUpstreamAuthorityError(
                "portfolio_strategy_lab.risk_listing_axis_incomplete"
            )
        return self.project(
            sessions=sessions,
            positions=np.asarray([index[v] for v in listing_ids], dtype=np.int64),
        )

    def allocation_projections(
        self, *, sessions: tuple[date, ...], listing_ids: tuple[str, ...]
    ) -> tuple[RiskAllocationProjection, ...]:
        """Each named session's per-name volatility on ``listing_ids``, a sealed Risk lane.

        Read from each session's covariance diagonal, never a whole matrix held; the lane is the
        Risk owner's projection, whose own check (`verify_content`) proves its identity and
        refuses a variance that is not positive. A session or listing the surface does not
        carry is a refusal, never a gap filled.
        """
        index = {listing: position for position, listing in enumerate(self.ordered_listing_ids)}
        if len(set(listing_ids)) != len(listing_ids) or any(v not in index for v in listing_ids):
            raise PortfolioUpstreamAuthorityError(
                "portfolio_strategy_lab.risk_listing_axis_incomplete"
            )
        positions: npt.NDArray[np.int64] = np.asarray(
            [index[v] for v in listing_ids], dtype=np.int64
        )
        wanted = {value: position for position, value in enumerate(sessions)}
        lanes: list[RiskAllocationProjection | None] = [None] * len(sessions)
        store = RiskArtifactStore(self.evidence_root)
        for chunk in self.surface.chunks:
            if not any(value in wanted for value in chunk.formation_sessions):
                continue
            matrices = unpack_chunk_matrices(store=store, chunk=chunk)
            for offset, session in enumerate(chunk.formation_sessions):
                position = wanted.get(session)
                if position is None:
                    continue
                variance = np.ascontiguousarray(np.diagonal(matrices[offset])[positions])
                lanes[position] = _allocation_projection(
                    surface=self.surface,
                    session=session,
                    listing_ids=listing_ids,
                    volatility=np.sqrt(variance, dtype=np.float64),
                )
        found = tuple(lane for lane in lanes if lane is not None)
        if len(found) != len(sessions):
            raise PortfolioUpstreamAuthorityError(
                "portfolio_strategy_lab.risk_session_axis_incomplete"
            )
        return found


def _allocation_projection(
    *,
    surface: RiskDevelopmentCovarianceSurface,
    session: date,
    listing_ids: tuple[str, ...],
    volatility: FloatArray,
) -> RiskAllocationProjection:
    """One session's volatility lane in the Risk owner's projection, its identity then proved."""

    lane = np.ascontiguousarray(volatility, dtype=np.float64)
    lane.setflags(write=False)
    projection = RiskAllocationProjection(
        surface_hash=str(surface.surface_hash),
        recipe_hash=str(surface.recipe_hash),
        formation_session=session,
        ordered_listing_ids=listing_ids,
        per_name_volatility=lane,
        projection_hash=canonical_hash(
            {
                "kind": "RiskAllocationProjection",
                "recipe_hash": str(surface.recipe_hash),
                "formation_session": session.isoformat(),
                "ordered_listing_ids": list(listing_ids),
                "per_name_volatility": hashlib.sha256(
                    np.ascontiguousarray(lane, dtype="<f8").tobytes()
                ).hexdigest(),
            }
        ),
    )
    projection.verify_content()
    return projection


def load_stage_six_covariance(
    *, evidence_root: Path, surface_hash: str
) -> ResolvedStageSixCovariance:
    """Open the one rematerialized surface a published graph already named.

    There is nothing to resolve: the graph states which surface was read, and a
    replay that searched for one again could find a different answer than the run
    did.
    """
    store = RiskArtifactStore(evidence_root)
    surface = RiskDevelopmentCovarianceSurface(
        **store.load_json(
            category=DEVELOPMENT_SURFACE_CATEGORY,
            uri=store.uri(DEVELOPMENT_SURFACE_CATEGORY, surface_hash),
            identity_field="surface_hash",
        )
    )
    covered = tuple(session for chunk in surface.chunks for session in chunk.formation_sessions)
    if covered != tuple(surface.formation_sessions):
        raise PortfolioUpstreamAuthorityError(
            "portfolio_strategy_lab.stage_six_covariance_axis_invalid"
        )
    input_binding = load_exact_risk_input_binding(
        store=store, input_binding_hash=str(surface.input_binding_hash)
    )
    if tuple(input_binding.formation_sessions) != tuple(surface.formation_sessions) or tuple(
        input_binding.ordered_listing_ids
    ) != tuple(surface.ordered_listing_ids):
        raise PortfolioUpstreamAuthorityError(
            "portfolio_strategy_lab.stage_six_risk_input_axis_invalid"
        )
    return ResolvedStageSixCovariance(
        evidence_root=Path(evidence_root),
        surface=surface,
        input_binding=input_binding,
        formation_sessions=tuple(surface.formation_sessions),
        ordered_listing_ids=tuple(surface.ordered_listing_ids),
    )


_TRANSITION_SOURCES: Final[Mapping[str, str]] = {
    "portfolio_backtesting.execution": "capabilities/portfolio_backtesting/execution.py",
    "portfolio_backtesting.reference_marks": (
        "capabilities/portfolio_backtesting/reference_marks.py"
    ),
    "portfolio_backtesting.segments": "capabilities/portfolio_backtesting/segments.py",
    "portfolio_backtesting.state": "capabilities/portfolio_backtesting/state.py",
}
"""The four modules that carry or value a book, under stable ids not paths.

``reference_marks`` joined them because the reference valuation moved into it. A
closure that named only the other three would let the arithmetic deciding what
every optimizer measures turnover against change without moving the identity of
a single published Campaign.
"""


@dataclass(frozen=True, slots=True)
class ResolvedReferenceMark:
    """One Campaign's reference-mark authority, opened once and shared onward.

    Resolver and executor read the *same object*. The defect this closes is not
    hypothetical: the binding was sealed from a surface the resolver opened, and
    the segment loop then valued the reference by a rule of its own -- so a
    Campaign could seal ``MARKED_TO_MARKET_AT_CLOSE_T`` into a published identity
    while every number behind it came from the open proxy. Nothing at the seal
    could see that, because the numbers are produced two files away.

    ``lane`` and ``binding`` are therefore built together, from one manifest, and
    travel together. The executor asserts they are the pair it was given.
    """

    binding: PortfolioStateTransitionBinding
    surface: SessionMarkSurface
    lane: ReferenceMarkLane
    ordered_listing_ids: tuple[str, ...]
    marked_sessions: tuple[date, ...]
    """The two axes the marks were read on, so the executor can check them.

    The formation-to-entry projection is deliberately *not* carried beside them.
    It lives on the lane, which is the only thing that uses it, and a second copy
    here could only ever agree with that one or silently disagree.
    """


def resolve_portfolio_state_transition_binding(
    *,
    surface: SessionMarkSurface,
    rebalance_clock: RebalanceClockBinding,
    playpen_root: Path,
    execution_method_binding_hash: str | None,
) -> PortfolioStateTransitionBinding:
    """Bind an installed cadence to one exact Market-owned close-mark surface."""
    transition_hash = switched_source_identity(
        {
            component: source_root(playpen_root) / "alphalattice" / relative
            for component, relative in _TRANSITION_SOURCES.items()
        },
        semantic_owner="portfolio_backtesting",
        numerical_role="PORTFOLIO_STATE_TRANSITION",
    )
    return PortfolioStateTransitionBinding.create(
        transition_implementation_hash=transition_hash,
        execution_method_binding_hash=execution_method_binding_hash,
        rebalance_clock=rebalance_clock,
        reference_mark_method=MARKED_TO_MARKET_AT_CLOSE_T,
        mark_manifest_ref=surface.manifest_uri,
        mark_surface_hash=surface.surface_hash,
        mark_epoch_hash=surface.epoch.epoch_hash,
        mark_price_basis=surface.price_basis,
        mark_source_watermark_hash=surface.source_watermark_hash,
        mark_availability_policy_hash=surface.availability_policy_hash,
        mark_availability_policy_id=surface.availability_policy_id,
    )


def resolve_reference_mark_lane(
    *,
    store: SessionMarkArtifactStore,
    surface: SessionMarkSurface,
    binding: PortfolioStateTransitionBinding,
    entry_session_by_formation: Mapping[date, date],
    formation_sessions: tuple[date, ...],
    ordered_listing_ids: tuple[str, ...],
) -> ResolvedReferenceMark:
    """Read the verified marks and build the lane the run will actually use.

    The marks are read on the *decision* axis, not on an entry axis, because the
    lane values a book at the open of the session it is being decided at.

    Every block is re-verified on the way out by the store, so a Campaign that
    reaches this line is marking against bytes that still hash to what the
    manifest sealed.
    """
    marks = store.read_marks(surface, sessions=formation_sessions, listing_ids=ordered_listing_ids)
    lane = ReferenceMarkLane(
        method=binding.reference_mark_method,
        marks_by_session=dict(zip(formation_sessions, marks, strict=True)),
        entry_session_by_formation=dict(entry_session_by_formation),
        state_transition_binding_hash=binding.binding_hash,
    )
    # Existence, never adjacency. The decision axis is the *scored* axis, and
    # the installed one skips four sessions its Alpha purge removed -- so two
    # scored formations can be adjacent by index and days apart on the exchange
    # calendar. Adjacency is checked per carry segment, by the engine that
    # carries the state.
    lane.require_entry_projection(formation_sessions)
    return ResolvedReferenceMark(
        binding=binding,
        surface=surface,
        lane=lane,
        ordered_listing_ids=ordered_listing_ids,
        marked_sessions=formation_sessions,
    )


def project_reference_mark_lane(
    *,
    resolved: ResolvedReferenceMark,
    formation_sessions: tuple[date, ...],
    ordered_listing_ids: tuple[str, ...],
) -> ResolvedReferenceMark:
    """Project one verified mark read to an admitted consumer axis."""
    marks = resolved.lane.marks_by_session
    if (
        marks is None
        or resolved.marked_sessions != formation_sessions
        or resolved.lane.state_transition_binding_hash != resolved.binding.binding_hash
    ):
        raise PortfolioUpstreamAuthorityError(
            "portfolio_strategy_lab.reference_mark_projection_not_this_axis"
        )
    positions = {listing_id: index for index, listing_id in enumerate(resolved.ordered_listing_ids)}
    try:
        columns: npt.NDArray[np.int64] = np.asarray(
            [positions[listing_id] for listing_id in ordered_listing_ids], dtype=np.int64
        )
        projected: dict[date, FloatArray] = {}
        for session in formation_sessions:
            values = np.ascontiguousarray(
                np.asarray(marks[session], dtype=np.float64)[columns], dtype=np.float64
            )
            values.setflags(write=False)
            projected[session] = values
    except (KeyError, IndexError) as error:
        raise PortfolioUpstreamAuthorityError(
            "portfolio_strategy_lab.reference_mark_projection_not_this_axis"
        ) from error
    lane = ReferenceMarkLane(
        method=resolved.lane.method,
        marks_by_session=projected,
        entry_session_by_formation=resolved.lane.entry_session_by_formation,
        state_transition_binding_hash=resolved.lane.state_transition_binding_hash,
    )
    lane.require_entry_projection(formation_sessions)
    return ResolvedReferenceMark(
        binding=resolved.binding,
        surface=resolved.surface,
        lane=lane,
        ordered_listing_ids=ordered_listing_ids,
        marked_sessions=formation_sessions,
    )


__all__ = [
    "ResolvedReferenceMark",
    "ResolvedStageSixCovariance",
    "load_stage_six_covariance",
    "project_reference_mark_lane",
    "resolve_portfolio_state_transition_binding",
]
