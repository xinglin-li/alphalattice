"""Pure numerical owner for robust sector-neutral Panel materialization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from typing import cast

import numpy as np
import pandas as pd

from alphalattice.foundation.feature_engine.contracts import (
    FeaturePanelBinding,
    PanelAdmissionSummary,
    canonical_hash,
)
from alphalattice.kernel.quant.cross_section import (
    MAD_SCALE,
    MIN_COVERAGE,
    MIN_SECTOR_SAMPLE,
    WINSOR_MULTIPLIER,
    equal_sector_demean,
    median_mad_winsor,
    robust_zscore,
)

SMALL_SECTOR_WARNING_BELOW = 10
PANEL_SESSION_BATCH_SIZE = 126
SESSION_LOCAL_POLICY_HASH = "d0379323f98f2d76d6e9ee90360c5db4601cee8a8f3a93a7dae8c3b5ea52f230"
"""Durable numerical/coverage identity; replay must not follow a future policy default."""
SOURCE_ELIGIBILITY_POLICY_HASH = "73adfa83c69789328c25427640e20686971881b108f778067cdeb358b3831373"


def allows_missing_source_rows(policy_hash: str) -> bool:
    """Report whether the recorded policy assesses missing base rows by finite coverage.

    Args:
        policy_hash: Recorded numerical policy authority of the Panel.

    Returns:
        Whether the policy is one of the installed session-local or source-eligibility policies.
    """
    return policy_hash in {SESSION_LOCAL_POLICY_HASH, SOURCE_ELIGIBILITY_POLICY_HASH}


def _array_digest(values: np.ndarray) -> str:
    """Content identity of one float matrix, NaN-stable and shape-bearing.

    ``tobytes`` alone would let two differently shaped arrays with the same
    buffer share an identity, and NaN payload bits are not guaranteed stable, so
    the mask travels separately from the finite values.
    """
    finite = np.isfinite(values)
    return cast(
        str,
        canonical_hash(
            {
                "kind": "PanelArrayDigest",
                "dtype": str(values.dtype),
                "shape": list(values.shape),
                "finite_mask": sha256(np.ascontiguousarray(finite).tobytes()).hexdigest(),
                "finite_values": sha256(
                    np.ascontiguousarray(np.where(finite, values, 0.0), dtype=np.float64).tobytes()
                ).hexdigest(),
            }
        ),
    )


def cross_section_policy_hash() -> str:
    """Hash the installed transformation, coverage, and row-identity policy.

    Returns:
        Canonical policy identity for ordered winsorization, equal-sector demeaning,
        robust Z scaling, coverage and sector thresholds, source eligibility, and row hashing.
    """
    return cast(
        str,
        canonical_hash(
            {
                "sequence": ["median_mad_winsor", "equal_sector_demean", "global_robust_zscore"],
                "mad_scale": MAD_SCALE,
                "winsor_multiplier": WINSOR_MULTIPLIER,
                "minimum_coverage": MIN_COVERAGE,
                "minimum_sector_sample": MIN_SECTOR_SAMPLE,
                "small_sector_warning_below": SMALL_SECTOR_WARNING_BELOW,
                "panel_row_hash": "duckdb_struct_json_sha256",
                "partial_universe": "fail_closed",
                "sector_reduction": "ordered_member_accumulation",
                "missing_base_rows": "finite_factor_coverage",
                "source_eligibility": "dated_exclusions_nominal_coverage_reference_sector_groups",
            }
        ),
    )


@dataclass(frozen=True)
class PanelFactorClipObservation:
    """What the winsor step changed for one factor, measured while it ran."""

    factor_id: str
    finite_input_count: int
    per_session_finite_counts: tuple[int, ...]
    per_session_clipped_counts: tuple[int, ...]
    boundary_identity: str


@dataclass(frozen=True)
class PanelMaterialization:
    """Carry numerical Panel output and the observations needed to seal its evidence.

    Attributes:
        rows: Materialized listing-session values.
        availability: Per-factor session coverage, qualification, and reasons.
        binding: Recorded Panel lineage and policy authority.
        receipt_hash: Identity of this materialization's receipt projection.
        complete: Completion decision of the selected numerical method.
        admission: As-of research admission summary.
        clip_observations: Per-factor clipping measurements taken during transformation.
        raw_input_identity: Identity of the ordered source arrays.
        transformed_identity: Identity of the ordered output arrays.
        ordered_sessions_hash: Identity of the session calculation axis.
        ordered_listing_ids_hash: Identity of the listing calculation axis.
    """

    rows: pd.DataFrame
    availability: list[dict[str, object]]
    binding: FeaturePanelBinding
    receipt_hash: str
    complete: bool
    admission: PanelAdmissionSummary
    clip_observations: tuple[PanelFactorClipObservation, ...] = ()
    """Per-factor clipping facts, measured inside the transformation.

    Observations rather than a sealed receipt: this owner knows what it clipped,
    but binding that to an installed recipe and a Panel identity is the
    producer's authority, not the numerical owner's.
    """

    raw_input_identity: str = ""
    transformed_identity: str = ""
    ordered_sessions_hash: str = ""
    ordered_listing_ids_hash: str = ""


class PanelCrossSectionKernel:
    """Materialize a Panel from explicit axes, sectors, base values and binding."""

    def materialize(
        self,
        *,
        feature_rows: list[dict[str, object]] | pd.DataFrame,
        active_listing_ids: tuple[str, ...],
        sector_by_listing_id: dict[str, str],
        factor_ids: tuple[str, ...],
        binding: FeaturePanelBinding,
        members_by_session: Mapping[date, Sequence[str]] | None = None,
        source_exclusions_by_session: Mapping[date, Sequence[str]] | None = None,
        legacy_epoch_session_counts: Mapping[date, int] | None = None,
    ) -> PanelMaterialization:
        """Transform every session's cross-section over its own members.

        ``active_listing_ids`` is the calculation axis, in calculation order.
        ``members_by_session`` names the axis listings each session's
        cross-section holds; absent, every session holds the whole axis.
        Sessions with the same members form an epoch and are transformed as
        one dense block of exactly those members, so a session's numbers are
        the numbers a Panel whose whole axis was that member set would
        produce -- what lets a partition computed under one membership be
        reused, bit for bit, by a build whose axis has since grown. A row is
        emitted for every member of a session and for nothing else: a listing
        that is not a member has no row (not a missing value), and a member
        whose base row is missing is assessed under the recorded policy:
        current builds count it against coverage; historical builds retain
        the complete-base-grid precondition.

        Args:
            feature_rows: Listing-session source feature values.
            active_listing_ids: Declared calculation axis in execution order.
            sector_by_listing_id: Sector authority for every axis listing.
            factor_ids: Ordered factor scope to transform.
            binding: Panel lineage and recorded numerical policy authority.
            members_by_session: Nominal session members, or the full uniform axis when omitted.
            source_exclusions_by_session: Evidenced exclusions by session, if present.
            legacy_epoch_session_counts: Recorded historical dense batch layout for policy replay.

        Returns:
            Transformed rows, availability and admission, measured clipping observations,
                and ordered input/output identities bound to the supplied Panel lineage.

        Raises:
            KeyError: A required listing, session, or factor source column is absent.
            ValueError: Rows are empty or duplicated, axes or sectors are invalid, temporal
                membership is incomplete, or source eligibility contradicts the recorded policy.
        """
        if isinstance(feature_rows, pd.DataFrame):
            is_empty = feature_rows.empty
        else:
            is_empty = not feature_rows
        if is_empty:
            raise ValueError("sector panel requires feature rows")
        if set(active_listing_ids) - set(sector_by_listing_id):
            raise ValueError("sector panel has no complete active sector revision")
        if not factor_ids or factor_ids != tuple(sorted(set(factor_ids))):
            raise ValueError("sector panel factor IDs must be sorted and unique")
        frame = (
            feature_rows.copy()
            if isinstance(feature_rows, pd.DataFrame)
            else pd.DataFrame(feature_rows)
        )
        frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.date
        frame = frame.loc[frame["listing_id"].isin(active_listing_ids)].copy()
        sessions = tuple(sorted(frame["session_date"].unique()))
        listing_positions = {
            listing_id: position for position, listing_id in enumerate(active_listing_ids)
        }
        universe = len(active_listing_ids)
        session_local = allows_missing_source_rows(binding.policy_hash)
        if legacy_epoch_session_counts is not None and (
            session_local
            or set(legacy_epoch_session_counts) != set(sessions)
            or any(count < 1 for count in legacy_epoch_session_counts.values())
        ):
            raise ValueError("feature_panel.legacy_replay_layout_invalid")
        member_mask: np.ndarray = np.ones((len(sessions), universe), dtype=bool)
        if members_by_session is not None:
            member_mask[:] = False
            for index, session in enumerate(sessions):
                try:
                    members = members_by_session[session]
                except KeyError as exc:
                    raise ValueError(
                        f"sector panel session {session.isoformat()} has no membership"
                    ) from exc
                try:
                    member_positions = [listing_positions[listing_id] for listing_id in members]
                except KeyError as exc:
                    raise ValueError(
                        f"sector panel member {exc.args[0]} is outside the calculation axis"
                    ) from exc
                if not member_positions or len(set(member_positions)) != len(member_positions):
                    raise ValueError("sector panel session members must be non-empty and unique")
                member_mask[index, member_positions] = True
        reference_mask = member_mask.copy()
        if source_exclusions_by_session:
            if binding.policy_hash != SOURCE_ELIGIBILITY_POLICY_HASH:
                raise ValueError("feature_panel.source_eligibility_policy_unavailable")
            for index, session in enumerate(sessions):
                for listing in source_exclusions_by_session.get(session, ()):
                    if listing not in listing_positions:
                        raise ValueError("feature_panel.source_exclusion_outside_axis")
                    reference_mask[index, listing_positions[listing]] = False
        if not reference_mask.any(axis=1).all():
            raise ValueError("feature_panel.reference_population_empty")
        epochs = _membership_epochs(reference_mask)
        epoch_sectors = {
            start: _sector_positions(
                tuple(active_listing_ids[position] for position in positions),
                sector_by_listing_id,
            )
            for start, _stop, positions in epochs
        }
        grid = pd.MultiIndex.from_product(
            (sessions, active_listing_ids), names=("session_date", "listing_id")
        )
        indexed = frame.set_index(["session_date", "listing_id"])
        if not indexed.index.is_unique:
            raise ValueError("sector panel feature rows contain duplicate listing sessions")
        aligned = indexed.reindex(grid)
        present_cells: np.ndarray = np.asarray(grid.isin(indexed.index), dtype=bool).reshape(
            len(sessions), universe
        )
        present = (present_cells | ~member_mask).all(axis=1)
        factor_cube = (
            aligned.loc[:, list(factor_ids)]
            .to_numpy(dtype=float)
            .reshape(len(sessions), universe, len(factor_ids))
        )
        # A non-finite raw value is missing, as NaN is, in every session statistic -- the
        # median, MAD, winsor and coverage alike -- so it never moves another name's value
        # (V517). The producers screen their own; the kernel does not rely on them.
        factor_cube = np.where(np.isfinite(factor_cube), factor_cube, np.nan)
        universe_size = member_mask.sum(axis=1)
        member_sessions, member_positions = np.nonzero(member_mask)
        rows = pd.DataFrame(
            {
                "listing_id": np.asarray(active_listing_ids, dtype=object)[member_positions],
                "session_date": np.asarray(
                    [session.isoformat() for session in sessions], dtype=object
                )[member_sessions],
            }
        )
        availability: list[dict[str, object]] = []
        clip_observations: list[PanelFactorClipObservation] = []
        raw_input_digests: list[tuple[str, str]] = []
        transformed_digests: list[tuple[str, str]] = []
        for factor_position, factor_id in enumerate(factor_ids):
            matrix = np.array(factor_cube[:, :, factor_position], dtype=float)
            matrix[~reference_mask] = np.nan
            valid = np.isfinite(matrix)
            computed = valid.sum(axis=1)
            coverage = computed / universe_size
            median = np.full(len(sessions), np.nan)
            mad = np.full(len(sessions), np.nan)
            lower = np.full(len(sessions), np.nan)
            upper = np.full(len(sessions), np.nan)
            residual_median = np.full(len(sessions), np.nan)
            residual_mad = np.full(len(sessions), np.nan)
            score = np.full((len(sessions), universe), np.nan)
            clipped_mask: np.ndarray = np.zeros((len(sessions), universe), dtype=bool)
            small_sector: np.ndarray = np.zeros(len(sessions), dtype=bool)
            small_sector_warning: np.ndarray = np.zeros(len(sessions), dtype=bool)
            sector_counts_by_session: list[dict[str, int]] = [{} for _ in sessions]
            for start, stop, positions in epochs:
                sectors = epoch_sectors[start]
                block = matrix[start:stop][:, positions]
                block_valid = np.isfinite(block)
                for sector, compact in sectors.items():
                    count = block_valid[:, compact].sum(axis=1)
                    small_sector[start:stop] |= count < MIN_SECTOR_SAMPLE
                    small_sector_warning[start:stop] |= (count >= MIN_SECTOR_SAMPLE) & (
                        count < SMALL_SECTOR_WARNING_BELOW
                    )
                    for offset, value in enumerate(count):
                        sector_counts_by_session[start + offset][sector] = int(value)
                winsor, block_median, block_mad, block_lower, block_upper = median_mad_winsor(block)
                median[start:stop] = block_median
                mad[start:stop] = block_mad
                lower[start:stop] = block_lower
                upper[start:stop] = block_upper
                # Measured here, where both sides of the clip exist at once. A
                # later recomputation from published rows could not tell a
                # clipped value from one that merely sat at the boundary.
                block_clipped = block_valid & (
                    (block < block_lower[:, None]) | (block > block_upper[:, None])
                )
                clipped_view = clipped_mask[start:stop]
                clipped_view[:, positions] = block_clipped
                residual = (
                    equal_sector_demean(
                        winsor,
                        sector_positions=tuple(sectors.values()),
                        session_local=session_local,
                    )
                    if legacy_epoch_session_counts is None
                    else np.full_like(winsor, np.nan)
                )
                if legacy_epoch_session_counts is not None:
                    # A partial legacy receipt can collapse a formerly strided
                    # epoch to one row, or merge retained rows from separate
                    # epochs. Reproduce its proved original layout per row.
                    strided: np.ndarray = np.asarray(
                        [legacy_epoch_session_counts[day] > 1 for day in sessions[start:stop]],
                        dtype=bool,
                    )
                    if strided.any():
                        residual[strided] = equal_sector_demean(
                            winsor[strided],
                            sector_positions=tuple(sectors.values()),
                            session_local=True,
                        )
                    for row in np.flatnonzero(~strided):
                        residual[row : row + 1] = equal_sector_demean(
                            winsor[row : row + 1], sector_positions=tuple(sectors.values())
                        )
                block_score, block_residual_median, block_residual_mad = robust_zscore(residual)
                residual_median[start:stop] = block_residual_median
                residual_mad[start:stop] = block_residual_mad
                score_view = score[start:stop]
                score_view[:, positions] = block_score
            clip_observations.append(
                PanelFactorClipObservation(
                    factor_id=factor_id,
                    finite_input_count=int(valid.sum()),
                    per_session_finite_counts=tuple(int(value) for value in valid.sum(axis=1)),
                    per_session_clipped_counts=tuple(
                        int(value) for value in clipped_mask.sum(axis=1)
                    ),
                    boundary_identity=cast(
                        str,
                        canonical_hash(
                            {
                                "kind": "PanelFactorClipBoundaries",
                                "factor_id": factor_id,
                                "bounds": [
                                    (
                                        float(lower[index]) if np.isfinite(lower[index]) else None,
                                        float(upper[index]) if np.isfinite(upper[index]) else None,
                                    )
                                    for index in range(len(sessions))
                                ],
                            }
                        ),
                    ),
                )
            )
            raw_input_digests.append((factor_id, _array_digest(matrix)))
            available = (
                (present | session_local)
                & (coverage >= MIN_COVERAGE)
                & ~small_sector
                & np.isfinite(mad)
                & (mad != 0.0)
                & np.isfinite(residual_mad)
                & (residual_mad != 0.0)
            )
            score[~available, :] = np.nan
            for index, session in enumerate(sessions):
                if not present[index] and not session_local:
                    reason = "active_manifest_base_row_missing"
                elif coverage[index] < MIN_COVERAGE:
                    reason = "coverage_below_98_percent"
                elif small_sector[index]:
                    reason = "sector_sample_below_5"
                elif not np.isfinite(mad[index]) or mad[index] == 0.0:
                    reason = "feature_mad_zero"
                elif not np.isfinite(residual_mad[index]) or residual_mad[index] == 0.0:
                    reason = "residual_mad_zero"
                else:
                    reason = None
                sector_counts = sector_counts_by_session[index]
                availability.append(
                    {
                        "session_date": session.isoformat(),
                        "factor_id": factor_id,
                        "universe_size": int(universe_size[index]),
                        "computed_count": int(computed[index]),
                        "coverage": float(coverage[index]),
                        "sector_counts": dict(sector_counts),
                        "winsor_lower": float(lower[index]) if np.isfinite(lower[index]) else None,
                        "winsor_upper": float(upper[index]) if np.isfinite(upper[index]) else None,
                        "residual_median": (
                            float(residual_median[index])
                            if np.isfinite(residual_median[index])
                            else None
                        ),
                        "residual_mad": (
                            float(residual_mad[index]) if np.isfinite(residual_mad[index]) else None
                        ),
                        "status": "available" if available[index] else "unavailable",
                        "reason": reason,
                        "small_sector_warning": bool(
                            available[index] and small_sector_warning[index]
                        ),
                        "small_sector_names": tuple(
                            sector
                            for sector, count in sector_counts.items()
                            if MIN_SECTOR_SAMPLE <= count < SMALL_SECTOR_WARNING_BELOW
                        ),
                        "panel_binding_hash": binding.panel_binding_hash,
                    }
                )
            transformed_digests.append((factor_id, _array_digest(score)))
            member_scores = score[member_mask]
            rows[factor_id] = pd.Series(member_scores).where(np.isfinite(member_scores), np.nan)
        latest_session = max(cast(str, item["session_date"]) for item in availability)
        sector_distribution = {
            sector: len(compact) for sector, compact in epoch_sectors[epochs[-1][0]].items()
        }
        admission = PanelAdmissionSummary.evaluate(
            as_of_session=pd.Timestamp(latest_session).date(),
            factor_ids=factor_ids,
            availability=availability,
            sector_distribution=sector_distribution,
        )
        complete = admission.research_admissible
        receipt_hash = canonical_hash(
            {
                "binding": binding.panel_binding_hash,
                "row_count": len(rows),
                "availability": [
                    {
                        key: value
                        for key, value in item.items()
                        if key not in {"sector_counts", "small_sector_names"}
                    }
                    for item in availability
                ],
                "admission": admission.summary_hash(),
            }
        )
        ordered_listing_ids_hash = panel_ordered_members_identity(
            active_listing_ids, sessions, epochs, uniform=bool(member_mask.all())
        )
        return PanelMaterialization(
            rows=rows,
            availability=availability,
            binding=binding,
            receipt_hash=receipt_hash,
            complete=complete,
            admission=admission,
            clip_observations=tuple(clip_observations),
            raw_input_identity=cast(
                str,
                canonical_hash({"kind": "PanelRawInput", "factors": raw_input_digests}),
            ),
            transformed_identity=cast(
                str,
                canonical_hash({"kind": "PanelTransformed", "factors": transformed_digests}),
            ),
            ordered_sessions_hash=cast(
                str,
                canonical_hash([session.isoformat() for session in sessions]),
            ),
            ordered_listing_ids_hash=ordered_listing_ids_hash,
        )


def _membership_epochs(member_mask: np.ndarray) -> tuple[tuple[int, int, np.ndarray], ...]:
    """Contiguous session ranges with one member set: (start, stop, axis positions)."""
    epochs: list[tuple[int, int, np.ndarray]] = []
    start = 0
    for index in range(1, len(member_mask) + 1):
        if index == len(member_mask) or not np.array_equal(member_mask[index], member_mask[start]):
            epochs.append((start, index, np.flatnonzero(member_mask[start])))
            start = index
    return tuple(epochs)


def panel_ordered_members_identity(
    axis: tuple[str, ...],
    sessions: tuple[date, ...],
    epochs: tuple[tuple[int, int, np.ndarray], ...],
    *,
    uniform: bool,
) -> str:
    """The batch's existing ordered-axis identity, shared by writer and replay."""
    identity = (
        canonical_hash(list(axis))
        if uniform
        else canonical_hash(
            {
                "kind": "PanelOrderedMembers",
                "axis": list(axis),
                "epochs": [
                    [
                        sessions[start].isoformat(),
                        sessions[stop - 1].isoformat(),
                        [int(position) for position in positions],
                    ]
                    for start, stop, positions in epochs
                ],
            }
        )
    )
    return cast(str, identity)


def _sector_positions(
    members: tuple[str, ...], sector_by_listing_id: Mapping[str, str]
) -> dict[str, np.ndarray]:
    """Each sector's members as positions into the epoch's compact member block."""
    return {
        sector: np.array(
            [
                position
                for position, listing_id in enumerate(members)
                if sector_by_listing_id[listing_id] == sector
            ],
            dtype=int,
        )
        for sector in sorted({sector_by_listing_id[listing_id] for listing_id in members})
    }


__all__ = [
    "PANEL_SESSION_BATCH_SIZE",
    "SMALL_SECTOR_WARNING_BELOW",
    "PanelCrossSectionKernel",
    "PanelFactorClipObservation",
    "PanelMaterialization",
    "cross_section_policy_hash",
    "panel_ordered_members_identity",
]
