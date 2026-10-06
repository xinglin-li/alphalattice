"""Causal sector event aggregation and compact context surface construction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, time
from hashlib import sha256
from math import exp, log
from typing import cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from alphalattice.kernel.quant.cross_section import (
    median_mad_winsor,
)
from alphalattice.kernel.quant.sector_history import sector_positions, sector_treatment_of
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.sector_treatment import (
    SECTOR_HISTORY_BACKFILLED,
    SectorHistoryTreatment,
)

from .contracts import (
    SectorContextManifest,
    SectorContextPolicy,
    seal_sector_context_contract,
)

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]

_REQUIRED = frozenset(
    {
        "formation_session",
        "formation_close_at",
        "holding_end_open_at",
        "listing_id",
        "fit_target",
    }
)


class SectorContextBoundaryError(ValueError):
    """Stable fail-closed sector context boundary."""


@dataclass(frozen=True, slots=True)
class SectorContextSurface:
    """Hold a validated context manifest and its read-only four-feature tensor.

    values has shape (formation sessions, sector identifiers, 4), in the exact manifest axis order.
    Construction checks shape and the non-writeable array flag.
    """

    manifest: SectorContextManifest
    values: FloatArray

    def __post_init__(self) -> None:
        """Require the manifest-aligned four-feature shape and a read-only array.

        Raises:
            SectorContextBoundaryError: The tensor shape or writeability differs from the context
                contract.
        """
        expected = (
            len(self.manifest.formation_sessions),
            len(self.manifest.sector_ids),
            4,
        )
        if self.values.shape != expected or self.values.flags.writeable:
            raise SectorContextBoundaryError("alpha_research.sector_context_shape_invalid")


def build_sector_context_policy(
    *,
    sector_revision: str,
    sector_history_treatment: SectorHistoryTreatment = SECTOR_HISTORY_BACKFILLED,
) -> SectorContextPolicy:
    """Seal the installed context constants against an exact classification revision.

    Args:
        sector_revision: Classification revision identity used by the context.
        sector_history_treatment: What its sessions read of the classification (V346).

    Returns:
        Fixed context policy and its canonical policy_hash.

    Raises:
        pydantic.ValidationError: The classification revision violates the policy contract.
    """
    return seal_sector_context_contract(
        SectorContextPolicy,
        "policy_hash",
        sector_revision=sector_revision,
        sector_history_treatment=sector_history_treatment,
    )


def _ordered(source: pa.Table) -> pa.Table:
    missing = sorted(_REQUIRED - set(source.schema.names))
    if missing:
        raise SectorContextBoundaryError(
            f"alpha_research.sector_context_required_column_missing:{','.join(missing)}"
        )
    ordered = source.take(
        pc.sort_indices(
            source,
            sort_keys=[("formation_session", "ascending"), ("listing_id", "ascending")],
        )
    ).combine_chunks()
    if ordered.num_rows == 0:
        raise SectorContextBoundaryError("alpha_research.sector_context_source_empty")
    if ordered.num_rows > 1:
        duplicate = pc.and_(
            pc.equal(
                ordered["formation_session"].slice(1),
                ordered["formation_session"].slice(0, ordered.num_rows - 1),
            ),
            pc.equal(
                ordered["listing_id"].slice(1),
                ordered["listing_id"].slice(0, ordered.num_rows - 1),
            ),
        )
        if bool(pc.any(duplicate).as_py()):
            raise SectorContextBoundaryError("alpha_research.sector_context_duplicate_row")
    return ordered


def _table(values: FloatArray, sessions: tuple[date, ...], sectors: tuple[str, ...]) -> pa.Table:
    flattened = values.reshape(-1, 4)
    columns: dict[str, pa.Array] = {
        "formation_session": pa.array(
            np.repeat(np.asarray(sessions, dtype=object), len(sectors)), type=pa.date32()
        ),
        "sector_id": pa.array(np.tile(np.asarray(sectors, dtype=object), len(sessions))),
    }
    for index, name in enumerate(
        ("sector_trend_20", "sector_surprise_0", "sector_surprise_1", "sector_surprise_5")
    ):
        column = flattened[:, index]
        columns[name] = pa.array(column, mask=~np.isfinite(column), type=pa.float64())
    return pa.table(columns).combine_chunks()


def _payload(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd", use_dictionary=False)
    return cast(bytes, sink.getvalue().to_pybytes())


def compile_sector_context_arrays(
    *,
    sessions: tuple[date, ...],
    holding_end_sessions: tuple[date, ...],
    listing_ids: tuple[str, ...],
    raw_log_returns: FloatArray,
    target_evidence_hash: str,
    sector_by_listing_id: Mapping[str, str],
    sector_revision: str,
    output_sessions: tuple[date, ...] | None = None,
    reference_eligible: npt.NDArray[np.bool_] | None = None,
) -> tuple[FloatArray, str]:
    """Compose known execution observations onto a possibly later formation axis."""
    formation_values = tuple(datetime.combine(value, time(16, 0)) for value in sessions)
    holding_values = tuple(datetime.combine(value, time(9, 30)) for value in holding_end_sessions)
    source = pa.table(
        {
            "formation_session": pa.array(
                np.repeat(np.asarray(sessions, dtype=object), len(listing_ids)), type=pa.date32()
            ),
            "formation_close_at": pa.array(
                np.repeat(np.asarray(formation_values, dtype=object), len(listing_ids)),
                type=pa.timestamp("us"),
            ),
            "holding_end_open_at": pa.array(
                np.repeat(np.asarray(holding_values, dtype=object), len(listing_ids)),
                type=pa.timestamp("us"),
            ),
            "listing_id": pa.array(np.tile(np.asarray(listing_ids, dtype=object), len(sessions))),
            # This existing field is LOG_EXECUTION_RETURN, never Target-Z.
            "fit_target": pa.array(raw_log_returns.reshape(-1), type=pa.float64()),
        }
    )
    surface, _table, _payload = compile_sector_context_surface(
        source_table=source,
        source_surface_hash=target_evidence_hash,
        policy=build_sector_context_policy(
            sector_revision=sector_revision,
            sector_history_treatment=sector_treatment_of(sector_by_listing_id),
        ),
        sector_by_listing_id=sector_by_listing_id,
        formation_sessions=output_sessions or sessions,
        reference_eligible=reference_eligible,
    )
    return surface.values, surface.manifest.manifest_hash


def _sector_event_returns(
    source: FloatArray,
    positions: tuple[IntArray, ...],
    *,
    minimum_members: int,
    stable_reduction: bool = False,
) -> FloatArray:
    protected, _median, _mad, _lower, _upper = median_mad_winsor(source)
    result: FloatArray = np.full((len(source), len(positions)), np.nan, dtype=np.float64)
    for index, members in enumerate(positions):
        block = protected[:, members]
        if stable_reduction:
            block = np.ascontiguousarray(block)
        counts = np.isfinite(block).sum(axis=1)
        means = np.nanmean(block, axis=1)
        admitted = counts >= minimum_members
        result[admitted, index] = means[admitted]
    return result


def compile_sector_context_surface(
    *,
    source_table: pa.Table,
    source_surface_hash: str,
    policy: SectorContextPolicy,
    sector_by_listing_id: Mapping[str, str],
    formation_sessions: tuple[date, ...] | None = None,
    reference_eligible: npt.NDArray[np.bool_] | None = None,
) -> tuple[SectorContextSurface, pa.Table, bytes]:
    """Publish only context observable by each formation close."""
    policy = SectorContextPolicy.model_validate(policy)
    ordered = _ordered(source_table)
    row_sessions = tuple(cast(list[date], ordered["formation_session"].to_pylist()))
    row_listings = tuple(str(value) for value in ordered["listing_id"].to_pylist())
    event_sessions = tuple(sorted(set(row_sessions)))
    output_sessions = formation_sessions or event_sessions
    if output_sessions != tuple(sorted(set(output_sessions))):
        raise SectorContextBoundaryError("alpha_research.sector_context_session_axis_invalid")
    listings = tuple(sorted(set(row_listings)))
    if ordered.num_rows != len(event_sessions) * len(listings):
        raise SectorContextBoundaryError("alpha_research.sector_context_source_incomplete")
    if set(listings) - set(sector_by_listing_id):
        raise SectorContextBoundaryError("alpha_research.sector_context_sector_map_incomplete")
    # Each run of event sessions aggregates the Sectors in force there (V346); the Sector axis
    # is every Sector some session reads, one run's while no reclassification falls inside it.
    runs = sector_positions(sector_by_listing_id, event_sessions, listings)
    sectors = tuple(
        sorted({sector for _rows, run_sectors, _members in runs for sector in run_sectors})
    )
    empty: IntArray = np.asarray([], dtype=np.int64)
    run_positions: list[tuple[slice, tuple[IntArray, ...]]] = []
    for rows, run_sectors, members_by_sector in runs:
        held = dict(zip(run_sectors, members_by_sector, strict=True))
        run_positions.append((rows, tuple(held.get(sector, empty) for sector in sectors)))
    source = np.asarray(
        ordered["fit_target"].to_numpy(zero_copy_only=False), dtype=np.float64
    ).reshape(len(event_sessions), len(listings))
    if bool(np.any(np.isinf(source))):
        raise SectorContextBoundaryError("alpha_research.sector_context_source_nonfinite")
    sector_returns: FloatArray = np.full(
        (len(event_sessions), len(sectors)), np.nan, dtype=np.float64
    )
    if reference_eligible is None:
        for rows, positions in run_positions:
            sector_returns[rows] = _sector_event_returns(
                source[rows], positions, minimum_members=policy.minimum_sector_members
            )
    else:
        if reference_eligible.shape != source.shape or reference_eligible.dtype != np.bool_:
            raise SectorContextBoundaryError("alpha_research.sector_context_reference_axis_invalid")
        for rows, positions in run_positions:
            first = rows.start or 0
            masks, inverse = np.unique(reference_eligible[rows], axis=0, return_inverse=True)
            for index, mask in enumerate(masks):
                days: IntArray = first + np.flatnonzero(inverse == index)
                members = np.flatnonzero(mask)
                if not len(members):
                    continue
                sector_returns[days] = _sector_event_returns(
                    np.ascontiguousarray(source[np.ix_(days, members)]),
                    tuple(np.flatnonzero(np.isin(members, p)) for p in positions),
                    minimum_members=policy.minimum_sector_members,
                    stable_reduction=True,
                )
        source_surface_hash = str(
            canonical_hash(
                {
                    "source": source_surface_hash,
                    "sessions": event_sessions,
                    "listings": listings,
                    "reference_mask": sha256(
                        np.ascontiguousarray(reference_eligible).tobytes()
                    ).hexdigest(),
                    "reference_reduction": "COMPACT_SESSION_ROWS",
                }
            )
        )

    formation_clock = np.asarray(ordered["formation_close_at"].to_pylist(), dtype=object).reshape(
        len(event_sessions), len(listings)
    )
    availability_clock = np.asarray(
        ordered["holding_end_open_at"].to_pylist(), dtype=object
    ).reshape(len(event_sessions), len(listings))
    if any(
        not isinstance(value, datetime)
        for value in (*formation_clock.reshape(-1), *availability_clock.reshape(-1))
    ):
        raise SectorContextBoundaryError("alpha_research.sector_context_clock_invalid")
    if any(
        len(set(row)) != 1
        for matrix in (formation_clock, availability_clock)
        for row in matrix.tolist()
    ):
        raise SectorContextBoundaryError("alpha_research.sector_context_clock_mismatch")
    if any(
        left >= right
        for left, right in zip(
            formation_clock.reshape(-1), availability_clock.reshape(-1), strict=True
        )
    ):
        raise SectorContextBoundaryError("alpha_research.sector_context_causal_order_invalid")

    trend_decay = exp(log(0.5) / policy.trend_half_life_sessions)
    surprise_decay = exp(log(0.5) / policy.surprise_half_life_sessions)
    surprise_weights = np.power(
        surprise_decay, np.arange(policy.surprise_window_sessions - 1, -1, -1)
    )
    trend: FloatArray = np.zeros(len(sectors), dtype=np.float64)
    observations: IntArray = np.zeros(len(sectors), dtype=np.int64)
    surprises: list[list[float]] = [[] for _ in sectors]
    values: FloatArray = np.full((len(output_sessions), len(sectors), 4), np.nan, dtype=np.float64)
    next_event = 0
    available_ats = availability_clock[:, 0]
    for formation_index, formation_session in enumerate(output_sessions):
        while (
            next_event < len(event_sessions)
            and available_ats[next_event].date() <= formation_session
        ):
            event = sector_returns[next_event]
            for sector_index, item in enumerate(event):
                if not np.isfinite(item):
                    continue
                prior = trend[sector_index] if observations[sector_index] else float(item)
                surprise = float(item) - prior
                trend[sector_index] = (
                    float(item)
                    if observations[sector_index] == 0
                    else trend_decay * prior + (1.0 - trend_decay) * float(item)
                )
                observations[sector_index] += 1
                surprises[sector_index].append(surprise)
            next_event += 1
        for sector_index in range(len(sectors)):
            history = surprises[sector_index]
            if observations[sector_index] < policy.minimum_history_sessions or len(history) < 2:
                continue
            latest = history[-policy.surprise_window_sessions :]
            weights = surprise_weights[-len(latest) :]
            values[formation_index, sector_index] = (
                trend[sector_index],
                history[-1],
                history[-2],
                float(np.dot(weights, latest) / weights.sum()),
            )
    values.setflags(write=False)
    table = _table(values, output_sessions, sectors)
    payload = _payload(table)
    table_hash = str(canonical_hash({"schema": str(table.schema), "rows": table.to_pylist()}))
    manifest = seal_sector_context_contract(
        SectorContextManifest,
        "manifest_hash",
        policy_hash=policy.policy_hash,
        source_surface_hash=source_surface_hash,
        formation_sessions=output_sessions,
        sector_ids=sectors,
        available_row_count=int(np.isfinite(values).all(axis=2).sum()),
        table_content_hash=table_hash,
        payload_sha256=sha256(payload).hexdigest(),
    )
    return SectorContextSurface(manifest=manifest, values=values), table, payload


__all__ = [
    "SectorContextBoundaryError",
    "SectorContextSurface",
    "build_sector_context_policy",
    "compile_sector_context_surface",
]
