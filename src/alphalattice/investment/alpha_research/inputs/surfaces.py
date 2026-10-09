"""Bounded program-scoped Arrow/NumPy surfaces for Alpha Research."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from itertools import chain
from typing import cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc

from alphalattice.foundation.factor_research.inputs.execution_target import (
    build_factor_target_policy,
    build_factor_target_policy_for_method,
    compile_factor_target_surface,
)
from alphalattice.foundation.feature_engine.panels.reader import FeaturePanelReadRequest
from alphalattice.kernel.shared_kernel.identity import canonical_hash

from ..targets.execution_outcome import AlphaTargetPolicy, compile_alpha_target_surface
from .folds import (
    AlphaArrayBoundaryError,
    AlphaCurrentRefitArrays,
    AlphaFoldArrayPlan,
    AlphaFoldArrays,
    build_alpha_fold_commitment,
)
from .observations import (
    AlphaArrayReadLedger,
    AlphaArrayReadObservation,
    AlphaArrayReadObserver,
)
from .training import build_alpha_training_input_binding

type FloatArray = npt.NDArray[np.float64]
type BoolArray = npt.NDArray[np.bool_]
type IntArray = npt.NDArray[np.intp]
type OrdinalArray = npt.NDArray[np.int32]
type HashArray = npt.NDArray[np.bytes_]

_MIB = 1024 * 1024
_GIB = 1024 * _MIB
_MINIMUM_RECOMMENDED_PHYSICAL_MEMORY_BYTES = 16 * _GIB
_PREFERRED_PHYSICAL_MEMORY_BYTES = 32 * _GIB


def _readonly(value: npt.NDArray[np.generic]) -> npt.NDArray[np.generic]:
    result = np.asarray(value)
    result.setflags(write=False)
    return result


def _training_binding_source_hashes(
    plan: AlphaFoldArrayPlan,
) -> tuple[str, str, str] | None:
    """Return complete production identities; partial legacy fixtures stay unbound."""

    try:
        values = (
            plan.foundation.foundation_hash,
            plan.foundation.feature_panel_snapshot_hash,
            plan.foundation.execution_outcome.snapshot_hash,
        )
    except AttributeError:
        return None
    if any(not isinstance(value, str) or len(value) != 64 for value in values):
        return None
    return values


def _total_physical_memory_bytes() -> int:
    if os.name == "nt":
        import ctypes

        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return int(status.total_physical)
    raw_sysconf = getattr(os, "sysconf", None)
    if raw_sysconf is None:
        raise OSError("physical-memory discovery is unavailable")
    sysconf = cast(Callable[[str], int], raw_sysconf)
    page_size = sysconf("SC_PAGE_SIZE")
    page_count = sysconf("SC_PHYS_PAGES")
    return int(page_size * page_count)


@dataclass(frozen=True, slots=True)
class AlphaArraySurfaceBudget:
    """Declare memory admission and conservative surface/derived/live-fold estimates.

    The record separates program caching from bounded-fold execution. Cache and peak limits,
    physical-memory recommendations and estimated retained/live arrays are reconciled by
    construction.
    """

    total_physical_memory_bytes: int
    cache_cap_bytes: int
    peak_rss_limit_bytes: int
    minimum_recommended_physical_memory_bytes: int
    preferred_physical_memory_bytes: int
    meets_minimum_recommendation: bool
    meets_preferred_recommendation: bool
    estimated_surface_bytes: int
    estimated_retained_derived_bytes: int
    estimated_worst_live_fold_bytes: int
    estimated_peak_working_set_bytes: int
    mode: str

    def __post_init__(self) -> None:
        """Require admitted memory limits, recommendations and working-set accounting.

        Raises:
            AlphaArrayBoundaryError: Memory limits/recommendation flags, estimate signs/sums or
                execution mode violate the budget contract.
        """
        if (
            self.total_physical_memory_bytes < 1
            or not 512 * _MIB <= self.cache_cap_bytes <= 2 * _GIB
            or not 4 * _GIB <= self.peak_rss_limit_bytes <= 6 * _GIB
            or self.minimum_recommended_physical_memory_bytes
            != _MINIMUM_RECOMMENDED_PHYSICAL_MEMORY_BYTES
            or self.preferred_physical_memory_bytes != _PREFERRED_PHYSICAL_MEMORY_BYTES
            or self.meets_minimum_recommendation
            != (self.total_physical_memory_bytes >= _MINIMUM_RECOMMENDED_PHYSICAL_MEMORY_BYTES)
            or self.meets_preferred_recommendation
            != (self.total_physical_memory_bytes >= _PREFERRED_PHYSICAL_MEMORY_BYTES)
            or self.estimated_surface_bytes < 1
            or min(
                self.estimated_retained_derived_bytes,
                self.estimated_worst_live_fold_bytes,
                self.estimated_peak_working_set_bytes,
            )
            < 0
            or self.estimated_peak_working_set_bytes
            != (
                self.estimated_surface_bytes
                + self.estimated_retained_derived_bytes
                + self.estimated_worst_live_fold_bytes
            )
            or self.mode not in {"PROGRAM_CACHE", "BOUNDED_FOLD"}
        ):
            raise AlphaArrayBoundaryError("alpha_research.array_surface_budget_invalid")

    @classmethod
    def estimate_surface_bytes(
        cls, *, session_count: int, listing_count: int, factor_count: int
    ) -> int:
        """Estimate dense feature/target/mask/provenance storage with 25 percent structural room.

        Args:
            session_count: Number of source sessions.
            listing_count: Listing count per session.
            factor_count: Float64 feature count per row.

        Returns:
            Conservative integer byte estimate, bounded below by one byte.
        """
        row_count = session_count * listing_count
        return max(
            1,
            int(row_count * (factor_count * 8 + 8 + 2 + 4 + 128 + 8) * 1.25),
        )

    def assert_live_surface_admitted(
        self, *, session_count: int, listing_count: int, factor_count: int
    ) -> None:
        """Require the estimated live surface to fit both cache and peak memory limits.

        Args:
            session_count: Number of requested source sessions.
            listing_count: Listing count per session.
            factor_count: Feature count per row.

        Raises:
            AlphaArrayBoundaryError: Estimated surface bytes exceed cache or peak limit.
        """
        estimated = self.estimate_surface_bytes(
            session_count=session_count,
            listing_count=listing_count,
            factor_count=factor_count,
        )
        if estimated > self.cache_cap_bytes or estimated > self.peak_rss_limit_bytes:
            raise AlphaArrayBoundaryError("alpha_research.array_surface_resource_limit_exceeded")

    def assert_bounded_operation_admitted(
        self,
        *,
        session_count: int,
        listing_count: int,
        factor_count: int,
        live_row_count: int,
    ) -> None:
        """Require the source surface and one live derived fold to fit the memory budget.

        Args:
            session_count: Source sessions in this bounded operation.
            listing_count: Listing count per session.
            factor_count: Feature count per row.
            live_row_count: Derived fold/refit rows simultaneously held with the surface.

        Raises:
            AlphaArrayBoundaryError: Source bytes exceed cache capacity or source plus live fold
                exceeds the peak limit.
        """
        surface = self.estimate_surface_bytes(
            session_count=session_count,
            listing_count=listing_count,
            factor_count=factor_count,
        )
        live = self.estimate_fold_bytes(
            row_count=live_row_count,
            factor_count=factor_count,
        )
        if surface > self.cache_cap_bytes or surface + live > self.peak_rss_limit_bytes:
            raise AlphaArrayBoundaryError("alpha_research.array_surface_resource_limit_exceeded")

    @staticmethod
    def estimate_fold_bytes(*, row_count: int, factor_count: int) -> int:
        """Estimate one derived fold feature/target/economic/mask/provenance working set.

        Args:
            row_count: Nonnegative derived row count.
            factor_count: Positive feature count per row.

        Returns:
            Conservative integer byte estimate with 25 percent structural room.

        Raises:
            AlphaArrayBoundaryError: Row count is negative or feature count is below one.
        """
        if row_count < 0 or factor_count < 1:
            raise AlphaArrayBoundaryError("alpha_research.array_surface_budget_invalid")
        # Feature values, fit target, economic return, two masks, and
        # conservative Python/hash overhead.
        return int(row_count * (factor_count * 8 + 16 + 2 + 128) * 1.25)

    @classmethod
    def derive(
        cls,
        *,
        session_count: int,
        listing_count: int,
        factor_count: int,
        retained_fold_row_counts: tuple[int, ...] = (),
        worst_live_fold_row_count: int = 0,
        total_physical_memory_bytes: int | None = None,
        cache_cap_bytes: int | None = None,
    ) -> AlphaArraySurfaceBudget:
        """Derive memory limits and select program caching or bounded-fold execution.

        Default cache capacity uses eight percent of physical memory, bounded to 512 MiB through 2
        GiB. Peak capacity uses 25 percent, bounded to 4 through 6 GiB. Bounded-fold mode drops
        estimated retained derived arrays from peak accounting.

        Args:
            session_count: Program source-session count.
            listing_count: Listings per session.
            factor_count: Features per row.
            retained_fold_row_counts: Derived fold populations proposed for program caching.
            worst_live_fold_row_count: Largest simultaneously live derived fold.
            total_physical_memory_bytes: Optional measured physical memory; otherwise read from the
                host.
            cache_cap_bytes: Optional admitted cache capacity overriding the derived default.

        Returns:
            Validated budget with resource estimates and selected execution mode.

        Raises:
            AlphaArrayBoundaryError: Estimates or explicit memory declarations violate the budget
                contract.
        """
        total = total_physical_memory_bytes or _total_physical_memory_bytes()
        cap = cache_cap_bytes or min(max(int(total * 0.08), 512 * _MIB), 2 * _GIB)
        peak_limit = min(max(int(total * 0.25), 4 * _GIB), 6 * _GIB)
        # Dense float64 feature/target values, validity masks, causal ordinal,
        # two fixed-width SHA-256 row hashes, indices, and 25% structural room.
        estimated = cls.estimate_surface_bytes(
            session_count=session_count,
            listing_count=listing_count,
            factor_count=factor_count,
        )
        retained = sum(
            cls.estimate_fold_bytes(row_count=value, factor_count=factor_count)
            for value in retained_fold_row_counts
        )
        worst_live = cls.estimate_fold_bytes(
            row_count=worst_live_fold_row_count,
            factor_count=factor_count,
        )
        estimated_peak = estimated + retained + worst_live
        mode = (
            "PROGRAM_CACHE" if estimated <= cap and estimated_peak <= peak_limit else "BOUNDED_FOLD"
        )
        if mode == "BOUNDED_FOLD":
            retained = 0
            estimated_peak = estimated + worst_live
        return cls(
            total_physical_memory_bytes=total,
            cache_cap_bytes=cap,
            peak_rss_limit_bytes=peak_limit,
            minimum_recommended_physical_memory_bytes=(_MINIMUM_RECOMMENDED_PHYSICAL_MEMORY_BYTES),
            preferred_physical_memory_bytes=_PREFERRED_PHYSICAL_MEMORY_BYTES,
            meets_minimum_recommendation=(total >= _MINIMUM_RECOMMENDED_PHYSICAL_MEMORY_BYTES),
            meets_preferred_recommendation=total >= _PREFERRED_PHYSICAL_MEMORY_BYTES,
            estimated_surface_bytes=estimated,
            estimated_retained_derived_bytes=retained,
            estimated_worst_live_fold_bytes=worst_live,
            estimated_peak_working_set_bytes=estimated_peak,
            mode=mode,
        )


@dataclass(frozen=True, slots=True)
class _FeatureSurface:
    sessions: tuple[date, ...]
    listings: tuple[str, ...]
    session_index: dict[date, int]
    listing_index: dict[str, int]
    values: FloatArray
    present: BoolArray
    row_hashes: HashArray
    read_count: int
    feature_panel_snapshot_hash: str | None = None
    ordered_feature_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _OutcomeSurface:
    sessions: tuple[date, ...]
    listings: tuple[str, ...]
    session_index: dict[date, int]
    listing_index: dict[str, int]
    values: FloatArray
    economic_values: FloatArray
    present: BoolArray
    holding_end_ordinals: OrdinalArray
    row_hashes: HashArray
    causal_outcome_snapshot_hash: str | None = None
    target_policy_hash: str | None = None


def _float_values(column: pa.Array | pa.ChunkedArray) -> FloatArray:
    array = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    values = np.asarray(array.to_numpy(zero_copy_only=False), dtype=np.float64)
    values.setflags(write=False)
    return cast(FloatArray, values)


def _positions(
    sessions: pa.Array | pa.ChunkedArray,
    listings: pa.Array | pa.ChunkedArray,
    *,
    session_index: Mapping[date, int],
    listing_index: Mapping[str, int],
) -> IntArray:
    """The surface row of every (session, listing) pair, in the pairs' own order.

    Resolved as two columns rather than one key at a time: each session and
    each listing is looked up in its index, and the row is the session's index
    times the listing count plus the listing's index, exactly as before. A
    pair naming a session or a listing the index does not carry is the same
    refusal a missing key was.
    """

    if not pa.types.is_date(sessions.type):
        raise AlphaArrayBoundaryError("alpha_research.array_surface_axis_mismatch")
    session_at = pc.index_in(
        pc.cast(sessions, pa.date32()), value_set=pa.array(tuple(session_index), pa.date32())
    )
    listing_at = pc.index_in(
        pc.cast(listings, pa.string()), value_set=pa.array(tuple(listing_index), pa.string())
    )
    if session_at.null_count or listing_at.null_count:
        raise AlphaArrayBoundaryError("alpha_research.array_surface_axis_mismatch")
    session_values: IntArray = np.fromiter(
        session_index.values(), dtype=np.intp, count=len(session_index)
    )
    listing_values: IntArray = np.fromiter(
        listing_index.values(), dtype=np.intp, count=len(listing_index)
    )
    result: IntArray = (
        session_values[session_at.to_numpy()] * len(listing_index)
        + listing_values[listing_at.to_numpy()]
    )
    return result


def _grid_positions(session_positions: IntArray, listing_count: int) -> IntArray:
    """One row per (session, listing) of the grid, session-major, listing-minor."""

    offsets: IntArray = np.arange(listing_count, dtype=np.intp)
    result: IntArray = (session_positions[:, None] * listing_count + offsets).ravel()
    return result


def _encoded_hashes(column: pa.Array | pa.ChunkedArray) -> HashArray:
    """Every row hash as its ASCII bytes in a fixed-width array.

    A hash that is not one to sixty-four ASCII bytes is refused, as it was
    when each value was encoded on its own; a null is spelled ``None`` as
    ``str`` spelled it.
    """

    array = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    if array.null_count:
        values = ["None" if value is None else str(value) for value in array.to_pylist()]
        array = pa.array(values, pa.string())
    lengths = pc.binary_length(pc.cast(array, pa.string()))
    if len(array) and (pc.min(lengths).as_py() < 1 or pc.max(lengths).as_py() > 64):
        raise AlphaArrayBoundaryError("alpha_research.array_surface_axis_mismatch")
    try:
        return np.asarray(array.to_numpy(zero_copy_only=False), dtype="S64")
    except UnicodeEncodeError as error:
        raise AlphaArrayBoundaryError("alpha_research.array_surface_axis_mismatch") from error


def _decode_hashes(
    values: HashArray, positions: IntArray, admitted: BoolArray
) -> tuple[str | None, ...]:
    """The stored hash of every admitted row, ``None`` for the rest, in row order."""

    selected: list[bytes] = values[positions].tolist()
    return tuple(
        value.decode("ascii") if ok else None
        for value, ok in zip(selected, admitted.tolist(), strict=True)
    )


_EPOCH_ORDINAL = date(1970, 1, 1).toordinal()


def _ordinals(column: pa.Array | pa.ChunkedArray) -> OrdinalArray:
    """``date.toordinal()`` of every session of a date column, as one array."""

    array = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    if array.null_count:
        raise AlphaArrayBoundaryError("alpha_research.array_surface_axis_mismatch")
    days = np.asarray(array.to_numpy(zero_copy_only=False), dtype="datetime64[D]").astype(np.int64)
    result: OrdinalArray = (days + _EPOCH_ORDINAL).astype(np.int32)
    return result


def _axis_positions(session_index: Mapping[date, int], sessions: Sequence[date]) -> IntArray:
    """The surface position of every named session, in the names' order."""

    try:
        result: IntArray = np.fromiter(
            (session_index[value] for value in sessions), dtype=np.intp, count=len(sessions)
        )
    except KeyError as error:
        raise AlphaArrayBoundaryError("alpha_research.array_surface_axis_mismatch") from error
    return result


@dataclass(frozen=True, slots=True)
class _AdmittedRows:
    """The (session, listing) names of the admitted cells of one grid, in row order.

    Session-major, listing-minor: the order the rows of every fold matrix have
    always had, read off the admitted mask in one pass rather than assembled
    one key at a time.
    """

    sessions: tuple[date, ...]
    listings: tuple[str, ...]

    @classmethod
    def of(
        cls, admitted: BoolArray, sessions: Sequence[date], listings: Sequence[str]
    ) -> _AdmittedRows:
        session_at, listing_at = np.nonzero(admitted)
        session_names = np.asarray(sessions, dtype=object)
        listing_names = np.asarray(listings, dtype=object)
        return cls(
            tuple(session_names[session_at].tolist()), tuple(listing_names[listing_at].tolist())
        )


def _declared_sample(plan: AlphaFoldArrayPlan) -> pa.Array | None:
    """The plan's listing axis, when it is a declared sample of the Panel's listings.

    An exploration study runs on a sample of its input's names (binding plan, B17): its
    authority's listing axis is a strict subset of the Panel's, and rows of the other
    names are left out. A plan on the Panel's own axis (its `listing_set_hash`) keeps
    refusing any row off that axis, as before.
    """

    reader = plan.feature_reader
    resolver = getattr(reader, "resolver", None)
    if resolver is not None:
        manifest = resolver.load_feature_panel_manifest(plan.feature_panel_manifest_ref)
        whole = manifest.get("listing_set_hash")
    else:
        # A prepared overlay covers its base Panel's axis and names it.
        axis = getattr(reader, "listing_ids", None)
        if axis is None:
            return None
        whole = canonical_hash(tuple(axis(plan.feature_panel_manifest_ref)))
    if canonical_hash(plan.ordered_listing_ids) == whole:
        return None
    return pa.array(plan.ordered_listing_ids, pa.string())


def _load_feature_surface(
    plan: AlphaFoldArrayPlan,
    sessions: tuple[date, ...],
    observer: AlphaArrayReadObserver,
) -> _FeatureSurface:
    source_hashes = _training_binding_source_hashes(plan)
    sample = _declared_sample(plan)
    session_index = {value: index for index, value in enumerate(sessions)}
    listing_index = {value: index for index, value in enumerate(plan.ordered_listing_ids)}
    maximum_row_count = len(sessions) * len(listing_index)
    values: FloatArray = np.empty((maximum_row_count, len(plan.base_feature_ids)), dtype=np.float64)
    values.fill(np.nan)
    present: BoolArray = np.zeros(maximum_row_count, dtype=np.bool_)
    hashes: HashArray = np.zeros(maximum_row_count, dtype="S64")
    request = FeaturePanelReadRequest(
        manifest_ref=plan.feature_panel_manifest_ref,
        start_session=sessions[0],
        end_session=sessions[-1],
        exact_sessions=sessions,
        factor_columns=tuple(sorted(plan.base_feature_ids)),
        include_row_hash=True,
    )
    row_count = chunk_count = byte_count = 0
    for batch in plan.feature_reader.batches(request):
        row_count += batch.num_rows
        chunk_count += 1
        byte_count += int(batch.nbytes)
        if sample is not None:
            batch = batch.filter(pc.is_in(batch.column("listing_id"), value_set=sample))
        names = batch.schema.names
        positions = _positions(
            batch.column(names.index("session_date")),
            batch.column(names.index("listing_id")),
            session_index=session_index,
            listing_index=listing_index,
        )
        if np.unique(positions).size != len(positions) or bool(present[positions].any()):
            raise AlphaArrayBoundaryError("alpha_research.array_surface_duplicate_row")
        for factor_index, factor_id in enumerate(plan.base_feature_ids):
            values[positions, factor_index] = _float_values(batch.column(names.index(factor_id)))
        hashes[positions] = _encoded_hashes(batch.column(names.index("row_hash")))
        present[positions] = True
    values.setflags(write=False)
    present.setflags(write=False)
    hashes.setflags(write=False)
    pa.default_memory_pool().release_unused()
    observer.observe(
        AlphaArrayReadObservation(
            operation="feature_batches",
            row_count=row_count,
            chunk_count=chunk_count,
            byte_count=byte_count,
        )
    )
    return _FeatureSurface(
        sessions=sessions,
        listings=plan.ordered_listing_ids,
        session_index=session_index,
        listing_index=listing_index,
        values=values,
        present=present,
        row_hashes=hashes,
        read_count=1,
        feature_panel_snapshot_hash=source_hashes[1] if source_hashes is not None else None,
        ordered_feature_ids=plan.base_feature_ids,
    )


def _load_outcome_surface(
    plan: AlphaFoldArrayPlan,
    sessions: tuple[date, ...],
    observer: AlphaArrayReadObserver,
) -> _OutcomeSurface:
    source_hashes = _training_binding_source_hashes(plan)
    table = plan.outcome_reader.read_development_sessions(
        plan.causal_outcome_manifest_ref, sessions
    )
    observer.observe(
        AlphaArrayReadObservation(
            operation="read_development_sessions",
            row_count=table.num_rows,
            chunk_count=max(1, len(table.to_batches())),
            byte_count=int(table.nbytes),
        )
    )
    required = {
        "formation_session",
        "listing_id",
        "simple_return",
        "row_hash",
        "holding_end_session",
    }
    if not required.issubset(table.schema.names):
        raise AlphaArrayBoundaryError("ALPHA_CAUSAL_OUTCOME_SCHEMA_MISMATCH")
    target_table: pa.Table | None = None
    if plan.target_policy is not None or plan.target_method is not None:
        manifest = plan.outcome_reader.load_manifest(
            plan.foundation.execution_outcome.snapshot_hash
        )
        source_target = compile_factor_target_surface(
            source_table=table,
            source_manifest=manifest,
            source_manifest_ref=plan.causal_outcome_manifest_ref,
            # The declaration follows the resolved seal when a development plan
            # carries one, so a five-session plan projects five-session rows
            # under a policy derived from the same installed method the seal
            # names. The frozen current/Goal route carries no seal and keeps its
            # one-session policy and legacy path byte-identical.
            policy=(
                build_factor_target_policy()
                if plan.outcome_method is None
                else build_factor_target_policy_for_method(plan.outcome_method.method_bound)
            ),
            outcome_method=plan.outcome_method,
        )
        if plan.sector_by_listing_id is None:
            raise AlphaArrayBoundaryError("alpha_research.target_sector_map_incomplete")
        # The history, when the plan holds one, travels whole.
        sector_map = plan.sector_by_listing_id
        if plan.target_method is not None:
            # One development route, and the method compiles itself. This used to
            # branch on the recipe family, which meant every new target
            # composition needed a new branch here -- in a module that has no
            # business knowing how many target methods exist. The method travels
            # whole, so what compiled these values is what the Host resolved.
            target_table = plan.target_method.compile_target_surface(
                source_table=source_target.table,
                sector_by_listing_id=sector_map,
            )
        else:
            target_policy = AlphaTargetPolicy.model_validate(plan.target_policy)
            target_table = compile_alpha_target_surface(
                source_table=source_target.table,
                policy=target_policy,
                sector_by_listing_id=sector_map,
                # The standardization this frozen-lane caller named, or the
                # lane's derived key when it named none.
                standardization_id=plan.target_standardization_id,
            )
    sample = _declared_sample(plan)
    if sample is not None:
        # A sample's names keep the targets the whole cross-section gives them, as their
        # features do; only the rows the sample fits on are kept.
        table = table.filter(pc.is_in(table["listing_id"], value_set=sample))
        if target_table is not None:
            target_table = target_table.filter(
                pc.is_in(target_table["listing_id"], value_set=sample)
            )
    session_index = {value: index for index, value in enumerate(sessions)}
    listing_index = {value: index for index, value in enumerate(plan.ordered_listing_ids)}
    positions = _positions(
        table["formation_session"],
        table["listing_id"],
        session_index=session_index,
        listing_index=listing_index,
    )
    if np.unique(positions).size != len(positions):
        raise AlphaArrayBoundaryError("alpha_research.array_surface_duplicate_row")
    row_count = len(sessions) * len(listing_index)
    values: FloatArray = np.full(row_count, np.nan, dtype=np.float64)
    economic_values: FloatArray = np.full(row_count, np.nan, dtype=np.float64)
    present: BoolArray = np.zeros(row_count, dtype=np.bool_)
    holding_end_ordinals: OrdinalArray = np.zeros(row_count, dtype=np.int32)
    hashes: HashArray = np.zeros(row_count, dtype="S64")
    if target_table is None:
        values[positions] = _float_values(table["simple_return"])
        economic_values[positions] = _float_values(table["simple_return"])
    else:
        target_positions = _positions(
            target_table["formation_session"],
            target_table["listing_id"],
            session_index=session_index,
            listing_index=listing_index,
        )
        if not np.array_equal(np.unique(target_positions), np.unique(positions)):
            raise AlphaArrayBoundaryError("alpha_research.target_axis_mismatch")
        values[target_positions] = _float_values(target_table["fit_target"])
        economic_values[target_positions] = _float_values(target_table["simple_economic_return"])
    present[positions] = True
    holding_end_ordinals[positions] = _ordinals(table["holding_end_session"])
    hashes[positions] = _encoded_hashes(table["row_hash"])
    for value in (values, economic_values, present, holding_end_ordinals, hashes):
        value.setflags(write=False)
    result = _OutcomeSurface(
        sessions=sessions,
        listings=plan.ordered_listing_ids,
        session_index=session_index,
        listing_index=listing_index,
        values=values,
        economic_values=economic_values,
        present=present,
        holding_end_ordinals=holding_end_ordinals,
        row_hashes=hashes,
        causal_outcome_snapshot_hash=source_hashes[2] if source_hashes is not None else None,
        target_policy_hash=(
            AlphaTargetPolicy.model_validate(plan.target_policy).policy_hash
            if plan.target_policy is not None
            else None
        ),
    )
    del table
    pa.default_memory_pool().release_unused()
    return result


def _matrix_for_keys(
    keys: tuple[tuple[date, str], ...],
    feature: _FeatureSurface,
    outcome: _OutcomeSurface | None,
    *,
    factor_count: int,
    outcome_available_through: date | None = None,
    copy_values: bool = False,
    include_hashes: bool = True,
) -> tuple[
    FloatArray,
    FloatArray,
    BoolArray,
    BoolArray,
    tuple[str | None, ...],
    tuple[str | None, ...],
    FloatArray,
]:
    """The matrices of rows named by (session, listing) keys, resolved on each surface."""

    sessions = pa.array([key[0] for key in keys], pa.date32())
    listings = pa.array([key[1] for key in keys], pa.string())
    feature_indices = _positions(
        sessions, listings, session_index=feature.session_index, listing_index=feature.listing_index
    )
    outcome_indices = (
        None
        if outcome is None
        else _positions(
            sessions,
            listings,
            session_index=outcome.session_index,
            listing_index=outcome.listing_index,
        )
    )
    return _matrix_for_rows(
        feature_indices,
        outcome_indices,
        feature,
        outcome,
        factor_count=factor_count,
        outcome_available_through=outcome_available_through,
        copy_values=copy_values,
        include_hashes=include_hashes,
    )


def _matrix_for_rows(
    feature_indices: IntArray,
    outcome_indices: IntArray | None,
    feature: _FeatureSurface,
    outcome: _OutcomeSurface | None,
    *,
    factor_count: int,
    outcome_available_through: date | None = None,
    copy_values: bool = False,
    include_hashes: bool = True,
) -> tuple[
    FloatArray,
    FloatArray,
    BoolArray,
    BoolArray,
    tuple[str | None, ...],
    tuple[str | None, ...],
    FloatArray,
]:
    """The matrices of one row set, named by its rows on each surface.

    ``feature_indices`` and ``outcome_indices`` are the same rows resolved on
    the feature surface and on the outcome surface, in the row set's order;
    ``outcome_indices`` is ``None`` when the rows have no outcome (a current
    formation), which yields all-missing targets.
    """

    if outcome is not None and (
        outcome_indices is None or len(outcome_indices) != len(feature_indices)
    ):
        raise AlphaArrayBoundaryError("alpha_research.array_surface_axis_mismatch")
    row_count = len(feature_indices)
    feature_present = feature.present[feature_indices]
    contiguous_feature = bool(feature_present.all()) and (
        not len(feature_indices)
        or np.array_equal(
            feature_indices,
            np.arange(feature_indices[0], feature_indices[0] + len(feature_indices)),
        )
    )
    if contiguous_feature and len(feature_indices) and not copy_values:
        feature_values = feature.values[
            feature_indices[0] : feature_indices[0] + len(feature_indices)
        ]
    elif contiguous_feature and len(feature_indices):
        feature_values = feature.values[feature_indices].copy()
    else:
        feature_values = np.full((row_count, factor_count), np.nan, dtype=np.float64)
        if bool(feature_present.any()):
            feature_values[feature_present] = feature.values[feature_indices[feature_present]]
    feature_hashes = (
        _decode_hashes(feature.row_hashes, feature_indices, feature_present)
        if include_hashes
        else ()
    )

    targets: FloatArray
    economic_targets: FloatArray
    outcome_hashes: tuple[str | None, ...]
    if outcome is None or outcome_indices is None:
        targets = np.full(row_count, np.nan, dtype=np.float64)
        economic_targets = np.full(row_count, np.nan, dtype=np.float64)
        outcome_hashes = (None,) * row_count if include_hashes else ()
    else:
        admitted = outcome.present[outcome_indices].copy()
        if outcome_available_through is not None:
            admitted_indices = np.flatnonzero(admitted)
            admitted[admitted_indices] = (
                outcome.holding_end_ordinals[outcome_indices[admitted_indices]]
                <= outcome_available_through.toordinal()
            )
        contiguous_outcome = bool(admitted.all()) and (
            not len(outcome_indices)
            or np.array_equal(
                outcome_indices,
                np.arange(outcome_indices[0], outcome_indices[0] + len(outcome_indices)),
            )
        )
        if contiguous_outcome and len(outcome_indices) and not copy_values:
            targets = outcome.values[outcome_indices[0] : outcome_indices[0] + len(outcome_indices)]
            economic_targets = outcome.economic_values[
                outcome_indices[0] : outcome_indices[0] + len(outcome_indices)
            ]
        elif contiguous_outcome and len(outcome_indices):
            targets = outcome.values[outcome_indices].copy()
            economic_targets = outcome.economic_values[outcome_indices].copy()
        else:
            targets = np.full(row_count, np.nan, dtype=np.float64)
            economic_targets = np.full(row_count, np.nan, dtype=np.float64)
            if bool(admitted.any()):
                targets[admitted] = outcome.values[outcome_indices[admitted]]
                economic_targets[admitted] = outcome.economic_values[outcome_indices[admitted]]
        outcome_hashes = (
            _decode_hashes(outcome.row_hashes, outcome_indices, admitted) if include_hashes else ()
        )
    feature_complete = np.isfinite(feature_values).all(axis=1)
    outcome_complete = np.isfinite(targets)
    return (
        cast(FloatArray, _readonly(feature_values)),
        cast(FloatArray, _readonly(targets)),
        cast(BoolArray, _readonly(feature_complete)),
        cast(BoolArray, _readonly(outcome_complete)),
        feature_hashes,
        outcome_hashes,
        cast(FloatArray, _readonly(economic_targets)),
    )


def _fold_array_bytes(value: AlphaFoldArrays) -> int:
    return sum(
        int(array.nbytes)
        for array in (
            value.training_features,
            value.training_targets,
            value.training_feature_complete,
            value.training_outcome_complete,
            value.validation_features,
            value.validation_targets,
            value.validation_feature_complete,
            value.validation_outcome_complete,
            *(
                array
                for array in (
                    value.training_economic_returns,
                    value.validation_economic_returns,
                )
                if array is not None
            ),
        )
    )


class AlphaFoldArrayLease:
    """Keep one bounded fold live only for the duration of its numerical work."""

    def __init__(self, workspace: AlphaProgramArrayWorkspace, fold_index: int) -> None:
        """Bind a fold lease to one workspace and split-plan index.

        Args:
            workspace: Array workspace owning cache/admission and lease lifetime.
            fold_index: Zero-based fold index requested on entry.
        """
        self._workspace = workspace
        self._fold_index = fold_index
        self._arrays: AlphaFoldArrays | None = None

    def __enter__(self) -> AlphaFoldArrays:
        """Admit the requested fold arrays through the workspace lease owner.

        Returns:
            Fold arrays retained for the active lease.

        Raises:
            AlphaArrayBoundaryError: Fold admission, resource limits or concurrent bounded lease
                rules fail.
        """
        self._arrays = self._workspace._enter_fold_lease(self._fold_index)
        return self._arrays

    def __exit__(self, *_args: object) -> None:
        """Release this lease reference and the workspace bounded-fold lifetime."""
        self._arrays = None
        self._workspace._exit_fold_lease()


class AlphaProgramArrayWorkspace:
    """Own one bounded decoded surface for all numerical jobs in a program."""

    def __init__(
        self,
        plan: AlphaFoldArrayPlan,
        *,
        train_session_count: int | None,
        cache_cap_bytes: int | None = None,
        total_physical_memory_bytes: int | None = None,
        read_observer: AlphaArrayReadObserver | None = None,
        formation_session: date | None = None,
    ) -> None:
        """Admit a program calendar and memory mode before loading reusable or bounded surfaces.

        Args:
            plan: Admitted fold array plan with source/calendar/feature authority.
            train_session_count: Current-refit training window, or None when refit is not
                configured.
            cache_cap_bytes: Optional explicit memory cache limit.
            total_physical_memory_bytes: Optional measured physical-memory capacity.
            read_observer: Optional observer of deterministic array-read counts.
            formation_session: Optional admitted Panel formation; defaults to its last session.

        Raises:
            AlphaArrayBoundaryError: Formation/calendar, current training geometry or memory budget
                cannot be admitted.
        """
        self.plan = plan
        self.train_session_count = train_session_count
        self._read_ledger = AlphaArrayReadLedger()
        self._read_observer = read_observer
        # The calendar the plan's split was cut from, not a second read of it.
        panel_sessions = plan.panel_sessions
        if not panel_sessions:
            raise AlphaArrayBoundaryError("ALPHA_CURRENT_FORMATION_UNAVAILABLE")
        if formation_session is not None and formation_session not in panel_sessions:
            raise AlphaArrayBoundaryError("ALPHA_CURRENT_FORMATION_UNAVAILABLE")
        self.formation_session = formation_session or panel_sessions[-1]
        self.current_training_sessions = self._current_training_sessions(train_session_count)
        fold_sessions = tuple(
            sorted(
                set(
                    chain.from_iterable(
                        (*window.train_sessions, *window.validation_sessions)
                        for window in plan.split_plan.windows
                    )
                )
            )
        )
        self.program_sessions = tuple(
            sorted({*fold_sessions, *self.current_training_sessions, self.formation_session})
        )
        fold_row_counts = tuple(
            (len(window.train_sessions) + len(window.validation_sessions))
            * len(plan.ordered_listing_ids)
            for window in plan.split_plan.windows
        )
        self.budget = AlphaArraySurfaceBudget.derive(
            session_count=len(self.program_sessions),
            listing_count=len(plan.ordered_listing_ids),
            factor_count=len(plan.base_feature_ids) + len(plan.additional_factor_ids),
            retained_fold_row_counts=fold_row_counts,
            worst_live_fold_row_count=max(fold_row_counts, default=0),
            total_physical_memory_bytes=total_physical_memory_bytes,
            cache_cap_bytes=cache_cap_bytes,
        )
        self._feature: _FeatureSurface | None = None
        self._outcome: _OutcomeSurface | None = None
        self._feature_read_count = 0
        self._outcome_read_count = 0
        self._requested_session_count = 0
        self._reused_session_count = 0
        self._complete_program_decode_count = 0
        self._surface_build_count = 0
        self._cache_hit_count = 0
        self._feature_bytes = 0
        self._outcome_bytes = 0
        self._last_sessions: tuple[date, ...] = ()
        self._bounded_feature: _FeatureSurface | None = None
        self._bounded_outcome: _OutcomeSurface | None = None
        self._fold_cache: dict[int, AlphaFoldArrays] = {}
        self._current_cache: AlphaCurrentRefitArrays | None = None
        self._active_bounded_lease = False
        self._retained_derived_bytes = 0
        if self.budget.mode == "PROGRAM_CACHE":
            self._feature, self._outcome = self._load(self.program_sessions)
            self._complete_program_decode_count = 1

    def _current_training_sessions(self, train_session_count: int | None) -> tuple[date, ...]:
        if train_session_count is None:
            return ()
        if train_session_count < 1:
            raise AlphaArrayBoundaryError("ALPHA_CURRENT_REFIT_TRAIN_WINDOW_INVALID")
        sessions: tuple[date, ...] = tuple(
            self.plan.outcome_reader.available_development_sessions(
                self.plan.causal_outcome_manifest_ref,
                holding_end_through=self.formation_session,
            )
        )
        observation = AlphaArrayReadObservation(
            operation="available_development_sessions",
            row_count=len(sessions),
            chunk_count=1,
            byte_count=0,
        )
        self._observe(observation)
        if len(sessions) < train_session_count:
            raise AlphaArrayBoundaryError("ALPHA_CURRENT_REFIT_TRAINING_HISTORY_INSUFFICIENT")
        return sessions[-train_session_count:]

    def _load(self, sessions: tuple[date, ...]) -> tuple[_FeatureSurface, _OutcomeSurface]:
        self.budget.assert_live_surface_admitted(
            session_count=len(sessions),
            listing_count=len(self.plan.ordered_listing_ids),
            factor_count=len(self.plan.base_feature_ids),
        )
        self._requested_session_count += len(sessions)
        if self._last_sessions:
            self._reused_session_count += len(set(self._last_sessions).intersection(sessions))
        feature = _load_feature_surface(self.plan, sessions, self)
        outcome = _load_outcome_surface(self.plan, sessions, self)
        self._feature_read_count += feature.read_count
        self._outcome_read_count += 1
        self._surface_build_count += 1
        self._feature_bytes += int(feature.values.nbytes)
        self._outcome_bytes += int(outcome.values.nbytes + outcome.economic_values.nbytes)
        self._last_sessions = sessions
        if self.budget.mode == "BOUNDED_FOLD":
            self._bounded_feature = feature
            self._bounded_outcome = outcome
        return feature, outcome

    def observe(self, observation: AlphaArrayReadObservation, /) -> None:
        """Record a typed array read in the internal ledger and optional observer.

        Args:
            observation: Nonnegative resource counts from a deterministic array operation.
        """
        self._observe(observation)

    def _observe(self, observation: AlphaArrayReadObservation) -> None:
        self._read_ledger.observe(observation)
        if self._read_observer is not None:
            self._read_observer.observe(observation)

    def _surfaces(self, sessions: tuple[date, ...]) -> tuple[_FeatureSurface, _OutcomeSurface]:
        if self._feature is not None and self._outcome is not None:
            self._cache_hit_count += 1
            self._requested_session_count += len(sessions)
            self._reused_session_count += len(sessions)
            return self._feature, self._outcome
        if (
            sessions == self._last_sessions
            and self._bounded_feature is not None
            and self._bounded_outcome is not None
        ):
            self._cache_hit_count += 1
            self._requested_session_count += len(sessions)
            self._reused_session_count += len(sessions)
            return self._bounded_feature, self._bounded_outcome
        return self._load(sessions)

    def _build_fold(self, fold_index: int) -> AlphaFoldArrays:
        cached = self._fold_cache.get(fold_index) if self.budget.mode == "PROGRAM_CACHE" else None
        if cached is not None:
            self._cache_hit_count += 1
            return cached
        try:
            window = self.plan.split_plan.windows[fold_index]
        except IndexError as error:
            raise AlphaArrayBoundaryError("ALPHA_FOLD_INDEX_INVALID") from error
        used_sessions = tuple(sorted((*window.train_sessions, *window.validation_sessions)))
        listing_count = len(self.plan.ordered_listing_ids)
        if self.budget.mode == "BOUNDED_FOLD":
            self.budget.assert_bounded_operation_admitted(
                session_count=len(used_sessions),
                listing_count=listing_count,
                factor_count=len(self.plan.base_feature_ids),
                live_row_count=len(used_sessions) * listing_count,
            )
        feature, outcome = self._surfaces(used_sessions)
        feature_session_indices: IntArray = np.fromiter(
            (feature.session_index[value] for value in used_sessions),
            dtype=np.intp,
            count=len(used_sessions),
        )
        outcome_session_indices: IntArray = np.fromiter(
            (outcome.session_index[value] for value in used_sessions),
            dtype=np.intp,
            count=len(used_sessions),
        )
        listing_offsets: IntArray = np.arange(listing_count, dtype=np.intp)
        feature_positions = feature_session_indices[:, None] * listing_count + listing_offsets
        outcome_positions = outcome_session_indices[:, None] * listing_count + listing_offsets
        feature_admitted = feature.present[feature_positions]
        outcome_admitted = outcome.present[outcome_positions]
        if not bool(feature_admitted.any(axis=0).all()):
            raise AlphaArrayBoundaryError("ALPHA_FEATURE_LISTING_AUTHORITY_MISMATCH")
        if not bool(outcome_admitted.any(axis=0).all()):
            raise AlphaArrayBoundaryError("ALPHA_OUTCOME_LISTING_AUTHORITY_MISMATCH")
        admitted = feature_admitted | outcome_admitted
        train_count = len(window.train_sessions)
        # The admitted cells of the grid in session-major, listing-minor order:
        # one row per admitted (session, listing), the rows each surface names
        # taken from the same cells.
        training_admitted = admitted[:train_count]
        validation_admitted = admitted[train_count:]
        training_rows = _AdmittedRows.of(
            training_admitted, window.train_sessions, self.plan.ordered_listing_ids
        )
        validation_rows = _AdmittedRows.of(
            validation_admitted, window.validation_sessions, self.plan.ordered_listing_ids
        )
        if not training_rows.sessions or not validation_rows.sessions:
            raise AlphaArrayBoundaryError("ALPHA_FOLD_HAS_NO_ALIGNED_ROWS")
        training = _matrix_for_rows(
            feature_positions[:train_count][training_admitted],
            outcome_positions[:train_count][training_admitted],
            feature,
            outcome,
            factor_count=len(self.plan.base_feature_ids),
            outcome_available_through=window.train_sessions[-1],
            copy_values=self.budget.mode == "BOUNDED_FOLD",
        )
        validation = _matrix_for_rows(
            feature_positions[train_count:][validation_admitted],
            outcome_positions[train_count:][validation_admitted],
            feature,
            outcome,
            factor_count=len(self.plan.base_feature_ids),
            copy_values=self.budget.mode == "BOUNDED_FOLD",
        )
        commitment = build_alpha_fold_commitment(window)
        training_row_sessions = training_rows.sessions
        source_hashes = _training_binding_source_hashes(self.plan)
        binding = (
            None
            if source_hashes is None
            else build_alpha_training_input_binding(
                scope="DEVELOPMENT_FOLD",
                foundation_hash=source_hashes[0],
                feature_panel_snapshot_hash=source_hashes[1],
                causal_outcome_snapshot_hash=source_hashes[2],
                target_policy=self.plan.target_policy,
                ordered_feature_ids=self.plan.base_feature_ids,
                feature_context_hash=None,
                training_row_sessions=training_row_sessions,
                training_row_listing_ids=training_rows.listings,
                prediction_row_sessions=validation_rows.sessions,
                prediction_row_listing_ids=validation_rows.listings,
                training_features=training[0],
                training_targets=training[1],
                training_mask=cast(BoolArray, _readonly(training[2] & training[3])),
                prediction_features=validation[0],
                prediction_targets=validation[1],
                prediction_mask=validation[2],
                training_feature_row_hashes=training[4],
                training_outcome_row_hashes=training[5],
                prediction_feature_row_hashes=validation[4],
                prediction_outcome_row_hashes=validation[5],
                training_economic_returns=training[6],
                prediction_economic_returns=validation[6],
                training_cutoff=window.train_sessions[-1],
                outcome_maturity_session=window.train_sessions[-1],
                prediction_anchor=window.validation_sessions[0],
                fold_commitment_hash=commitment.commitment_hash,
            )
        )
        result = AlphaFoldArrays(
            commitment=commitment,
            ordered_factor_ids=self.plan.base_feature_ids,
            training_sessions=tuple(window.train_sessions),
            training_listing_ids=training_rows.listings,
            training_features=training[0],
            training_targets=training[1],
            training_feature_complete=training[2],
            training_outcome_complete=training[3],
            validation_sessions=tuple(window.validation_sessions),
            validation_row_sessions=validation_rows.sessions,
            validation_listing_ids=validation_rows.listings,
            validation_features=validation[0],
            validation_targets=validation[1],
            validation_feature_complete=validation[2],
            validation_outcome_complete=validation[3],
            validation_feature_row_hashes=validation[4],
            validation_outcome_row_hashes=validation[5],
            training_economic_returns=training[6],
            validation_economic_returns=validation[6],
            training_row_sessions=training_row_sessions,
            training_feature_row_hashes=training[4],
            training_outcome_row_hashes=training[5],
            training_input_binding=binding,
        )
        if self.budget.mode == "PROGRAM_CACHE":
            self._fold_cache[fold_index] = result
            self._retained_derived_bytes += _fold_array_bytes(result)
            if (
                self.budget.estimated_surface_bytes + self._retained_derived_bytes
                > self.budget.peak_rss_limit_bytes
            ):
                self._fold_cache.pop(fold_index, None)
                self._retained_derived_bytes -= _fold_array_bytes(result)
                raise AlphaArrayBoundaryError(
                    "alpha_research.array_surface_resource_limit_exceeded"
                )
        return result

    def load_fold(self, fold_index: int) -> AlphaFoldArrays:
        """Compatibility read for cache mode; bounded callers must use a lease."""
        if self.budget.mode == "BOUNDED_FOLD":
            raise AlphaArrayBoundaryError("alpha_research.array_surface_lease_required")
        return self._build_fold(fold_index)

    def fold_lease(self, fold_index: int) -> AlphaFoldArrayLease:
        """Create a fold lease whose array admission occurs when entered.

        Args:
            fold_index: Requested zero-based split-plan fold index.

        Returns:
            Lease context for that fold; creating the lease does not load it.
        """
        return AlphaFoldArrayLease(self, fold_index)

    def _enter_fold_lease(self, fold_index: int) -> AlphaFoldArrays:
        if self.budget.mode == "BOUNDED_FOLD":
            if self._active_bounded_lease:
                raise AlphaArrayBoundaryError("alpha_research.array_surface_lease_conflict")
            self._active_bounded_lease = True
        try:
            return self._build_fold(fold_index)
        except Exception:
            if self.budget.mode == "BOUNDED_FOLD":
                self._active_bounded_lease = False
                self._bounded_feature = None
                self._bounded_outcome = None
                self._last_sessions = ()
            raise

    def _exit_fold_lease(self) -> None:
        if self.budget.mode == "BOUNDED_FOLD":
            if not self._active_bounded_lease:
                raise AlphaArrayBoundaryError("alpha_research.array_surface_lease_invalid")
            self._active_bounded_lease = False
            self._bounded_feature = None
            self._bounded_outcome = None
            self._last_sessions = ()

    def prepare_current_refit(self) -> AlphaCurrentRefitArrays:
        """Prepare current training and formation arrays with matured outcomes and exact provenance.

        Program-cache mode reuses current arrays. Bounded mode checks source-plus-live admission,
        builds aligned session-major/listing-minor rows and releases bounded source references after
        assembly. Available source authority seals the resulting current-refit input binding.

        Returns:
            Current training/features/targets/masks and formation features on the exact declared
            axes.

        Raises:
            AlphaArrayBoundaryError: A bounded lease is active, the refit window is absent or
                memory/source admission fails.
        """
        if self._active_bounded_lease:
            raise AlphaArrayBoundaryError("alpha_research.array_surface_lease_conflict")
        if self._current_cache is not None:
            self._cache_hit_count += 1
            return self._current_cache
        if self.train_session_count is None or not self.current_training_sessions:
            raise AlphaArrayBoundaryError("ALPHA_CURRENT_REFIT_TRAIN_WINDOW_INVALID")
        requested = tuple(sorted({*self.current_training_sessions, self.formation_session}))
        if self.budget.mode == "BOUNDED_FOLD":
            self.budget.assert_bounded_operation_admitted(
                session_count=len(requested),
                listing_count=len(self.plan.ordered_listing_ids),
                factor_count=len(self.plan.base_feature_ids),
                live_row_count=(
                    (len(self.current_training_sessions) + 1) * len(self.plan.ordered_listing_ids)
                ),
            )
        feature, outcome = self._surfaces(requested)
        listing_count = len(self.plan.ordered_listing_ids)
        # Every (session, listing) of the training window and of the formation
        # session, session-major, listing-minor.
        training_rows = _AdmittedRows.of(
            np.ones((len(self.current_training_sessions), listing_count), dtype=np.bool_),
            self.current_training_sessions,
            self.plan.ordered_listing_ids,
        )
        current_rows = _AdmittedRows.of(
            np.ones((1, listing_count), dtype=np.bool_),
            (self.formation_session,),
            self.plan.ordered_listing_ids,
        )
        training = _matrix_for_rows(
            _grid_positions(
                _axis_positions(feature.session_index, self.current_training_sessions),
                listing_count,
            ),
            _grid_positions(
                _axis_positions(outcome.session_index, self.current_training_sessions),
                listing_count,
            ),
            feature,
            outcome,
            factor_count=len(self.plan.base_feature_ids),
            outcome_available_through=self.formation_session,
            copy_values=self.budget.mode == "BOUNDED_FOLD",
        )
        current = _matrix_for_rows(
            _grid_positions(
                _axis_positions(feature.session_index, (self.formation_session,)), listing_count
            ),
            None,
            feature,
            None,
            factor_count=len(self.plan.base_feature_ids),
            copy_values=self.budget.mode == "BOUNDED_FOLD",
        )
        training_row_sessions = training_rows.sessions
        training_mask = cast(BoolArray, _readonly(training[2] & training[3]))
        source_hashes = _training_binding_source_hashes(self.plan)
        binding = (
            None
            if source_hashes is None
            else build_alpha_training_input_binding(
                scope="CURRENT_REFIT",
                foundation_hash=source_hashes[0],
                feature_panel_snapshot_hash=source_hashes[1],
                causal_outcome_snapshot_hash=source_hashes[2],
                target_policy=self.plan.target_policy,
                ordered_feature_ids=self.plan.base_feature_ids,
                feature_context_hash=None,
                training_row_sessions=training_row_sessions,
                training_row_listing_ids=training_rows.listings,
                prediction_row_sessions=current_rows.sessions,
                prediction_row_listing_ids=current_rows.listings,
                training_features=training[0],
                training_targets=training[1],
                training_mask=training_mask,
                prediction_features=current[0],
                prediction_targets=current[1],
                prediction_mask=current[2],
                training_feature_row_hashes=training[4],
                training_outcome_row_hashes=training[5],
                prediction_feature_row_hashes=current[4],
                prediction_outcome_row_hashes=current[5],
                training_economic_returns=training[6],
                prediction_economic_returns=current[6],
                training_cutoff=self.current_training_sessions[-1],
                outcome_maturity_session=self.formation_session,
                prediction_anchor=self.formation_session,
                fold_commitment_hash=None,
            )
        )
        result = AlphaCurrentRefitArrays(
            ordered_factor_ids=self.plan.base_feature_ids,
            ordered_listing_ids=self.plan.ordered_listing_ids,
            training_sessions=self.current_training_sessions,
            training_cutoff=self.current_training_sessions[-1],
            formation_session=self.formation_session,
            training_features=training[0],
            training_targets=training[1],
            training_mask=training_mask,
            current_features=current[0],
            current_feature_complete=current[2],
            training_row_sessions=training_row_sessions,
            training_feature_row_hashes=training[4],
            training_outcome_row_hashes=training[5],
            current_feature_row_hashes=current[4],
            training_economic_returns=training[6],
            training_input_binding=binding,
        )
        if self.budget.mode == "PROGRAM_CACHE":
            self._current_cache = result
        else:
            self._bounded_feature = None
            self._bounded_outcome = None
            self._last_sessions = ()
        return result

    def close(self) -> None:
        """Release cached source/derived arrays and reset bounded lease accounting."""
        self._feature = None
        self._outcome = None
        self._bounded_feature = None
        self._bounded_outcome = None
        self._fold_cache.clear()
        self._current_cache = None
        self._last_sessions = ()
        self._active_bounded_lease = False
        self._retained_derived_bytes = 0

    def __enter__(self) -> AlphaProgramArrayWorkspace:
        """Enter this already admitted program array workspace.

        Returns:
            This workspace; construction has already selected and admitted its memory mode.
        """
        return self

    def __exit__(self, *_args: object) -> None:
        """Release this workspace cached arrays when its context ends."""
        self.close()


def prepare_alpha_program_array_workspace(
    plan: AlphaFoldArrayPlan,
    *,
    train_session_count: int | None,
    cache_cap_bytes: int | None = None,
    total_physical_memory_bytes: int | None = None,
    read_observer: AlphaArrayReadObserver | None = None,
    formation_session: date | None = None,
) -> AlphaProgramArrayWorkspace:
    """Construct an admitted reusable or bounded Alpha program array workspace.

    Args:
        plan: Admitted fold array plan with source/calendar/feature authority.
        train_session_count: Current-refit training window, or None when refit is not configured.
        cache_cap_bytes: Optional explicit memory cache limit.
        total_physical_memory_bytes: Optional measured physical-memory capacity.
        read_observer: Optional observer of deterministic array-read counts.
        formation_session: Optional admitted Panel formation; defaults to its last session.

    Returns:
        Workspace with declared memory mode and exact program/current formation scope.

    Raises:
        AlphaArrayBoundaryError: Program formation, training geometry or memory/source admission
            fails.
    """
    return AlphaProgramArrayWorkspace(
        plan,
        train_session_count=train_session_count,
        cache_cap_bytes=cache_cap_bytes,
        total_physical_memory_bytes=total_physical_memory_bytes,
        read_observer=read_observer,
        formation_session=formation_session,
    )


__all__ = [
    "AlphaArraySurfaceBudget",
    "AlphaFoldArrayLease",
    "AlphaProgramArrayWorkspace",
    "prepare_alpha_program_array_workspace",
]
