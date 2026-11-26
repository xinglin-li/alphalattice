"""One verified join between a fold's predictions and the rows they predict."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc

from .development_artifacts import AlphaDevelopmentArtifactStore
from .development_contracts import (
    AlphaCandidateNumericalFoldResult,
    AlphaDevelopmentExecutionReceipt,
    AlphaDevelopmentFoldSurface,
    CanonicalScoreSurfaceError,
    canonical_score_value_identity,
)

if TYPE_CHECKING:
    from .development_evidence import AlphaDevelopmentVerifiedGraph


@dataclass(frozen=True, slots=True)
class DevelopmentScoreRows:
    sessions: tuple[date, ...]
    listings: tuple[str, ...]
    scores: npt.NDArray[np.float64]
    available: npt.NDArray[np.bool_]
    score_chunk_hash: str


def read_development_score_rows(
    store: AlphaDevelopmentArtifactStore, *, numerical_result_hash: str, fold_surface_hash: str
) -> DevelopmentScoreRows:
    """Read published bytes only; target qualification stays with each consumer.

    Each chunk is resolved -- read and proved -- once: the store hands the rows
    of the resolution that loading the child performed.
    """

    numerical, scores = store.read_candidate_numerical_fold_result(numerical_result_hash)
    fold, validation = store.read_fold_surface(fold_surface_hash)
    return _join_score_rows(numerical, scores, fold, validation)


def development_score_rows_of(
    graph: AlphaDevelopmentVerifiedGraph, *, numerical_result_hash: str, fold_surface_hash: str
) -> DevelopmentScoreRows:
    """The same join over the rows a verified walk already proved.

    A graph is what one walk answered inside one request; joining from it
    proves no chunk a second time within that request and reads nothing the
    walk did not verify. A child or surface the graph does not carry is
    refused rather than read from the store behind the walk's back.
    """

    numerical = next(
        (value for value in graph.folds if value.numerical_result_hash == numerical_result_hash),
        None,
    )
    fold = graph.fold_surfaces.get(fold_surface_hash)
    validation = graph.validation_tables.get(fold_surface_hash)
    if numerical is None or fold is None or validation is None:
        raise CanonicalScoreSurfaceError("alpha_research.score_candidate_or_folds_missing")
    return _join_score_rows(
        numerical, graph.score_tables.get(numerical_result_hash), fold, validation
    )


def _join_score_rows(
    numerical: AlphaCandidateNumericalFoldResult,
    scores: pa.Table | None,
    fold: AlphaDevelopmentFoldSurface,
    validation: pa.Table,
) -> DevelopmentScoreRows:
    """One fold's proved score rows joined to the proved rows they predict."""

    if numerical.score_chunk is None or scores is None:
        raise CanonicalScoreSurfaceError("alpha_research.canonical_score_fold_chunk_missing")
    if numerical.fold_commitment_hash != fold.fold_commitment_hash:
        raise CanonicalScoreSurfaceError("alpha_research.score_fold_commitment_mismatch")
    if scores.num_rows != validation.num_rows:
        raise CanonicalScoreSurfaceError("alpha_research.canonical_score_axis_row_mismatch")
    # Both row axes were proved canonical (0..n-1, no nulls) when the chunks
    # were resolved; compared as arrays rather than as two Python lists.
    if not np.array_equal(
        scores["row_index"].to_numpy(zero_copy_only=False),
        validation["row_index"].to_numpy(zero_copy_only=False),
    ):
        raise CanonicalScoreSurfaceError("alpha_research.canonical_score_row_index_mismatch")
    sessions = tuple(validation["formation_session"].to_pylist())
    listings = tuple(str(v) for v in validation["listing_id"].to_pylist())
    if len(set(zip(sessions, listings, strict=True))) != len(sessions):
        raise CanonicalScoreSurfaceError("alpha_research.score_duplicate_row")
    values = np.asarray(
        scores["score"].combine_chunks().to_numpy(zero_copy_only=False), dtype=np.float64
    )
    flags = scores["availability"].combine_chunks()
    available = np.asarray(
        pc.fill_null(pc.equal(flags, "SCORED"), False).to_numpy(zero_copy_only=False),
        dtype=np.bool_,
    )
    if (
        set(pc.unique(flags).to_pylist()) - {"SCORED", "FEATURE_INCOMPLETE"}
        or np.isinf(values).any()
        or not np.array_equal(np.isfinite(values), available)
    ):
        raise CanonicalScoreSurfaceError("alpha_research.score_availability_mismatch")
    values.setflags(write=False)
    available.setflags(write=False)
    return DevelopmentScoreRows(
        sessions, listings, values, available, numerical.score_chunk.content_hash
    )


@dataclass(frozen=True, slots=True)
class DevelopmentScoreMatrix:
    sessions: tuple[date, ...]
    listings: tuple[str, ...]
    scores: npt.NDArray[np.float64]
    available: npt.NDArray[np.bool_]
    refs: tuple[tuple[str, str], ...]
    value_hash: str


@dataclass(frozen=True, slots=True)
class DevelopmentScoreSupport:
    """The full verified availability surface without an assembled score matrix."""

    sessions: tuple[date, ...]
    listings: tuple[str, ...]
    available: npt.NDArray[np.bool_]
    refs: tuple[tuple[str, str], ...]


def read_experiment_score_matrix(
    store: AlphaDevelopmentArtifactStore,
    receipt: AlphaDevelopmentExecutionReceipt,
    candidate_id: str,
) -> DevelopmentScoreMatrix:
    """The receipt reader must first verify the complete candidate/fold lineage."""
    entries = tuple(v for v in receipt.child_lineage if v.candidate_id == candidate_id)
    if tuple(v.fold_index for v in entries) != tuple(range(len(receipt.fold_surface_hashes))):
        raise CanonicalScoreSurfaceError("alpha_research.score_candidate_or_folds_missing")
    refs = tuple(
        (v.numerical_result_hash, receipt.fold_surface_hashes[v.fold_index]) for v in entries
    )
    parts = tuple(
        read_development_score_rows(store, numerical_result_hash=n, fold_surface_hash=f)
        for n, f in refs
    )
    sessions = tuple(sorted({s for p in parts for s in p.sessions}))
    listings = tuple(sorted({s for p in parts for s in p.listings}))
    matrix, available = assemble_score_matrix(parts, sessions=sessions, listings=listings)
    matrix.setflags(write=False)
    available.setflags(write=False)
    return DevelopmentScoreMatrix(
        sessions, listings, matrix, available, refs, canonical_score_value_identity(matrix)
    )


def read_experiment_score_support(
    graph: AlphaDevelopmentVerifiedGraph,
    candidate_id: str,
) -> DevelopmentScoreSupport:
    """The candidate's complete support surface from the rows a verified walk proved.

    The comparer keeps only availability: which cells the candidate scored,
    never the score values. Every fold's rows come from the graph the request
    verified, so no chunk is proved twice inside that request and nothing is
    read that the walk did not prove.
    """

    receipt = graph.receipt
    entries = tuple(value for value in receipt.child_lineage if value.candidate_id == candidate_id)
    if tuple(value.fold_index for value in entries) != tuple(
        range(len(receipt.fold_surface_hashes))
    ):
        raise CanonicalScoreSurfaceError("alpha_research.score_candidate_or_folds_missing")
    refs = tuple(
        (value.numerical_result_hash, receipt.fold_surface_hashes[value.fold_index])
        for value in entries
    )
    parts = tuple(
        development_score_rows_of(
            graph, numerical_result_hash=numerical_result_hash, fold_surface_hash=fold_surface_hash
        )
        for numerical_result_hash, fold_surface_hash in refs
    )
    sessions = tuple(sorted({session for part in parts for session in part.sessions}))
    listings = tuple(sorted({listing for part in parts for listing in part.listings}))
    available = assemble_score_support(parts, sessions=sessions, listings=listings)
    available.setflags(write=False)
    return DevelopmentScoreSupport(sessions, listings, available, refs)


def assemble_score_matrix(
    parts: tuple[DevelopmentScoreRows, ...],
    *,
    sessions: tuple[date, ...],
    listings: tuple[str, ...],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """One axis-aware assembly for canonical readback and authored research.

    Each part's rows are placed by the position of their session and listing
    on the given axes; a row whose session or listing is not on an axis, and a
    cell any two rows share (within a part or across parts), are refused. The
    positions are resolved per part as arrays and every cell is written once,
    the same values in the same cells as placing the rows one at a time.
    """

    session_positions = {s: i for i, s in enumerate(sessions)}
    listing_positions = {s: i for i, s in enumerate(listings)}
    matrix: npt.NDArray[np.float64] = np.full(
        (len(sessions), len(listings)), np.nan, dtype=np.float64
    )
    available: npt.NDArray[np.bool_] = np.zeros(matrix.shape, dtype=np.bool_)
    cells: list[npt.NDArray[np.intp]] = []
    for p in parts:
        if not (len(p.sessions) == len(p.listings) == len(p.scores) == len(p.available)):
            raise CanonicalScoreSurfaceError("alpha_research.canonical_score_axis_row_mismatch")
        try:
            rows: npt.NDArray[np.intp] = np.fromiter(
                (session_positions[s] for s in p.sessions), dtype=np.intp, count=len(p.sessions)
            )
            columns: npt.NDArray[np.intp] = np.fromiter(
                (listing_positions[s] for s in p.listings), dtype=np.intp, count=len(p.listings)
            )
        except KeyError as error:
            raise CanonicalScoreSurfaceError(
                "alpha_research.canonical_score_slice_axis_unbound"
            ) from error
        flat = rows * len(listings) + columns
        if np.unique(flat).size != flat.size:
            raise CanonicalScoreSurfaceError("alpha_research.score_duplicate_row")
        cells.append(flat)
        matrix.flat[flat] = p.scores
        available.flat[flat] = p.available
    if cells:
        all_cells = np.concatenate(cells)
        if np.unique(all_cells).size != all_cells.size:
            raise CanonicalScoreSurfaceError("alpha_research.score_duplicate_row")
    return matrix, available


def assemble_score_support(
    parts: tuple[DevelopmentScoreRows, ...],
    *,
    sessions: tuple[date, ...],
    listings: tuple[str, ...],
) -> npt.NDArray[np.bool_]:
    """Assemble the same exact availability semantics without score values."""

    session_positions = {session: index for index, session in enumerate(sessions)}
    listing_positions = {listing: index for index, listing in enumerate(listings)}
    available: npt.NDArray[np.bool_] = np.zeros((len(sessions), len(listings)), dtype=np.bool_)
    seen: set[tuple[date, str]] = set()
    for part in parts:
        for session, listing, valid in zip(
            part.sessions, part.listings, part.available, strict=True
        ):
            if (session, listing) in seen:
                raise CanonicalScoreSurfaceError("alpha_research.score_duplicate_row")
            seen.add((session, listing))
            try:
                available[session_positions[session], listing_positions[listing]] = valid
            except KeyError as error:
                raise CanonicalScoreSurfaceError(
                    "alpha_research.canonical_score_slice_axis_unbound"
                ) from error
    return available
