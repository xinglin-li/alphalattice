"""Target-independent average-linkage redundancy evidence for Factor Research."""

from __future__ import annotations

import math
from itertools import combinations, pairwise
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.foundation.factor_research.evaluation.rank_correlation import (
    period_pair_correlation_matrix,
)
from alphalattice.foundation.factor_research.programs.sealed import seal_contract
from alphalattice.kernel.shared_kernel.identity import canonical_hash

type FloatArray = npt.NDArray[np.float64]


class FactorRedundancyBoundaryError(ValueError):
    """Stable failure raised when redundancy evidence is incomplete."""


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class FactorRedundancyPolicy(_Contract):
    """Sealed rules for pairwise similarity and average-linkage clustering."""

    kind: Literal["FactorRedundancyPolicy"] = "FactorRedundancyPolicy"
    correlation_semantics: Literal["MEDIAN_DAILY_PAIRWISE_COMMON_SPEARMAN"] = (
        "MEDIAN_DAILY_PAIRWISE_COMMON_SPEARMAN"
    )
    distance_semantics: Literal["ONE_MINUS_ABSOLUTE_RHO"] = "ONE_MINUS_ABSOLUTE_RHO"
    linkage: Literal["AVERAGE"] = "AVERAGE"
    minimum_common_listings: int = Field(default=100, ge=2)
    minimum_formal_periods: int = Field(default=20, ge=1)
    distance_cut: float = Field(default=0.2, gt=0.0, lt=1.0)
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_policy(self) -> FactorRedundancyPolicy:
        """Verify the redundancy policy's content hash.

        Returns:
            The validated policy.

        Raises:
            ValueError: If its hash does not match its rules.
        """
        expected = canonical_hash(self.model_dump(mode="json", exclude={"policy_hash"}))
        if self.policy_hash != expected:
            raise ValueError("Factor redundancy policy hash is invalid")
        return self


class FactorRedundancyPairEvidence(_Contract):
    """Sealed common-period correlation and distance for one factor pair."""

    kind: Literal["FactorRedundancyPairEvidence"] = "FactorRedundancyPairEvidence"
    left_factor_id: str = Field(min_length=1)
    right_factor_id: str = Field(min_length=1)
    common_formal_period_count: int = Field(ge=0)
    median_cross_section_spearman: float | None
    distance: float | None
    pair_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_pair(self) -> FactorRedundancyPairEvidence:
        """Verify pair order, resolved distance and content hash.

        Returns:
            The validated pair evidence.

        Raises:
            ValueError: If the pair order, distance or hash is inconsistent.
        """
        if self.left_factor_id >= self.right_factor_id:
            raise ValueError("Factor redundancy pair order is invalid")
        if (self.median_cross_section_spearman is None) != (self.distance is None):
            raise ValueError("Factor redundancy pair resolution is inconsistent")
        if self.median_cross_section_spearman is not None:
            if not -1.0 <= self.median_cross_section_spearman <= 1.0:
                raise ValueError("Factor redundancy correlation is invalid")
            expected_distance = 1.0 - abs(self.median_cross_section_spearman)
            if self.distance is None or not math.isclose(
                self.distance, expected_distance, rel_tol=0.0, abs_tol=1e-15
            ):
                raise ValueError("Factor redundancy distance is invalid")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"pair_hash"}))
        if self.pair_hash != expected:
            raise ValueError("Factor redundancy pair hash is invalid")
        return self


class FactorRedundancyCluster(_Contract):
    """Sealed member set and within-cluster distance evidence."""

    kind: Literal["FactorRedundancyCluster"] = "FactorRedundancyCluster"
    cluster_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    member_factor_ids: tuple[str, ...] = Field(min_length=1)
    resolved_within_pair_count: int = Field(ge=0)
    maximum_within_pair_distance: float | None
    cluster_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_cluster(self) -> FactorRedundancyCluster:
        """Verify member order, distance range and cluster hash.

        Returns:
            The validated cluster.

        Raises:
            ValueError: If its members, distance or hash are invalid.
        """
        if self.member_factor_ids != tuple(sorted(set(self.member_factor_ids))):
            raise ValueError("Factor redundancy cluster members are not canonical")
        if self.maximum_within_pair_distance is not None and not (
            0.0 <= self.maximum_within_pair_distance <= 1.0
        ):
            raise ValueError("Factor redundancy cluster distance is invalid")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"cluster_hash"}))
        if self.cluster_hash != expected:
            raise ValueError("Factor redundancy cluster hash is invalid")
        return self


class FactorRedundancyStructure(_Contract):
    """Sealed factor-axis partition into deterministic redundancy clusters."""

    kind: Literal["FactorRedundancyStructure"] = "FactorRedundancyStructure"
    feature_panel_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_panel_manifest_ref: str = Field(min_length=1)
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    factor_ids: tuple[str, ...] = Field(min_length=1)
    formal_period_count: int = Field(ge=1)
    pair_evidence: tuple[FactorRedundancyPairEvidence, ...]
    clusters: tuple[FactorRedundancyCluster, ...] = Field(min_length=1)
    structure_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_structure(self) -> FactorRedundancyStructure:
        """Verify the factor axis, complete pair family and structure hash.

        Returns:
            The validated redundancy structure.

        Raises:
            ValueError: If the axis, pairs, partition or hash are inconsistent.
        """
        if self.factor_ids != tuple(sorted(set(self.factor_ids))):
            raise ValueError("Factor redundancy axis is not canonical")
        expected_pairs = len(self.factor_ids) * (len(self.factor_ids) - 1) // 2
        if len(self.pair_evidence) != expected_pairs:
            raise ValueError("Factor redundancy pair family is incomplete")
        flattened = tuple(
            sorted(member for cluster in self.clusters for member in cluster.member_factor_ids)
        )
        if flattened != self.factor_ids:
            raise ValueError("Factor redundancy clusters do not partition the factor axis")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"structure_hash"}))
        if self.structure_hash != expected:
            raise ValueError("Factor redundancy structure hash is invalid")
        return self


def build_factor_redundancy_policy(
    *,
    minimum_common_listings: int = 100,
    minimum_formal_periods: int = 20,
    distance_cut: float = 0.2,
) -> FactorRedundancyPolicy:
    """Seal the common-listing, period and distance thresholds.

    Args:
        minimum_common_listings: Paired listings needed for a period.
        minimum_formal_periods: Periods needed for a resolved pair.
        distance_cut: Maximum distance admitted to a cluster.

    Returns:
        The content-bound redundancy policy.
    """
    return seal_contract(
        FactorRedundancyPolicy,
        "policy_hash",
        minimum_common_listings=minimum_common_listings,
        minimum_formal_periods=minimum_formal_periods,
        distance_cut=distance_cut,
    )


def _float_column(table: pa.Table, name: str) -> FloatArray:
    return cast(
        FloatArray,
        np.asarray(table[name].combine_chunks().to_numpy(zero_copy_only=False), dtype=np.float64),
    )


def _pair_evidence(
    *,
    table: pa.Table,
    factor_ids: tuple[str, ...],
    policy: FactorRedundancyPolicy,
) -> tuple[FactorRedundancyPairEvidence, ...]:
    ordered = table.take(
        pc.sort_indices(
            table,
            sort_keys=[("session_date", "ascending"), ("listing_id", "ascending")],
        )
    ).combine_chunks()
    session_column = ordered["session_date"]
    row_count = ordered.num_rows
    if row_count > 1:
        same_session = pc.equal(session_column.slice(1), session_column.slice(0, row_count - 1))
        same_listing = pc.equal(
            ordered["listing_id"].slice(1), ordered["listing_id"].slice(0, row_count - 1)
        )
        if bool(pc.any(pc.and_(same_session, same_listing)).as_py()):
            raise FactorRedundancyBoundaryError("factor_research.redundancy_duplicate_feature_row")
    values = np.column_stack([_float_column(ordered, factor_id) for factor_id in factor_ids])
    # Period boundaries: the sorted rows are grouped by session, and a period
    # starts wherever the session changes.
    session_days = session_column.combine_chunks().to_numpy(zero_copy_only=False)
    period_starts: npt.NDArray[np.intp] = np.flatnonzero(session_days[1:] != session_days[:-1]) + 1
    boundaries = [0, *period_starts.tolist(), row_count]
    # One row per pair, one column per period, NaN where the period did not
    # resolve the pair: the same per-pair series the per-period dictionaries
    # held, in the same period order, without a Python append per pair.
    left_index, right_index = np.triu_indices(len(factor_ids), 1)
    observations: FloatArray = np.full(
        (left_index.size, len(boundaries) - 1), np.nan, dtype=np.float64
    )
    for period, (start, stop) in enumerate(pairwise(boundaries)):
        correlations, _counts = period_pair_correlation_matrix(
            values[start:stop],
            minimum_common_count=policy.minimum_common_listings,
        )
        observations[:, period] = correlations[left_index, right_index]
    evidence: list[FactorRedundancyPairEvidence] = []
    for pair, (left, right) in enumerate(zip(left_index, right_index, strict=True)):
        series = observations[pair][np.isfinite(observations[pair])]
        median = float(np.median(series)) if series.size >= policy.minimum_formal_periods else None
        distance = None if median is None else 1.0 - abs(median)
        evidence.append(
            seal_contract(
                FactorRedundancyPairEvidence,
                "pair_hash",
                left_factor_id=factor_ids[int(left)],
                right_factor_id=factor_ids[int(right)],
                common_formal_period_count=int(series.size),
                median_cross_section_spearman=median,
                distance=distance,
            )
        )
    return tuple(evidence)


def _average_linkage_clusters(
    *,
    factor_ids: tuple[str, ...],
    pairs: tuple[FactorRedundancyPairEvidence, ...],
    distance_cut: float,
) -> tuple[tuple[str, ...], ...]:
    unresolved = tuple(pair for pair in pairs if pair.distance is None)
    if unresolved:
        raise FactorRedundancyBoundaryError("factor_research.redundancy_pair_unresolved")
    base = {
        (pair.left_factor_id, pair.right_factor_id): cast(float, pair.distance) for pair in pairs
    }
    clusters: list[tuple[str, ...]] = [(factor_id,) for factor_id in factor_ids]

    def pair_key(left: str, right: str) -> tuple[str, str]:
        return (left, right) if left < right else (right, left)

    def distance(left: tuple[str, ...], right: tuple[str, ...]) -> float:
        values = [base[pair_key(a, b)] for a in left for b in right]
        return float(np.mean(np.asarray(values, dtype=np.float64)))

    while True:
        candidates = [
            (distance(left, right), left, right)
            for left_index, left in enumerate(clusters)
            for right in clusters[left_index + 1 :]
        ]
        if not candidates:
            break
        best_distance, left, right = min(candidates, key=lambda item: item)
        if best_distance > distance_cut:
            break
        clusters.remove(left)
        clusters.remove(right)
        clusters.append(tuple(sorted((*left, *right))))
        clusters.sort()
    return tuple(clusters)


def build_factor_redundancy_structure(
    *,
    feature_table: pa.Table,
    feature_panel_snapshot_hash: str,
    feature_panel_manifest_ref: str,
    factor_ids: tuple[str, ...],
    policy: FactorRedundancyPolicy,
) -> FactorRedundancyStructure:
    """Build complete pair evidence and deterministic average-linkage clusters."""
    policy = FactorRedundancyPolicy.model_validate(policy)
    if factor_ids != tuple(sorted(set(factor_ids))):
        raise FactorRedundancyBoundaryError("factor_research.redundancy_factor_axis_invalid")
    required = {"session_date", "listing_id", *factor_ids}
    missing = sorted(required - set(feature_table.schema.names))
    if missing:
        raise FactorRedundancyBoundaryError(
            f"factor_research.redundancy_feature_column_missing:{','.join(missing)}"
        )
    sessions = tuple(sorted(set(feature_table["session_date"].to_pylist())))
    if not sessions:
        raise FactorRedundancyBoundaryError("factor_research.redundancy_surface_empty")
    pair_evidence = _pair_evidence(table=feature_table, factor_ids=factor_ids, policy=policy)
    cluster_members = _average_linkage_clusters(
        factor_ids=factor_ids,
        pairs=pair_evidence,
        distance_cut=policy.distance_cut,
    )
    distance_by_pair: dict[tuple[str, str], float | None] = {
        (pair.left_factor_id, pair.right_factor_id): pair.distance for pair in pair_evidence
    }
    clusters: list[FactorRedundancyCluster] = []
    for members in cluster_members:
        within: list[float] = []
        for left, right in combinations(members, 2):
            value = distance_by_pair[(left, right) if left < right else (right, left)]
            if value is not None:
                within.append(value)
        cluster_id = canonical_hash(
            {"member_factor_ids": members, "policy_hash": policy.policy_hash}
        )
        clusters.append(
            seal_contract(
                FactorRedundancyCluster,
                "cluster_hash",
                cluster_id=cluster_id,
                member_factor_ids=members,
                resolved_within_pair_count=len(within),
                maximum_within_pair_distance=max(within) if within else None,
            )
        )
    return seal_contract(
        FactorRedundancyStructure,
        "structure_hash",
        feature_panel_snapshot_hash=feature_panel_snapshot_hash,
        feature_panel_manifest_ref=feature_panel_manifest_ref,
        policy_hash=policy.policy_hash,
        factor_ids=factor_ids,
        formal_period_count=len(sessions),
        pair_evidence=pair_evidence,
        clusters=tuple(clusters),
    )


__all__ = [
    "FactorRedundancyBoundaryError",
    "FactorRedundancyCluster",
    "FactorRedundancyPairEvidence",
    "FactorRedundancyPolicy",
    "FactorRedundancyStructure",
    "build_factor_redundancy_policy",
    "build_factor_redundancy_structure",
]
