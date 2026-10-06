"""Strict artifact-ready contracts for WP40A Validation protocol."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from pydantic import Field, model_validator

from alphalattice.kernel.shared_kernel.domain.base import (
    DomainModel,
    Sha256Hex,
    UtcDatetime,
)
from alphalattice.kernel.shared_kernel.domain.serialization import canonical_json_bytes, sha256_hex
from alphalattice.kernel.validation.enums import (
    SplitMode,
)

NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(ge=1)]
RevisionIndex = Annotated[int, Field(ge=0)]
BoundedRevisionIndex = Annotated[int, Field(ge=0, le=2)]
RemainingRevisionBudget = Annotated[int, Field(ge=0, le=2)]
Uint64 = Annotated[int, Field(ge=0, le=18_446_744_073_709_551_615)]
ResampleCount = Annotated[int, Field(ge=100, le=10_000)]
BootstrapDtype = Literal["uint32-le"]
BootstrapBitGenerator = Literal["PCG64"]


def _logical_hash(value: object) -> str:
    return sha256_hex(canonical_json_bytes(value))


class RebalanceEvidencePoint(DomainModel):
    """Session date and availability clocks for one rebalance observation."""

    session_date: date
    evidence_available_at: UtcDatetime
    benchmark_available_at: UtcDatetime | None


class ValidationTimeline(DomainModel):
    """Ordered rebalance observations admitted to split construction."""

    points: tuple[RebalanceEvidencePoint, ...] = Field(min_length=1)


class ResearchSplitSpec(DomainModel):
    """Walk-forward window sizes, frozen clock and holdout commitment."""

    mode: SplitMode
    as_of_timestamp: UtcDatetime
    train_sessions: PositiveInt
    validation_sessions: PositiveInt
    step_sessions: PositiveInt
    purge_sessions: NonNegativeInt
    embargo_sessions: NonNegativeInt
    holdout_sessions: PositiveInt
    minimum_folds: PositiveInt

    @model_validator(mode="after")
    def validate_step(self) -> ResearchSplitSpec:
        """Require each step to cover its validation and embargo windows."""
        if self.step_sessions < self.validation_sessions + self.embargo_sessions:
            raise ValueError("step_sessions must cover validation and embargo sessions")
        return self


class SplitWindow(DomainModel):
    """Training, purge, validation and embargo sessions for one fold."""

    fold_index: NonNegativeInt
    train_sessions: tuple[date, ...] = Field(min_length=1)
    purge_sessions: tuple[date, ...]
    validation_sessions: tuple[date, ...] = Field(min_length=1)
    embargo_sessions: tuple[date, ...]


class SealedHoldoutDescriptor(DomainModel):
    """Count, boundaries and hash of the sealed tail sessions."""

    session_count: PositiveInt
    first_session: date
    last_session: date
    sessions_hash: Sha256Hex

    @model_validator(mode="after")
    def validate_range(self) -> SealedHoldoutDescriptor:
        """Reject a descriptor whose final session precedes its first."""
        if self.last_session < self.first_session:
            raise ValueError("holdout range is reversed")
        return self


class ResearchSplitPlan(DomainModel):
    """Committed walk-forward folds and sealed tail descriptor."""

    spec: ResearchSplitSpec
    timeline_hash: Sha256Hex
    windows: tuple[SplitWindow, ...] = Field(min_length=1)
    sealed_holdout: SealedHoldoutDescriptor
    split_hash: Sha256Hex

    @model_validator(mode="after")
    def validate_hash(self) -> ResearchSplitPlan:
        """Require the split hash to bind every declared window and input."""
        expected = research_split_hash(
            spec=self.spec,
            timeline_hash=self.timeline_hash,
            windows=self.windows,
            sealed_holdout=self.sealed_holdout,
        )
        if self.split_hash != expected:
            raise ValueError("research split hash is invalid")
        return self


def timeline_logical_hash(timeline: ValidationTimeline) -> str:
    """Hash the ordered points of a validation timeline."""
    return _logical_hash({"points": timeline.points})


def holdout_sessions_hash(sessions: tuple[date, ...]) -> str:
    """Hash the ordered sessions of a sealed tail."""
    return _logical_hash({"sessions": sessions})


def research_split_hash(
    *,
    spec: ResearchSplitSpec,
    timeline_hash: str,
    windows: tuple[SplitWindow, ...],
    sealed_holdout: SealedHoldoutDescriptor,
) -> str:
    """Hash a split specification, timeline, windows and sealed tail."""
    return _logical_hash(
        {
            "sealed_holdout": sealed_holdout,
            "spec": spec,
            "timeline_hash": timeline_hash,
            "windows": windows,
        }
    )


__all__ = [
    "RebalanceEvidencePoint",
    "ResearchSplitPlan",
    "ResearchSplitSpec",
    "SealedHoldoutDescriptor",
    "SplitWindow",
    "ValidationTimeline",
    "holdout_sessions_hash",
    "research_split_hash",
    "timeline_logical_hash",
]
