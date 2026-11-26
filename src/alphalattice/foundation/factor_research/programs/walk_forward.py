"""Label-availability-aware one-session Factor Research split compiler."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

import pyarrow.compute as pc
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.foundation.factor_research.inputs.execution_target import FactorTargetSurface
from alphalattice.foundation.factor_research.programs.sealed import seal_contract
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.validation.contracts import (
    RebalanceEvidencePoint,
    ResearchSplitPlan,
    ResearchSplitSpec,
    ValidationTimeline,
)
from alphalattice.kernel.validation.enums import SplitMode
from alphalattice.kernel.validation.splitting import build_research_split, verify_plan_hash


class FactorWalkForwardBoundaryError(ValueError):
    """Stable failure raised before a Factor split can be admitted."""


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class FactorWalkForwardPolicy(_Contract):
    """Declare rolling development folds while keeping the reserved holdout unread.

    Attributes:
        kind: Split-policy discriminator.
        mode: Rolling training-window mode.
        prediction_horizon_sessions: One-session prediction horizon.
        train_sessions: Number of source sessions in each training interval.
        purge_sessions: One-session separation for training labels.
        validation_sessions: Number of sessions in each validation interval.
        step_sessions: Distance between consecutive fold starts.
        embargo_sessions: Additional excluded sessions after validation.
        sealed_holdout_sessions: Count reserved from development use.
        minimum_folds: Required number of complete development folds.
        policy_hash: Canonical identity of the split declaration.
    """

    kind: Literal["FactorWalkForwardPolicy"] = "FactorWalkForwardPolicy"
    mode: Literal["ROLLING"] = "ROLLING"
    prediction_horizon_sessions: Literal[1] = 1
    train_sessions: int = Field(default=756, ge=2)
    purge_sessions: Literal[1] = 1
    validation_sessions: int = Field(default=252, ge=1)
    step_sessions: int = Field(default=252, ge=1)
    embargo_sessions: int = Field(default=0, ge=0)
    sealed_holdout_sessions: int = Field(default=252, ge=1)
    minimum_folds: int = Field(default=3, ge=1)
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_policy(self) -> FactorWalkForwardPolicy:
        """Verify fold spacing and the split policy's canonical identity.

        Returns:
            This validated immutable contract.

        Raises:
            ValueError: Step is shorter than validation plus embargo, or the policy hash is invalid.
        """
        if self.step_sessions < self.validation_sessions + self.embargo_sessions:
            raise ValueError("Factor walk-forward step is too short")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"policy_hash"}))
        if self.policy_hash != expected:
            raise ValueError("Factor walk-forward policy hash is invalid")
        return self


class FactorWalkForwardPlan(_Contract):
    """Bind formal development folds to verified training-label availability.

    Attributes:
        kind: Split-plan discriminator.
        policy_hash: Qualified rolling policy.
        target_surface_hash: Target surface used for label schedule verification.
        formal_split: Verified formal split including its protected holdout commitment.
        formal_session_count: Total sessions in the admitted panel calendar.
        consumed_development_session_count: Unique sessions used by complete training/validation
            folds.
        label_availability_hash: Canonical evidence of consumed training labels and availability
            instants.
        plan_hash: Canonical identity of the bound Factor plan.
    """

    kind: Literal["FactorWalkForwardPlan"] = "FactorWalkForwardPlan"
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_surface_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    formal_split: ResearchSplitPlan
    formal_session_count: int = Field(ge=1)
    consumed_development_session_count: int = Field(ge=1)
    label_availability_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_plan(self) -> FactorWalkForwardPlan:
        """Verify the formal split hash and bound Factor plan identity.

        Returns:
            This validated immutable contract.

        Raises:
            ValueError: The formal split or canonical plan hash is invalid.
        """
        verify_plan_hash(self.formal_split)
        expected = canonical_hash(self.model_dump(mode="json", exclude={"plan_hash"}))
        if self.plan_hash != expected:
            raise ValueError("Factor walk-forward plan hash is invalid")
        return self


def build_factor_walk_forward_policy(
    **overrides: int,
) -> FactorWalkForwardPolicy:
    """Seal the installed rolling split defaults with declared integer overrides.

    Args:
        **overrides: Policy fields to replace before immutable contract validation.

    Returns:
        Validated rolling policy with its canonical identity.

    Raises:
        ValueError: An override is unknown or violates horizon, bounds, or fold spacing.
    """
    values = {
        "train_sessions": 756,
        "validation_sessions": 252,
        "step_sessions": 252,
        "embargo_sessions": 0,
        "sealed_holdout_sessions": 252,
        "minimum_folds": 3,
        **overrides,
    }
    return seal_contract(FactorWalkForwardPolicy, "policy_hash", **values)


_SCHEDULE_COLUMNS = ("holding_end_session", "holding_end_open_at", "formation_close_at")


def _target_schedule(surface: FactorTargetSurface) -> dict[date, tuple[date, datetime, datetime]]:
    """One label-availability triple per formation session, proved unique in Arrow.

    Every row of a formation must name the same holding end, availability
    instant and formation close; the check is a per-formation distinct count
    of each column with nulls counted (one distinct value per column is
    exactly one distinct triple), and only the one row per formation crosses
    into Python.
    """
    table = surface.table.select(["formation_session", *_SCHEDULE_COLUMNS])
    distinct = table.group_by("formation_session").aggregate(
        [(column, "count_distinct", pc.CountOptions(mode="all")) for column in _SCHEDULE_COLUMNS]
    )
    for column in _SCHEDULE_COLUMNS:
        if not pc.all(pc.equal(distinct[f"{column}_count_distinct"], 1)).as_py():
            raise FactorWalkForwardBoundaryError("factor_research.label_availability_axis_mismatch")
    first = table.group_by("formation_session", use_threads=False).aggregate(
        [(column, "first") for column in _SCHEDULE_COLUMNS]
    )
    formations: list[date] = first["formation_session"].to_pylist()
    holding_ends: list[date] = first["holding_end_session_first"].to_pylist()
    available: list[datetime] = first["holding_end_open_at_first"].to_pylist()
    formation_closes: list[datetime] = first["formation_close_at_first"].to_pylist()
    return {
        formation: (holding_end, available_at, formation_close_at)
        for formation, holding_end, available_at, formation_close_at in zip(
            formations, holding_ends, available, formation_closes, strict=True
        )
    }


def _formal_split(
    *,
    panel_sessions: tuple[date, ...],
    frozen_at: datetime,
    policy: FactorWalkForwardPolicy,
) -> ResearchSplitPlan:
    timeline = ValidationTimeline(
        points=tuple(
            RebalanceEvidencePoint(
                session_date=session,
                evidence_available_at=frozen_at,
                benchmark_available_at=frozen_at,
            )
            for session in panel_sessions
        )
    )
    return build_research_split(
        timeline,
        ResearchSplitSpec(
            mode=SplitMode.ROLLING,
            as_of_timestamp=frozen_at,
            train_sessions=policy.train_sessions,
            validation_sessions=policy.validation_sessions,
            step_sessions=policy.step_sessions,
            purge_sessions=policy.purge_sessions,
            embargo_sessions=policy.embargo_sessions,
            holdout_sessions=policy.sealed_holdout_sessions,
            minimum_folds=policy.minimum_folds,
        ),
    )


def factor_walk_forward_development_sessions(
    *,
    panel_sessions: tuple[date, ...],
    frozen_at: datetime,
    policy: FactorWalkForwardPolicy,
) -> tuple[date, ...]:
    """Return only sessions consumed by complete development folds.

    Args:
        panel_sessions: Nonempty sorted unique admitted session calendar.
        frozen_at: Timezone-aware evidence clock used by the formal split compiler.
        policy: Qualified rolling development policy.

    Returns:
        Sorted unique union of complete folds' training and validation sessions.

    Raises:
        FactorWalkForwardBoundaryError: The clock is naive or the calendar is invalid.
        ValueError: The policy or formal split cannot be admitted.
    """
    if frozen_at.tzinfo is None or frozen_at.utcoffset() is None:
        raise FactorWalkForwardBoundaryError("factor_research.split_clock_invalid")
    if not panel_sessions or panel_sessions != tuple(sorted(set(panel_sessions))):
        raise FactorWalkForwardBoundaryError("factor_research.panel_calendar_invalid")
    policy = FactorWalkForwardPolicy.model_validate(policy)
    plan = _formal_split(
        panel_sessions=panel_sessions,
        frozen_at=frozen_at,
        policy=policy,
    )
    return tuple(
        sorted(
            {
                session
                for window in plan.windows
                for session in (*window.train_sessions, *window.validation_sessions)
            }
        )
    )


def compile_factor_walk_forward_plan(
    *,
    panel_sessions: tuple[date, ...],
    target_surface: FactorTargetSurface,
    frozen_at: datetime,
    policy: FactorWalkForwardPolicy,
) -> FactorWalkForwardPlan:
    """Build the formal split, then prove every consumed label is available.

    Args:
        panel_sessions: Sorted unique admitted calendar used for formal fold positions.
        target_surface: Target rows proving one execution/availability schedule per formation.
        frozen_at: Timezone-aware evidence clock for the formal split.
        policy: Qualified rolling policy including the unread holdout commitment.

    Returns:
        Sealed formal plan with target lineage, population counts, and training-label
        availability evidence. Every training label ends and is available by validation close.

    Raises:
        FactorWalkForwardBoundaryError: Clock/calendar, schedule uniqueness, consumed
            labels, purge separation, or label availability fails its causal boundary.
        ValueError: The policy or formal split is invalid.
        KeyError: A required schedule column or holding-end calendar position is absent.
    """
    if frozen_at.tzinfo is None or frozen_at.utcoffset() is None:
        raise FactorWalkForwardBoundaryError("factor_research.split_clock_invalid")
    if not panel_sessions or panel_sessions != tuple(sorted(set(panel_sessions))):
        raise FactorWalkForwardBoundaryError("factor_research.panel_calendar_invalid")
    policy = FactorWalkForwardPolicy.model_validate(policy)
    formal = _formal_split(
        panel_sessions=panel_sessions,
        frozen_at=frozen_at,
        policy=policy,
    )
    schedule = _target_schedule(target_surface)
    consumed = tuple(
        sorted(
            {
                session
                for window in formal.windows
                for session in (*window.train_sessions, *window.validation_sessions)
            }
        )
    )
    if any(session not in schedule for session in consumed):
        raise FactorWalkForwardBoundaryError("factor_research.label_availability_incomplete")
    panel_positions = {session: index for index, session in enumerate(panel_sessions)}
    availability_rows: list[dict[str, object]] = []
    for window in formal.windows:
        validation_start = window.validation_sessions[0]
        validation_start_close = schedule[validation_start][2]
        for session in window.train_sessions:
            holding_end, available_at, _formation_close = schedule[session]
            if panel_positions[holding_end] > panel_positions[validation_start]:
                raise FactorWalkForwardBoundaryError("factor_research.label_purge_incomplete")
            if available_at > validation_start_close:
                raise FactorWalkForwardBoundaryError("factor_research.label_future_data")
            availability_rows.append(
                {
                    "fold_index": window.fold_index,
                    "formation_session": session,
                    "holding_end_session": holding_end,
                    "available_at": available_at,
                }
            )
    label_availability_hash = canonical_hash(availability_rows)
    return seal_contract(
        FactorWalkForwardPlan,
        "plan_hash",
        policy_hash=policy.policy_hash,
        target_surface_hash=target_surface.manifest.surface_hash,
        formal_split=formal,
        formal_session_count=len(panel_sessions),
        consumed_development_session_count=len(consumed),
        label_availability_hash=label_availability_hash,
    )


__all__ = [
    "FactorWalkForwardBoundaryError",
    "FactorWalkForwardPlan",
    "FactorWalkForwardPolicy",
    "build_factor_walk_forward_policy",
    "compile_factor_walk_forward_plan",
    "factor_walk_forward_development_sessions",
]
