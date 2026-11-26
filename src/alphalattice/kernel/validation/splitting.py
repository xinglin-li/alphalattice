"""Deterministic walk-forward split construction."""

from __future__ import annotations

from alphalattice.kernel.validation.contracts import (
    ResearchSplitPlan,
    ResearchSplitSpec,
    SealedHoldoutDescriptor,
    SplitWindow,
    ValidationTimeline,
    holdout_sessions_hash,
    research_split_hash,
    timeline_logical_hash,
)
from alphalattice.kernel.validation.enums import SplitMode
from alphalattice.kernel.validation.errors import ValidationProtocolError


def _error(message: str, code: str) -> ValidationProtocolError:
    return ValidationProtocolError(message, code=code)


def _validate_timeline(timeline: ValidationTimeline, spec: ResearchSplitSpec) -> None:
    previous = None
    for point in timeline.points:
        if previous is not None and point.session_date <= previous:
            raise _error(
                "rebalance sessions must be strictly increasing and unique",
                "validation.timeline_invalid",
            )
        previous = point.session_date
        if point.benchmark_available_at is None:
            raise _error(
                f"benchmark availability is missing for {point.session_date}",
                "validation.benchmark_incomplete",
            )
        if (
            point.evidence_available_at > spec.as_of_timestamp
            or point.benchmark_available_at > spec.as_of_timestamp
        ):
            raise _error(
                f"evidence is unavailable at the frozen clock for {point.session_date}",
                "validation.future_data",
            )


def verify_plan_hash(plan: ResearchSplitPlan) -> None:
    """Reject a split plan whose content does not match its declared hash.

    Args:
        plan: Split plan to verify.

    Raises:
        ValidationProtocolError: If the split hash does not match the plan.

    """
    expected = research_split_hash(
        spec=plan.spec,
        timeline_hash=plan.timeline_hash,
        windows=plan.windows,
        sealed_holdout=plan.sealed_holdout,
    )
    if expected != plan.split_hash:
        raise _error("split plan hash is invalid", "validation.attempt_input_mismatch")


def build_research_split(
    timeline: ValidationTimeline,
    spec: ResearchSplitSpec,
) -> ResearchSplitPlan:
    """Build one explicit walk-forward plan and sealed tail commitment."""
    if not isinstance(timeline, ValidationTimeline) or not isinstance(spec, ResearchSplitSpec):
        raise _error(
            "split inputs must use ValidationTimeline and ResearchSplitSpec",
            "validation.invalid_split_spec",
        )
    _validate_timeline(timeline, spec)
    sessions = tuple(point.session_date for point in timeline.points)
    holdout_start = len(sessions) - spec.holdout_sessions
    if holdout_start <= 0:
        raise _error(
            "timeline cannot provide the requested sealed holdout",
            "validation.insufficient_history",
        )
    final_development_boundary = holdout_start - spec.purge_sessions
    validation_start = spec.train_sessions + spec.purge_sessions
    windows: list[SplitWindow] = []
    fold_index = 0
    while True:
        validation_end = validation_start + spec.validation_sessions
        embargo_end = validation_end + spec.embargo_sessions
        if embargo_end > final_development_boundary:
            break
        train_end = validation_start - spec.purge_sessions
        train_start = 0
        if spec.mode is SplitMode.ROLLING:
            train_start = train_end - spec.train_sessions
        if train_start < 0 or train_end - train_start < spec.train_sessions:
            break
        windows.append(
            SplitWindow(
                fold_index=fold_index,
                train_sessions=sessions[train_start:train_end],
                purge_sessions=sessions[train_end:validation_start],
                validation_sessions=sessions[validation_start:validation_end],
                embargo_sessions=sessions[validation_end:embargo_end],
            )
        )
        fold_index += 1
        validation_start += spec.step_sessions
    if len(windows) < spec.minimum_folds:
        raise _error(
            "timeline cannot provide the requested complete walk-forward folds",
            "validation.insufficient_history",
        )
    holdout = sessions[holdout_start:]
    descriptor = SealedHoldoutDescriptor(
        session_count=len(holdout),
        first_session=holdout[0],
        last_session=holdout[-1],
        sessions_hash=holdout_sessions_hash(holdout),
    )
    timeline_hash = timeline_logical_hash(timeline)
    ordered_windows = tuple(windows)
    return ResearchSplitPlan(
        spec=spec,
        timeline_hash=timeline_hash,
        windows=ordered_windows,
        sealed_holdout=descriptor,
        split_hash=research_split_hash(
            spec=spec,
            timeline_hash=timeline_hash,
            windows=ordered_windows,
            sealed_holdout=descriptor,
        ),
    )


__all__ = ["build_research_split", "verify_plan_hash"]
