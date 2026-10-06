"""Durable Host-owned task control for long desktop research work."""

from .contracts import (
    ResearchGoal,
    ResearchPlan,
    StageFailureCause,
    TaskCommand,
    TaskCommandKind,
    TaskEvidence,
    TaskExecution,
    TaskExecutionCompatibility,
    TaskHeartbeatSignal,
    TaskInputEnvelope,
    TaskLifecycle,
    TaskRecord,
    TaskSafeProjection,
    TaskStageReceipt,
    WorkItemDefinition,
    WorkItemLifecycle,
    WorkItemState,
)
from .projection import render_task_projection

__all__ = [
    "ResearchGoal",
    "ResearchPlan",
    "StageFailureCause",
    "TaskCommand",
    "TaskCommandKind",
    "TaskEvidence",
    "TaskExecution",
    "TaskExecutionCompatibility",
    "TaskHeartbeatSignal",
    "TaskInputEnvelope",
    "TaskLifecycle",
    "TaskRecord",
    "TaskSafeProjection",
    "TaskStageReceipt",
    "WorkItemDefinition",
    "WorkItemLifecycle",
    "WorkItemState",
    "render_task_projection",
]
