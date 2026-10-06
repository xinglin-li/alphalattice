"""Generic read-only operational evidence consumed by Project Guanyin."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class GuardianHealth(StrEnum):
    """Describe observed task liveness and terminal state in a read-only guardian.

    STARTING and HEALTHY distinguish startup from observed liveness. A stale heartbeat
    is an operational signal; terminal succeeded, blocked and deferred states retain
    the task owner's decision authority.
    """

    STARTING = "STARTING"
    HEALTHY = "HEALTHY"
    LIVENESS_STALE = "LIVENESS_STALE"
    TERMINAL_SUCCEEDED = "TERMINAL_SUCCEEDED"
    TERMINAL_BLOCKED = "TERMINAL_BLOCKED"
    TERMINAL_DEFERRED = "TERMINAL_DEFERRED"


class HumanRecoveryOption(_Contract):
    """Present an explicit human recovery choice without performing it.

    Attributes:
        option_id: Inspect failure, retry the same admission, or return to mandate revision.
        available: Whether this choice is currently offered.
        requires_user_confirmation: Whether selection requires explicit user confirmation.
        reason: Bounded user-safe explanation of availability.
    """

    option_id: Literal[
        "INSPECT_FAILURE",
        "RETRY_SAME_ADMISSION",
        "RETURN_TO_FRONT_DESK_TO_REVISE_MANDATE",
    ]
    available: bool
    requires_user_confirmation: bool
    reason: str = Field(min_length=1, max_length=300)


class HealthSignal(_Contract):
    """Seal a task liveness observation and its heartbeat age.

    Attributes:
        task_id: Task whose operational health is observed.
        status: Observed guardian health state.
        observed_at: Timezone-aware observation clock.
        last_heartbeat_at: Optional clock of the latest heartbeat.
        heartbeat_age_seconds: Optional nonnegative measured heartbeat age.
        signal_hash: Canonical signal identity excluding this hash.
    """

    task_id: str
    status: GuardianHealth
    observed_at: datetime
    last_heartbeat_at: datetime | None
    heartbeat_age_seconds: float | None = Field(default=None, ge=0)
    signal_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> HealthSignal:
        """Require an aware observation time and matching signal hash."""
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("Guardian health clock must be timezone-aware")
        if self.signal_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"signal_hash"})
        ):
            raise ValueError("Guardian health signal hash is invalid")
        return self


class RuntimeIncident(_Contract):
    """Seal a user-safe operational incident without active remediation authority.

    Attributes:
        incident_id: Canonical identity of the incident fields.
        incident_code: Bounded operational failure or liveness code.
        detected_at: Timezone-aware incident detection clock.
        stage_id: Optional affected task stage.
        user_safe_detail: Bounded explanation suitable for user readback.
        shadow_only: Always true: this record reports rather than remediates.
    """

    incident_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    incident_code: str = Field(min_length=3, max_length=120)
    detected_at: datetime
    stage_id: str | None
    user_safe_detail: str = Field(min_length=1, max_length=500)
    shadow_only: Literal[True] = True

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> RuntimeIncident:
        """Require an aware detection time and matching incident ID."""
        if self.detected_at.tzinfo is None or self.detected_at.utcoffset() is None:
            raise ValueError("Guardian incident clock must be timezone-aware")
        identity = self.model_dump(mode="json", exclude={"incident_id"})
        if self.incident_id != canonical_hash(identity):
            raise ValueError("Guardian incident identity is invalid")
        return self


class RuntimeGuardianProjection(_Contract):
    """Bind read-only guardian health to the admitted task and execution scope.

    The G0 projection records no active remediation attempts. Unhealthy states require
    an incident, and verified stage counts cannot exceed the task plan. Recovery
    choices are presented to a human rather than executed by this projection.

    Attributes:
        task_id: Task represented by the readback.
        subject_request_hash: Admitted subject request identity.
        subject_execution_binding_hash: Admitted subject execution identity.
        task_projection_hash: Task-control projection used as source evidence.
        health: Content-hashed liveness or terminal signal.
        verified_stage_count: Number of stages with verified completion evidence.
        total_stage_count: Positive number of planned stages.
        current_stage_id: Optional current stage handle.
        model_execution_attempt_count: Bounded model attempt count.
        model_recovery_mode: Whether protocol repair or budget recovery was used.
        last_verified_artifact_ref: Optional most recent verified artifact handle.
        incidents: Shadow operational incidents supporting unhealthy status.
        human_recovery_options: Explicit user-facing recovery choices.
        active_remediation_attempt_count: Always zero in read-only G0 mode.
        updated_at: Aware projection clock.
        projection_hash: Canonical identity of all other projection fields.
    """

    kind: Literal["RuntimeGuardianProjection"] = "RuntimeGuardianProjection"
    task_id: str
    subject_request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    subject_execution_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    guardian_mode: Literal["G0_READ_ONLY"] = "G0_READ_ONLY"
    task_projection_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    health: HealthSignal
    verified_stage_count: int = Field(ge=0)
    total_stage_count: int = Field(ge=1)
    current_stage_id: str | None
    model_execution_attempt_count: int = Field(ge=0, le=2)
    model_recovery_mode: Literal["NOT_USED", "PROTOCOL_REPAIR", "BUDGET_RECOVERY"]
    last_verified_artifact_ref: str | None
    incidents: tuple[RuntimeIncident, ...]
    human_recovery_options: tuple[HumanRecoveryOption, ...]
    active_remediation_attempt_count: Literal[0] = 0
    updated_at: datetime
    projection_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> RuntimeGuardianProjection:
        """Validate the clock, stage counts, incidents and projection hash."""
        if self.updated_at.tzinfo is None or self.updated_at.utcoffset() is None:
            raise ValueError("Guardian projection clock must be timezone-aware")
        if self.verified_stage_count > self.total_stage_count:
            raise ValueError("Guardian verified stages exceed task plan")
        if (
            self.health.status
            in {
                GuardianHealth.TERMINAL_BLOCKED,
                GuardianHealth.TERMINAL_DEFERRED,
                GuardianHealth.LIVENESS_STALE,
            }
            and not self.incidents
        ):
            raise ValueError("Guardian unhealthy projection requires an incident")
        if self.projection_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"projection_hash"})
        ):
            raise ValueError("Guardian projection hash is invalid")
        return self


__all__ = [
    "GuardianHealth",
    "HealthSignal",
    "HumanRecoveryOption",
    "RuntimeGuardianProjection",
    "RuntimeIncident",
]
