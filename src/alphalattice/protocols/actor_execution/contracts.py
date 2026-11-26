"""Framework-neutral provenance for optional human or model actors."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash


class ProjectionAudit(BaseModel):
    """Receipt-safe audit for one admitted actor-facing projection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    projection_id: str
    semantics_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    utf8_bytes: int = Field(ge=1)
    token_estimate_method: Literal["CL100K_PROXY"] = "CL100K_PROXY"
    cl100k_proxy_tokens: int = Field(ge=1)


class ModelAttemptContextAudit(BaseModel):
    """Projection and provider usage evidence for one model-actor attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    attempt: int = Field(ge=1, le=2)
    projections: tuple[ProjectionAudit, ...] = Field(min_length=1)
    total_projection_bytes: int = Field(ge=1)
    total_cl100k_proxy_tokens: int = Field(ge=1)
    provider_prompt_tokens: int | None = Field(default=None, ge=0)
    provider_completion_tokens: int | None = Field(default=None, ge=0)
    provider_cache_hit_tokens: int | None = Field(default=None, ge=0)
    provider_cache_miss_tokens: int | None = Field(default=None, ge=0)
    model_calls: int | None = Field(default=None, ge=0)
    latency_seconds: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_totals(self) -> ModelAttemptContextAudit:
        """Require reported totals to equal the admitted projection sums."""
        if self.total_projection_bytes != sum(value.utf8_bytes for value in self.projections):
            raise ValueError("model context audit byte total is invalid")
        if self.total_cl100k_proxy_tokens != sum(
            value.cl100k_proxy_tokens for value in self.projections
        ):
            raise ValueError("model context audit token total is invalid")
        return self


class AgentExecutionBinding(BaseModel):
    """Host-authored provenance for one accepted structured actor response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str
    mode: str
    profile_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_protocol: str
    response_protocol_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    concrete_schema_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class ActorKind(StrEnum):
    """The admitted producer of one typed, non-authoritative proposal."""

    HUMAN = "HUMAN"
    INSTALLED_AGENT = "INSTALLED_AGENT"
    EXTERNAL_AUTOMATION = "EXTERNAL_AUTOMATION"
    HOST_FALLBACK = "HOST_FALLBACK"


class ActorSubmissionBinding(BaseModel):
    """Host-authored provenance for one proposal submitted for sealing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    actor_kind: ActorKind
    actor_id: str = Field(min_length=1, max_length=160)
    submission_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    agent_execution: AgentExecutionBinding | None = None
    binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_actor_evidence(self) -> Self:
        """Require agent evidence only for agents and verify the binding hash."""
        has_agent_evidence = self.agent_execution is not None
        if has_agent_evidence != (self.actor_kind is ActorKind.INSTALLED_AGENT):
            raise ValueError("only installed Agent submissions carry Agent execution evidence")
        expected = canonical_hash(self.model_dump(mode="json", exclude={"binding_hash"}))
        if self.binding_hash != expected:
            raise ValueError("actor submission binding identity is invalid")
        return self


def seal_actor_submission(
    *,
    actor_kind: ActorKind,
    actor_id: str,
    submission_hash: str,
    agent_execution: AgentExecutionBinding | None = None,
) -> ActorSubmissionBinding:
    """Seal provenance around already-canonical domain submission content.

    This deliberately owns no domain policy: a Desk validates the submission
    before calling it.  The binding only says who supplied those exact bytes.
    """
    draft = ActorSubmissionBinding.model_construct(
        actor_kind=actor_kind,
        actor_id=actor_id,
        submission_hash=submission_hash,
        agent_execution=agent_execution,
        binding_hash="0" * 64,
    )
    return cast(
        ActorSubmissionBinding,
        ActorSubmissionBinding.model_validate(
            {
                **draft.model_dump(mode="json", exclude={"binding_hash"}),
                "binding_hash": canonical_hash(
                    draft.model_dump(mode="json", exclude={"binding_hash"})
                ),
            }
        ),
    )


__all__ = [
    "ActorKind",
    "ActorSubmissionBinding",
    "AgentExecutionBinding",
    "ModelAttemptContextAudit",
    "ProjectionAudit",
    "seal_actor_submission",
]
