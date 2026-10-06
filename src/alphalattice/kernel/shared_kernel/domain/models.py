"""Strict domain records used by the engineering foundation."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from alphalattice.kernel.shared_kernel.domain.base import (
    DomainModel,
    NonEmptyString,
    ShortString,
    UtcDatetime,
    Uuid4,
)

NonNegativeInt = Annotated[int, Field(ge=0)]


class RunIdentity(DomainModel):
    """Identify one experiment run and its creation time.

    Attributes:
        run_id: Run UUID.
        experiment_id: Experiment UUID owning the run.
        created_at: UTC-aware creation time.
    """

    run_id: Uuid4
    experiment_id: Uuid4
    created_at: UtcDatetime


class TypedFailure(DomainModel):
    """Carry a stable refusal and its retry, cause and artifact context.

    Attributes:
        category: Declared failure category.
        code: Stable machine-readable refusal code.
        message: Human-readable failure detail.
        retryable: Whether the failure is classified as retryable.
        cause_chain: Ordered lower-level failure descriptions.
        artifact_refs: Artifact UUIDs associated with the failure.
    """

    category: ShortString
    code: ShortString
    message: NonEmptyString
    retryable: bool = False
    cause_chain: tuple[NonEmptyString, ...] = ()
    artifact_refs: tuple[Uuid4, ...] = ()
