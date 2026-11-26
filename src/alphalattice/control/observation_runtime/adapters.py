"""Small adapters that translate existing owner facts into safe observations."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from .contracts import (
    ObservationAuthority,
    ObservationDraft,
    ObservationRetentionClass,
    ObservationSensitivity,
)
from .policy import EXTERNAL_ACTIVITY_SCHEMA, ObservationPolicyRegistry


def safe_observation_draft(
    *,
    policies: ObservationPolicyRegistry,
    schema_kind: str,
    occurred_at: datetime,
    source_kind: str,
    source_id: str,
    source_sequence: int,
    payload: dict[str, Any],
    authority: ObservationAuthority,
    retention_class: ObservationRetentionClass,
    task_id: str | None = None,
    run_id: str | None = None,
    stage_id: str | None = None,
    span_id: str | None = None,
    parent_span_id: str | None = None,
    correlation_ids: tuple[str, ...] = (),
    supersedes_observation_id: str | None = None,
) -> ObservationDraft:
    """Create one inline public-safe draft under its code-owned schema policy."""
    policy = policies.policy(schema_kind, 1)
    if schema_kind == EXTERNAL_ACTIVITY_SCHEMA:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(encoded) > policy.max_inline_bytes:
            # Keep the old policy identity for exact retries of already stored
            # events; wider admitted documents use their own versioned policy.
            policy = policies.policy(schema_kind, 2)
    return ObservationDraft(
        schema_kind=schema_kind,
        schema_version=policy.schema_version,
        occurred_at=occurred_at,
        source_kind=source_kind,
        source_id=source_id,
        source_sequence=source_sequence,
        task_id=task_id,
        run_id=run_id,
        stage_id=stage_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        correlation_ids=tuple(sorted(set(correlation_ids))),
        authority=authority,
        sensitivity=ObservationSensitivity.PUBLIC_SAFE,
        retention_class=retention_class,
        inline_safe_payload=payload,
        policy_hash=policy.policy_hash,
        supersedes_observation_id=supersedes_observation_id,
    )


__all__ = ["safe_observation_draft"]
