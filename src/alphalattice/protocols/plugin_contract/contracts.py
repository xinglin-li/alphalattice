"""The plugin contract: what a capability declares, runs with, is asked and answers.

One contract serves both execution modes. A built-in capability runs in the Host's kept workers
(TRUSTED_POOL); a user plugin runs in its own sandboxed process tree (SANDBOX_SUBPROCESS). No
plugin writes the ledger: it reads a catalog snapshot and writes only its staging folder, and the
Host commits what it staged. The speculation fields (effect mode, snapshot anchor, idempotency
key, speculative flag, priority) are part of the contract from its first version.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, cast

from pydantic import AfterValidator, JsonValue, NonNegativeInt, PlainSerializer, PositiveInt

from alphalattice.capabilities.domain_contracts.domain.base import (
    Blake3Identity,
    DomainModel,
    ShortString,
    Uuid4,
)
from alphalattice.kernel.identity import canonical_hash


def _frozen(value: object) -> object:
    """A JSON value that cannot change after validation; a non-finite number is refused."""
    if isinstance(value, dict):
        return MappingProxyType({key: _frozen(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_frozen(item) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("plugin_contract.json_value_not_finite")
    return value


def _thawed(value: object) -> JsonValue:
    """The frozen value in its JSON shape: an object and an array, as the wire carries them."""
    if isinstance(value, Mapping):
        return {key: _thawed(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thawed(item) for item in value]
    return cast(JsonValue, value)


def _frozen_object(value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    return cast(Mapping[str, JsonValue], _frozen(dict(value)))


JsonObject = Annotated[
    Mapping[str, JsonValue], AfterValidator(_frozen_object), PlainSerializer(_thawed)
]


class EffectMode(StrEnum):
    """What a capability's run may change, so the Host knows what it may run ahead of time."""

    READ_ONLY = "READ_ONLY"
    STAGING_WRITE = "STAGING_WRITE"
    LEDGER_COMMIT = "LEDGER_COMMIT"


class ExecutionMode(StrEnum):
    """Where a capability runs: the Host's kept workers, or its own sandboxed process tree."""

    TRUSTED_POOL = "TRUSTED_POOL"
    SANDBOX_SUBPROCESS = "SANDBOX_SUBPROCESS"


class Priority(StrEnum):
    """Whether a run answers a request now, or fills idle capacity (speculation, maintenance)."""

    FOREGROUND = "FOREGROUND"
    BACKGROUND = "BACKGROUND"


class ReceiptStatus(StrEnum):
    """How one invocation ended."""

    OK = "ok"
    CACHE_HIT = "cache_hit"
    CANCELLED = "cancelled"
    FAILED = "failed"


class Lease(DomainModel):
    """The resources one run may hold; the network is off unless the lease grants it."""

    memory_bytes: PositiveInt
    threads: PositiveInt
    timeout_ms: PositiveInt
    network: bool = False


_KEY_INPUTS = frozenset({"capability_id", "code_role_hash", "params", "catalog_snapshot_id"})


class _ActionKeyInputs(DomainModel):
    """The four values one run's result is keyed by."""

    capability_id: ShortString
    code_role_hash: Blake3Identity
    params: JsonObject
    catalog_snapshot_id: ShortString

    def derived_key(self) -> str:
        """The tagged BLAKE3 digest naming the run's result: the same four inputs, one result.

        It hashes the validated values as the wire carries them, in canonical form, so the
        parameters' key order never changes it and a reader of the invocation derives it again.
        """
        return canonical_hash(self.model_dump(mode="json", include=_KEY_INPUTS))


class ActionInvocation(_ActionKeyInputs):
    """One request for a capability's code against one catalog snapshot, with its key."""

    idempotency_key: Blake3Identity
    is_speculative: bool = False

    def model_post_init(self, context: object, /) -> None:
        """An invocation carries only the key its own validated inputs derive."""
        if self.idempotency_key != self.derived_key():
            raise ValueError("plugin_contract.invocation_key_mismatch")


class ActionSpec(DomainModel):
    """What a capability declares: its identity, effect, execution mode, lease and schemas."""

    capability_id: ShortString
    plugin_id: ShortString
    code_role_hash: Blake3Identity
    effect_mode: EffectMode
    execution_mode: ExecutionMode
    lease: Lease
    input_schema: JsonObject
    output_schema: JsonObject

    def invoke(
        self,
        params: Mapping[str, JsonValue],
        catalog_snapshot_id: str,
        *,
        is_speculative: bool = False,
    ) -> ActionInvocation:
        """The invocation of this capability's code, keyed by its validated inputs."""
        inputs = _ActionKeyInputs(
            capability_id=self.capability_id,
            code_role_hash=self.code_role_hash,
            params=params,
            catalog_snapshot_id=catalog_snapshot_id,
        )
        return ActionInvocation(
            **inputs.model_dump(),
            idempotency_key=inputs.derived_key(),
            is_speculative=is_speculative,
        )


class ExecutionContext(DomainModel):
    """What one run is given: its snapshot, a read-only data root, its staging folder, its lease."""

    invocation_id: Uuid4
    catalog_snapshot_id: ShortString
    readonly_data_root: Path
    staging_dir: Path
    lease: Lease
    is_speculative: bool = False
    priority: Priority = Priority.FOREGROUND


class ActionReceipt(DomainModel):
    """How one invocation ended, what it answered and staged, and what it cost."""

    invocation_id: Uuid4
    idempotency_key: Blake3Identity
    status: ReceiptStatus
    output: JsonObject | None = None
    staging_manifest: ShortString | None = None
    provenance_ids: tuple[ShortString, ...] = ()
    wall_ms: NonNegativeInt
    peak_bytes: NonNegativeInt
