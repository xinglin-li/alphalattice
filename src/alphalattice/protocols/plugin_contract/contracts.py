"""The plugin contract: what a capability declares, runs with, is asked and answers.

One contract serves both execution modes. A built-in capability runs in the Host's kept workers
(TRUSTED_POOL); a user plugin runs in its own sandboxed process tree (SANDBOX_SUBPROCESS). No
plugin writes the ledger: it reads a catalog snapshot and writes only its staging folder, and the
Host commits what it staged. The speculation fields (effect mode, snapshot anchor, idempotency
key, speculative flag, priority) are part of the contract from its first version.

A run's key names its result by what the result depends on: the capability, its code, its
parameters with an implicit as-of resolved, the version of each input table it declares and the
contract's version. The catalog snapshot it reads at is not among them, since every commit
advances it: a write to a table the capability never reads leaves the key, so the result stays
reusable.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Final, cast

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


def _frozen_versions(value: Mapping[str, int]) -> Mapping[str, int]:
    return MappingProxyType(dict(value))


InputVersions = Annotated[
    Mapping[ShortString, NonNegativeInt],
    AfterValidator(_frozen_versions),
    PlainSerializer(dict),
]
"""Each declared input table's version at a run's read snapshot: the last catalog snapshot at
or before it that changed the table, as the catalog owner reads it (`StorePort.catalog_read`)."""

CONTRACT_VERSION: Final = 1
"""The contract's version, among a run's key inputs: a run under another contract never takes
this one's result."""


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


_KEY_INPUTS = frozenset(
    {"capability_id", "code_role_hash", "params", "input_versions", "contract_version"}
)


class _ActionKeyInputs(DomainModel):
    """The values one run's result is keyed by."""

    capability_id: ShortString
    code_role_hash: Blake3Identity
    params: JsonObject
    input_versions: InputVersions
    contract_version: PositiveInt = CONTRACT_VERSION

    def derived_key(self) -> str:
        """The tagged BLAKE3 digest naming the run's result: the same inputs, one result.

        It hashes the validated values as the wire carries them, in canonical form, so the
        parameters' key order never changes it and a reader of the invocation derives it again.
        """
        return canonical_hash(self.model_dump(mode="json", include=_KEY_INPUTS))


class ActionInvocation(_ActionKeyInputs):
    """One request for a capability's code, read at one catalog snapshot, with its key."""

    catalog_snapshot_id: ShortString
    idempotency_key: Blake3Identity
    is_speculative: bool = False

    def model_post_init(self, context: object, /) -> None:
        """An invocation carries only the key its own validated inputs derive."""
        if self.idempotency_key != self.derived_key():
            raise ValueError("plugin_contract.invocation_key_mismatch")


class CatalogRead(DomainModel):
    """One read of the catalog: the snapshot a run reads at and each declared input's version at
    that same snapshot, read together by the catalog owner, so a key never pairs one snapshot's
    data with another's versions."""

    snapshot_id: ShortString
    input_versions: InputVersions


class ActionSpec(DomainModel):
    """What a capability declares: its identity, effect, execution mode, lease, schemas, the
    catalog tables it reads (`inputs`) and the parameter its as-of defaults to today in, if any
    (`as_of_param`)."""

    capability_id: ShortString
    plugin_id: ShortString
    code_role_hash: Blake3Identity
    effect_mode: EffectMode
    execution_mode: ExecutionMode
    lease: Lease
    input_schema: JsonObject
    output_schema: JsonObject
    inputs: tuple[ShortString, ...] = ()
    as_of_param: ShortString | None = None

    def invoke(
        self,
        params: Mapping[str, JsonValue],
        read: CatalogRead,
        *,
        as_of: date | None = None,
        is_speculative: bool = False,
    ) -> ActionInvocation:
        """The invocation of this capability's code at `read`'s snapshot, keyed by its validated
        inputs.

        `read` holds one version for each declared input at the snapshot it names. An as-of the
        parameters leave implicit is resolved into them from `as_of`, the caller's today, before
        the key; one with no `as_of` is refused by name.
        """
        if set(read.input_versions) != set(self.inputs):
            raise ValueError("plugin_contract.input_versions_mismatch")
        resolved = dict(params)
        if self.as_of_param is not None and self.as_of_param not in resolved:
            if as_of is None:
                raise ValueError("plugin_contract.as_of_unresolved")
            resolved[self.as_of_param] = as_of.isoformat()
        inputs = _ActionKeyInputs(
            capability_id=self.capability_id,
            code_role_hash=self.code_role_hash,
            params=resolved,
            input_versions=read.input_versions,
        )
        return ActionInvocation(
            **inputs.model_dump(),
            catalog_snapshot_id=read.snapshot_id,
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
