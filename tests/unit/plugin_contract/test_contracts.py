"""The plugin contract's four records: round trip, frozen, strict refusal by name, and the key."""

from __future__ import annotations

import json
import operator
from pathlib import Path
from uuid import uuid1, uuid4

import pytest
from pydantic import BaseModel, ValidationError

from alphalattice.protocols.plugin_contract import (
    ActionInvocation,
    ActionReceipt,
    ActionSpec,
    CatalogRead,
    EffectMode,
    ExecutionContext,
    ExecutionMode,
    Lease,
    Priority,
    ReceiptStatus,
)

TEST_SOURCE = "src/alphalattice/protocols/plugin_contract/contracts.py"

ROLE = "blake3:" + "1" * 64
INVOCATION = uuid4()
LEASE = Lease(memory_bytes=1 << 30, threads=2, timeout_ms=60_000)
SPEC = ActionSpec(
    capability_id="factor.custom_momentum",
    plugin_id="sample-factor",
    code_role_hash=ROLE,
    effect_mode=EffectMode.STAGING_WRITE,
    execution_mode=ExecutionMode.SANDBOX_SUBPROCESS,
    lease=LEASE,
    input_schema={"type": "object", "properties": {"window": {"type": "integer"}}},
    output_schema={"type": "object"},
    inputs=("raw_bar",),
)
PARAMS = {"window": 20, "universe": ["A", "B"], "options": {"lag": 1, "demean": True}}
VERSIONS = {"raw_bar": 7}


def read(snapshot: str = "market:42", **versions: int) -> CatalogRead:
    """One catalog read: the snapshot and each declared input's version at it."""
    return CatalogRead(snapshot_id=snapshot, input_versions=VERSIONS | versions)


INVOKED = SPEC.invoke(PARAMS, read(), is_speculative=True)
CONTEXT = ExecutionContext(
    invocation_id=INVOCATION,
    catalog_snapshot_id="market:42",
    readonly_data_root=Path("data"),
    staging_dir=Path("staging") / "run",
    lease=LEASE,
    is_speculative=True,
    priority=Priority.BACKGROUND,
)
RECEIPT = ActionReceipt(
    invocation_id=INVOCATION,
    idempotency_key=INVOKED.idempotency_key,
    status=ReceiptStatus.OK,
    output={"rows": 3},
    staging_manifest="manifest.json",
    provenance_ids=("market:42",),
    wall_ms=12,
    peak_bytes=4096,
)
RECORDS: tuple[BaseModel, ...] = (LEASE, SPEC, read(), INVOKED, CONTEXT, RECEIPT)

REFUSALS = (
    ("a number as text", Lease, {"memory_bytes": "1"}, "int_type"),
    ("no threads", Lease, {"threads": 0}, "greater_than"),
    ("an unknown effect", ActionSpec, {"effect_mode": "WRITE"}, "enum"),
    ("an unknown field", ActionSpec, {"entry": "x:y"}, "extra_forbidden"),
    ("a sha256 role hash", ActionSpec, {"code_role_hash": "sha256:" + "1" * 64}, "pattern"),
    ("a non-finite parameter", ActionInvocation, {"params": {"window": 1e999}}, "not_finite"),
    ("a malformed key", ActionInvocation, {"idempotency_key": "blake3:1"}, "pattern"),
    (
        "another run's key",
        ActionInvocation,
        {"idempotency_key": "blake3:" + "0" * 64},
        "key_mismatch",
    ),
    ("another contract's run", ActionInvocation, {"contract_version": 2}, "key_mismatch"),
    ("a negative version", ActionInvocation, {"input_versions": {"raw_bar": -1}}, "greater_than"),
    ("a UUIDv1 invocation", ExecutionContext, {"invocation_id": str(uuid1())}, "UUIDv4"),
    ("an unknown priority", ExecutionContext, {"priority": "LOW"}, "enum"),
    ("negative wall time", ActionReceipt, {"wall_ms": -1}, "greater_than_equal"),
)


@pytest.mark.proves("plugin_contract.a_run_is_keyed_by_its_declared_inputs_not_the_catalog")
def test_the_contract_round_trips_refuses_by_name_and_keys_each_run() -> None:
    for record in RECORDS:
        kind = type(record)
        if kind is not ActionSpec:  # a spec's round trip is each capability's, in its seam's suite
            assert kind.model_validate_json(record.model_dump_json()) == record
            assert kind.model_validate(record.model_dump()) == record
        with pytest.raises(ValidationError, match="frozen_instance"):
            record.__setattr__("schema_version", "1")

    by_kind = {type(record): record for record in RECORDS}
    for case, kind, change, name in REFUSALS:  # Refused in the wire form a plugin sends.
        with pytest.raises(ValidationError) as refused:
            kind.model_validate_json(
                json.dumps(json.loads(by_kind[kind].model_dump_json()) | change)
            )
        assert name in str(refused.value), case

    key = INVOKED.idempotency_key  # Carried on the wire, and derived again by its reader.
    assert json.loads(INVOKED.model_dump_json())["idempotency_key"] == key
    # Another contract version's run: its key names other inputs, so it is never reused here.
    assert INVOKED.model_copy(update={"contract_version": 2}).derived_key() != key

    # Neither the payload nor what its key names can change after validation.
    params = INVOKED.params
    assert not hasattr(params["universe"], "append")
    frozen_values = (params, params["options"], SPEC.input_schema["properties"], RECEIPT.output)
    for frozen in (*frozen_values, INVOKED.input_versions):
        with pytest.raises(TypeError):
            operator.setitem(frozen, "window", 21)
