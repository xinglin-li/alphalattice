"""The plugin contract seam's conformance suite: every capability keeps the contract.

One suite per seam, in the form of dbt's adapter tests: the test below runs once per
implementation, the sample registered here and every capability the product declares in the
seam's entry-point group, so a new capability inherits the whole suite and cannot land without
passing it (the commit gate refuses a group no suite declares as its SEAM).
"""

from __future__ import annotations

import tomllib
from collections.abc import Callable
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from alphalattice.control.product_host.plugins.registry import PluginRegistry, VerifiedPlugin
from alphalattice.kernel.identity import is_blake3
from alphalattice.protocols.plugin_contract import (
    ActionSpec,
    CatalogRead,
    EffectMode,
    ExecutionMode,
    Lease,
)

TEST_SOURCE = "src/alphalattice/protocols/plugin_contract/contracts.py"
SEAM = "alphalattice.capabilities"
"""The entry-point group whose capabilities this suite collects, the seam it names."""

OTHER_ROLE = "blake3:" + "6" * 64
SAMPLE = ActionSpec(
    capability_id="sample.dated_window",
    plugin_id="sample",
    code_role_hash="blake3:" + "5" * 64,
    effect_mode=EffectMode.STAGING_WRITE,
    execution_mode=ExecutionMode.SANDBOX_SUBPROCESS,
    lease=Lease(memory_bytes=1 << 28, threads=1, timeout_ms=10_000),
    input_schema={"type": "object", "properties": {"window": {"type": "integer"}}},
    output_schema={"type": "object"},
    inputs=("raw_bar", "listing"),
    as_of_param="as_of",
)


def _declared(name: str) -> Callable[[Path], ActionSpec]:
    def build(workspace: Path) -> ActionSpec:
        captured: list[ActionSpec] = []

        def capture(plugin: VerifiedPlugin) -> SimpleNamespace:
            captured.append(plugin.spec)
            return SimpleNamespace(is_alive=lambda: True)

        PluginRegistry(workspace).activate(name, capture)
        (spec,) = captured
        assert spec.capability_id == name
        return spec

    return build


_GROUP = tomllib.loads((Path(__file__).resolve().parents[3] / "pyproject.toml").read_text("utf-8"))
IMPLEMENTATIONS: dict[str, Callable[[Path], ActionSpec]] = {
    "sample": lambda _workspace: SAMPLE,
    **{name: _declared(name) for name in _GROUP["project"].get("entry-points", {}).get(SEAM, {})},
}


@pytest.mark.proves("plugin_contract.every_capability_keeps_the_contract")
@pytest.mark.parametrize("implementation", IMPLEMENTATIONS, ids=tuple(IMPLEMENTATIONS))
def test_every_capability_keeps_the_plugin_contract(implementation: str, tmp_path: Path) -> None:
    """A capability's spec survives the contract's wire form, each of its runs is keyed by its
    inputs alone, and a run its declared inputs or as-of do not admit is refused by name."""
    spec = IMPLEMENTATIONS[implementation](tmp_path)
    assert ActionSpec.model_validate_json(spec.model_dump_json()) == spec
    today, params = date(2026, 11, 27), {"window": 20, "universe": ["A", "B"]}
    versions = {table: n + 1 for n, table in enumerate(spec.inputs)}

    def run(params=params, as_of=today, snapshot="lake:42", spec=spec, **moved: int):
        read = CatalogRead(snapshot_id=snapshot, input_versions=versions | moved)
        return spec.invoke(params, read, as_of=as_of)

    first = run()
    # Each case: a run, and whether it shares the first run's result.
    cases = [
        ("reordered parameters", run(dict(reversed(params.items()))), True),
        ("another snapshot, trimmed", run(snapshot=" lake:43 "), True),
        ("another parameter", run(params | {"window": 21}), False),
        ("another code", run(spec=spec.model_copy(update={"code_role_hash": OTHER_ROLE})), False),
        ("another capability", run(spec=spec.model_copy(update={"capability_id": "x"})), False),
        *((f"a commit to {table}", run(**{table: 9}), False) for table in spec.inputs),
    ]
    if spec.as_of_param:
        explicit = params | {spec.as_of_param: today.isoformat()}
        cases += [
            ("the implicit as-of resolved to today", run(explicit, as_of=None), True),
            ("the implicit as-of on another day", run(as_of=date(2026, 11, 30)), False),
        ]
        with pytest.raises(ValueError, match="as_of_unresolved"):
            run(as_of=None)
    with pytest.raises(ValueError, match="input_versions_mismatch"):
        run(foreign=1)
    assert is_blake3(first.idempotency_key) and first.idempotency_key == first.derived_key()
    for case, other, same in cases:
        assert (other.idempotency_key == first.idempotency_key) is same, case
