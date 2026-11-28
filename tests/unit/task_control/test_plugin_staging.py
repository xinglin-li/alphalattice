"""A plugin run's staging folder, the Host's checks on what it staged, and its one commit."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Collection
from concurrent.futures import Future
from pathlib import Path

import pytest

from alphalattice.capabilities.domain_contracts.domain.errors import WorkspaceSecurityError
from alphalattice.control.task_control.plugin_staging import (
    OUTPUT,
    PROVENANCE,
    PluginStaging,
    StagingRefused,
)
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.protocols.plugin_contract import (
    ActionReceipt,
    ActionSpec,
    CatalogRead,
    EffectMode,
    ExecutionContext,
    ExecutionMode,
    Lease,
    ReceiptStatus,
)

TEST_SOURCE = "src/alphalattice/control/task_control/plugin_staging.py"

ROLE = "blake3:" + "1" * 64
SPEC = ActionSpec(
    capability_id="factor.custom_momentum",
    plugin_id="sample-factor",
    code_role_hash=ROLE,
    effect_mode=EffectMode.STAGING_WRITE,
    execution_mode=ExecutionMode.SANDBOX_SUBPROCESS,
    lease=Lease(memory_bytes=1 << 30, threads=2, timeout_ms=60_000),
    input_schema={"type": "object"},
    output_schema={
        "type": "object",
        "properties": {"score": {"type": "number"}},
        "required": ["score"],
    },
    inputs=("raw_bar",),
)
READ = CatalogRead(snapshot_id="lake:42", input_versions={"raw_bar": 7})
Plugin = Callable[[ExecutionContext], ActionReceipt]


def link_directory(link: Path, target: Path) -> None:
    """A directory symlink: Windows makes one in developer mode or with the right, as CI has."""
    link.symlink_to(target, target_is_directory=True)


@pytest.mark.proves("task_control.a_plugin_run_commits_only_what_its_staging_proves")
def test_a_staged_run_commits_once_only_what_its_key_and_schema_prove(tmp_path: Path) -> None:
    """An admitted run commits its checked answer once and every caller gets that answer, while
    each inadmissible, foreign, linked, reserved, invalid or stale run is refused by name."""
    workspace, artifacts, outside = tmp_path / "workspace", tmp_path / "artifacts", tmp_path / "x"
    outside.write_text("{}", encoding="utf-8")
    workspace.mkdir()
    latest = {"raw_bar": 7}

    def catalog(tables: Collection[str], snapshot: str | None) -> CatalogRead:
        if snapshot is None:
            return CatalogRead(snapshot_id="lake:43", input_versions=dict(latest))
        older = {"lake:41": 6}  # a kept snapshot before the latest version
        return CatalogRead(snapshot_id=snapshot, input_versions={"raw_bar": older.get(snapshot, 7)})

    resolver = ArtifactResolver(artifacts)
    staging = PluginStaging(
        workspace, tmp_path / "data", resolver, WorkspaceMutationGate(), catalog
    )
    elsewhere, behind = tmp_path / "elsewhere", tmp_path / "behind"
    elsewhere.mkdir()
    (behind / "staging").mkdir(parents=True)
    link_directory(behind / "staging" / "plugin-runs", elsewhere)  # a linked parent of each run
    linked_runs = PluginStaging(
        behind, tmp_path / "data", resolver, WorkspaceMutationGate(), catalog
    )
    invocation = SPEC.invoke({"window": 20}, READ)
    calls: list[ExecutionContext] = []
    attached: list[object] = []

    def plugin(answer: object, key: str = invocation.idempotency_key, also: str = "") -> Plugin:
        def run(context: ExecutionContext) -> ActionReceipt:
            calls.append(context)
            (context.staging_dir / OUTPUT).write_text(json.dumps(answer), encoding="utf-8")
            if also == "link":
                os.link(outside, context.staging_dir / "data.json")
            elif also == "provenance":
                (context.staging_dir / PROVENANCE).write_text("{}", encoding="utf-8")
            elif also == "attach":  # requests for the same key while this run is in flight
                with pytest.raises(StagingRefused) as foreign:
                    staging.request(other_code, invocation, plugin({"score": 9.0}))
                running = staging.request(SPEC, invocation, plugin({"score": 9.0}))
                attached.extend([foreign.value.code, running.cancel(), running])
            return ActionReceipt(
                invocation_id=context.invocation_id,
                idempotency_key=key,
                status=ReceiptStatus.OK,
                wall_ms=1,
                peak_bytes=1,
            )

        return run

    def refused(
        run: Plugin, spec: ActionSpec = SPEC, before=lambda: None, read=invocation, owner=staging
    ) -> str:
        before()
        with pytest.raises(StagingRefused) as error:
            owner.request(spec, read, run).result()
        return error.value.code

    other_code = SPEC.model_copy(update={"code_role_hash": "blake3:" + "2" * 64})
    # A reference the default resolver would fetch: a file holding a schema that accepts all.
    remote = SPEC.model_copy(update={"output_schema": {"$ref": outside.as_uri()}})
    unbound = SPEC.invoke({"window": 20}, CatalogRead(snapshot_id="lake:41", input_versions=latest))
    refusals = [
        refused(
            plugin({"score": 1.0}), SPEC.model_copy(update={"effect_mode": EffectMode.READ_ONLY})
        ),
        refused(plugin({"score": 1.0}), other_code),
        refused(plugin({"score": 1.0}), read=unbound),
        refused(plugin({"score": 1.0}, key="blake3:" + "3" * 64)),
        refused(plugin({"score": "high"})),
        refused(plugin({"score": 1.0}), remote),
        refused(plugin({"score": 1.0}, also="link")),
        refused(plugin({"score": 1.0}, also="provenance")),
        refused(plugin({"score": 1.0}), before=lambda: latest.update(raw_bar=8)),
        refused(plugin({"score": 1.0}), owner=linked_runs),
    ]
    published = artifacts / "plugin-result"
    nothing_committed = not published.exists() or not any(published.iterdir())
    latest["raw_bar"] = 7
    committed = staging.request(SPEC, invocation, plugin({"score": 1.5}, also="attach")).result()
    again = staging.request(SPEC, invocation, plugin({"score": 2.5})).result()
    linked = tmp_path / "linked"
    linked.mkdir()
    os.link(outside, linked / OUTPUT)
    with pytest.raises(WorkspaceSecurityError):  # the artifact owner refuses it unexposed
        resolver.publish_plugin_result(linked, "blake3:" + "4" * 64)
    link_directory(tmp_path / "walk-root", elsewhere)  # a linked walk root over an empty folder
    with pytest.raises(WorkspaceSecurityError):
        resolver.publish_plugin_result(tmp_path / "walk-root", "blake3:" + "5" * 64)

    stored = sorted(path.name for path in next(published.iterdir()).iterdir())
    assert refusals == [
        "task_control.plugin_not_staging_write",
        "task_control.plugin_role_hash_mismatch",
        "task_control.plugin_read_unbound",
        "task_control.plugin_provenance_mismatch",
        "task_control.plugin_output_invalid",
        "task_control.plugin_output_invalid",
        "task_control.plugin_staged_entry_invalid",
        "task_control.plugin_staged_entry_invalid",
        "task_control.plugin_inputs_changed",
        "task_control.plugin_staging_unproved",
    ]
    # Eight runs: an inadmissible request runs no code, and the attached request runs none.
    assert nothing_committed and len(calls) == 8
    assert committed.output == again.output == {"score": 1.5} and stored == [OUTPUT, PROVENANCE]
    assert committed.artifact == again.artifact
    assert committed.artifact.content_hash == invocation.idempotency_key
    assert attached[:2] == ["task_control.plugin_role_hash_mismatch", False]
    running = attached[2]
    assert isinstance(running, Future) and running.result() is committed
    assert len(list(published.iterdir())) == 1 and not any(elsewhere.iterdir())
    assert not any((workspace / "staging" / "plugin-runs").iterdir())
