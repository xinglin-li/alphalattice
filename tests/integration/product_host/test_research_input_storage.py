"""Input storage and cleanup mechanics; no synthetic file is claimed as numerical evidence."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
import threading
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from alphalattice.capabilities.data.errors import DataError
from alphalattice.capabilities.domain_contracts.identity import canonical_hash
from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceExperimentInput,
    publish_research_workspace_manifest,
)
from alphalattice.control.product_host.research_authoring.factor_inputs import (
    FactorInputBundle,
    _copy_input,
    factor_input_paths,
    file_digest,
    legacy_bindings,
    read_factor_bundle,
)
from alphalattice.control.product_host.storage.input_references import ResearchInputStorage
from alphalattice.foundation.data_layer import history
from alphalattice.foundation.data_layer.log import uuid7
from tests.portfolio_strategy_lab.local_web_support import _manifest

if sys.platform == "win32":
    import _winapi


def test_observation_capacity_takes_a_fresh_sample_when_the_workspace_reopens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """a retained sample from another session cannot set this session's allowance."""

    from alphalattice.control.workspace_runtime.storage import capacity

    monkeypatch.setattr(
        capacity.shutil, "disk_usage", lambda _path: SimpleNamespace(free=1_000_000)
    )
    artifact = tmp_path / "artifacts/model.bin"
    artifact.parent.mkdir()
    artifact.write_bytes(b"a" * 1000)
    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        storage = ResearchInputStorage(session)
        first = storage.capacity_cap_bytes()
        assert storage.capacity_cap_bytes() == first
    artifact.write_bytes(b"a" * 5096)
    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        storage = ResearchInputStorage(session)
        assert storage.capacity_cap_bytes() == first + 4096
        assert storage.capacity_cap_bytes() == first + 4096


@pytest.mark.parametrize(
    ("free", "write", "cap", "refused"),
    [
        (200, 3.7, "auto", None),  # a laptop's first preparation
        (30, 3.7, "auto", None),  # one a tenth of the free disk would not hold
        (20, 0.8, "auto", None),  # a CI runner's fixture preparation
        (4, 3.7, "auto", "storage.disk_space_insufficient"),  # no recovery headroom left
        (30, 10, "auto", "storage.managed_capacity_exceeded"),  # past a quarter of free disk
        (200, 3.7, str(2**30), "storage.managed_capacity_exceeded"),  # an operator's own cap
    ],
)
def test_an_automatic_cap_admits_a_first_preparation_the_disk_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, free, write, cap, refused
) -> None:
    """requirement: under the automatic cap one write within a quarter of free disk is admitted
    where the disk holds it with headroom; each refusal has its way on, and is retryable."""
    from alphalattice.control.product_host.composition.research_experiments import (
        retryable_block,
    )
    from alphalattice.control.product_host.storage.inventory import (
        StorageInventoryError,
        require_storage_capacity,
    )
    from alphalattice.control.workspace_runtime.storage import capacity
    from alphalattice.interface.local_application.cli_contract import refusal_words

    gib = 2**30
    monkeypatch.setattr(
        capacity.shutil, "disk_usage", lambda _path: SimpleNamespace(free=int(free * gib))
    )
    capacity.StorageCapStore(tmp_path).write(
        cap, chosen_by="HUMAN", chosen_at=datetime(2026, 10, 14, tzinfo=UTC)
    )
    basis = capacity.StorageCapStore(tmp_path).capacity(measured_data_bytes=0).automatic_basis
    assert capacity.SHARE_WORDS[capacity.FREE_SHARE] in basis and "percent" not in basis
    if refused is None:
        require_storage_capacity(tmp_path, additional_bytes=int(write * gib))
        return
    with pytest.raises(StorageInventoryError) as caught:
        require_storage_capacity(tmp_path, additional_bytes=int(write * gib))
    assert caught.value.failure_code == refused and refusal_words(refused)["next_action"]
    assert retryable_block(f"{refused}:stage") == refused


def _counting_reads(monkeypatch: pytest.MonkeyPatch, inject=None) -> list[int]:
    """Count every whole-object read of `file_digests`; `inject` runs once, at the
    first read -- after every path is open and grouped, before any byte is read."""

    import alphalattice.control.product_host.research_authoring.factor_inputs as inputs

    original = inputs.file_sha256
    reads = [0]
    # The objects are digested on a pool: the first read's inject holds the
    # others until it has run, or a second read also sees a count of zero.
    first_read = threading.Lock()

    def counted(path, **keywords):
        with first_read:
            if reads[0] == 0 and inject is not None:
                inject()
            reads[0] += 1
        return original(path, **keywords)

    monkeypatch.setattr(inputs, "file_sha256", counted)
    return reads


def _try_replace(source: Path, target: Path) -> bool:
    """Atomically point `target` at `source`'s bytes; False where the platform
    refuses to replace a file that is held open (Windows)."""

    try:
        os.replace(source, target)
    except PermissionError:
        return False
    return True


def test_file_digests_bind_every_path_to_the_object_whose_bytes_were_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from alphalattice.control.product_host.research_authoring.factor_inputs import (
        file_digest,
        file_digests,
    )

    shared = b"shared-object-bytes-" * 100
    a, b, c = tmp_path / "A", tmp_path / "B", tmp_path / "C"
    a.write_bytes(shared)
    os.link(a, b)
    c.write_bytes(b"independent-bytes---" * 100)  # the same size, other bytes
    reads = _counting_reads(monkeypatch)
    digests = file_digests([a, b, c, a])
    # One read per object: the two links once, the same-size other file itself.
    assert reads[0] == 2
    # A file inside its racy tick is read again by the next call.
    assert file_digests([a, b, c]) == digests
    assert reads[0] == 4
    assert digests[a] == digests[b] == file_digest(a) and digests[c] == file_digest(c)
    assert digests[c] != digests[a]


def test_an_alias_replaced_after_grouping_never_answers_another_objects_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Control: Alias replaced after grouping never answers another objects digest."""

    from alphalattice.control.product_host.research_authoring.factor_inputs import (
        file_digest,
        file_digests,
    )
    from alphalattice.protocols.research_authoring.contracts import AuthoringError

    a, b = tmp_path / "A", tmp_path / "B"
    a.write_bytes(b"shared-object-bytes-" * 100)
    os.link(a, b)
    other = tmp_path / "other"
    other.write_bytes(b"independent-bytes---" * 100)
    replaced = []
    _counting_reads(monkeypatch, inject=lambda: replaced.append(_try_replace(other, b)))
    try:
        digests = file_digests([a, b])
    except AuthoringError as error:
        assert replaced == [True]
        assert str(error) == "research_experiment.input_file_replaced_during_verification"
        assert file_digest(b) != file_digest(a)
    else:
        assert replaced == [False], "a replaced alias must not be answered with A's digest"
        assert b.read_bytes() == a.read_bytes()
        assert digests[b] == file_digest(b) == digests[a]
    # The same window where the platform does let the path move: B's path
    # names another object by the time the read is over.
    real_stat = Path.stat

    def moved(path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        if path == b:
            return os.stat_result((*result[:1], result.st_ino + 1, *result[2:]))
        return result

    monkeypatch.setattr(Path, "stat", moved)
    with pytest.raises(AuthoringError, match="input_file_replaced_during_verification"):
        file_digests([a, b])


def test_an_alias_removed_after_grouping_is_not_answered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Control: Alias removed after grouping is not answered."""
    from alphalattice.control.product_host.research_authoring.factor_inputs import (
        file_digest,
        file_digests,
    )
    from alphalattice.protocols.research_authoring.contracts import AuthoringError

    a, b = tmp_path / "A", tmp_path / "B"
    a.write_bytes(b"shared-object-bytes-" * 100)
    os.link(a, b)
    removed = []

    def remove_b() -> None:
        try:
            b.unlink()
        except PermissionError:
            removed.append(False)
        else:
            removed.append(True)

    _counting_reads(monkeypatch, inject=remove_b)
    try:
        digests = file_digests([a, b])
    except AuthoringError as error:
        assert removed == [True]
        assert str(error) == "research_experiment.input_file_removed_during_verification"
    else:
        assert removed == [False]
        assert digests[b] == file_digest(b) == digests[a]
    with pytest.raises(AuthoringError, match="input_file_unavailable"):
        file_digests([a, tmp_path / "absent"])
    # Begin the injected post-open removal with an existing alias on every platform.
    if removed == [True]:
        os.link(a, b)
    real_stat = Path.stat

    def gone(path, *args, **kwargs):
        if path == b:
            raise FileNotFoundError(str(path))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", gone)
    with pytest.raises(AuthoringError, match="input_file_removed_during_verification"):
        file_digests([a, b])


def test_bundle_pool_alias_is_verified_as_one_object_and_still_refuses_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from alphalattice.control.product_host.research_authoring.factor_inputs import (
        compatible_factor_inputs,
        read_factor_bundle,
    )
    from alphalattice.protocols.research_authoring.contracts import AuthoringError

    # Old file-bound identities remain readable; new bindings name the catalog artifact and digest.
    digest = "c" * 64
    artifact, other_artifact = uuid7(), uuid7()
    chunk = f"artifacts/data-operations/execution-outcomes/development/chunks/{digest}.parquet"
    common = {
        "panel_snapshot_hash": "a" * 64,
        "outcome_snapshot_hash": "b" * 64,
        "sessions": (date(2024, 1, 2),),
    }
    old = FactorInputBundle.create(**common, files=((chunk, "d" * 64), ("panel", "e" * 64)))
    new = FactorInputBundle.create(
        **common,
        files=(("panel", "e" * 64),),
        binding_version=2,
        outcomes=((artifact, digest),),
    )
    changed = FactorInputBundle.create(
        **common,
        files=(("panel", "e" * 64),),
        binding_version=2,
        outcomes=((other_artifact, "f" * 64),),
    )
    different_artifact = FactorInputBundle.create(
        **common,
        files=(("panel", "e" * 64),),
        binding_version=2,
        outcomes=((other_artifact, digest),),
    )
    with pytest.raises(ValueError, match="input_manifest_invalid"):
        FactorInputBundle.create(
            **common,
            files=(("panel", "e" * 64),),
            binding_version=2,
            outcomes=(("sha256:" + digest, digest),),
        )
    import pyarrow as pa

    rows = pa.table({"listing": ["one", "two"], "return": [None, 0.1]})
    read_old = lambda _path: rows
    read_new = lambda _artifact: rows
    assert old.binding_hash != new.binding_hash
    assert not compatible_factor_inputs(old, new)  # No row proof is no compatibility.
    assert compatible_factor_inputs(old, new, read_legacy=read_old, read_catalog=read_new)
    assert not compatible_factor_inputs(old, changed, read_legacy=read_old, read_catalog=read_new)
    assert not compatible_factor_inputs(new, different_artifact)

    def at_snapshot(
        snapshot: int, index_hash: str = digest, content: str = "blake3:" + "c" * 64
    ) -> FactorInputBundle:
        return FactorInputBundle.create(
            **common,
            files=tuple(sorted((*new.files, ("market-data.duckdb", index_hash)))),
            binding_version=3,
            outcomes=new.outcomes,
            market_selection_id=artifact,
            market_catalog_snapshot=snapshot,
            market_index_hash=content,
        )

    selected = at_snapshot(7)
    assert "market_selection_id" not in new.model_dump()
    assert FactorInputBundle.model_validate_json(selected.model_dump_json()) == selected
    assert selected.binding_hash == at_snapshot(7, "a" * 64).binding_hash
    assert compatible_factor_inputs(selected, at_snapshot(7, "a" * 64))
    # The sealed index's rows bind it (its Feature values among them); its file's padding does not.
    assert selected.binding_hash != at_snapshot(7, content="blake3:" + "d" * 64).binding_hash
    assert not compatible_factor_inputs(selected, at_snapshot(7, content="blake3:" + "d" * 64))
    assert selected.binding_hash != at_snapshot(8).binding_hash
    assert not compatible_factor_inputs(selected, at_snapshot(8))
    assert not compatible_factor_inputs(selected, new)
    assert not compatible_factor_inputs(
        old, new, read_legacy=read_old, read_catalog=lambda _: rows.slice(0, 1)
    )
    # A binding this data root does not keep is refused by name, never read from elsewhere.
    with pytest.raises(DataError) as missing:
        read_factor_bundle(tmp_path, "a" * 64, verify=False)
    assert missing.value.failure.code == "data_layer.input_missing"

    (bundle,) = _inputs(tmp_path, count=1)
    source = factor_input_paths(tmp_path, bundle.binding_hash)[0]
    alias = source / "market-data.duckdb"
    pool = source.parent.parent / "databases" / bundle.database_snapshot_hash
    pool = pool / "market-data.duckdb"
    assert os.stat(alias).st_ino == os.stat(pool).st_ino
    reads = _counting_reads(monkeypatch)
    assert read_factor_bundle(tmp_path, bundle.binding_hash) == bundle
    # Three objects (the manifest copy, the Parquet object, the database
    # object named by its alias and its pool path), three reads.
    assert reads[0] == 3
    forged = pool.with_name("forged.duckdb")  # beside the view, on its volume
    forged.write_bytes(b"database-X")  # the same size as the pooled "database-0"
    saved = pool.read_bytes()
    os.replace(forged, pool)
    try:
        assert os.stat(alias).st_ino != os.stat(pool).st_ino
        with pytest.raises(AuthoringError, match="input_database_tampered"):
            read_factor_bundle(tmp_path, bundle.binding_hash)
    finally:
        pool.unlink()
        os.link(alias, pool)
    assert pool.read_bytes() == saved
    assert read_factor_bundle(tmp_path, bundle.binding_hash) == bundle
    # The window through the bundle owner: the pool path replaced after the
    # paths are open and grouped, before any byte is read.
    forged.write_bytes(b"database-X")
    replaced = []
    _counting_reads(monkeypatch, inject=lambda: replaced.append(_try_replace(forged, pool)))
    try:
        result = read_factor_bundle(tmp_path, bundle.binding_hash)
    except AuthoringError as error:
        assert replaced == [True]
        assert str(error) == "research_experiment.input_file_replaced_during_verification"
    else:
        assert replaced == [False] and result == bundle
        assert pool.read_bytes() == saved
    if replaced == [True]:
        pool.unlink()
        os.link(alias, pool)
    else:
        forged.unlink()
    assert read_factor_bundle(tmp_path, bundle.binding_hash) == bundle


def test_managed_inventory_preserves_file_identity_and_link_refusal(tmp_path):
    from alphalattice.control.product_host.storage import inventory

    folder = tmp_path / "artifacts" / "nested"
    folder.mkdir(parents=True)
    item = folder / "value.bin"
    item.write_bytes(b"bounded inventory fixture")
    stat = item.stat()
    assert inventory.managed_file_inventory(tmp_path) == (
        ("artifacts/nested/value.bin", stat.st_size, f"{stat.st_dev}:{stat.st_ino}"),
    )
    # A real junction, which Windows makes without link privileges (a symlink
    # stands in elsewhere): the walk refuses it by the entry's own reparse tag.
    outside = tmp_path / "outside"
    outside.mkdir()
    if sys.platform == "win32":
        _winapi.CreateJunction(str(outside), str(folder / "linked"))
    else:
        (folder / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(inventory.StorageInventoryError) as refused:
        inventory.managed_file_inventory(tmp_path)
    assert refused.value.failure_code == "storage.retention_root_mismatch"


def _inputs(root: Path, *, count: int = 3):
    """Bound inputs as 0.2.8 binds them: each staged beside the workspace as the writer stages
    it, then kept once on the data root behind the door; the workspace holds its publication
    records and manifest."""
    manifest = _manifest("input-storage")
    bundles = []
    source = root / "staging"
    source.mkdir()
    panel = {"active_listing_count": 120}
    for index in range(count):
        database = source / "market-data.duckdb"
        database.write_bytes(f"database-{index}".encode())
        parquet = source / "shared.parquet"
        parquet.write_bytes(b"shared-immutable-object")
        panel_file = source / "panel.json"
        panel_file.write_text(json.dumps(panel), encoding="utf-8")
        files = (
            ("artifacts/feature-panel/manifests/" + "a" * 64 + ".json", file_digest(panel_file)),
            ("data.parquet", file_digest(parquet)),
            ("market-data.duckdb", file_digest(database)),
        )
        bundle = FactorInputBundle.create(
            panel_snapshot_hash="a" * 64,
            outcome_snapshot_hash="b" * 64,
            sessions=(date(2024, 1, 2),),
            files=files,
            database_snapshot_hash=file_digest(database),
        )
        bound = root.parent / f"{root.name}-bind"  # laid out as a copy, outside the workspace
        staged = bound / "research-inputs" / bundle.binding_hash
        for name, _digest in files:
            original = (
                database
                if name == "market-data.duckdb"
                else parquet
                if name == "data.parquet"
                else panel_file
            )
            _copy_input(root, original, staged / "source" / name)
        (staged / "manifest.json").write_text(bundle.model_dump_json(), encoding="utf-8")
        (kept,) = [files for h, files in legacy_bindings(bound) if h == bundle.binding_hash]
        history.keep_research_input(bundle.binding_hash, kept)
        updated = manifest.with_bindings(
            experiment_inputs=(
                ResearchWorkspaceExperimentInput(
                    input_id="factor", binding_hash=bundle.binding_hash
                ),
            )
        )
        values = {
            "input_id": "factor",
            "prior_binding_hash": bundles[-1].binding_hash if bundles else None,
            "binding_hash": bundle.binding_hash,
            "prior_manifest_hash": manifest.manifest_hash,
            "next_manifest_hash": updated.manifest_hash,
        }
        receipt = canonical_hash(values)
        records = root / "research-inputs/publications"
        records.mkdir(parents=True, exist_ok=True)
        (records / f"{receipt}.json").write_text(
            json.dumps({**values, "receipt_hash": receipt}), encoding="utf-8"
        )
        manifest = updated
        bundles.append(bundle)
    publish_research_workspace_manifest(root, manifest)
    return bundles


def test_revision_lineage_reuse_staleness_and_retention(tmp_path: Path, monkeypatch):
    """Tiny byte fixtures prove control/retention, not numerical materialization."""
    from alphalattice.control.product_host.data_preparation import input_capture
    from alphalattice.control.product_host.research_authoring.input_revisions import (
        ResearchInputRevision,
        ResearchInputSource,
    )
    from alphalattice.control.task_control.runner import StageDisposition, StageExecutionResult
    from alphalattice.interface.local_application.dispatcher import LocalBackgroundDispatcher

    a, b, c, default = _inputs(tmp_path, count=4)
    manifest_bytes = (tmp_path / "research-workspace.json").read_bytes()
    source = ResearchInputSource(
        binding_hash="1" * 64,
        manifest_revision="2" * 64,
        data_revision_hash="3" * 64,
        panel_snapshot_hash="a" * 64,
        panel_through=date(2024, 1, 2),
        outcome_watermark_hash="4" * 64,
        outcome_recipe_hash="5" * 64,
        outcome_policy_hash="6" * 64,
        materializer_hash="7" * 64,
    )
    calls = []
    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        app = input_capture.ResearchInputCaptureApplication(
            session, clock=lambda: datetime.now(UTC)
        )
        current = source
        monkeypatch.setattr(app, "_source", lambda root=None: source if root else current)
        assert app.plan("factor")["status"] == "REUSED_EXACT"
        for index, bundle in enumerate((a, b, c), start=1):
            # Same through-date; corrected source journals must still rotate capture identity.
            current = source.model_copy(update={"data_revision_hash": str(index + 4) * 64})
            preview = app.plan("factor")
            assert preview["summary"]["database_snapshot_bytes"] == 0
            plan = app.last_plan
            assert preview["status"] == "CONFIRMATION_REQUIRED" and plan is not None
            assert preview["next_requests"]["confirm"] == {
                "operation": "RESEARCH_INPUT_CONFIRM",
                "research_input_plan_hash": preview["plan_hash"],
            }
            with monkeypatch.context() as scoped:
                scoped.setattr(app, "_source", lambda: source)
                before = len(session.task_control_registry.tasks())
                with pytest.raises(ValueError, match="source_changed_preview_again"):
                    app.admit(plan)
                assert len(session.task_control_registry.tasks()) == before

            def materialize(*_args, bundle=bundle, **_kwargs):
                calls.append(bundle.binding_hash)
                return ResearchWorkspaceExperimentInput(
                    input_id="factor", binding_hash=bundle.binding_hash
                )

            monkeypatch.setattr(input_capture, "publish_prepared_factor_inputs", materialize)
            task = app.admit(plan).task_id
            if index in (1, 2):

                def unavailable(*_args, **_kwargs):
                    raise OSError("research_input.synthetic_capacity_failure")

                with monkeypatch.context() as scoped:
                    if index == 1:
                        scoped.setattr(input_capture, "publish_prepared_factor_inputs", unavailable)
                    else:
                        execute_stage = app.execute_stage

                        def publish_then_block(
                            execute_stage=execute_stage, captured_task=task, **kwargs
                        ):
                            execute_stage(**kwargs)
                            assert app.revisions.for_task(captured_task) is not None
                            return StageExecutionResult(
                                StageDisposition.BLOCKED,
                                failure_code="research_input.synthetic_capacity_failure",
                            )

                        scoped.setattr(app, "execute_stage", publish_then_block)
                    app.execute(task)
                assert app.readback(task)["status"] == "BLOCKED"
                retry = app.plan("factor")
                assert retry["status"] == "CONFIRMATION_REQUIRED", retry
                assert retry["next_requests"]["confirm"] == {
                    "operation": "RESEARCH_INPUT_CONFIRM",
                    "research_input_plan_hash": retry["plan_hash"],
                }
                if index == 2:
                    publication = app.revisions.for_task(task)
                    assert publication is not None
                    assert retry["next_requests"]["controls"] == {
                        "operation": "EXPERIMENT_CONTROLS",
                        "research_input_id": publication.input_id,
                        "input_binding_hash": publication.binding_hash,
                    }
                dispatcher = LocalBackgroundDispatcher(session.task_control_registry)
                try:
                    result = app.confirm(retry["plan_hash"], caller="HUMAN", dispatcher=dispatcher)
                    assert result["task_id"] == str(task)
                    dispatcher.drain_for_tests()
                finally:
                    dispatcher.close()
            app.execute(task)
            assert app.readback(task)["status"] == "SUCCEEDED", app.readback(task)
            assert app.plan("factor")["status"] == "REUSED_EXACT"
        assert calls == [a.binding_hash, b.binding_hash, c.binding_hash]
        assert (tmp_path / "research-workspace.json").read_bytes() == manifest_bytes
        records = app.revisions.lineage("factor")
        assert len(records) == 3
        saved = tmp_path / "research-inputs/publications" / f"{records[0].receipt_hash}.json"
        body = saved.read_bytes()
        saved.unlink()
        try:
            with pytest.raises(ValueError, match="ancestor_missing"):
                app.revisions.lineage("factor")
        finally:
            saved.write_bytes(body)
        # A correctly rehashed receipt still cannot substitute the captured source.
        forged = ResearchInputRevision.create(
            **{**records[0].model_dump(exclude={"receipt_hash"}), "source": current}
        )
        path = saved.parent / f"{forged.receipt_hash}.json"
        path.write_text(forged.model_dump_json(), encoding="utf-8")
        try:
            with pytest.raises(ValueError, match="publication_task_mismatch"):
                app.revisions.publications()
        finally:
            path.unlink()
        substituted = ResearchInputRevision.create(
            **{
                **records[-1].model_dump(exclude={"receipt_hash"}),
                "source": records[-1].source,
                "binding_hash": default.binding_hash,
            }
        )
        original_record = saved.parent / f"{records[-1].receipt_hash}.json"
        original_bytes = original_record.read_bytes()
        original_record.unlink()
        replacement = saved.parent / f"{substituted.receipt_hash}.json"
        replacement.write_text(substituted.model_dump_json(), encoding="utf-8")
        try:
            with pytest.raises(ValueError, match="publication_evidence_mismatch"):
                app.revisions.lineage("factor")
        finally:
            replacement.unlink()
            original_record.write_bytes(original_bytes)
        # The inputs are kept on the data root, outside the cleanup: no input row, no input
        # target, and after a confirmed cleanup every version is still selectable.
        storage = ResearchInputStorage(session)
        assert storage.readback()["inputs"] == []
        plan = storage.plan()
        assert plan["bindings"] == []
        assert not any(target.startswith("research-inputs/") for target in plan["targets"])
        storage.confirm(plan["plan_hash"], caller="HUMAN")
        assert app.revisions.versions()["status"] == "AVAILABLE"
        for bundle in (a, b):
            assert app.revisions.select("factor", bundle.binding_hash).binding_hash == (
                bundle.binding_hash
            )


def test_a_bundle_damaged_before_its_publication_leaves_no_receipt(tmp_path: Path, monkeypatch):
    """A bundle damaged before its publication leaves no receipt."""

    from alphalattice.control.product_host.data_preparation import input_capture
    from alphalattice.control.product_host.research_authoring.input_revisions import (
        ResearchInputRevisions,
        ResearchInputSource,
    )
    from alphalattice.interface.local_application.dispatcher import LocalBackgroundDispatcher

    assert "verify" not in inspect.signature(ResearchInputRevisions.publish).parameters
    bundle, *_rest = _inputs(tmp_path, count=2)
    source = ResearchInputSource(
        binding_hash="1" * 64,
        manifest_revision="2" * 64,
        data_revision_hash="3" * 64,
        panel_snapshot_hash="a" * 64,
        panel_through=date(2024, 1, 2),
        outcome_watermark_hash="4" * 64,
        outcome_recipe_hash="5" * 64,
        outcome_policy_hash="6" * 64,
        materializer_hash="7" * 64,
    )
    sealed = factor_input_paths(tmp_path, bundle.binding_hash)[0] / "market-data.duckdb"
    good = sealed.read_bytes()
    publications = tmp_path / "research-inputs/publications"
    before = sorted(path.name for path in publications.glob("*.json"))
    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        app = input_capture.ResearchInputCaptureApplication(
            session, clock=lambda: datetime.now(UTC)
        )
        current = source.model_copy(update={"data_revision_hash": "5" * 64})
        monkeypatch.setattr(app, "_source", lambda root=None: source if root else current)
        preview = app.plan("factor")
        assert preview["status"] == "CONFIRMATION_REQUIRED"
        assert preview["summary"]["database_snapshot_bytes"] == 0
        plan = app.last_plan
        assert plan is not None

        def bind_then_damage(*_args, **_kwargs):
            sealed.write_bytes(b"damaged after the binding, before the publication")
            return ResearchWorkspaceExperimentInput(
                input_id="factor", binding_hash=bundle.binding_hash
            )

        monkeypatch.setattr(input_capture, "publish_prepared_factor_inputs", bind_then_damage)
        task = app.admit(plan).task_id
        app.execute(task)
        readback = app.readback(task)
        assert readback["status"] == "BLOCKED", readback
        assert readback["failure_code"] == "research_experiment.input_file_tampered"
        assert sorted(path.name for path in publications.glob("*.json")) == before
        assert app.revisions.for_task(task) is None
        assert app.revisions.lineage("factor") == ()
        # Restored bytes, the same Task: one receipt, written after the read.
        sealed.write_bytes(good)
        monkeypatch.setattr(
            input_capture,
            "publish_prepared_factor_inputs",
            lambda *_a, **_k: ResearchWorkspaceExperimentInput(
                input_id="factor", binding_hash=bundle.binding_hash
            ),
        )
        retry = app.plan("factor")
        assert retry["status"] == "CONFIRMATION_REQUIRED"
        dispatcher = LocalBackgroundDispatcher(session.task_control_registry)
        try:
            result = app.confirm(retry["plan_hash"], caller="HUMAN", dispatcher=dispatcher)
            assert result["task_id"] == str(task)
            dispatcher.drain_for_tests()
        finally:
            dispatcher.close()
        app.execute(task)
        assert app.readback(task)["status"] == "SUCCEEDED"
        records = app.revisions.lineage("factor")
        assert [v.binding_hash for v in records] == [bundle.binding_hash]
        assert len(sorted(path.name for path in publications.glob("*.json"))) == len(before) + 1
        assert app.plan("factor")["status"] == "REUSED_EXACT"


@pytest.mark.proves("product_host.storage_cleanup_takes_only_confirmed_eligible_targets")
def test_a_cleanup_takes_only_its_eligible_targets_never_an_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A cleanup takes only a person's confirmation of its eligible targets, never a research input,
    a forged target or a redirected path, and holds every other change while it is interrupted."""
    # Its resume after an interruption is issue 1119-C2-7.
    from alphalattice.foundation.feature_engine.panels.retention_residue import (
        PanelRetentionResidueOwner,
    )

    a, b = _inputs(tmp_path, count=2)
    chunk = _panel_generations(tmp_path, monkeypatch)
    monkeypatch.setattr(PanelRetentionResidueOwner, "reconcile_physical_availability", len)
    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        storage = ResearchInputStorage(session)
        with pytest.raises(ValueError, match=r"storage.input_pin_retired"):
            storage.pin(a.binding_hash, pinned=True, caller="HUMAN")
        with pytest.raises(ValueError, match=r"storage.input_pin_unavailable"):
            storage.pin("9" * 64, pinned=True, caller="HUMAN")
        plan = storage.plan()
        assert plan["bindings"] == []
        assert list(plan["targets"]) == [chunk("a"), chunk("d"), chunk("e")]
        with pytest.raises(ValueError, match="human_confirmation"):
            storage.confirm(plan["plan_hash"], caller="INSTALLED_AGENT")
        # A self-consistent forged plan still cannot include a protected chunk.
        forged = {k: v for k, v in plan.items() if k not in {"status", "plan_hash", "limitations"}}
        forged["targets"] = {chunk("c"): file_digest(tmp_path / chunk("c"))}
        forged_hash = storage.owner.publish_input_plan(forged)
        with pytest.raises(RuntimeError, match="not eligible"):
            storage.confirm(forged_hash, caller="HUMAN")
        # A redirected path must not turn an admitted unlink into a deletion of another
        # same-workspace file, even with identical bytes.
        victim = tmp_path / "market-data.duckdb"
        victim.write_bytes(b"must survive")
        redirected = next(iter(plan["targets"]))
        resolve = Path.resolve
        with monkeypatch.context() as scoped:
            scoped.setattr(
                Path,
                "resolve",
                lambda p, *a, **kw: victim if p == tmp_path / redirected else resolve(p, *a, **kw),
            )
            with pytest.raises(RuntimeError, match="outside input storage"):
                storage.owner.apply_input_plan(
                    plan["plan_hash"],
                    references_hash=plan["references_hash"],
                    eligible_targets=plan["targets"],
                )
        assert victim.read_bytes() == b"must survive"
        unlink = Path.unlink
        removed = []

        def fail_after_one(path, *args, **kwargs):
            if path.is_relative_to(tmp_path / "artifacts/feature-panel/chunks"):
                if removed:
                    raise OSError("simulated cleanup interruption")
                removed.append(path)
            return unlink(path, *args, **kwargs)

        with monkeypatch.context() as scoped:
            scoped.setattr(Path, "unlink", fail_after_one)
            with pytest.raises(OSError, match="interruption"):
                storage.confirm(plan["plan_hash"], caller="HUMAN")
        assert storage.readback()["status"] == "RECOVERY_REQUIRED"
        with pytest.raises(RuntimeError, match="Resume approved cleanup"):
            storage.pin(a.binding_hash, pinned=True, caller="HUMAN")
        with pytest.raises(RuntimeError, match="Resume approved cleanup"):
            storage.plan()
        # The started cleanup resumes even after its references changed, as when a release
        # retires a reference field the plan was journalled with; its targets stay checked.
        assert plan["references_hash"] != canonical_hash({"retired": plan["references_hash"]})
        resumed = storage.owner.apply_input_plan(
            plan["plan_hash"],
            references_hash=canonical_hash({"retired": plan["references_hash"]}),
            eligible_targets=plan["targets"],
        )
        assert resumed > 0
        assert storage.confirm(plan["plan_hash"], caller="HUMAN")["deleted_paths"] == 0
        for bundle in (a, b):  # the inputs are untouched by any cleanup
            assert read_factor_bundle(tmp_path, bundle.binding_hash) == bundle


def test_storage_readback_breaks_managed_bytes_down_by_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Storage readback breaks managed bytes down by root."""

    from alphalattice.control.product_host.storage.input_references import MANAGED_ROOTS
    from alphalattice.control.product_host.storage.inventory import managed_file_inventory

    _inputs(tmp_path)
    package = tmp_path / "authority" / "evidence-cro" / "package" / "issuer-registry.json"
    package.parent.mkdir(parents=True)
    package.write_bytes(b"{}" * 512)
    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        view = ResearchInputStorage(session).readback()
    roots = view["by_root"]
    assert list(roots) == list(dict.fromkeys(role for _prefix, role in MANAGED_ROOTS))
    assert sum(v["physical_bytes"] for v in roots.values()) == view["managed_bytes"]
    assert sum(v["logical_bytes"] for v in roots.values()) == view["logical_bytes"]
    files = managed_file_inventory(tmp_path)
    assert sum(v["files"] for v in roots.values()) == len(files)
    assert roots["IMMUTABLE_INPUTS"]["files"] == sum(
        1 for name, _size, _identity in files if name.startswith("research-inputs/")
    )
    assert roots["IMMUTABLE_INPUTS"]["physical_bytes"] > 0
    assert roots["WORKING_DATABASE"] == {"physical_bytes": 0, "logical_bytes": 0, "files": 0}
    assert roots["STAGING"]["files"] == 3  # the fixture's staging copies
    assert roots["AUTHORITY"] == {"physical_bytes": 1024, "logical_bytes": 1024, "files": 1}
    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        _check_storage_status_categories(tmp_path / "footprint", ResearchInputStorage(session))
    _check_task_scratch(tmp_path / "scratch-workspace")
    _check_reclaim_command(tmp_path, monkeypatch)


def _check_storage_status_categories(root: Path, storage: ResearchInputStorage) -> None:
    from alphalattice.control.product_host.composition import task_operations
    from alphalattice.control.workspace_runtime.storage.footprint import files, owner_for
    from alphalattice.interface.local_application.portfolio_research import (
        PortfolioResearchRequestDocument as Request,
    )

    for name, payload in (("blobs/body", b"body"), ("staging/temporary", b"scratch")):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    expected: dict[str, int] = {}
    for path, (allocated, _length, _identity) in files(root):
        owner = owner_for("data-root", path.relative_to(root).as_posix())
        assert owner is not None
        expected[owner[1]] = expected.get(owner[1], 0) + allocated
    assert len(expected) == 2
    measured = []

    def storage_status():
        measured.append(True)
        return storage.status()

    operations = SimpleNamespace(
        status=lambda *_a, **_kw: {},
        storage=SimpleNamespace(status=storage_status),
    )
    task_id = uuid7()
    with history.own_root(root):
        for wait_seconds in (None, 20):
            plain = Request(operation="STATUS", task_id=task_id, wait_seconds=wait_seconds)
            assert (
                task_operations.execute(
                    SimpleNamespace(
                        operations=operations, request=plain.to_operation_request(), caller="HUMAN"
                    )
                )
                == {}
            )
        assert measured == []
        explicit = Request(operation="STATUS", task_id=task_id, status_section="storage")
        answer = task_operations.execute(
            SimpleNamespace(
                operations=operations, request=explicit.to_operation_request(), caller="HUMAN"
            )
        )
    assert measured == [True]
    assert answer is not None and "storage" in answer
    assert answer["storage"]["bytes_by_category"] == expected
    assert answer["storage"]["reclaimable_bytes"] == storage.plan()["reclaimable_bytes"]


def _check_task_scratch(workspace: Path) -> None:
    import psutil

    from alphalattice.control.task_control.scratch import (
        own_task_scratch,
        reclaim_task_scratch,
        task_scratch_directory,
    )

    live = psutil.Process()
    previous = str(uuid7())
    root = workspace / "staging/task-scratch"
    outside = workspace / "do-not-reclaim"
    outside.mkdir(parents=True)
    (outside / "retained").write_text("keep", encoding="utf-8")

    def scratch(epoch: str | None, pid: int) -> Path:
        path = root / str(uuid7())
        path.mkdir(parents=True)
        (path / "payload").write_bytes(b"tmp")
        if epoch is not None:
            identity = path.stat()
            (path / "owner.json").write_text(
                json.dumps(
                    dict(
                        task_id=path.name,
                        host_epoch=epoch,
                        pid=pid,
                        created=live.create_time(),
                        device=identity.st_dev,
                        inode=identity.st_ino,
                    )
                ),
                encoding="utf-8",
            )
        return path

    dead_pid = 2**31 - 1
    paths = {
        name: scratch(epoch, pid)
        for name, epoch, pid in (
            ("killed", previous, dead_pid),
            ("previous", previous, live.pid),
            ("unproved", None, live.pid),
            ("outside-link", previous, live.pid),
        )
    }
    link = paths["outside-link"] / "alias"
    if sys.platform == "win32":
        _winapi.CreateJunction(str(outside), str(link))
    else:
        link.symlink_to(outside, target_is_directory=True)
    with WorkspaceApplicationSession.acquire(workspace) as session:
        for case in ("killed", "previous", "outside-link"):
            assert not paths[case].exists(), case
        assert paths["unproved"].exists()
        assert (outside / "retained").read_text(encoding="utf-8") == "keep"
        assert str(paths["unproved"].name) in session.scratch_cleanup["unproved"]
        current = session.scratch_epoch
        missing_process = scratch(current, dead_pid)
        with own_task_scratch(workspace, uuid7(), current):
            running = task_scratch_directory()
            assert running is not None and running.is_dir()
            read = reclaim_task_scratch(workspace, current)
            assert read["removed"] == [missing_process.name]
            assert not missing_process.exists()
            assert running.is_dir()


def _check_reclaim_command(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import io

    from alphalattice.control.product_host.composition.operation_route_families import host_door
    from alphalattice.control.product_host.composition.portfolio_research_operations import (
        PortfolioResearchOperations,
    )
    from alphalattice.control.product_host.composition.research_workspace import (
        read_research_workspace_manifest,
    )
    from alphalattice.control.task_control.runner import TaskHeartbeatReader
    from alphalattice.foundation.feature_engine.panels.retention_residue import (
        PanelRetentionResidueOwner,
    )
    from alphalattice.interface.local_application import cli, client
    from alphalattice.interface.local_application.cli_contract import (
        REQUEST_PROVENANCE,
        RequestProvenance,
    )
    from alphalattice.interface.local_application.dispatcher import LocalBackgroundDispatcher
    from alphalattice.interface.local_application.goals import GoalSession
    from alphalattice.interface.local_application.portfolio_research import (
        PortfolioResearchRequestDocument as Request,
    )
    from alphalattice.investment.alpha_research.publication.artifacts import (
        AlphaCurrentArtifactStore,
    )

    parser = cli.command_parser(frozenset({"storage"}))
    assert ("storage", "reclaim") in cli.CLIENT_COMMANDS
    chunk = _panel_generations(workspace, monkeypatch)
    monkeypatch.setattr(PanelRetentionResidueOwner, "reconcile_physical_availability", lambda _: 0)
    pinned = (
        workspace / "artifacts/alpha-research/current/workspace-observation-history/pinned.parquet"
    )
    pinned.parent.mkdir(parents=True)
    pinned.write_bytes(b"pinned")
    pin = "f" * 64
    file = SimpleNamespace(path=pinned, sha256=file_digest(pinned), bytes=pinned.stat().st_size)

    def retention(_self, referenced_heads=(), *, active_scope_hashes=None):
        held = pin in referenced_heads
        return SimpleNamespace(
            head_hashes=(pin,),
            roots=(file,) if held else (),
            targets=() if held else (file,),
        )

    monkeypatch.setattr(
        AlphaCurrentArtifactStore, "workspace_observation_history_retention", retention
    )
    with WorkspaceApplicationSession.acquire(workspace) as session:
        storage = ResearchInputStorage(session)
        pins = {pin: 1}
        receipt_hash = canonical_hash(pins)
        storage.owner._publish("input-pin-receipts", receipt_hash, pins)
        (storage.owner.governance_root / "input-pins.json").write_text(
            json.dumps({"pins": pins, "receipt_hash": receipt_hash}), encoding="utf-8"
        )
        host = PortfolioResearchOperations(
            dispatcher=LocalBackgroundDispatcher(status_port=session.task_control_registry),
            workspace_manifest=read_research_workspace_manifest(workspace),
            workspace_session=session,
            heartbeats=TaskHeartbeatReader(session.runtime_path),
            door=host_door(),
        )
        opened = host.execute(
            Request(
                operation="GOAL_OPEN",
                change_reason="Configure the storage fixture",
                goal_declaration={
                    "title": "Workspace care",
                    "objective": "Reclaim eligible storage.",
                    "kind": "RESEARCH",
                    "criteria": [{"criterion_id": "storage", "text": "Keep pinned objects."}],
                    "deliverables": [
                        {
                            "deliverable_id": "cleanup",
                            "kind": "RESULT",
                            "description": "Cleanup result.",
                        },
                    ],
                },
            ).to_operation_request()
        )
        goal_id = UUID(opened["goal_id"])
        member = GoalSession(vendor="codex", session_id="storage-fixture")
        host.goals.store.bind(member, goal_id)
        provenance = RequestProvenance(
            vendor=member.vendor, session=member.session_id, goal_id=str(goal_id)
        )

        def request(document):
            token = REQUEST_PROVENANCE.set(provenance)
            try:
                return host.execute(
                    Request.model_validate(document).to_operation_request(),
                    caller="INSTALLED_AGENT",
                )
            finally:
                REQUEST_PROVENANCE.reset(token)

        class Terminal(io.StringIO):
            def isatty(self) -> bool:
                return True

        connection = SimpleNamespace(request=request)
        monkeypatch.setattr(sys, "stdin", io.StringIO())
        preview = client.reclaim_storage(connection, None)
        assert list(preview["targets"]) == [chunk("a"), chunk("d"), chunk("e")]
        hash_only = ["--confirm", preview["plan_hash"]]
        with pytest.raises(SystemExit):
            cli._client_fields(parser, parser.parse_args(["storage", "reclaim", *hash_only]))
        words = "Yes, please apply that cleanup plan."
        question = preview["ask_now"]
        approval = (*hash_only, "--asked", question, "--person-said", words)
        monkeypatch.setattr(sys, "stdin", Terminal("no\n"))
        for case, flags, count in (
            ("declined", (), 0),
            ("pinned", approval, 3),
            ("repeated", (), 0),
        ):
            args = parser.parse_args(["storage", "reclaim", *flags])
            confirmation = cli._client_fields(parser, args)["storage_reclaim_confirmation"]
            answer = client.reclaim_storage(connection, confirmation)
            assert answer.get("deleted_paths") == count, (case, answer)
            assert pinned.read_bytes() == b"pinned", case
            approvals = [
                event["relayed"]
                for event in host.goals.store.attributed(goal_id)
                if event.get("operation") == "STORAGE_CONFIRM"
            ]
            if case == "declined":
                assert not approvals and all((workspace / p).exists() for p in preview["targets"])
            else:
                assert [(row["question"], row["words"]) for row in approvals] == [(question, words)]


def test_no_link_fallback_checks_capacity_before_copy(tmp_path: Path, monkeypatch):
    import alphalattice.control.product_host.research_authoring.factor_inputs as inputs
    from alphalattice.control.product_host.storage.inventory import StorageInventoryError

    source = tmp_path / "source.parquet"
    source.write_bytes(b"immutable source")
    target = tmp_path / "research-inputs/alias/source.parquet"
    copied = []

    def no_link(*_args, **_kwargs):
        raise OSError("hardlinks unsupported")

    def full(*_args, **_kwargs):
        raise StorageInventoryError("storage.disk_space_insufficient", "no recovery headroom")

    monkeypatch.setattr(inputs.os, "link", no_link)
    monkeypatch.setattr(inputs, "require_storage_capacity", full)
    monkeypatch.setattr(inputs.shutil, "copy2", lambda *args: copied.append(args))
    with pytest.raises(StorageInventoryError):
        _copy_input(tmp_path, source, target)
    assert copied == [] and not target.exists()


def _panel_generations(root: Path, monkeypatch: pytest.MonkeyPatch):
    """Four Panel generations (`c` current, `b` before the last data update) and a chunk no
    generation published (`e`); returns each chunk's workspace path by its letter."""
    from alphalattice.control.data_platform.maintenance.registry import (
        DuckDbWorkspaceMaintenanceRegistry,
    )
    from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
    from alphalattice.foundation.feature_engine.storage.repositories import PanelStateRepository

    panel_root = root / "artifacts/feature-panel"
    (panel_root / "manifests").mkdir(parents=True)
    (panel_root / "chunks").mkdir()
    for name in "abcd":
        (panel_root / "manifests" / f"{name * 64}.json").write_text("{}")
        (panel_root / "chunks" / f"{name * 64}.parquet").write_bytes(name.encode())
    (panel_root / "chunks" / f"{'e' * 64}.parquet").write_bytes(b"e")
    monkeypatch.setattr(
        PanelStateRepository,
        "feature_panel_snapshot_for_active",
        lambda *_: {"snapshot_hash": "c" * 64},
    )
    monkeypatch.setattr(
        DuckDbWorkspaceMaintenanceRegistry,
        "read_data_update_receipt",
        lambda *_: SimpleNamespace(
            before=SimpleNamespace(panel_hash="b" * 64),
            after=SimpleNamespace(panel_hash="c" * 64),
            content_hash="e" * 64,
        ),
    )
    monkeypatch.setattr(
        ArtifactResolver,
        "load_feature_panel_manifest",
        lambda _self, uri: {"chunks": [{"chunk_hash": uri.rsplit("/", 1)[-1]}]},
    )
    return lambda name: f"artifacts/feature-panel/chunks/{name * 64}.parquet"


@pytest.mark.proves("product_host.panel_cleanup_keeps_current_and_previous_generations")
def test_source_snapshot_roots_keep_current_previous_and_release_the_rest(tmp_path, monkeypatch):
    """The current generation and the one before the last data update are kept, every other
    chunk released; a busy workspace releases none, and without a data update's receipt every
    generation is kept and only a chunk no generation published is released."""
    from alphalattice.control.data_platform.maintenance.registry import (
        DuckDbWorkspaceMaintenanceRegistry,
    )
    from alphalattice.control.product_host.storage.retention import file_content_hash

    chunk = _panel_generations(tmp_path, monkeypatch)
    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        storage = ResearchInputStorage(session)
        view = storage._source_references(busy=False)
        assert list(view["targets"]) == [chunk("a"), chunk("d"), chunk("e")]
        assert view["targets"][chunk("e")] == file_content_hash(tmp_path / chunk("e"))
        assert storage._source_references(busy=True)["targets"] == {}
        monkeypatch.setattr(
            DuckDbWorkspaceMaintenanceRegistry, "read_data_update_receipt", lambda *_: None
        )
        assert list(storage._source_references(busy=False)["targets"]) == [chunk("e")]


def test_the_plan_links_a_retained_retrieval_model_copy_to_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The plan links a retained retrieval model copy to the store."""

    from alphalattice.capabilities.knowledge import model_store
    from alphalattice.control.product_host.composition.evidence_review_workspace import (
        EvidenceReviewArtifactBinding,
        EvidenceReviewWorkspaceManifest,
    )
    from alphalattice.control.product_host.composition.research_workspace import (
        ResearchWorkspaceEvidenceReview,
        read_research_workspace_manifest,
    )

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _inputs(workspace)
    files = {
        "encoder": {"onnx/model.onnx": b"encoder graph", "sentencepiece.bpe.model": b"spm"},
        "reranker": {"config.json": b"{}", "onnx/model.onnx": b"reranker graph"},
    }
    store, packs = tmp_path / "store", []
    for name, payload in files.items():
        manifest = tuple(sorted((f, hashlib.sha256(b).hexdigest()) for f, b in payload.items()))
        pack = model_store.PackDefinition(
            name, f"tests/{name}", "0" * 40, manifest, "MIT", "ONNX_CPU", 42
        )
        for file, data in payload.items():
            held = tmp_path / f"source-{name}" / file
            held.parent.mkdir(parents=True, exist_ok=True)
            held.write_bytes(data)
            copied = workspace / "evidence-cro-authority" / "semantic-model"
            copied = copied / (file if name == "encoder" else f"reranker/{file}")
            copied.parent.mkdir(parents=True, exist_ok=True)
            copied.write_bytes(data)
        fetcher = model_store.local_directory_fetcher(tmp_path / f"source-{name}")
        model_store.install_pack(store, pack, fetcher)
        packs.append(pack)
    monkeypatch.setenv("ALPHALATTICE_MODEL_STORE", str(store))
    monkeypatch.setattr(
        model_store,
        "recipe_pack_statuses",
        lambda root, _recipe: tuple(model_store.pack_status(root, pack) for pack in packs),
    )
    child = EvidenceReviewArtifactBinding(
        relative_path="evidence-cro-authority/registry.json",
        file_sha256="a" * 64,
        content_hash="b" * 64,
    )
    authority = EvidenceReviewWorkspaceManifest.create(
        authority_id="stand-in",
        issuer_registry=child,
        listing_authority=child,
        recorded_documents=child,
        semantic_model_relative_path="evidence-cro-authority/semantic-model",
        semantic_capability_hash="c" * 64,
    )
    payload = authority.model_dump_json().encode("utf-8")
    (workspace / "evidence-cro-authority" / "authority.json").write_bytes(payload)
    publish_research_workspace_manifest(
        workspace,
        read_research_workspace_manifest(workspace).with_bindings(
            evidence_review=ResearchWorkspaceEvidenceReview(
                relative_path="evidence-cro-authority/authority.json",
                file_sha256=hashlib.sha256(payload).hexdigest(),
            )
        ),
    )
    copy_bytes = sum(len(b) for payload in files.values() for b in payload.values())
    with WorkspaceApplicationSession.acquire(workspace) as session:
        storage = ResearchInputStorage(session)
        plan = storage.plan()
        assert plan["relinks"] == {"evidence-cro-authority/semantic-model": copy_bytes}
        done = storage.confirm(plan["plan_hash"], caller="HUMAN")
        assert done["relinked"] == {"evidence-cro-authority/semantic-model": copy_bytes}
        model = workspace / "evidence-cro-authority" / "semantic-model"
        assert not model_store.retained_copy(model)
        assert (model / "reranker" / "onnx" / "model.onnx").read_bytes() == b"reranker graph"
        assert "relinks" not in storage.plan()
