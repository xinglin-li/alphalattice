"""Input retention mechanics; no synthetic file is claimed as numerical evidence."""

from __future__ import annotations

import json
import os
import sys
import threading
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

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
    file_digest,
    read_factor_bundle,
)
from alphalattice.control.product_host.storage.input_references import ResearchInputStorage
from alphalattice.control.task_control.contracts import (
    ResearchGoal,
    ResearchPlan,
    TaskInputEnvelope,
    WorkItemDefinition,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from tests.portfolio_strategy_lab.local_web_support import _manifest


def test_observation_capacity_takes_a_fresh_sample_when_the_workspace_reopens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """V680: a retained sample from another session cannot set this session's allowance."""
    from types import SimpleNamespace

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


def _counting_reads(monkeypatch: pytest.MonkeyPatch, inject=None) -> list[int]:
    """Count every whole-object read of `file_digests`; `inject` runs once, at the
    first read -- after every path is open and grouped, before any byte is read."""

    import alphalattice.control.product_host.research_authoring.factor_inputs as inputs

    original = inputs._digest_stream
    reads = [0]
    # The objects are digested on a pool: the first read's inject holds the
    # others until it has run, or a second read also sees a count of zero.
    first_read = threading.Lock()

    def counted(stream):
        with first_read:
            if reads[0] == 0 and inject is not None:
                inject()
            reads[0] += 1
        return original(stream)

    monkeypatch.setattr(inputs, "_digest_stream", counted)
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
    # Nothing survives a call: the next call reads every object again.
    assert file_digests([a, b, c]) == digests
    assert reads[0] == 4
    assert digests[a] == digests[b] == file_digest(a) and digests[c] == file_digest(c)
    assert digests[c] != digests[a]


def test_an_alias_replaced_after_grouping_never_answers_another_objects_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The window: every path is open and grouped, no byte is read yet; the
    non-representative link B is pointed at an independent file. Where the
    platform lets that happen the call refuses B by name; where it holds the
    open object in place the replacement itself is refused and B still reads
    as the object that was verified."""

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
    # The same window where the platform does let the path go: B no longer
    # names any object by the time the read is over.
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
        read_factor_bundle,
    )
    from alphalattice.protocols.research_authoring.contracts import AuthoringError

    (bundle,) = _inputs(tmp_path, count=1)
    source = tmp_path / "research-inputs" / bundle.binding_hash / "source"
    alias = source / "market-data.duckdb"
    pool = tmp_path / "research-inputs/databases" / bundle.database_snapshot_hash
    pool = pool / "market-data.duckdb"
    assert os.stat(alias).st_ino == os.stat(pool).st_ino
    reads = _counting_reads(monkeypatch)
    assert read_factor_bundle(tmp_path, bundle.binding_hash) == bundle
    # Three objects (the manifest copy, the Parquet object, the database
    # object named by its alias and its pool path), three reads.
    assert reads[0] == 3
    forged = tmp_path / "forged.duckdb"
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
        import _winapi

        _winapi.CreateJunction(str(outside), str(folder / "linked"))
    else:
        (folder / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(inventory.StorageInventoryError) as refused:
        inventory.managed_file_inventory(tmp_path)
    assert refused.value.failure_code == "storage.retention_root_mismatch"


def _inputs(root: Path, *, count: int = 3):
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
        folder = root / "research-inputs" / bundle.binding_hash
        for name, digest in files:
            original = (
                database
                if name == "market-data.duckdb"
                else parquet
                if name == "data.parquet"
                else panel_file
            )
            _copy_input(root, original, folder / "source" / name, digest)
        (folder / "manifest.json").write_text(bundle.model_dump_json(), encoding="utf-8")
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
        history = root / "research-inputs/publications"
        history.mkdir(exist_ok=True)
        (history / f"{receipt}.json").write_text(
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
    from alphalattice.control.product_host.storage.retention import StorageRetentionError
    from alphalattice.control.task_control.runner import StageDisposition, StageExecutionResult
    from alphalattice.interface.local_application.cli_contract import worded_refusal
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
        storage = ResearchInputStorage(session)
        rows = {v["binding_hash"]: v["roots"] for v in storage.readback()["inputs"]}
        assert rows[a.binding_hash] == []
        assert "PREVIOUS_ROLLBACK" in rows[b.binding_hash]
        assert "CURRENT_RESEARCH_VERSION" in rows[c.binding_hash]
        assert "CURRENT_ACTIVE" in rows[default.binding_hash]
        # The installed binding and a succeeded historical publication both
        # name their exact missing manifest, without discarding the references.
        before = storage.readback()
        for binding in (default.binding_hash, c.binding_hash):
            folder = tmp_path / "research-inputs" / binding
            aside = tmp_path / f"held-{binding}"
            folder.rename(aside)
            try:
                with pytest.raises(StorageRetentionError) as refused:
                    storage.readback()
                subject = f"research-inputs/{binding}"
                code = f"storage.input_binding_missing:{subject}"
                assert refused.value.failure_code == code
                answer = worded_refusal({"status": "REFUSED", "failure_code": code})
                assert f"{subject}/manifest.json" in answer["detail"]
                assert "backup restore" in answer["detail"]
                assert answer["next_action"] == "RESTORE_THE_WORKSPACE_FROM_A_BACKUP"
            finally:
                aside.rename(folder)
            restored = storage.readback()
            assert restored["inputs"] == before["inputs"]
            assert restored["references_hash"] == before["references_hash"]
        plan = storage.plan()
        assert plan["bindings"] == [a.binding_hash]
        storage.confirm(plan["plan_hash"], caller="HUMAN")
        assert app.revisions.versions()["status"] == "AVAILABLE"
        with pytest.raises(RuntimeError, match="released"):
            app.revisions.select("factor", a.binding_hash)
        assert app.revisions.select("factor", b.binding_hash).binding_hash == b.binding_hash


def test_a_bundle_damaged_before_its_publication_leaves_no_receipt(tmp_path: Path, monkeypatch):
    """counterexample: the receipt must follow the whole read, never precede it.

    The binding completes and the manifest is written last; before the
    revision is published, the sealed database in the target bundle is
    damaged (the source workspace is unchanged). The Task must block with
    no publication record and no lineage movement, and the same Task, once
    the Human confirms again over restored bytes, publishes exactly one
    record. The publisher has no switch that records a revision unread.
    """
    import inspect

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
    sealed = tmp_path / "research-inputs" / bundle.binding_hash / "source" / "market-data.duckdb"
    good = sealed.read_bytes()
    publications = tmp_path / "research-inputs/publications"
    before = sorted(path.name for path in publications.glob("*.json"))
    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        app = input_capture.ResearchInputCaptureApplication(
            session, clock=lambda: datetime.now(UTC)
        )
        current = source.model_copy(update={"data_revision_hash": "5" * 64})
        monkeypatch.setattr(app, "_source", lambda root=None: source if root else current)
        assert app.plan("factor")["status"] == "CONFIRMATION_REQUIRED"
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


def test_roots_pins_stale_cleanup_shared_objects_and_metadata_survive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    a, b, c = _inputs(tmp_path)
    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        storage = ResearchInputStorage(session)
        publication = next((tmp_path / "research-inputs/publications").glob("*.json"))
        saved = publication.read_bytes()
        try:
            value = json.loads(saved)
            value["binding_hash"] = "f" * 64
            publication.write_text(json.dumps(value), encoding="utf-8")
            with pytest.raises(RuntimeError) as refused:
                storage.plan()
            assert refused.value.failure_code == "storage.retention_root_mismatch"
        finally:
            publication.write_bytes(saved)
        plan = storage.plan()
        assert plan["bindings"] == [a.binding_hash]
        storage.pin(a.binding_hash, pinned=True, caller="HUMAN")
        with pytest.raises(ValueError, match="human_confirmation"):
            storage.confirm(plan["plan_hash"], caller="INSTALLED_AGENT")
        with pytest.raises(RuntimeError, match="changed"):
            storage.confirm(plan["plan_hash"], caller="HUMAN")
        assert storage.plan()["targets"] == {}
        storage.pin(a.binding_hash, pinned=False, caller="HUMAN")
        plan = storage.plan()
        # A self-consistent forged plan still cannot include a protected input.
        forged = {k: v for k, v in plan.items() if k not in {"status", "plan_hash", "limitations"}}
        forged["targets"] = {
            f"research-inputs/{c.binding_hash}/source/market-data.duckdb": c.database_snapshot_hash
        }
        forged_hash = storage.owner.publish_input_plan(forged)
        with pytest.raises(RuntimeError, match="not eligible"):
            storage.confirm(forged_hash, caller="HUMAN")
        # A redirected alias must not turn an admitted input unlink into a
        # deletion of another same-workspace file, even with identical bytes.
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
            if path.is_relative_to(tmp_path / "research-inputs"):
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
            storage.pin(c.binding_hash, pinned=True, caller="HUMAN")
        with pytest.raises(RuntimeError, match="Resume approved cleanup"):
            storage.plan()
        result = storage.confirm(plan["plan_hash"], caller="HUMAN")
        assert result["deleted_paths"] > 0
        assert read_factor_bundle(tmp_path, b.binding_hash) == b
        assert read_factor_bundle(tmp_path, c.binding_hash) == c
        assert read_factor_bundle(tmp_path, a.binding_hash, verify=False) == a
        with pytest.raises(RuntimeError, match="released"):
            read_factor_bundle(tmp_path, a.binding_hash)
        assert storage.confirm(plan["plan_hash"], caller="HUMAN")["deleted_paths"] == 0


def test_storage_readback_breaks_managed_bytes_down_by_root(tmp_path: Path):
    """The page's storage summary is the inventory's own accounting, by role: the roles
    sum to the managed total (a shared object counted once, under the first root that
    names it), every reference counts logically, a role the workspace does not have is
    zero, not absent, and every store that holds the workspace's content is counted (V208):
    an installed authority package among them."""

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


def test_an_unfinished_task_protects_its_input_without_a_foundation(tmp_path: Path):
    a, _b, _c = _inputs(tmp_path)
    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        envelope = TaskInputEnvelope.create(
            task_kind="research_experiment",
            input_schema_id="test-input",
            payload={"plan": {"binding": {"binding_hash": a.binding_hash}}},
        )
        goal = ResearchGoal.create(
            goal_kind="TEST_RECOVERY",
            input_hash=envelope.input_hash,
            deliverable_kind="evidence",
            summary="Keep input while work is owed.",
        )
        workflow = ResearchPlan.create(
            goal_hash=goal.goal_hash,
            workflow_definition_hash="c" * 64,
            verifier_catalog_hash="d" * 64,
            work_items=(
                WorkItemDefinition.create(
                    stage_id="check_input", dependency_ids=(), verifier_id="test.input"
                ),
            ),
        )
        session.task_control_registry.admit(
            input_envelope=envelope, goal=goal, plan=workflow, observed_at=datetime.now(UTC)
        )
        view = ResearchInputStorage(session).readback()
        assert (
            "IN_FLIGHT_RECOVERY"
            in next(v for v in view["inputs"] if v["binding_hash"] == a.binding_hash)["roots"]
        )
        assert ResearchInputStorage(session).plan()["targets"] == {}


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
        _copy_input(tmp_path, source, target, file_digest(source))
    assert copied == [] and not target.exists()


def test_source_snapshot_roots_keep_current_previous_and_rooted_and_release_the_rest(
    tmp_path, monkeypatch
):
    """Root routing only: injected reader results are not numerical/hash evidence.

    requirement (V207): a generation no research input captured is kept while it is the current
    or the previous one; past the previous, and a chunk no generation published, is released
    under the confirmed plan; without a data update's receipt none is past the previous.
    """
    from types import SimpleNamespace

    from alphalattice.control.data_platform.maintenance.registry import (
        DuckDbWorkspaceMaintenanceRegistry,
    )
    from alphalattice.control.product_host.storage.retention import file_content_hash
    from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
    from alphalattice.foundation.feature_engine.storage.repositories import PanelStateRepository

    panel_root = tmp_path / "artifacts/feature-panel"
    (panel_root / "manifests").mkdir(parents=True)
    (panel_root / "chunks").mkdir()
    bundles = {}
    for name in "abcd":
        panel_hash = name * 64
        (panel_root / "manifests" / f"{panel_hash}.json").write_text("{}")
        chunk = panel_root / "chunks" / f"{panel_hash}.parquet"
        chunk.write_bytes(name.encode())
        if name != "d":
            bundle = FactorInputBundle.create(
                panel_snapshot_hash=panel_hash,
                outcome_snapshot_hash="f" * 64,
                sessions=(date(2024, 1, 2),),
                files=((chunk.relative_to(tmp_path).as_posix(), file_digest(chunk)),),
            )
            bundles[bundle.binding_hash] = bundle
    orphan = panel_root / "chunks" / f"{'e' * 64}.parquet"  # a composition no one published
    orphan.write_bytes(b"e")
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
    with WorkspaceApplicationSession.acquire(tmp_path) as session:
        storage = ResearchInputStorage(session)
        roots = {h: set() for h in bundles}
        chunk = "artifacts/feature-panel/chunks/{}.parquet".format
        view = storage._source_references(bundles, roots, busy=False)
        assert list(view["targets"]) == [chunk("a" * 64), chunk("d" * 64), chunk("e" * 64)]
        assert view["targets"][chunk("e" * 64)] == file_content_hash(orphan)
        assert storage._source_references(bundles, roots, busy=True)["targets"] == {}
        a = next(h for h, b in bundles.items() if b.panel_snapshot_hash == "a" * 64)
        roots[a].add("USER_PINNED")
        pinned = storage._source_references(bundles, roots, busy=False)["targets"]
        assert list(pinned) == [chunk("d" * 64), chunk("e" * 64)]
        monkeypatch.setattr(
            DuckDbWorkspaceMaintenanceRegistry, "read_data_update_receipt", lambda *_: None
        )
        unreceipted = storage._source_references(bundles, roots, busy=False)["targets"]
        assert list(unreceipted) == [chunk("b" * 64), chunk("e" * 64)]


def test_the_plan_links_a_retained_retrieval_model_copy_to_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """requirement (V208): an Evidence workspace on the retained recipe that holds its own model
    copy is offered, in the storage plan a person confirms, the copy's link to the machine's
    store when the store holds its packs; the confirmation links it and the next plan offers
    nothing more. The packs here are small stand-ins for the retained recipe's two."""

    import hashlib

    from alphalattice.control.product_host.composition.evidence_review_workspace import (
        EvidenceReviewArtifactBinding,
        EvidenceReviewWorkspaceManifest,
    )
    from alphalattice.control.product_host.composition.research_workspace import (
        ResearchWorkspaceEvidenceReview,
        read_research_workspace_manifest,
    )
    from alphalattice.kernel.knowledge import model_store

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
