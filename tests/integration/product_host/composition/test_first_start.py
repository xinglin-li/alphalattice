"""The ordinary Host's first 0.2.8 start upgrades every old workspace on one acknowledged plan."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from collections.abc import Callable
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from alphalattice.control.data_platform.maintenance.contracts import (
    MaintenanceTrigger,
    WorkspaceMaintenanceRequest,
)
from alphalattice.control.data_platform.maintenance.registry import (
    DuckDbWorkspaceMaintenanceRegistry,
)
from alphalattice.control.product_host.composition import first_start
from alphalattice.control.product_host.composition.local_web_session import (
    LocalPortfolioWebSession,
)
from alphalattice.control.product_host.composition.research_workspace import (
    initialize_research_workspace,
    named_workspace_id,
    publish_research_workspace_manifest,
    read_research_workspace_manifest,
)
from alphalattice.control.product_host.storage.backup import WorkspaceBackups
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.foundation.data_layer import history, history_execution_outcomes, history_startup
from alphalattice.foundation.data_layer.upgrade import PROGRESS, Family, UpgradeRunner
from alphalattice.interface.local_application import cli
from alphalattice.interface.local_application.client import LocalResearchConnection
from alphalattice.kernel.clock import utc_now
from alphalattice.kernel.identity import blake3_filename
from tests._support.identity_fixtures import fixture_digest
from tests.portfolio_strategy_lab.host_requests import _json, _request

TEST_SOURCE = "src/alphalattice/control/product_host/composition/first_start.py"
OUTCOMES = Path("artifacts/data-operations/execution-outcomes")


def _held_family(stops: list[str], asked: list[str] | None = None) -> Callable[[], Family]:
    """A family whose old data is `old-store.json`; it stops once after its first workspace."""

    def make() -> Family:
        def step(runner: Any) -> None:
            for workspace in runner.workspaces:
                old = workspace / "old-store.json"
                if old.is_file():
                    runner.map_id("held", workspace.name, f"new-{workspace.name}")
                    (workspace / "new-store.json").write_bytes(old.read_bytes())
                    if stops:
                        raise RuntimeError(stops.pop())

        def verify(runner: Any) -> bool:
            return all(
                (w / "new-store.json").read_bytes() == (w / "old-store.json").read_bytes()
                for w in runner.workspaces
            )

        def holds(workspace: Path) -> bool:  # old data its new store lacks
            if asked is not None:
                asked.append("held")
            old, new = workspace / "old-store.json", workspace / "new-store.json"
            return old.is_file() and (not new.is_file() or new.read_bytes() != old.read_bytes())

        return Family(
            "held",
            step,
            verify,
            holds,
            trust_paths=("/old-store.json", "/new-store.json"),
            shared_inputs=("/old-store.json",),
        )

    return make


def _old(workspace: Path, name: str) -> None:
    """A workspace that still holds an old store of its own, for the held family to upgrade."""
    initialize_research_workspace(workspace)
    (workspace / "old-store.json").write_text(json.dumps({"rows": [name]}), encoding="utf-8")


def _progress(subject: str) -> list[dict[str, Any]]:
    """The run's progress as the stopped Host left it on the machine's log."""
    log = history.data_root() / "machine" / "log.sqlite"
    with closing(sqlite3.connect(f"{log.as_uri()}?mode=ro", uri=True)) as con:
        rows = con.execute(
            "SELECT attrs FROM event WHERE type = ? AND subject_id = ? ORDER BY seq",
            (PROGRESS, subject),
        ).fetchall()
    return [json.loads(row[0]) for row in rows]


def _cli(capsys: pytest.CaptureFixture[str], workspace: Path, *line: str) -> tuple[int, Any]:
    """One CLI command line against the Host serving `workspace`: its exit code and answer."""
    try:
        code = cli.main(["--workspace", str(workspace), *line], serve=lambda _: 99)
    except SystemExit as refused:  # a refusal exits with its code
        code = int(refused.code or 0)
    return code, json.loads(capsys.readouterr().out)["data"]


def _started(workspace: Path) -> LocalPortfolioWebSession:
    session = LocalPortfolioWebSession.from_workspace(workspace)
    session.start()
    return session


@pytest.mark.proves("product_host.first_start_upgrades_every_old_workspace")
def test_the_first_start_upgrades_every_old_workspace_on_one_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """requirement: the first 0.2.8 start upgrades every old workspace on one acknowledged plan."""
    _upgrades_on_one_plan(tmp_path, monkeypatch, capsys)


def _upgrades_on_one_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Upgrade three held old workspaces on one plan: two at the first start, one later."""
    monkeypatch.setenv("ALPHALATTICE_BACKUP_ROOT", str(tmp_path / "backups"))
    stops = ["stopped mid-family"]
    monkeypatch.setattr(first_start, "FAMILIES", (_held_family(stops),))
    made = first_start.FAMILIES
    families = [make().name for make in made]
    project = tmp_path / "project" / "workspaces"
    first, second, later = project / "first", project / "second", project / "later"
    _old(first, "first")
    _old(second, "second")
    # Second is from before the kept bound, its store holding a row stamped after the wall reads.
    ahead = utc_now() + timedelta(hours=2)
    DuckDbWorkspaceMaintenanceRegistry(
        second / "market-data.duckdb", gate=WorkspaceMutationGate()
    ).admit(
        WorkspaceMaintenanceRequest.create(
            market_profile_id="earlier-release",
            target_market_session=date(2026, 7, 31),
            knowledge_cutoff_at=ahead,
            trigger=MaintenanceTrigger.STARTUP,
            membership_revision=fixture_digest("membership"),
            data_policy_hash=fixture_digest("data-policy"),
            feature_policy_hash=fixture_digest("feature-policy"),
        ),
        observed_at=ahead,
    )
    (second / ".alphalattice-writer.lock").write_bytes(b"0")  # the lock byte alone

    # The first start shows its plan and serves nothing else; nothing moves before acknowledgement.
    host = _started(first)
    try:
        session = _json(host, "/api/session")
        assert session["execution_mode"] == "UPGRADE"
        assert set(session["routes"]) == {"UPGRADE_OVERVIEW", "UPGRADE_ACKNOWLEDGE"}
        status, _headers, body = _request(host, "/api/tasks")
        refused = json.loads(body)
        assert (status, refused["failure_code"]) == (503, "upgrade.in_progress")
        overview = _json(host, "/api/upgrade")
        plan = overview["first_start"]
        assert plan["state"] == "PENDING" and plan["families"] == families
        assert plan["workspaces"] == [str(first.resolve()), str(second.resolve())]
        assert [(b["kind"], b["generation"]) for b in plan["backups"]] == [
            ("machine", None),
            ("workspace", None),
            ("workspace", None),
        ]
        assert all(b["free_after"] > 0 for b in plan["backups"])
        assert not (tmp_path / "backups").exists()  # nothing is written before acknowledgement
        status, _headers, body = _request(
            host, "/api/upgrade/acknowledge", method="POST", payload={"upgrade_set_hash": "0" * 64}
        )
        assert (status, json.loads(body)["failure_code"]) == (400, "upgrade.plan_changed")
        # The CLI reads the same plan through the connection this Host publishes, and refuses
        # every other operation with the progress. None of it writes the log, whose head the
        # data root's backup must still hold when the run checks it.
        log = history.holding()
        assert log is not None
        events = log.read("SELECT count(*) FROM event")[0][0]
        code, shown = _cli(capsys, first, "upgrade", "show")
        same = ("state", "workspaces", "subject")
        assert code == 0 and {k: shown["first_start"][k] for k in same} == {
            k: plan[k] for k in same
        }
        # The compact answer pages a list over eight items; its section read is the whole list.
        code, listed = _cli(capsys, first, "upgrade", "show", "--section", "first_start.families")
        assert code == 0 and listed["value"] == plan["families"]
        assert [b["repository"] for b in shown["first_start"]["backups"]] == [
            b["repository"] for b in plan["backups"]
        ]
        code, task_list = _cli(capsys, first, "task", "list")
        assert code != 0 and task_list["failure_code"] == "upgrade.in_progress"
        assert task_list["upgrade"] == plan
        assert log.read("SELECT count(*) FROM event")[0][0] == events

        # Acknowledged through the CLI: the backups are written, and the run stops mid-family.
        plan_hash = overview["next_requests"]["acknowledge"]["upgrade_set_hash"]
        code, running = _cli(capsys, first, "upgrade", "acknowledge", plan_hash)
        assert code == 0 and running["first_start"]["state"] == "RUNNING"
        assert not host.upgraded(timeout=600)
        stopped = _json(host, "/api/upgrade")["first_start"]
        assert stopped["state"] == "FAILED" and stopped["family_failed"] == "held"
        assert _request(host, "/api/tasks")[0] == 503
    finally:
        host.stop()
    assert not LocalResearchConnection.path(first).exists()  # withdrawn with the restricted Host
    started = [e for e in _progress("0.2.8") if e["kind"] == "started"]
    assert len(started) == 1 and len(started[0]["backups"]["workspaces"]) == 2
    # Second's backup is stamped by its own clock, seeded past its store's latest row.
    kept = WorkspaceBackups(
        second,
        workspace_id=str(named_workspace_id(second)),
        root=Path(plan["backups"][2]["repository"]),
    )
    assert [generation.created_at > ahead for generation in kept.generations()] == [True]

    # The next start resumes the recorded plan with no new acknowledgement, then serves in full.
    host = _started(first)
    try:
        assert host.first_start is not None and host.first_start.plan is not None
        assert host.first_start.plan.plan_hash == plan_hash  # the same plan, already acknowledged
        assert host.upgraded(timeout=600)
        assert _json(host, "/api/session")["workspace_id"]
        assert _request(host, "/api/tasks")[0] == 200
    finally:
        host.stop()
    progress = _progress("0.2.8")
    assert [e["kind"] for e in progress if e["kind"] == "started"] == ["started"]
    assert [(e["family"], e["verdict"]) for e in progress if e["kind"] == "family_done"] == [
        (name, "ALL EQUAL") for name in families
    ]
    assert progress[-1]["kind"] == "completed"

    # A third old workspace that appears afterwards is upgraded alone at its open.
    _old(later, "later")
    host = _started(later)
    try:
        own = _json(host, "/api/upgrade")
        assert own["first_start"]["subject"].startswith("0.2.8/")
        assert own["first_start"]["workspaces"] == [str(later.resolve())]
        acknowledge = own["next_requests"]["acknowledge"]
        _json(host, "/api/upgrade/acknowledge", method="POST", payload=acknowledge)
        assert host.upgraded(timeout=600)
        assert _request(host, "/api/tasks")[0] == 200
    finally:
        host.stop()
    alone = _progress(own["first_start"]["subject"])
    assert [(e["family"], e["verdict"]) for e in alone if e["kind"] == "family_done"] == [
        (name, "ALL EQUAL") for name in families
    ]
    # The upgraded first workspace opens with no plan: the machine's run covers it.
    host = _started(first)
    try:
        assert host.first_start is None and _request(host, "/api/tasks")[0] == 200
    finally:
        host.stop()


@pytest.mark.proves("product_host.first_start_asks_a_clean_workspace_once")
def test_a_workspace_holding_no_old_data_is_asked_once_per_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A start asks the families once of a workspace holding no old data and its next start
    asks nothing; another folder, a backup restored into a clean one's emptied folder, and any
    workspace after a declaration changes are asked."""
    monkeypatch.setenv("ALPHALATTICE_BACKUP_ROOT", str(tmp_path / "backups"))
    asked: list[str] = []

    def make() -> Family:
        def holds(workspace: Path) -> bool:
            asked.append(workspace.name)
            return False

        return Family("held", lambda _runner: None, lambda _runner: True, holds)

    monkeypatch.setattr(first_start, "FAMILIES", (make,))
    plain, other = tmp_path / "plain", tmp_path / "other"
    initialize_research_workspace(plain)
    initialize_research_workspace(other)

    def opened(workspace: Path) -> None:
        host = _started(workspace)
        try:
            assert host.first_start is None and _request(host, "/api/tasks")[0] == 200
        finally:
            host.stop()

    opened(plain)
    opened(plain)
    assert asked == ["plain"]
    opened(other)
    assert asked == ["plain", "other"]
    # The supported restore writes another workspace's backup into plain's emptied folder.
    backups = WorkspaceBackups(
        other, workspace_id=named_workspace_id(other), root=tmp_path / "vault"
    )
    backups.create(reason="REQUEST")
    shutil.rmtree(plain)
    plain.mkdir()
    backups.restore(plain)
    opened(plain)
    assert asked == ["plain", "other", "plain"]
    opened(plain)
    assert asked == ["plain", "other", "plain"]
    monkeypatch.setattr(first_start, "DECLARATIONS", first_start.DECLARATIONS + " 130-added.json")
    opened(plain)
    assert asked == ["plain", "other", "plain", "plain"]
    assert [e["kind"] for e in _progress(first_start.CLEAN)] == ["none_due"] * 4


@pytest.mark.proves("product_host.first_start_upgrades_every_old_workspace")
@pytest.mark.parametrize("witness", ("manifest-rewrite", "bounded-history"))
def test_first_start_reuses_only_the_recorded_family_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, witness: str
) -> None:
    from alphalattice.capabilities.data.errors import DataError
    from alphalattice.foundation.data_layer import create_data_root
    from alphalattice.foundation.data_layer.log import HostLog

    root, workspace, manifestless = (tmp_path / name for name in ("data", "workspace", "unnamed"))
    create_data_root(root)
    _old(workspace, "first")
    manifestless.mkdir()
    monkeypatch.setenv("ALPHALATTICE_BACKUP_ROOT", str(tmp_path / "backups"))
    asked: list[str] = []
    stops: list[str] = []

    def other() -> Family:
        def step(runner: Any) -> None:
            for folder in runner.workspaces:
                output = folder / "new-store.json"
                stamp = output.stat()
                os.utime(output, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1_000_000_000))

        def holds(_workspace: Path) -> bool:
            asked.append("other")
            return False

        return Family("other", step, lambda _runner: True, holds)

    monkeypatch.setattr(first_start, "FAMILIES", (_held_family(stops, asked), other))
    cases = (
        ("fresh", workspace, None, ("held",), ("held", "other")),
        ("steady", workspace, None, (), ()),
        ("shared view added", workspace, None, (), ()),
        ("changed", workspace, "other", ("held",), ("held",)),
        ("manifestless", manifestless, None, ("held", "other"), ()),
        ("manifestless reopened", manifestless, None, (), ()),
        ("steady after history", workspace, None, (), ()),
        ("interrupted", workspace, "interrupted source", ("held",), ("held",)),
        ("resumed", workspace, None, (), ("held",)),
    )
    with HostLog.open(root) as log:
        read = log.read
        progress_reads: list[tuple[int, int]] = []
        steady_rows = 0

        def observe(sql: str, args: tuple[Any, ...] = ()) -> Any:
            if "FROM event" not in sql or not args or args[0] != PROGRESS:
                return read(sql, args)
            steps = 0

            def tick() -> int:
                nonlocal steps
                steps += 1
                return 0

            with log.reading() as db:
                db.set_progress_handler(tick, 1)
                try:
                    rows = read(sql, args)
                finally:
                    db.set_progress_handler(None, 0)
            progress_reads.append((len(rows), steps))
            return rows

        monkeypatch.setattr(log, "read", observe)
        for case, folder, changed, probes, families in cases:
            source = folder / "old-store.json"
            if case == "shared view added":
                os.link(source, tmp_path / "kept-input-view.json")
            if changed is not None:
                prior_time = source.stat()
                source.write_text(json.dumps({"rows": [changed]}), encoding="utf-8")
                if case == "changed":
                    # A byte change with size and write time preserved still ends its trust.
                    os.utime(source, ns=(prior_time.st_atime_ns, prior_time.st_mtime_ns))
                    if witness == "manifest-rewrite":
                        manifest = read_research_workspace_manifest(folder)
                        publish_research_workspace_manifest(
                            folder, manifest.with_bindings(committee_agenda_limit=1)
                        )
                        assert named_workspace_id(folder) == manifest.workspace_id
            if case == "steady after history" and witness == "bounded-history":
                progress_reads.clear()
                assert first_start.FirstStart(root, log, folder).plan is None
                [before_history] = progress_reads
                prefix = first_start.workspace_plan_prefix(folder)
                for attempt in range(64):
                    runner = UpgradeRunner(
                        root,
                        log,
                        (folder,),
                        (Family("other", lambda _: None, lambda _: True),),
                        lambda *_, scope=folder: {"workspaces": [{"workspace": str(scope)}]},
                        subject=f"{prefix}fixture-{attempt}",
                    )
                    assert runner.run()
            asked.clear()
            progress_reads.clear()
            start = first_start.FirstStart(root, log, folder)
            assert len(progress_reads) == 1, case
            assert tuple(asked) == probes, case
            if case == "steady":
                steady_rows = progress_reads[0][0]
            if case == "steady after history" and witness == "bounded-history":
                assert progress_reads[0][0] <= steady_rows, case
                assert progress_reads[0] == before_history, "completed plans under this workspace"
            if not families:
                assert start.plan is None, case
                continue
            assert start.plan is not None and start.plan.families == families, case
            if case == "resumed":
                assert start.plan.recorded and start.state == "RUNNING"
            else:
                start.admit(start.plan.plan_hash)
            if case == "interrupted":
                stops.append("stopped before verification")
                with pytest.raises(DataError, match="stopped at step"):
                    start.run()
                assert not start.view()["first_start"]["families_done"]
            else:
                start.run()
                receipt = start.view()["first_start"]
                assert tuple(receipt["families_done"]) == families
                assert receipt["upgrade_seconds"] >= 0
                if case == "fresh":
                    current = first_start.FirstStart(root, log, workspace)
                    facts = current._trusted[str(workspace.resolve()), "held"]
                    committed = facts["files"]["/new-store.json"]["metadata"]
                    assert committed == list(
                        first_start.FileDigestTrust.stamp(workspace / "new-store.json")
                    )


@pytest.mark.proves("product_host.first_start_asks_nothing_of_catalog_held_outcomes")
@pytest.mark.parametrize(
    "case",
    [
        "the catalog holds it",
        "the catalog is created after discovery",
        "nowhere",
        "its profile without rows",
        "more rows than its manifest",
        "a folder in its place",
        "a catalog fault",
        "no catalog reader",
    ],
)
def test_a_committed_chunk_with_no_file_is_upgraded_only_if_the_catalog_lacks_it(
    tmp_path: Path, case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fileless chunk whose rows the catalog holds asks nothing of the first start; one it
    lacks is planned and refused, and any other fault stops the start by its own error."""
    from datetime import UTC, date, datetime

    import pyarrow as pa

    from alphalattice.capabilities.data.errors import DataError
    from alphalattice.capabilities.domain_contracts.identity import canonical_hash
    from alphalattice.foundation.causal_outcomes.execution.contracts import (
        CausalExecutionOutcomeChunk,
        CausalExecutionOutcomeManifest,
        CausalExecutionOutcomeMarker,
    )

    outcomes, address = tmp_path / OUTCOMES, "playpen://data-operations/execution-outcomes/"

    def publish(category: str, payload: dict[str, Any], name: str) -> str:
        """Write `payload` as the outcome store's files hold it; returns its address."""
        (outcomes / category).mkdir(parents=True, exist_ok=True)
        (outcomes / category / f"{blake3_filename(payload[name])}.json").write_text(
            json.dumps(payload)
        )
        return f"{address}{category}/{payload[name]}"

    day = date(2025, 1, 2)
    chunks = {}
    metadata_hash = fixture_digest("execution outcome chunk metadata")
    for split in ("DEVELOPMENT", "SEALED_HOLDOUT"):
        content_hash = fixture_digest(("execution outcome chunk", split))
        location = split.casefold().replace("_", "-")
        chunks[split] = CausalExecutionOutcomeChunk(
            split=split,
            year=2025,
            row_count=1,
            first_formation_session=day,
            last_formation_session=day,
            content_hash=content_hash,
            metadata_hash=metadata_hash,
            uri=f"{address}{location}/chunks/{content_hash}",
        )
    values = {
        "research_cadence": "DAILY",
        "market_as_of": date(2025, 1, 5),
        "listing_ids": ("listing",),
        "listing_set_hash": canonical_hash(("listing",)),
        "schedule_hash": fixture_digest("execution outcome schedule"),
        "ordered_session_triples_hash": fixture_digest("ordered execution sessions"),
        "source_rows_semantic_hash": fixture_digest("execution outcome source rows"),
        "data_validity_class": "CURRENT_UNIVERSE_RESEARCH_ONLY",
        "development_chunks": (chunks["DEVELOPMENT"],),
        "sealed_holdout_chunks": (chunks["SEALED_HOLDOUT"],),
        "development_formation_count": 1,
        "sealed_holdout_formation_count": 1,
        "limitations": ("test",),
    }
    draft = CausalExecutionOutcomeManifest.model_construct(**values, snapshot_hash="")
    manifest = CausalExecutionOutcomeManifest(
        **values,
        snapshot_hash=canonical_hash(draft.model_dump(mode="json", exclude={"snapshot_hash"})),
    )
    ref = publish("manifests", manifest.model_dump(mode="json"), "snapshot_hash")
    fields = {
        "kind": "CausalExecutionOutcomeMarker",
        "snapshot_hash": manifest.snapshot_hash,
        "manifest_ref": ref,
        "schedule_hash": manifest.schedule_hash,
        "listing_set_hash": manifest.listing_set_hash,
    }
    marker = CausalExecutionOutcomeMarker(**fields, marker_hash=canonical_hash(fields))
    publish("markers", marker.model_dump(mode="json"), "marker_hash")

    if case == "no catalog reader":
        monkeypatch.setattr(history_startup, "duckdb_session", lambda *_args, **_kwargs: None)
    history.open_history("first-start", "first start", tmp_path, tmp_path / "data-root")
    try:
        family = first_start.execution_outcomes()
        if case == "no catalog reader":
            with pytest.raises(DataError) as refusal:
                family.holds(tmp_path)
            assert refusal.value.failure.code == "data_layer.history_unavailable"
            return
        if case == "the catalog is created after discovery":
            with pytest.raises(DataError, match="outcome body is not in the catalog"):
                history.read_execution_outcome(chunks["DEVELOPMENT"].content_hash)
        port = history.store()
        assert port is not None
        lake = port.lake
        empty = lake.sql("FROM lake.results.execution_outcome LIMIT 0").to_arrow_table()
        schema, stamp = empty.drop(["artifact_id"]).schema, datetime(2025, 1, 2, tzinfo=UTC)
        kinds = (
            (pa.types.is_string, "x"),
            (pa.types.is_date, day),
            (pa.types.is_timestamp, stamp),
            (pa.types.is_integer, 0),
        )
        row = {f.name: next((v for kind, v in kinds if kind(f.type)), None) for f in schema}
        rows = {"nowhere": None, "its profile without rows": 0}.get(case, 1)
        rows = 2 if case == "more rows than its manifest" else rows
        for chunk in chunks.values() if rows is not None else ():
            listed = [row | {"listing_id": f"listing-{n}"} for n in range(rows)]
            history.write_execution_outcome(
                chunk.content_hash, pa.Table.from_pylist(listed, schema)
            )
        if case == "a folder in its place":
            (
                outcomes
                / "development"
                / "chunks"
                / f"{blake3_filename(chunks['DEVELOPMENT'].content_hash)}.parquet"
            ).mkdir(parents=True)
        if case == "a catalog fault":

            def fault(_digest: str) -> str:
                raise DataError("named twice", code="data_layer.outcome_incomplete")

            monkeypatch.setattr(history_execution_outcomes, "execution_outcome_artifact_id", fault)
        unheld = case in ("nowhere", "its profile without rows")
        if case == "the catalog is created after discovery":
            import duckdb

            try:
                assert family.verify(SimpleNamespace(workspaces=(tmp_path,))) is True
            except duckdb.CatalogException as error:
                pytest.fail(f"first verification could not read the committed catalog: {error}")
        if unheld or case == "the catalog holds it":
            assert family.holds(tmp_path) is unheld
        if unheld:
            with pytest.raises(DataError, match="not in the catalog"):
                family.verify(SimpleNamespace(workspaces=(tmp_path,)))
        stops = {"more rows than its manifest": ValueError, "a folder in its place": OSError}
        if case in (*stops, "a catalog fault"):
            with pytest.raises(stops.get(case, DataError)):
                family.holds(tmp_path)
    finally:
        history.close_history()
