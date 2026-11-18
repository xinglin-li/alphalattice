"""The ordinary Host's first 0.2.8 start upgrades every old workspace on one acknowledged plan."""

from __future__ import annotations

import json
import shutil
import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from alphalattice.control.product_host.composition import first_start
from alphalattice.control.product_host.composition.local_web_session import (
    LocalPortfolioWebSession,
)
from alphalattice.control.product_host.composition.research_workspace import (
    initialize_research_workspace,
)
from alphalattice.interface.local_application import cli
from alphalattice.interface.local_application.client import LocalResearchConnection
from alphalattice.kernel.data_layer import history
from alphalattice.kernel.data_layer.upgrade import PROGRESS, Family
from tests.portfolio_strategy_lab.local_web_support import _json, _request

TEST_SOURCE = "src/alphalattice/control/product_host/composition/first_start.py"
OUTCOMES = Path("artifacts/data-operations/execution-outcomes")


def _held_family(stops: list[str]) -> Callable[[], Family]:
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
            old, new = workspace / "old-store.json", workspace / "new-store.json"
            return old.is_file() and (not new.is_file() or new.read_bytes() != old.read_bytes())

        return Family("held", step, verify, lambda _mode: None, holds)

    return make


def _old(source: Path | None, workspace: Path, name: str) -> None:
    """A 0.2.7 workspace: a QA copy's manifest and outcome store, or a held store of its own."""
    if source is None:
        initialize_research_workspace(workspace)
        (workspace / "old-store.json").write_text(json.dumps({"rows": [name]}), encoding="utf-8")
        return
    workspace.mkdir(parents=True)
    shutil.copy2(source / "research-workspace.json", workspace)
    shutil.copytree(source / OUTCOMES, workspace / OUTCOMES)


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
@pytest.mark.parametrize("corpus", ["held", pytest.param("qa", marks=pytest.mark.real_evidence)])
def test_the_first_start_upgrades_every_old_workspace_on_one_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    capsys: pytest.CaptureFixture[str],
    corpus: str,
) -> None:
    """requirement: the first 0.2.8 start upgrades every old workspace on one acknowledged plan."""
    monkeypatch.setenv("ALPHALATTICE_BACKUP_ROOT", str(tmp_path / "backups"))
    stops = ["stopped mid-family"]
    if corpus == "held":
        sources: list[Path | None] = [None, None, None]
        monkeypatch.setattr(first_start, "FAMILIES", (_held_family(stops),))
    else:
        qa = request.getfixturevalue("evidence_roots").require("alpha_development_outcomes")
        sources = [qa / "alpha-base", qa / "alpha-after-1", qa / "alpha-before-1"]
        write = history.write_execution_outcome

        def stopping(*args: Any, **kwargs: Any) -> str:
            written = write(*args, **kwargs)
            if stops:
                raise RuntimeError(stops.pop())
            return written

        monkeypatch.setattr(history, "write_execution_outcome", stopping)
    made = first_start.FAMILIES
    families = [make().name for make in made]
    project = tmp_path / "project" / "workspaces"
    first, second, later = project / "first", project / "second", project / "later"
    _old(sources[0], first, "first")
    _old(sources[1], second, "second")

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
        same = ("state", "workspaces", "families", "subject")
        assert code == 0 and {k: shown["first_start"][k] for k in same} == {
            k: plan[k] for k in same
        }
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
        assert stopped["state"] == "FAILED" and stopped["family_failed"] == families[0]
        assert _request(host, "/api/tasks")[0] == 503
    finally:
        host.stop()
    assert not LocalResearchConnection.path(first).exists()  # withdrawn with the restricted Host
    started = [e for e in _progress("0.2.8") if e["kind"] == "started"]
    assert len(started) == 1 and len(started[0]["backups"]["workspaces"]) == 2

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

    # A third old workspace that appears afterwards is upgraded alone at its open. Every QA
    # workspace holds the same committed chunks, so the third QA copy has nothing the catalog
    # lacks: it opens with no plan, and its old reads already equal the catalog's.
    _old(sources[2], later, "later")
    host = _started(later)
    try:
        if corpus == "qa":
            assert host.first_start is None and _request(host, "/api/tasks")[0] == 200
            assert all(make().verify(SimpleNamespace(workspaces=(later,))) for make in made)
            return
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

    from alphalattice.foundation.causal_outcomes.execution.contracts import (
        CausalExecutionOutcomeChunk,
        CausalExecutionOutcomeManifest,
        CausalExecutionOutcomeMarker,
    )
    from alphalattice.kernel.data.errors import DataError
    from alphalattice.kernel.shared_kernel.identity import canonical_hash

    outcomes, address = tmp_path / OUTCOMES, "playpen://data-operations/execution-outcomes/"

    def publish(category: str, payload: dict[str, Any], name: str) -> str:
        """Write `payload` as the outcome store's files hold it; returns its address."""
        (outcomes / category).mkdir(parents=True, exist_ok=True)
        (outcomes / category / f"{payload[name]}.json").write_text(json.dumps(payload))
        return f"{address}{category}/{payload[name]}"

    day = date(2025, 1, 2)
    chunks = {
        split: CausalExecutionOutcomeChunk(
            split=split,
            year=2025,
            row_count=1,
            first_formation_session=day,
            last_formation_session=day,
            content_hash=digit * 64,
            metadata_hash="f" * 64,
            uri=f"{address}{split.casefold().replace('_', '-')}/chunks/{digit * 64}",
        )
        for split, digit in (("DEVELOPMENT", "a"), ("SEALED_HOLDOUT", "b"))
    }
    values = {
        "research_cadence": "DAILY",
        "market_as_of": date(2025, 1, 5),
        "listing_ids": ("listing",),
        "listing_set_hash": canonical_hash(("listing",)),
        "schedule_hash": "1" * 64,
        "ordered_session_triples_hash": "2" * 64,
        "source_rows_semantic_hash": "3" * 64,
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

    history.open_history("first-start", "first start", tmp_path, tmp_path / "data-root")
    try:
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
            (outcomes / "development" / "chunks" / f"{'a' * 64}.parquet").mkdir(parents=True)
        if case == "a catalog fault":

            def fault(_digest: str) -> str:
                raise DataError("named twice", code="data_layer.outcome_incomplete")

            monkeypatch.setattr(history, "execution_outcome_artifact_id", fault)
        if case == "no catalog reader":  # DuckDB, the data extra, is absent
            monkeypatch.setattr(history, "reading_session", lambda *_args, **_kwargs: None)
        family = first_start.execution_outcomes()
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
        if case in (*stops, "a catalog fault", "no catalog reader"):
            with pytest.raises(stops.get(case, DataError)):
                family.holds(tmp_path)
    finally:
        history.close_history()
