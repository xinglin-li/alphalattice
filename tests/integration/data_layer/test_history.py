"""The log a process holds, and the Host's history in it: opened at start, or refused by name."""

from __future__ import annotations

import json
import socket
import sqlite3
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from alphalattice.control.product_host.composition.experience_read import (
    read_experience,
    read_experience_brief,
    read_experience_class,
    recall_experience,
    record_experience,
)
from alphalattice.control.task_control.registry import (
    DuckDbTaskControlRegistry,
    TaskAdmission,
    resolve_task_control_database,
)
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.interface.local_application.answers import (
    ExperienceAnswer,
    ExperienceRecallAnswer,
    ExperienceRecordAnswer,
)
from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer import (
    encoders,
    experiences,
    history,
    history_runtime,
    history_startup,
    recall,
)
from alphalattice.kernel.data_layer.log import HostLog
from tests.workspace_task_runner.task_control_support import task_contract

TEST_SOURCE = "src/alphalattice/kernel/data_layer/history.py"


def _read_while_registry_writes(
    folder: Path, monkeypatch: pytest.MonkeyPatch, read_view: Callable[[], dict[str, Any]]
) -> dict[str, Any]:
    """Drive the existing history read across a real guarded Task registry write."""
    registry = DuckDbTaskControlRegistry(
        resolve_task_control_database(folder), workspace_id="w1", gate=WorkspaceMutationGate()
    )
    envelope, goal, plan = task_contract()
    log = history.held()
    original_reading, original_flush = log.reading, log.flush
    writer_has_guard, reader_at_guard, writer_finished = (threading.Event() for _ in range(3))
    reader: threading.Thread | None = None

    def before_reader_guard() -> None:
        if threading.current_thread() is reader:
            reader_at_guard.set()
            assert writer_finished.wait(3), "history and registry waited in opposite orders"

    @contextmanager
    def reading() -> Iterator[sqlite3.Connection]:
        before_reader_guard()
        with original_reading() as connection:
            yield connection

    def flush() -> None:
        before_reader_guard()
        original_flush()

    def write() -> TaskAdmission:
        try:
            with registry.read_scope():
                writer_has_guard.set()
                assert reader_at_guard.wait(5), "history never attempted the log guard"
                return registry.admit(
                    input_envelope=envelope, goal=goal, plan=plan, observed_at=datetime.now(UTC)
                )

        finally:
            writer_finished.set()

    def read() -> dict[str, Any]:
        nonlocal reader
        reader = threading.current_thread()
        assert writer_has_guard.wait(5), "the registry never acquired the log guard"
        return read_view()

    with monkeypatch.context() as patch, ThreadPoolExecutor(max_workers=2) as pool:
        patch.setattr(log, "reading", reading)
        patch.setattr(log, "flush", flush)
        writing, result = pool.submit(write), pool.submit(read)
        answer = result.result(timeout=10)
        admitted = writing.result(timeout=10)
    assert registry.task(admitted.record.task_id) == admitted.record
    return answer


def _assert_implicit_open_follows_copy_exit(
    copy_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closing_copy, sampled_copy = threading.Event(), threading.Event()
    read_root = history.data_root

    def sample_root() -> Path:
        root = read_root()
        if closing_copy.is_set() and root == copy_root:
            sampled_copy.set()
        return root

    def open_implicit() -> HostLog:
        assert closing_copy.wait(5), "the copy never reached its close"
        return history.held()

    try:
        with monkeypatch.context() as patch, ThreadPoolExecutor(max_workers=1) as worker:
            patch.setattr(history_runtime, "data_root", sample_root)
            with history.own_root(copy_root):
                copy = history.held()
                close_copy = copy.close

                def close() -> None:
                    closing_copy.set()
                    sampled_copy.wait(5)
                    close_copy()

                patch.setattr(copy, "close", close)
                opened = worker.submit(open_implicit)
            assert sampled_copy.is_set(), "the opener never read the root during copy close"
            assert opened.result(timeout=5) is history.holding(), "the opener kept the retired copy"
    finally:
        history.close_history()


@pytest.mark.proves("data_layer.host_history_records_or_refuses")
@pytest.mark.proves("data_layer.a_class_read_is_a_search_memory_span")
def test_a_host_records_its_workspace_and_a_second_host_is_refused_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """requirement: a Host records its workspace once by its folder, its steps and facts as given,
    and closes its history with its process's last Host; a Host that cannot hold or open its log is
    refused by name, recording nothing and holding no lock."""
    history.close_history()  # the process's own root, if a test before this one took it
    root = tmp_path / "o'brien" / "data"  # a quote in the path is text in DuckDB's ATTACH
    monkeypatch.setenv(history.DATA_ROOT_ENV, str(root))  # the data root this process names
    folder = tmp_path / "workspace"
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    with monkeypatch.context() as no_network:

        def no_socket(*args, **kwargs):
            raise AssertionError("an unconfigured Host opened a socket")

        no_network.setattr(socket, "socket", no_socket)
        history.open_history("w1", "one", folder, root)
        try:
            assert history.workspace_on_log(folder, "one") == "w1"
            # Goal conclusions need no trial; filters use the note's own fields.
            history.record("conclusion.drawn", ("goal", "no-trial"), text="Goal conclusion")
            history.record(
                "lesson.recorded",
                ("goal", "no-trial"),
                {"hypothesis_id": "h1", "method": "m1"},
                text="Attributed lesson",
            )
            request = SimpleNamespace(
                goal_id="no-trial",
                experience_hypothesis=None,
                experience_method=None,
                experience_workspace="w1",
                history_limit=25,
            )
            answer = _read_while_registry_writes(
                folder, monkeypatch, lambda: read_experience(request)
            )
            assert {n["text"] for n in answer.get("notes", [])} == {
                "Goal conclusion",
                "Attributed lesson",
            }
            answer = ExperienceAnswer.model_validate(answer)
            assert answer.trials == [] and len({n["note_id"] for n in answer.notes}) == 2
            request.experience_hypothesis, request.experience_method = "h1", "m1"
            assert [n["text"] for n in read_experience(request)["notes"]] == ["Attributed lesson"]
            request.experience_method = "other"
            assert read_experience(request)["notes"] == []
            request.experience_hypothesis = request.experience_method = None
            request.experience_workspace = "w2"
            assert read_experience(request)["notes"] == []
            request.experience_workspace, request.history_limit = "w1", 1
            assert len(read_experience(request)["notes"]) == 1
            exists = "SELECT 1 FROM goal WHERE goal_id = 'g'"
            goal = ("goal.set", ("goal", "g"), {"title": "t"})
            with ThreadPoolExecutor(8) as pool:  # one Host's workers record the same goal at once
                seqs = list(pool.map(lambda _: history.record_new(exists, (), *goal), range(8)))
            assert sum(seq is not None for seq in seqs) == 1
            # A note on the goal records with its verdict and run; one missing a part, or on a
            # trial the log does not hold, is refused by name.
            note = {"hypothesis": "H", "method": "M", "conclusion": "C", "lessons": ["L"]}
            review = SimpleNamespace(review_subject="goal:g", review_verdict="success")
            recorded = record_experience(SimpleNamespace(**vars(review), review_note=note), "A")
            recorded = ExperienceRecordAnswer.model_validate(recorded)
            assert (recorded.status, recorded.verdict, recorded.place) == (
                "RECORDED",
                "success",
                [],
            )
            blank = SimpleNamespace(**vars(review), review_note=note | {"method": " "})
            assert record_experience(blank, "A")["failure_code"] == "experience.note_incomplete"
            review.review_subject = "trial:none"
            unheld = record_experience(SimpleNamespace(**vars(review), review_note=note), "A")
            assert unheld["failure_code"] == "experience.unit_unknown"
            # A class read of a trial the log does not hold is refused by its own code, through
            # the reader's process, never as the history being unavailable.
            ask = SimpleNamespace(experience_trial="none", experience_width=None)
            unknown = read_experience_class(SimpleNamespace(**vars(ask), experience_known_at=None))
            assert unknown["failure_code"] == "experience.trial_unknown"
            ask = SimpleNamespace(goal_id="none", experience_budget=None, experience_known_at=None)
            assert read_experience_brief(ask)["failure_code"] == "experience.goal_unknown"
            # A recall finds the goal's review note, rendered, as the history knows it; with the
            # retrieval models not installed it is refused by name.
            import re
            import zlib

            import numpy as np

            class Encoder:  # a stand-in for the static model: shared words or Han characters
                def __init__(self, _pack: Path) -> None:
                    pass

                def embed_passages(self, texts):
                    rows = np.full((len(texts), 512), 1e-3, dtype=np.float32)
                    for row, text in enumerate(texts):
                        for word in re.findall(r"[一-鿿]|\w+", text.lower()):
                            rows[row, zlib.crc32(word.encode()) % 512] += 1
                    return rows / np.linalg.norm(rows, axis=1, keepdims=True)

                def embed_query(self, text):
                    return self.embed_passages((text,))[0]

            class Reranker:
                def __init__(self, _pack: Path) -> None:
                    pass

                def score(self, query, passages):
                    return tuple(float(len(passage)) for passage in passages)

            history.render_experiences()  # the note's experience, unless its renderer wrote it
            ask = SimpleNamespace(
                experience_query="C",
                experience_workspace=None,
                experience_known_at=None,
                experience_limit=None,
            )
            from alphalattice.kernel.knowledge import model_store

            with monkeypatch.context() as models:
                models.setattr(encoders, "StaticEncoder", Encoder)
                models.setattr(encoders, "Reranker", Reranker)
                models.setattr(model_store, "verify_directory", lambda _dir, _pack: None)
                found = ExperienceRecallAnswer.model_validate(recall_experience(ask))
                assert found.score_type == "LINEAR_FUSION" and len(found.hits) == 1
                assert found.hits[0]["experience_id"] and found.hits[0]["classes"] == {}
            recall.close()

            with monkeypatch.context() as models:  # a pack not installed whole
                models.setattr(model_store, "verify_directory", lambda _dir, _pack: "missing")
                refused = recall_experience(ask)
            assert refused["failure_code"] == "experience.recall_models_missing"
            recall.close()
            # A new selection is made whether or not world can be read: this Host's reader holds
            # no world catalog, so no market condition is recorded and nothing is refused.
            window = {"cutoff_us": 1, "universe": "sp500"}
            made = history.selection(window, actor=("agent", "claude"))
            # A trial's kept return series names a trial the log holds, else is refused by name.
            with pytest.raises(DataError, match="no trial") as refused:
                history.record_return_series("none", ["2015-01"], [0.01])
            assert refused.value.failure.code == "data_layer.trial_unknown"
            # A render holds the log guard. A registry writer holding that guard can still look
            # up the Host state under the state lock; the render does not hold it.
            inside, go, held = threading.Event(), threading.Event(), {}

            def render_pending(*_: object) -> list[str]:
                inside.set()
                go.wait(5)
                return []

            with monkeypatch.context() as patch:
                patch.setattr(experiences, "render_pending", render_pending)
                render = threading.Thread(target=history.render_experiences, daemon=True)
                render.start()
                assert inside.wait(5)
                held["lock"] = history._lock.acquire(timeout=3)
                if held["lock"]:
                    history._lock.release()
                go.set()
                render.join(5)
            assert held["lock"] and not render.is_alive()
            attrs = {"operation": "STATUS", "caller": "HUMAN", "subject": {"goal_id": "g1"}}
            assert history.begin("product.operation", attrs | {"request_fields": ["task_id"]})
            failed = history.begin("product.operation", attrs)
            history.finish(
                failed, "product.operation", attrs | {"failure_code": "x.refused"}, failed=True
            )
            at = datetime(2026, 11, 15, 12, tzinfo=UTC)
            with history.step("host.request", {}, operation="STATUS"):
                history.fact("agent.activity", {"event_kind": "n"}, text="Read it.", at=at)
            history.record("hypothesis.stated", ("hypothesis", "h1"), {"goal_id": "g"}, text="H")
            issued = history.record_hypothesis("g", "claim", "Claim", "researcher")
            assert history.record_hypothesis("g", "claim", "Claim", "researcher") == issued
            with pytest.raises(ValueError, match=r"goal\.duplicate_identifier"):
                history.record_hypothesis("g", "claim", "Different", "researcher")
            history.record(
                "workspace.created", ("workspace", "w2"), {"name": "two"}, workspace="w2"
            )
            history.record("hypothesis.stated", ("hypothesis", "h2"), text="Other", workspace="w2")
            assert history.hypothesis_goal("h1", expected_goal="g") == "g"
            for hypothesis, expected, reason in (
                ("missing", None, "hypothesis_unknown"),
                ("h2", None, "hypothesis_workspace"),
                ("h1", "other", "hypothesis_goal"),
            ):
                with pytest.raises(ValueError, match=reason):
                    history.hypothesis_goal(hypothesis, expected_goal=expected)
            # a second Host, same process
            history.open_history("w3", "three", tmp_path / "second", root)
            history.release_history()
            assert history.recording()  # the first Host still records
            history.release_history()
            assert not history.recording() and history.holding() is None
        finally:
            history.close_history()
    _assert_implicit_open_follows_copy_exit(tmp_path / "closing-copy", monkeypatch)
    # The same process shares the attached catalog but closes each reader's cursor.
    import duckdb

    from alphalattice.kernel.data_layer.root import catalog

    assert catalog(root) is not None
    monkeypatch.setattr(history_runtime, "data_root", lambda: root)
    surfaces = []
    try:
        for _ in range(2):
            with history.catalog_session() as cursor:
                assert cursor.execute("SELECT 1").fetchone() == (1,)
                surfaces.append(history._catalog_surfaces[root])
            with pytest.raises(duckdb.ConnectionException, match="closed"):
                cursor.execute("SELECT 1")
        assert surfaces[0] is surfaces[1]
    finally:
        history._catalog_surfaces.pop(root).close()
    # The Host keeps the stage's prepared reader and the writer's transaction separate.
    query = (
        "SELECT value FROM lake.results.series WHERE artifact_id IN (SELECT unnest(?::VARCHAR[]))"
    )
    artifact = "test's artifact"
    with history.own_root(tmp_path / "results-root") as results_root:
        history.open_history("catalog-test", "Catalog test", tmp_path / "results", results_root)
        with history.catalog_projection("series", query, ([artifact],)) as result:
            assert result is None  # the write owner has yet to open the catalog
        port = history.store()
        assert port is not None
        port.lake.execute("BEGIN TRANSACTION").complete()
        try:
            port.lake.execute(
                "INSERT INTO lake.results.series VALUES (?, 'net', DATE '2024-11-18', 1.5)",
                (artifact,),
            ).complete()
            with history.catalog_projection("series", query, ([artifact],)) as result:
                assert result is not None and result.fetchall() == []
        except BaseException:
            port.lake.execute("ROLLBACK").complete()
            raise
        port.lake.execute("COMMIT").complete()
        with history.catalog_projection("series", query, (["another artifact"],)) as result:
            assert result is not None and result.fetchall() == []
        with history.catalog_projection("series", query, ([artifact],)) as result:
            assert result is not None and result.fetchall() == [(1.5,)]
    with closing(sqlite3.connect(root / "machine" / "log.sqlite")) as con:
        assert con.execute(
            "SELECT count(*) FROM event WHERE type = 'experience.read'"
        ).fetchone() == (5,)
        # The class read and the brief are one search_memory step each in the log, failed with
        # their reads, holding neither the query nor the records.
        found = con.execute(
            "SELECT phase, json_remove(attrs, '$.definition_ids')"
            " FROM event WHERE type = 'search_memory' ORDER BY seq"
        ).fetchall()
        memory = {"gen_ai.operation.name": "search_memory", "gen_ai.memory.store.id": "experience"}
        condition = con.execute(
            "SELECT l.name, m.value FROM artifact a JOIN measure m USING (artifact_id)"
            " JOIN definition_lineage l ON l.definition_id = m.metric_id"
            " WHERE a.selection_id = ? ORDER BY l.name",
            (made,),
        ).fetchall()
        assert made is not None and condition == []
        assert [(phase, json.loads(attrs)) for phase, attrs in found] == [("failed", memory)] * 2
        rows = con.execute(
            "SELECT workspace_id, name, folder FROM workspace ORDER BY workspace_id"
        ).fetchall()
        second = str((tmp_path / "second").resolve())
        assert rows == [
            ("w1", "one", str(folder.resolve())),
            ("w2", "two", None),  # recorded for the hypothesis in another workspace
            ("w3", "three", second),
        ]
        rows = con.execute(
            "SELECT type, phase, span_id, started_us IS NOT NULL, attrs FROM event"
            " WHERE type IN ('product.operation', 'host.request') ORDER BY seq"
        ).fetchall()
        assert [row[:2] + row[3:4] for row in rows] == [
            ("product.operation", "started", False),  # the open one, logged before any end
            ("product.operation", "started", False),
            ("product.operation", "failed", True),
            ("host.request", "completed", True),
        ]
        assert json.loads(rows[1][4]) == attrs  # nested, as given
        assert json.loads(rows[2][4]) == attrs | {"failure_code": "x.refused"}  # its own, as given
        assert rows[1][2] == rows[2][2]
        fact = con.execute(
            "SELECT e.span_id, e.occurred_us, t.text FROM event e JOIN event_text t USING (seq)"
            " WHERE e.type = 'agent.activity'"
        ).fetchone()
        assert fact == (rows[3][2], int(at.timestamp() * 1_000_000), "Read it.")
        owned = (
            "SELECT name, owner FROM definition WHERE family = 'artifact_type' AND owner = 'host'"
        )
        assert sorted(con.execute(owned).fetchall()) == [
            ("market_condition", "host"),
            ("research_evidence", "host"),
            ("return_series", "host"),
        ]
    with HostLog.open(root), pytest.raises(DataError) as held:
        history.open_history("w1", "one", folder, root)
    assert held.value.failure.code == "data_layer.host_running"
    assert history.DATA_ROOT_ENV in str(held.value)  # the way on: a root of its own
    assert history.record("goal.set", ("goal", "g2"), {"title": "t"}) is None
    damaged = tmp_path / "damaged"  # a log that is not a database
    (damaged / "machine").mkdir(parents=True)
    (damaged / "machine" / "log.sqlite").write_bytes(b"not a database" * 100)
    with pytest.raises(DataError) as unopened:
        history.open_history("w1", "one", folder, damaged)
    assert unopened.value.failure.code == "data_layer.log_unavailable"
    assert not history.recording()
    assert history.read_experience(limit=1) is None
    monkeypatch.setattr(history_startup, "reading_session", lambda where: 1 / 0)
    with pytest.raises(DataError) as views:
        history.open_history("w1", "one", folder, root)
    assert views.value.failure.code == "data_layer.history_unavailable"
    assert not history.recording()
    HostLog.open(root).close()  # the refusal released the data root's lock
