"""The Host's research history: the data root opened at start, or deferred by name."""

from __future__ import annotations

import socket
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest

from alphalattice.control.product_host.composition.experience_read import read_experience
from alphalattice.interface.local_application.answers import ExperienceAnswer
from alphalattice.kernel.data_layer import history
from alphalattice.kernel.data_layer.log import HostLog

TEST_SOURCE = "src/alphalattice/kernel/data_layer/history.py"


@pytest.mark.proves("data_layer.host_history_records_or_defers")
def test_a_host_records_its_workspace_and_a_second_host_defers_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first Host records its workspace and the evidence type, and a row its workers record at
    once is recorded once; while another Host holds the data root, or its log or views do not
    open, a Host's history is deferred by name, records nothing and holds no lock."""
    root = tmp_path / "o'brien" / "data"  # a quote in the path is text in DuckDB's ATTACH
    monkeypatch.delenv("ALPHALATTICE_OTLP_ENDPOINT", raising=False)
    with monkeypatch.context() as no_network:

        def no_socket(*args, **kwargs):
            raise AssertionError("an unconfigured Host opened a socket")

        no_network.setattr(socket, "socket", no_socket)
        assert history.open_history("w1", "one", root) is None
        try:
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
            answer = read_experience(request)
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
            with history.step("host.request", {}, operation="read"):
                pass
        finally:
            history.close_history()
    with closing(sqlite3.connect(root / "machine" / "log.sqlite")) as con:
        assert con.execute(
            "SELECT count(*) FROM event WHERE type = 'experience.read'"
        ).fetchone() == (5,)
        assert con.execute(
            "SELECT workspace_id, name FROM workspace ORDER BY workspace_id"
        ).fetchall() == [("w1", "one"), ("w2", "two")]
        owned = (
            "SELECT name, owner FROM definition WHERE family = 'artifact_type' AND owner = 'host'"
        )
        assert con.execute(owned).fetchall() == [(history.EVIDENCE_TYPE, "host")]
    with HostLog.open(root):
        assert history.open_history("w1", "one", root) == history.HISTORY_DEFERRED
        assert history.record("goal.set", ("goal", "g2"), {"title": "t"}) is None
    damaged = tmp_path / "damaged"  # a log that is not a database: the Host serves on without it
    (damaged / "machine").mkdir(parents=True)
    (damaged / "machine" / "log.sqlite").write_bytes(b"not a database" * 100)
    assert history.open_history("w1", "one", damaged) == history.HISTORY_UNAVAILABLE
    assert not history.recording()
    assert history.read_experience(limit=1) is None
    monkeypatch.setattr(history, "reading_session", lambda where: 1 / 0)  # views that fail
    assert history.open_history("w1", "one", root) == history.HISTORY_UNAVAILABLE
    assert not history.recording()
    HostLog.open(root).close()  # the deferral released the data root's lock
