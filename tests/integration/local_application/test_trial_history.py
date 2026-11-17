"""A sealed research run inside a Host is a trial in the history, and the history replays."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from alphalattice.control.research_program.authoring.document import load_authoring_document
from alphalattice.control.task_control.registry import (
    DuckDbTaskControlRegistry,
    resolve_task_control_database,
)
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.kernel.data_layer import history
from alphalattice.kernel.data_layer.log import replay
from alphalattice.kernel.data_layer.project import differences
from alphalattice.kernel.data_layer.reading import experience, reading_session
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.actor_execution.contracts import ActorKind
from tests.integration.research_program.test_program_execution import _FIXTURE, _workflow
from tests.researcher_methodology_surface.real_workspace import RealRiskWorkspace
from tests.workspace_task_runner.task_control_support import task_contract

TEST_SOURCE = "src/alphalattice/interface/local_application/trial_history.py"


@pytest.mark.proves("local_application.research_runs_are_recorded_as_trials")
def test_a_run_and_its_rerun_are_two_trials_of_one_selection_and_recipe(
    real_risk_workspace: RealRiskWorkspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first run's evidence is its trial's artifact; a rerun reusing it is a second trial
    with no artifact of its own; the history replays ALL EQUAL."""
    root = tmp_path / "data"
    document = load_authoring_document(_FIXTURE.read_text(encoding="utf-8"))
    workflow: Any = _workflow(real_risk_workspace, tmp_path)
    who = {"actor_kind": ActorKind.HUMAN, "actor_id": "researcher"}
    workflow.preflight(document, **who)
    sealed = workflow.prepare(document, **who)
    monkeypatch.setenv(history.DATA_ROOT_ENV, str(root))  # the Host's data root
    history.open_history("w-risk", "risk", tmp_path)
    try:
        registry = DuckDbTaskControlRegistry(
            resolve_task_control_database(tmp_path), gate=WorkspaceMutationGate()
        )
        envelope, goal, plan = task_contract()
        task = registry.admit(
            input_envelope=envelope,
            goal=goal,
            plan=plan,
            observed_at=datetime(2026, 11, 16, tzinfo=UTC),
        ).record
        attributed = replace(sealed, hypothesis_id="h1", task_id=str(task.task_id))
        assert canonical_hash(sealed.document) == canonical_hash(attributed.document)
        assert sealed.program == attributed.program
        history.record("goal.set", ("goal", "g1"), {"title": "A goal"})
        history.record("hypothesis.stated", ("hypothesis", "h1"), {"goal_id": "g1"}, text="A claim")
        first = workflow.execute_prepared(attributed)
        again, _ = workflow.run_sealed(document, **who)
    finally:
        history.close_history()
    assert (first.disposition, again.numerical_call_count) == ("COMPUTED", 0)
    with closing(sqlite3.connect(root / "machine" / "log.sqlite")) as con:
        trials = con.execute(
            "SELECT state, result_artifact, seeds, goal_id, hypothesis_id, recipe_key, task_id "
            "FROM trial ORDER BY registered_seq"
        ).fetchall()
        assert [row[:5] for row in trials] == [
            ("completed", first.evidence_hash, "[0]", "g1", "h1"),
            ("completed", None, "[0]", None, None),
        ]
        assert trials[0][5] == trials[1][5]
        assert [row[6] for row in trials] == [str(task.task_id), None]
        (trace_id,) = con.execute(
            "SELECT e.trace_id FROM trial t JOIN event e ON e.seq=t.registered_seq "
            "WHERE t.hypothesis_id='h1'"
        ).fetchone()
        assert trace_id
        assert con.execute("SELECT task_id FROM task").fetchall() == [(str(task.task_id),)]
        assert con.execute("SELECT count(*) FROM selection").fetchone()[0] == 1
        recipe = con.execute(
            "SELECT l.name FROM recipe r JOIN definition_lineage l ON l.definition_id = r.method_id"
        ).fetchall()
        assert recipe == [(first.kind,)]
        artifact = con.execute(
            "SELECT l.name, a.storage, a.trial_id IS NOT NULL FROM artifact a"
            " JOIN definition_lineage l ON l.definition_id = a.type_id"
        )
        assert artifact.fetchall() == [("research_evidence", "none", 1)]
        with closing(replay(con, tmp_path / "replayed.sqlite")) as target:
            assert differences(con, target) == []
    with closing(reading_session(root)) as view:
        assert view is not None
        (row,) = experience(view, hypothesis="h1")
        assert row["trial_id"] and row["trace_id"] == trace_id
        assert row["task_id"] == str(task.task_id)
