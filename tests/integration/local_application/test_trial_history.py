"""A sealed research run inside a Host is a trial in the history, and the history replays."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from alphalattice.capabilities.domain_contracts.identity import canonical_hash
from alphalattice.control.research_program.authoring.document import load_authoring_document
from alphalattice.control.task_control.registry import (
    DuckDbTaskControlRegistry,
    resolve_task_control_database,
)
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.foundation.data_layer import history
from alphalattice.foundation.data_layer.log import replay
from alphalattice.foundation.data_layer.project import differences
from alphalattice.foundation.data_layer.reading import experience, reading_session
from alphalattice.foundation.market_data_ops.storage import world
from alphalattice.protocols.actor_execution.contracts import ActorKind
from tests.integration.research_program.test_program_execution import _FIXTURE, _workflow
from tests.researcher_methodology_surface.real_workspace import RealRiskWorkspace
from tests.workspace_task_runner.task_control_support import task_contract

TEST_SOURCE = "src/alphalattice/interface/local_application/trial_history.py"


@pytest.mark.proves("local_application.research_runs_are_recorded_as_trials")
def test_a_run_and_its_rerun_are_two_trials_of_one_selection_and_recipe(
    real_risk_workspace: RealRiskWorkspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rerun reusing a run's evidence is a trial without an artifact, and another run producing
    equal evidence is a trial with its own."""
    root = tmp_path / "data"
    document = load_authoring_document(_FIXTURE.read_text(encoding="utf-8"))
    workflow: Any = _workflow(real_risk_workspace, tmp_path)
    who = {"actor_kind": ActorKind.HUMAN, "actor_id": "researcher"}
    workflow.preflight(document, **who)
    monkeypatch.setenv(history.DATA_ROOT_ENV, str(root))  # the Host's data root
    sealed = workflow.prepare(document, **who)
    history.open_history("w-risk", "risk", tmp_path)
    from alphalattice.foundation.data_layer.telemetry import decode
    from alphalattice.interface.local_application import trial_history

    started = trial_history.started

    def native_observation(sealed):
        trial = started(sealed)
        fields = {
            "service.name": "codex",
            "conversation.id": "trial-session",
            "alphalattice.trial_id": trial.trial_id,
            "model": "synthetic-model",
            "gen_ai.usage.input_tokens": 15,
            "gen_ai.usage.cache_read.input_tokens": 3,
            "gen_ai.usage.cache_write.input_tokens": 0,
            "gen_ai.usage.output_tokens": 5,
        }
        body = {
            "resourceLogs": [
                {
                    "scopeLogs": [
                        {
                            "logRecords": [
                                {
                                    "attributes": [
                                        {
                                            "key": key,
                                            "value": {
                                                "intValue"
                                                if isinstance(value, int)
                                                else "stringValue": str(value)
                                            },
                                        }
                                        for key, value in fields.items()
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        assert (
            history.receive_agent_telemetry(
                decode("logs", json.dumps(body).encode(), "application/json", workspace_id="w-risk")
            )
            == 1
        )
        return trial

    monkeypatch.setattr(trial_history, "started", native_observation)
    try:
        world.admit_universe(real_risk_workspace.workspace / "market-data.duckdb", "w-risk")
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
        trial_history.ended(trial_history.started(attributed), first)  # equal bytes, produced
    finally:
        history.close_history()
    assert (first.disposition, again.numerical_call_count) == ("COMPUTED", 0)
    with closing(sqlite3.connect(root / "machine" / "log.sqlite")) as con:
        trials = con.execute(
            "SELECT state, (SELECT a.content_digest FROM artifact a"
            " WHERE a.artifact_id = result_artifact)"
            ", seeds, s.goal_id, hypothesis_id, recipe_key, task_id"
            " FROM trial JOIN trial_scope s USING (trial_id) ORDER BY registered_seq"
        ).fetchall()
        assert [row[:5] for row in trials] == [
            ("completed", first.evidence_hash, "[0]", "g1", "h1"),
            ("completed", None, "[0]", None, None),
            ("completed", first.evidence_hash, "[0]", "g1", "h1"),
        ]
        assert trials[0][5] == trials[1][5]
        sums = con.execute(
            "SELECT attrs FROM event WHERE type = 'trial' AND phase = 'completed' ORDER BY seq"
        ).fetchall()
        assert [json.loads(row[0])["agent_telemetry"] for row in sums[:2]] == [
            {
                "status": "MEASURED",
                "model_seconds": None,
                "responses": 1,
                "input_tokens": 12,
                "cache_read_tokens": 3,
                "cache_write_tokens": 0,
                "output_tokens": 5,
            }
        ] * 2
        assert [row[6] for row in trials] == [str(task.task_id), None, str(task.task_id)]
        (trace_id,) = con.execute(
            "SELECT e.trace_id FROM trial t JOIN event e ON e.seq=t.registered_seq "
            "WHERE t.hypothesis_id='h1' ORDER BY t.registered_seq"
        ).fetchone()
        assert trace_id
        assert con.execute("SELECT task_id FROM task").fetchall() == [(str(task.task_id),)]
        assert con.execute("SELECT count(*) FROM selection").fetchone()[0] == 1
        assert (
            con.execute("SELECT cutoff_us FROM selection").fetchone()[0]
            == sealed.envelope.sessions.as_of.selection_cutoff_us
        )
        recipe = con.execute(
            "SELECT l.name FROM recipe r JOIN definition_lineage l ON l.definition_id = r.method_id"
        ).fetchall()
        assert recipe == [(first.kind,)]
        artifact = con.execute(
            "SELECT l.name, a.storage, a.trial_id IS NOT NULL FROM artifact a"
            " JOIN definition_lineage l ON l.definition_id = a.type_id ORDER BY a.artifact_id"
        )
        # Risk stores one covariance artifact per formation. The two computed trials
        # still each own their evidence, and the reuse trial owns none.
        assert artifact.fetchall() == (
            [("market_condition", "none", 0)]
            + [("risk_covariance_formation", "table", 0)] * len(first.formation_sessions)
            + [("research_evidence", "none", 1)] * 2
        )
        with closing(replay(con, tmp_path / "replayed.sqlite")) as target:
            assert differences(con, target) == []
    with closing(reading_session(root)) as view:
        assert view is not None
        row = next(r for r in experience(view, hypothesis="h1") if r["trace_id"] == trace_id)
        assert row["trial_id"] and len(experience(view, hypothesis="h1")) == 2
        assert row["task_id"] == str(task.task_id)
