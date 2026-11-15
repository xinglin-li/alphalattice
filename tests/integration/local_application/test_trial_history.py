"""A sealed research run inside a Host is a trial in the history, and the history replays."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from alphalattice.control.research_program.authoring.document import load_authoring_document
from alphalattice.kernel.data_layer import history
from alphalattice.kernel.data_layer.log import replay
from alphalattice.kernel.data_layer.project import differences
from alphalattice.protocols.actor_execution.contracts import ActorKind
from tests.integration.research_program.test_program_execution import _FIXTURE, _workflow
from tests.researcher_methodology_surface.real_workspace import RealRiskWorkspace

TEST_SOURCE = "src/alphalattice/interface/local_application/trial_history.py"


@pytest.mark.proves("local_application.research_runs_are_recorded_as_trials")
def test_a_run_and_its_rerun_are_two_trials_of_one_selection_and_recipe(
    real_risk_workspace: RealRiskWorkspace, tmp_path: Path
) -> None:
    """The first run's evidence is its trial's artifact; a rerun reusing it is a second trial
    with no artifact of its own; the history replays ALL EQUAL."""
    root = tmp_path / "data"
    document = load_authoring_document(_FIXTURE.read_text(encoding="utf-8"))
    workflow: Any = _workflow(real_risk_workspace, tmp_path)
    who = {"actor_kind": ActorKind.HUMAN, "actor_id": "researcher"}
    workflow.preflight(document, **who)
    assert history.open_history("w-risk", "risk", root) is None
    try:
        first, _ = workflow.run_sealed(document, **who)
        again, _ = workflow.run_sealed(document, **who)
    finally:
        history.close_history()
    assert (first.disposition, again.numerical_call_count) == ("COMPUTED", 0)
    with closing(sqlite3.connect(root / "machine" / "log.sqlite")) as con:
        trials = con.execute(
            "SELECT state, result_artifact, seeds FROM trial ORDER BY registered_seq"
        ).fetchall()
        assert trials == [("completed", first.evidence_hash, "[0]"), ("completed", None, "[0]")]
        assert con.execute("SELECT count(*) FROM selection").fetchone()[0] == 1
        recipe = con.execute("SELECT method FROM recipe").fetchall()
        assert recipe == [(first.kind,)]
        artifact = con.execute("SELECT type, storage, trial_id IS NOT NULL FROM artifact")
        assert artifact.fetchall() == [("research_evidence", "none", 1)]
        with closing(replay(con, tmp_path / "replayed.sqlite")) as target:
            assert differences(con, target) == []
