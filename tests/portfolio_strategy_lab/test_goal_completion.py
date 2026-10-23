"""A sealed completion returns its report without repeating reference verification."""

from contextlib import nullcontext
from unittest.mock import patch
from uuid import UUID

import pytest

from alphalattice.interface.local_application.cli_contract import envelope, worded
from tests.portfolio_strategy_lab.goal_support import Request, _attach, _reference, _submission
from tests.portfolio_strategy_lab.goal_support import goal_app as goal_app


@pytest.mark.parametrize("inline, fail", [(True, False), (False, False), (True, True)])
def test_sealed_reply_renders_once_and_keeps_its_seal_on_failure(goal_app, inline, fail):
    app, opened, calls, *_ = goal_app
    if not inline:
        _attach(app, opened)
    submitted = _submission(references=[_reference()] if inline else [])
    failure = (
        patch(
            "alphalattice.interface.local_application.experiment_report.render_goal",
            side_effect=RuntimeError("fixture"),
        )
        if fail
        else nullcontext()
    )
    calls.clear()
    with failure, patch.object(app, "_identity", wraps=app._identity) as identities:
        answer = app.operate(
            Request(operation="GOAL_SUBMIT", goal_id=UUID(int=1), goal_submission=submitted),
            "HUMAN",
        )
    assert answer["status"] == "COMPLETE" and answer["outcome"] == submitted["outcome"]
    assert "failure_code" not in answer and identities.call_count == 1
    assert calls == [("EXPERIMENT_READBACK", "HUMAN")]
    assert ("report_error" if fail else "report") in answer
    wrapped = envelope(operation="GOAL_SUBMIT", outcome="OK", body=answer, elapsed_seconds=0)
    assert wrapped["closure_detail"] == worded(answer["closure_message_id"])
    sealed_hash = answer["goal_hash"]
    if fail:
        assert answer["report_error"]["failure_code"] == "goal.closure.report_unavailable"
        report = app.operate(Request(**answer["next_requests"]["export"]), "HUMAN")
    else:
        report = answer["report"]
    assert report["status"] == "GOAL_EXPORTED" and report["goal_hash"] == sealed_hash
    assert app.store.head(UUID(int=1)).goal_hash == sealed_hash
    assert report["goal"]["submission"]["summary"] == submitted["summary"]
    assert submitted["summary"] in report["html"]
