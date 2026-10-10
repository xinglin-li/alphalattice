"""Feedback grading holds the next decision to the original offered answer."""

import pytest

from devtools.evaluation import FeedbackGrader, Trace, TraceEvent, collect_feedback

NOW = "2026-01-02T10:00:00+00:00"
SEND = "alphalattice request --from answer.json --action next"
SOURCE = {"data": {"next_requests": {"next": {"operation": "STATUS"}}}}


def evaluate(
    source=SOURCE,
    *,
    decision="send",
    sent=(),
    asked=(),
    disclosure=(),
    steps=(),
    unusable=None,
):
    event = TraceEvent(
        NOW, NOW, {"alphalattice.response": source, "alphalattice.path": "status"}, 1
    )
    row = collect_feedback(
        event, "sample.feedback", "Read status.", NOW, decision, "Status", unusable
    )
    answer = {
        "commands": list(sent),
        "fields": [],
        "steps": list(steps),
        "acts": [],
        "ask": list(asked),
        "refuse": list(disclosure),
        "decider": row.decider,
    }
    trace = Trace((TraceEvent(NOW, NOW, {"alphalattice.answer": answer}, 1),))
    catalog = {("status", "show"): {"--id"}}
    return row, FeedbackGrader(catalog).grade(row, trace)


def test_literal_commands_and_only_selected_unusable_offers_are_checked():
    """A proven bad offer fails only when selected, and literal commands stay verbatim."""
    literal = "alphalattice status show --id stable"
    source = {**SOURCE, "next_commands": {"read": literal}}
    assert evaluate(source, sent=[literal], unusable={"next": "refused"})[1]["passed"]
    assert not evaluate(source, sent=[literal + " "])[1]["passed"]
    result = evaluate(source, sent=[SEND], unusable={"next": "refused"})[1]
    assert {issue["loop_part"] for issue in result["reasons"]} == {"feedback"}


@pytest.mark.parametrize("state", ["BLOCKED", "CANCELLED", "RECOVERY_REQUIRED"])
@pytest.mark.parametrize("operation", ["ACTIVITY_WAIT", "WAKE_REGISTER"])
def test_an_offered_wait_on_a_stopped_task_is_a_feedback_defect(state, operation):
    """Both structured waiting operations refuse every stopped lifecycle."""
    source = {"data": {"lifecycle": state}, "next_requests": {"next": {"operation": operation}}}
    result = evaluate(source, sent=[SEND])[1]
    assert result["reasons"] == [{"reason": "cli:wait_on_stopped_task", "loop_part": "feedback"}]


def test_question_and_stop_decisions_require_their_own_evidence():
    """Questions require permission and one line; stopping requires a disclosure."""
    source = {**SOURCE, "first_use": {"ask_now": ["Use wider budget?"]}}
    assert evaluate(source, sent=[SEND])[1]["passed"]
    assert evaluate(source, decision="ask", asked=["Use wider budget?"])[1]["passed"]
    assert not evaluate(source, decision="ask", asked=["Why?\nHow?"])[1]["passed"]
    assert not evaluate(decision="ask", asked=["Confirm?"])[1]["passed"]
    assert evaluate(decision="stop", disclosure=["Unavailable."])[1]["passed"]
    assert evaluate(decision="stop", disclosure=["restart Host"])[1]["passed"]
    assert evaluate(decision="stop", steps=["Report limitation."])[1]["passed"]
    assert not evaluate(decision="stop")[1]["passed"]
    assert not evaluate(source, sent=[SEND], asked=["Confirm?"])[1]["passed"]
    stopped = {"data": {"lifecycle": "BLOCKED"}}
    assert not evaluate(stopped, decision="stop", steps=["Wait for the Task to settle."])[1][
        "passed"
    ]
    assert evaluate(stopped, decision="stop", steps=["Do not wait for the Task."])[1]["passed"]
    assert evaluate(stopped, decision="stop", steps=["Delay until retry."])[1]["passed"]
