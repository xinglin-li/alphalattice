"""Offered requests and fault classifications retain their observation scope."""

import pytest

from devtools.evaluation.round_trip import RoundTripObserver, TruthFacts


@pytest.mark.parametrize("code", ["", "request.operation_field_invalid"])
def test_recovery_offers_are_sent_verbatim_once_per_scope(code):
    """A filled recovery offer is tried once and a refusal identifies its feedback fault."""
    observer = RoundTripObserver()
    sent = []
    offer = {"operation": "PREVIEW", "recovery_task_id": "sample-task"}
    body = {"next_requests": {"preview": offer}}

    def admits(request):
        sent.append(request)
        return {"failure_code": code}

    def truth():
        return observer.truth(
            "STATUS",
            body,
            TruthFacts(previews=("PREVIEW",)),
            offered_requests=lambda answer: answer["next_requests"],
            stopped_problem=lambda *args, **kwargs: None,
            make_request=dict,
            admits=admits,
            failure_code=str,
        )

    for _ in range(2):
        observer.answered(
            "STATUS",
            body,
            {},
            answer_problem=lambda *args: None,
            untyped_failure=lambda failure: False,
            truth=truth,
        )
    result = observer.finish()
    assert sent == [offer]
    assert result.failure_kind == ("answer" if code else None)
    assert bool(result.failures) == bool(code)
    assert all(
        issue["loop_part"] == "feedback" and code in issue["reason"] for issue in result.failures
    )
    assert observer.finish().failures == ()


def test_untyped_faults_are_reported_as_system_failures_without_leaking_scopes():
    """Control: Untyped faults report one system failure and clear the next observation scope."""
    observer = RoundTripObserver()
    observer.raised(
        "STATUS",
        "handler_failed",
        untyped_failure=lambda code: code == "handler_failed",
        refusal_words=lambda code: None,
    )
    result = observer.finish()
    assert result.failure_kind == "untyped"
    assert result.failures == ({"reason": "STATUS raised handler_failed", "loop_part": "system"},)
    assert observer.finish().failure_kind is None


STOPPED = "a RECOVERY_REQUIRED Task's answer offers no way on (Task 0193-stopped)"
TEST = "tests/b.py::test_after"


@pytest.mark.proves("evaluation.answer_observer_judges_its_own_hosts")
@pytest.mark.parametrize(
    ("owner", "judged"),
    [
        ("tests/a.py::test_before", False),
        ("tests/b.py::test_other", False),
        (TEST, True),
        ("tests/b.py", True),
        ("tests", True),
        ("", True),
        (None, True),
    ],
    ids=[
        "earlier-test",
        "sibling-test",
        "this-test",
        "its-module",
        "its-package",
        "session",
        "unknown",
    ],
)
def test_a_tests_teardown_judges_only_the_hosts_its_scopes_started(owner, judged):
    """A stopped Task's answer fails the test whose scope started its Host, naming the Task."""
    observer = RoundTripObserver()
    observer.begin(TEST)
    observer.answered(
        "RUN",
        {"task_id": "0193-stopped"},
        {},
        answer_problem=lambda *args: STOPPED,
        untyped_failure=lambda failure: False,
        owner=owner,
    )
    result = observer.finish()
    assert result.failures == (({"reason": STOPPED, "loop_part": "feedback"},) if judged else ())
    assert observer.foreign == (
        [] if judged else [f"{STOPPED} (a Host of {owner!r}, during {TEST})"]
    )


def _stopped(observer, owner):
    observer.answered(
        "RUN",
        {"task_id": "0193-stopped"},
        {},
        answer_problem=lambda *args: STOPPED,
        untyped_failure=lambda failure: False,
        owner=owner,
        context="Task 0193-stopped",
    )


@pytest.mark.proves("evaluation.answer_observer_judges_its_own_hosts")
def test_an_answer_between_two_tests_waits_for_the_test_that_owns_its_host():
    """A Host's answer after its test closed fails no later test; a module Host's is judged next."""
    observer = RoundTripObserver()
    observer.begin("tests/a.py::test_before")
    assert observer.finish().failure_kind is None
    _stopped(observer, "tests/a.py::test_before")  # its test has closed
    _stopped(observer, "tests/b.py")  # a module Host, set up before the next test opens
    observer.begin("tests/b.py::test_after")
    named = f"{STOPPED} (Task 0193-stopped)"
    assert observer.finish().failures == ({"reason": named, "loop_part": "feedback"},)
    assert observer.foreign == [
        f"{named} (a Host of 'tests/a.py::test_before', during tests/b.py::test_after)"
    ]
