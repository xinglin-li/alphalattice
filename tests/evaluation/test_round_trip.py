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
    """A fault is classified once and the next observation scope starts empty."""
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
