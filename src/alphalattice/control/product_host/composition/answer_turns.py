"""A keyed answer's one deadline over the existing per-answer turns."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from time import monotonic
from typing import Any

from alphalattice.kernel.shared_kernel.identity import canonical_hash

# The client waits 120 seconds; leave 30 seconds for the response and other work.
SUBMISSION_WAIT_SECONDS = 90.0
ANSWER_FILING: ContextVar[Callable[[Any], None] | None] = ContextVar("answer_filing", default=None)
"""What a correction's filing records first, when its submitter set it: the answer's intent on its
own thread, so a filing cut short after its record is recovered on retry, never counted twice."""
SUBMISSION_DEADLINE: ContextVar[float | None] = ContextVar(
    "answer_submission_deadline", default=None
)


class AnswerTurnBusy(Exception):
    """The existing answer turn could not be acquired before the request's deadline."""


@contextmanager
def bounded_answer_turn(
    review: Any, binding: Mapping[str, object], deadline: float
) -> Iterator[None]:
    """Use the review owner's existing binding lock without waiting past this request."""
    key = str(canonical_hash(binding))
    with review._answer_turns_lock:
        turn = review._answer_turns.setdefault(key, threading.Lock())
    remaining = deadline - monotonic()
    if remaining <= 0 or not turn.acquire(timeout=remaining):
        raise AnswerTurnBusy
    try:
        yield
    finally:
        turn.release()
