"""A keyed answer's one deadline over the existing per-answer turns."""

from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from time import monotonic
from typing import Any

from alphalattice.kernel.shared_kernel.identity import canonical_hash

# The client waits 120 seconds; leave 30 seconds for the response and other work.
SUBMISSION_WAIT_SECONDS = 90.0
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
