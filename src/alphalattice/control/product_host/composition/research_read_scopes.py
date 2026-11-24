"""Per-operation verification scopes shared by research and trial readers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from alphalattice.capabilities.domain_contracts.bounded_memo import BoundedMemo

TRIALS_DIRECTORY = Path("runtime") / "feature-trials"


_VERIFIED_EVIDENCE: ContextVar[BoundedMemo[tuple[object, ...], object] | None] = ContextVar(
    "research_experiment_verified_evidence", default=None
)


@contextmanager
def verified_study_evidence() -> Iterator[None]:
    """Verify each study's evidence once per operation, then release.

    A study page or a research case reads several studies, and a comparison inside
    it reads two of them again; each read walked the whole sealed graph and
    re-derived every chunk's content hash (a case gathering
    studies verified the same two Alpha studies twice in one request). Inside this
    scope a Task's evidence is verified once per (Task, record version, readback
    mode) and the verified result reused for the rest of the operation. An
    operation run inside another (a case reading its studies) joins the scope
    already open.

    Across requests, a read outside full proof answers from the owner's ledger: the
    last full verification of the same Task record under the same installed
    implementation. Its sealed files were proved when the study wrote them; an
    export proves them again in full, and the scrub finds a changed one and drops
    the study's ledger entry (EV2). Its answer says so (``verification_basis``).
    """
    if _VERIFIED_EVIDENCE.get() is not None:
        yield
        return
    token = _VERIFIED_EVIDENCE.set(BoundedMemo())
    try:
        yield
    finally:
        scope = _VERIFIED_EVIDENCE.get()
        if scope is not None:
            scope.clear()
        _VERIFIED_EVIDENCE.reset(token)


_TRIAL_READS: ContextVar[BoundedMemo[str, dict[str, Any]] | None] = ContextVar(
    "feature_trial_reads", default=None
)


@contextmanager
def trial_reads_once() -> Iterator[None]:
    """One request reads each completed trial once.

    A goal's submission and its read resolve the trial a reference names and the review packet
    another names, and the packet reads the same trial again, its comparison included. Inside
    this scope a completed trial answers from its first read in the request. An operation run
    inside another joins the scope already open; the reads are released with it.
    """
    if _TRIAL_READS.get() is not None:
        yield
        return
    token = _TRIAL_READS.set(BoundedMemo())
    try:
        yield
    finally:
        _TRIAL_READS.reset(token)
