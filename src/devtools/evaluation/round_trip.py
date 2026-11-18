"""Observe answer contracts and offered requests through injected public surfaces."""

from __future__ import annotations

import json
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from .forms import GradingIssue, IssuePart


@dataclass(frozen=True)
class TruthFacts:
    """Facts projected by an adapter; no runtime owner is read by the observer."""

    lifecycle: str | None = None
    previews: Collection[str | None] = ()
    dated_operation: str | None = None
    formation_session: str | None = None


@dataclass(frozen=True)
class AdmissionPolicy:
    """The caller's operation vocabulary for plans with no new runnable admission."""

    confirmation_admissions: Mapping[tuple[str, str], Collection[str]] = field(default_factory=dict)
    existing_work_states: Mapping[str, Collection[str]] = field(default_factory=dict)
    no_plan_states: Collection[tuple[str, str]] = ()
    runnable_states: Collection[str] = ()
    inaccessible_confirmation: tuple[str, str] | None = None


def truth_problems(
    operation: str,
    body: dict[str, Any],
    facts: TruthFacts,
    *,
    offered_requests: Callable[..., Mapping[str, dict[str, Any]]],
    stopped_problem: Callable[..., str | None],
    make_request: Callable[[dict[str, Any]], Any],
    admits: Callable[[Any], Any],
    failure_code: Callable[[Exception], str],
    resent: set[str],
    dated_facts: Callable[[], tuple[str | None, str | None]] | None = None,
) -> Iterator[str]:
    """Check stopped-work truth, recovery admission and a delegated update's date."""
    if facts.lifecycle is not None and (
        problem := stopped_problem(body, facts.lifecycle, way_on=False)
    ):
        yield f"{operation}: {problem}"
    offers = offered_requests(body)
    for name, offer in offers.items():
        key = json.dumps(offer, sort_keys=True, default=str)
        if (
            "recovery_task_id" not in offer
            or offer["operation"] not in facts.previews
            or key in resent
        ):
            continue
        resent.add(key)
        try:
            answer = admits(make_request(offer))
            code = str(answer.get("failure_code") or "") if isinstance(answer, dict) else ""
        except Exception as error:
            code = failure_code(error)
        base = code.partition(":")[0]
        if "not_offered" in base or base.endswith("_invalid") or "operation_field" in base:
            yield f"{operation} offers `{name}`, which sent back as it stands is refused: {code}"
    dated_operation, formation_session = (
        dated_facts()
        if dated_facts is not None
        else (facts.dated_operation, facts.formation_session)
    )
    if dated_operation is not None:
        for name, offer in offers.items():
            dated = offer.get("observed_through")
            if offer.get("operation") == dated_operation and (
                formation_session is None or dated != formation_session
            ):
                yield f"{operation} offers `{name}`, an update that is not the first use's date"


def plan_admission_problem(
    operation: str,
    body: Any,
    replans: Iterable[tuple[str | None, str]],
    routes: Mapping[str, Mapping[str, str]],
    policy: AdmissionPolicy,
    *,
    offered_requests: Callable[..., Mapping[str, dict[str, Any]]],
    request_problem: Callable[..., str | None],
    choices: Callable[..., Any],
    validate_request: Callable[[dict[str, Any]], Any],
    refused: Callable[..., bool],
) -> str | None:
    """Require each runnable preview to offer a filled admission on a session POST route."""
    expected = {
        admitting for preview, admitting in replans if preview == operation and preview != admitting
    }
    if not expected or not isinstance(body, dict):
        return None
    status = body.get("status")
    confirmed = next(
        (
            admissions
            for (preview, state), admissions in policy.confirmation_admissions.items()
            if operation == preview and status == state
        ),
        None,
    )
    if confirmed is not None:
        expected = set(confirmed)
    offers = [q for q in offered_requests(body).values() if q.get("operation") in expected]
    if not offers:
        if _has_no_new_admission(operation, body, status, policy, refused):
            return None
        return f"{operation} offers no filled admission among {sorted(expected)}"
    for offer in offers:
        if problem := _filled_admission_problem(
            operation, offer, routes, request_problem, choices, validate_request
        ):
            return problem
    return None


def _filled_admission_problem(
    operation: str,
    offer: dict[str, Any],
    routes: Mapping[str, Mapping[str, str]],
    request_problem: Callable[..., str | None],
    choices: Callable[..., Any],
    validate_request: Callable[[dict[str, Any]], Any],
) -> str | None:
    """Hold a filled admission to its public request contract and session POST route."""
    problem = request_problem(offer)
    if problem or choices(offer):
        return f"{operation} admission is unfilled or invalid: {problem or choices(offer)}"
    validate_request(offer)
    route = routes.get(offer["operation"])
    if not route or route["method"] != "POST" or not route["path"]:
        return f"{operation} admission has no session POST route: {offer['operation']}"
    return None


def _has_no_new_admission(
    operation: str,
    body: dict[str, Any],
    status: Any,
    policy: AdmissionPolicy,
    refused: Callable[..., bool],
) -> bool:
    """Only the caller's declared terminal, existing-work and inaccessible states are exempt."""
    existing_work = False
    if operation in policy.existing_work_states:
        try:
            UUID(str(body.get("task_id")))
            existing_work = status in policy.existing_work_states[operation]
        except ValueError:
            pass
    # Unknown states fail closed even when both a plan hash and its offer are absent.
    return bool(
        (refused(body) and status not in policy.runnable_states)
        or (operation, status) in policy.no_plan_states
        or (
            (operation, status) == policy.inaccessible_confirmation
            and body.get("confirmation_available") is False
            and body.get("source_access_failure")
        )
        or existing_work
    )


@dataclass(frozen=True)
class RoundTripResult:
    """One observation scope's findings for the caller's failure and report adapters."""

    failure_kind: str | None
    failures: tuple[GradingIssue, ...]
    trips: tuple[GradingIssue, ...]
    unworded: tuple[GradingIssue, ...]


def _issues(lines: Iterable[tuple[str, IssuePart]]) -> tuple[GradingIssue, ...]:
    return tuple({"reason": reason, "loop_part": part} for reason, part in lines)


class RoundTripObserver:
    """Hold contracts, failure wording and recovery offers within one observation scope."""

    def __init__(self, *, report: bool = False) -> None:
        self.report = report
        self.problems: dict[str, IssuePart] = {}
        self.untyped: list[str] = []
        self.unworded: list[str] = []
        self.trips: list[str] = []
        self.resent: set[str] = set()
        self.sending = False
        self.scoped = False
        self.test: str | None = None
        self.pending: list[tuple[str | None, str, str, IssuePart]] = []
        """Findings heard while no test was open, each with its Host's owner, routed at `begin`."""
        self.foreign: list[str] = []
        """Findings in answers from Hosts the open test's scopes did not start, said, not judged."""

    def begin(self, test: str) -> None:
        """Open a test's scope: it judges what the Hosts it or a wider fixture started say, and
        what they said while no test was open; another scope's finding goes to `foreign`."""
        self.scoped, self.test = True, test
        pending, self.pending = self.pending, []
        for owner, kind, line, part in pending:
            self._record(owner, kind, line, part)

    def owns(self, owner: str | None) -> bool:
        """Whether the open test judges a Host started in `owner`'s scope: the test itself, or
        its class, module, package or session (""). A Host whose owner is unknown is judged."""
        test = self.test or ""
        return owner is None or owner in ("", test) or test.startswith((owner + "::", owner + "/"))

    def _record(self, owner: str | None, kind: str, line: str, part: IssuePart) -> None:
        """Keep a finding for the open test, for the next one while closed, or as foreign."""
        if self.scoped and self.test is None:
            self.pending.append((owner, kind, line, part))
        elif self.scoped and not self.owns(owner):
            self.foreign.append(f"{line} (a Host of {owner!r}, during {self.test})")
        elif kind == "untyped":
            self.untyped.append(line)
        elif kind == "unworded":
            self.unworded.append(line)
        else:
            self.problems.setdefault(line, part)

    def truth(
        self, operation: str, body: dict[str, Any], facts: TruthFacts, **checks: Any
    ) -> Iterator[str]:
        """Apply the truth rules with this scope's offer deduplication."""
        return truth_problems(operation, body, facts, resent=self.resent, **checks)

    def raised(
        self,
        operation: str,
        code: str,
        *,
        untyped_failure: Callable[[str], bool],
        refusal_words: Callable[[str], Any],
        owner: str | None = None,
        context: str | None = None,
    ) -> None:
        """Record the public classification and wording of an operation's raised failure; each
        line names `context` (its Task, workspace and Host) when given."""
        said = f"{operation} raised {code}" + (f" ({context})" if context else "")
        if untyped_failure(code):
            self._record(owner, "untyped", said, "system")
        elif refusal_words(code) is None:
            self._record(owner, "unworded", said, "feedback")

    def answered(
        self,
        operation: str,
        body: Any,
        given: Mapping[str, Any],
        *,
        answer_problem: Callable[..., str | None],
        untyped_failure: Callable[[str], bool],
        truth: Callable[[], Iterable[str]] | None = None,
        plan_admission: Callable[[], str | None] | None = None,
        owner: str | None = None,
        context: str | None = None,
    ) -> None:
        """Observe one answer; sending a recovery offer cannot recursively send more.

        Each finding names `context` (its Task, workspace and Host) when given. A Host the open
        test's scopes did not start is not judged, and its offers are not sent.
        """
        named = (lambda line: f"{line} ({context})") if context else (lambda line: line)
        if (problem := answer_problem(operation, body, given)) is not None:
            self._record(owner, "problem", named(problem), "feedback")
        if self.scoped and self.test is not None and not self.owns(owner):
            return
        if isinstance(body, dict) and not self.sending and truth is not None:
            self.sending = True
            try:
                if self.report:
                    self.trips.extend(truth())
                else:
                    for problem in truth():
                        self._record(owner, "problem", named(problem), "feedback")
            finally:
                self.sending = False
        if plan_admission is not None and (problem := plan_admission()):
            self._record(owner, "problem", named(problem), "cli")
        if isinstance(body, dict):
            code = str(body.get("failure_code") or body.get("refused") or "")
            if untyped_failure(code):
                self._record(owner, "untyped", named(f"{operation} answered {code}"), "system")

    def finish(
        self, *, allow_untyped: bool = False, report_unworded: bool = False
    ) -> RoundTripResult:
        """Drain findings in their failure precedence, clear per-scope offer state and close the
        test's scope: what its Hosts say after this waits for the next test's `begin`."""
        self.resent.clear()
        if self.scoped:
            self.test = None
        trips = _issues((line, "feedback") for line in self.trips)
        self.trips.clear()
        untyped = tuple(dict.fromkeys(self.untyped))
        self.untyped.clear()
        if untyped and not allow_untyped:
            self.problems.clear()
            self.unworded.clear()
            return RoundTripResult(
                "untyped", _issues((line, "system") for line in untyped), trips, ()
            )
        unworded = tuple(dict.fromkeys(self.unworded))
        self.unworded.clear()
        if unworded and not report_unworded:
            for line in unworded:
                self.problems.setdefault(f"{line} without words or a way on", "feedback")
        found = _issues(self.problems.items())
        self.problems.clear()
        return RoundTripResult(
            "answer" if found else None,
            found,
            trips,
            _issues((line, "feedback") for line in unworded),
        )
