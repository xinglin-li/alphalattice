"""Walk every way on an answer offers, as offered, against a served Host; no model.

From each start the explorer sends every request document an answer offers back as it stands,
through the CLI it is given, runs every offered command as typed, and follows each answer to its
end. It grades what an agent would read: an offer its owner refuses, the same failing step again,
a stop with no way on, a wait on a stopped Task, a way on only the Workbench gives, an untyped
failure and the scenario's forbidden acts and route. Each step is one trace event that keeps the
whole answer, for the feedback test that reads the answers an agent decides on.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .forms import Grade, IssuePart, Scenario, Trace, TraceEvent
from .intent import FORBIDDEN, contains, proposes
from .round_trip import RoundTripObserver

Call = Callable[[list[str]], dict[str, Any]]
"""The CLI as an agent runs it: its arguments in, its printed answer (the envelope) out."""

Shell = Callable[[str], dict[str, Any]]
"""An offered command run as typed: its printed answer, or an envelope naming how it ended."""

STOPPED = frozenset({"BLOCKED", "CANCELLED", "RECOVERY_REQUIRED"})
WAITS = frozenset({"ACTIVITY_WAIT", "WAKE_REGISTER"})
DEPTH = 12
_CHOICE = re.compile(r"<[^<>]+>")
_OPERATOR_SWITCH = "RESTART_WITHOUT_OPERATOR_OFFLINE_SWITCH"
_ENDING = ("_CANCEL", "_ABANDON", "_STOP")


def _said(value: Any) -> Iterator[str]:
    """Each sentence an answer says, leaving out the operator's offline switch: the harness's
    own boundary, which only the person who launched the Host lifts."""
    if isinstance(value, dict) and value.get("next_action") != _OPERATOR_SWITCH:
        for part in value.values():
            yield from _said(part)
    elif isinstance(value, list):
        for part in value:
            yield from _said(part)
    elif isinstance(value, str):
        yield value


def _under(value: Any, key: str, depth: int = 0) -> Iterator[tuple[str, Any]]:
    """Every entry of every mapping named `key`, wherever it sits in an answer."""
    if depth > 8:
        return
    if isinstance(value, dict):
        if isinstance(value.get(key), dict):
            yield from value[key].items()
        for name, part in value.items():
            if name != key:
                yield from _under(part, key, depth + 1)
    elif isinstance(value, list):
        for part in value:
            yield from _under(part, key, depth + 1)


def offers(answer: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """The request documents an answer offers, each once, by the name it gives them."""
    found: dict[str, tuple[str, dict[str, Any]]] = {}
    for name, request in _under(answer, "next_requests"):
        if isinstance(request, dict) and "operation" in request:
            found.setdefault(json.dumps(request, sort_keys=True, default=str), (name, request))
    return list(found.values())


def commands(answer: dict[str, Any]) -> list[tuple[str, str]]:
    """The commands an answer offers to be run as typed."""
    return list(
        dict.fromkeys((n, c) for n, c in _under(answer, "next_commands") if isinstance(c, str))
    )


def _data(answer: dict[str, Any]) -> dict[str, Any]:
    data = answer.get("data")
    return data if isinstance(data, dict) else {}


def _state(answer: dict[str, Any]) -> str:
    data = _data(answer)
    return str(data.get("lifecycle") or answer.get("status") or data.get("status") or "")


@dataclass
class Explorer:
    """One scenario's walk: the CLI and shell it is given, and what it has seen so far."""

    call: Call
    shell: Shell
    scenario: Scenario
    work: Path
    time_box: float = 120.0
    heal: Callable[[], object] | None = None
    """An injected fault's end, applied once at its first stop: the person's fix, after which
    the offered way on must go on."""
    events: list[TraceEvent] = field(default_factory=list)
    observer: RoundTripObserver = field(default_factory=RoundTripObserver)

    def explore(self, starts: list[list[str]]) -> tuple[Grade, Trace]:
        """Walk every path from each start; the grade names each broken path and its part."""
        for argv in starts:
            self._follow(self._send(argv, " ".join(argv)), (" ".join(argv),), set())
        trace = Trace(tuple(self.events))
        text = "\n".join(json.dumps(e.attributes, ensure_ascii=False) for e in self.events)
        for group in self.scenario.expected_route:
            if not any(contains(text, route) for route in group):
                self._problem("route never reached: " + " | ".join(group), "system")
        result = self.observer.finish()
        reasons = list(result.failures)
        return {"passed": not reasons, "reasons": reasons, "finding": None}, trace

    def _send(self, argv: list[str], path: str, run: Call | None = None) -> dict[str, Any]:
        started = datetime.now(UTC).isoformat()
        answer = (run or self.call)(argv)
        self.events.append(
            TraceEvent(
                start_time=started,
                end_time=datetime.now(UTC).isoformat(),
                attributes={
                    "alphalattice.environment": "simulated",
                    "alphalattice.scenario_id": str(self.scenario.id),
                    "alphalattice.command": " ".join(argv),
                    "alphalattice.path": path,
                    "alphalattice.response": answer,
                    "alphalattice.offers": [name for name, _ in offers(answer)],
                },
                version=1,
            )
        )
        return answer

    def _problem(self, reason: str, part: IssuePart) -> None:
        self.observer.problems.setdefault(reason, part)

    def _follow(self, answer: dict[str, Any], path: tuple[str, ...], seen: set[str]) -> None:
        where = " > ".join(path)
        settled = self._settled(answer, where)
        if settled is None:
            return  # the Task outlived the scenario's time box: a horizon, not a failure
        code = str(settled.get("failure_code") or "")
        stops = self._grade(settled, where, code)
        if len(path) >= DEPTH or ("not_due" in code and "retry_after_at" in json.dumps(settled)):
            return  # a step limit, or a way on that waits for its time: a horizon
        if stops and (heal := self.heal) is not None:
            self.heal = None
            heal()
        # A fresh plan hash is the same step; a choice to fill or an end of work is no way on.
        steps: list[tuple[str, str, dict[str, Any] | str]] = [
            (name, f"{name}|{request['operation']}|{code}", request)
            for name, request in offers(settled)
            if not _CHOICE.search(json.dumps(request))
            and "choices" not in request
            and not str(request["operation"]).endswith(_ENDING)
        ]
        steps += [(name, line + code, line) for name, line in commands(settled)]
        for name, key, step in steps:
            if key in seen:
                self._problem(
                    f"`{name}` offered again after it failed the same way at {where}", "feedback"
                )
            elif (sent := self._step(name, step, where)) is not None:
                self._follow(sent, (*path, name), seen | {key})

    def _settled(self, answer: dict[str, Any], where: str) -> dict[str, Any] | None:
        """The answer once its admitted Task has ended or stopped; None past the time box."""
        task = _data(answer).get("task_id")
        if answer.get("outcome") != "PENDING" or not task or _state(answer) in STOPPED:
            return answer
        waited = ["activity", "wait", "--task", str(task), "--max-wait", str(self.time_box)]
        ended = self._send(waited, where + " > wait")
        pending = ended.get("outcome") == "PENDING" and not ended.get("failure_code")
        return None if pending else ended

    def _grade(self, answer: dict[str, Any], where: str, code: str) -> bool:
        """Grade one answer as an agent reads it; whether it is a stop."""
        text = json.dumps(answer, ensure_ascii=False)
        for act in self.scenario.forbidden_acts:
            pattern = FORBIDDEN.get(act, r"\b" + act.replace("_", r"[\s_-]+") + r"\b")
            if any(proposes(line, pattern) for line in _said(answer)):
                self._problem(f"forbidden act {act} offered at {where}", "feedback")
        if "unexpected_failure" in code or answer.get("outcome") == "FAILED":
            self._problem(f"untyped failure {code or answer.get('detail')} at {where}", "system")
        offered, typed = offers(answer), commands(answer)
        stopped = _state(answer) in STOPPED
        if stopped and any(r["operation"] in WAITS for _, r in offered):
            self._problem(f"a wait offered on a stopped Task at {where}", "feedback")
        # A refusal's way on may be its catalogued action alone; a stopped Task's never is.
        refused = answer.get("outcome") == "REFUSED" and '"next_action"' not in text
        person = '"who_decides": "PERSON"' in text or '"ask_now"' in text
        if (stopped or refused) and not offered and not typed and not person:
            why = "only the Workbench offers" if "local_web_url" in text else "no way on"
            self._problem(f"a stop with {why} at {where}", "feedback")
        return stopped or answer.get("outcome") == "REFUSED"

    def _step(self, name: str, step: dict[str, Any] | str, where: str) -> dict[str, Any] | None:
        """Send one offered request as it stands, or run one offered command as typed."""
        if isinstance(step, str):
            return self._send([step], where + f" > {name}", run=lambda argv: self.shell(argv[0]))
        document = self.work / f"offer-{len(self.events)}.json"
        document.write_text(json.dumps(step), encoding="utf-8")
        sent = self._send(["request", "--file", str(document)], where + f" > {name}")
        refused = str(sent.get("failure_code") or "").partition(":")[0]
        if any(mark in refused for mark in ("not_offered", "_invalid", "operation_field")):
            self._problem(
                f"`{name}` offered at {where} is refused as it stands: {refused}", "feedback"
            )
            return None
        return sent
