"""The scenario person: every native release gate's one answerer of the agent's questions.

tau2-bench's simulated user, adapted (https://github.com/sierra-research/tau2-bench, MIT): the
person answers from the scenario's script first, then from its fact sheet, and never invents a
fact. A question neither covers is asked by the runner's own ``ask`` command, with a bound (an
ASK); its answer joins the sheet. Past the bound the person says it does not know, except for a
Host product question the scenario does not name nonblocking: that stops the run at once, by name,
because only a person may give that consent. Asking is graded, never failed: every question is an
event the moment it is asked.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

UNKNOWN = "I don't know; decide and tell me what you chose."
EVENTS = "questions.jsonl"
ASK_ENV = "ALPHALATTICE_PERSON_ASK"
"""A JSON argv whose ``{ask}`` items name the question's record; a missing setting asks no one."""


@dataclass(frozen=True)
class Answer:
    words: str | None
    source: str
    stop: str | None = None


def ask_command(environment: dict[str, str] | None = None) -> list[str] | None:
    """The configured ASK command, or None; an unreadable setting refuses by name. Its runner
    replaces ``{ask}`` with the question's record and ``{question}`` with its first 120
    characters."""
    raw = (os.environ if environment is None else environment).get(ASK_ENV)
    if not raw:
        return None
    try:
        argv = json.loads(raw)
    except ValueError as error:
        raise ValueError(f"scenario_person.ask_unreadable: {ASK_ENV} is not a JSON argv") from error
    if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
        raise ValueError(f"scenario_person.ask_unreadable: {ASK_ENV} is not a JSON argv")
    return argv


def asked(rows: Iterable[dict[str, Any]]) -> str | None:
    """The agent's own question to the person, when its turn ends on one (Claude Code or Codex)."""
    text = None
    for row in rows:
        if row.get("type") == "result" and isinstance(row.get("result"), str):
            text = row["result"]
        item = row.get("item") if row.get("type") == "item.completed" else None
        if isinstance(item, dict) and item.get("type") == "agent_message":
            text = item.get("text") if isinstance(item.get("text"), str) else text
    last = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    return last[-1] if last and last[-1].rstrip("*_` ").endswith(("?", "\uff1f")) else None


class ScenarioPerson:
    """One run's person; ``sheet`` is the scenario's ``person`` block, ``out`` the run's folder, and
    ``ask`` posts a question's record (its path and text) and says whether the post was taken."""

    def __init__(
        self,
        sheet: dict[str, Any],
        out: Path,
        *,
        ask: Callable[[Path, str], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.facts = list(sheet.get("facts") or [])
        self.nonblocking = set(sheet.get("nonblocking") or [])
        self.bound = float(sheet.get("ask_bound_seconds", 600))
        self.interval = float(sheet.get("ask_interval_seconds", 15))
        self.role = str(sheet.get("role", ""))
        self.out, self.ask, self.clock, self.sleep = out, ask, clock, sleep

    def record(self, event: str, **fields: Any) -> None:
        row = dict(event=event, at=datetime.now(UTC).isoformat(timespec="seconds"), **fields)
        self.out.mkdir(parents=True, exist_ok=True)
        with (self.out / EVENTS).open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def scripted(self, question: str, words: str, *, code: str | None, session: str) -> None:
        """A question the runner's script already answered is still a recorded question."""
        qid = uuid.uuid4().hex[:12]
        self.record("asked", id=qid, question=question, code=code, session=session)
        self.record("answered", id=qid, source="script", answer=words, waited_s=0.0)

    def answer(self, question: str, *, code: str | None = None, session: str = "") -> Answer:
        qid, began = uuid.uuid4().hex[:12], self.clock()
        self.record("asked", id=qid, question=question, code=code, session=session)
        found = self._sheet(question, code)
        if found is not None:
            reply = Answer(found["answer"], "sheet")
            self.record(
                "answered",
                id=qid,
                source="sheet",
                answer=reply.words,
                waited_s=0.0,
                avoidable=bool(found.get("avoidable")),
            )
            return reply
        words = self._ask(qid, question, code, session)
        if words is not None:
            reply = Answer(words, "asked")
            with (self.out / "sheet-proposals.jsonl").open(
                "a", encoding="utf-8", newline="\n"
            ) as stream:
                stream.write(
                    json.dumps(dict(code=code, match=question, answer=words), ensure_ascii=False)
                    + "\n"
                )
        elif code is not None and code not in self.nonblocking:
            reply = Answer(None, "stop", f"scenario_person.product_question_unanswered:{code}")
        else:
            reply = Answer(UNKNOWN, "unknown")
        self.record(
            "answered",
            id=qid,
            source=reply.source,
            answer=reply.words,
            stop=reply.stop,
            waited_s=round(self.clock() - began, 1),
        )
        return reply

    def after_turn(
        self,
        turn: dict[str, Any],
        rows: Iterable[dict[str, Any]],
        *,
        scripted: dict[str, tuple[str, str]],
        words: Callable[[str], str],
        answered: set[str],
        offered: set[str],
        session: str,
    ) -> tuple[str, bool, str | None]:
        """A routed native turn's next prompt, whether it ends, and the stop's name. ``turn`` holds
        the runner's ``prompt``, ``terminal``, its unanswered Host ``coded`` questions,
        ``unavailable`` and ``acted``; ``scripted`` maps each code the script just answered to
        (question, words).
        A Host question takes precedence over the agent's words, which often relay it, and an act
        answers its own turn."""
        for code, (question, said) in sorted(scripted.items()):
            self.scripted(question, said, code=code, session=session)
        prompt, terminal, coded = turn["prompt"], turn["terminal"], turn["coded"]
        if terminal:
            replies = []
            for code in coded:
                reply = self.answer(words(code), code=code, session=session)
                if reply.stop:
                    return prompt, True, reply.stop
                replies.append(str(reply.words))
                answered.add(code)  # a Host that lists it again is QUESTION_REASKED, its defect
                offered.discard(code)
            return "\n".join(replies) or prompt, not replies, None
        quiet = prompt == "Continue." and not (coded or turn["acted"] or turn["unavailable"])
        if quiet and (question := asked(rows)):
            return str(self.answer(question, session=session).words), False, None
        return prompt, terminal, None

    def _sheet(self, question: str, code: str | None) -> dict[str, Any] | None:
        folded = question.casefold()
        for fact in self.facts:
            if code is not None and fact.get("code") == code:
                return fact
            if fact.get("match") and str(fact["match"]).casefold() in folded:
                return fact
        return None

    def _ask(self, qid: str, question: str, code: str | None, session: str) -> str | None:
        """Post the question with its bound; take the first answer written within it."""
        if not self.ask:
            return None
        folder = self.out / "person-asks"
        folder.mkdir(parents=True, exist_ok=True)
        answer, record = folder / f"{qid}.answer.txt", folder / f"{qid}.md"
        record.write_text(
            f"Scenario person question {qid} (code {code or 'none'}, session {session}):\n\n"
            f"> {question}\n\n"
            f"The scenario person is: {self.role or 'the scenario person'}.\n"
            f"Answer in the person's role by writing the words to {answer} within "
            f"{int(self.bound)} seconds. Past that, the person says: {UNKNOWN!r}"
            + ("" if code is None or code in self.nonblocking else " and the run stops, by name")
            + ".\n",
            encoding="utf-8",
            newline="\n",
        )
        posted = self.ask(record, question)
        self.record("posted", id=qid, taken=posted)
        if not posted:
            return None
        deadline = self.clock() + self.bound
        while self.clock() < deadline:
            if answer.is_file() and (words := answer.read_text(encoding="utf-8").strip()):
                return words
            self.sleep(self.interval)
        return None


def asking(out: Path) -> dict[str, Any]:
    """The run's asking, graded and never failed: how many, answered from where, how long they
    waited, and how many the agent could have decided alone (a sheet answer marked avoidable)."""
    path = out / EVENTS
    rows = (
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if path.is_file()
        else []
    )
    answered = [r for r in rows if r["event"] == "answered"]
    return dict(
        questions=sum(r["event"] == "asked" for r in rows),
        by_source=dict(Counter(r["source"] for r in answered)),
        avoidable=sum(bool(r.get("avoidable")) for r in answered),
        waited_s=round(sum(r.get("waited_s") or 0 for r in answered), 1),
        stops=[r["stop"] for r in answered if r.get("stop")],
    )


def routed(
    person: ScenarioPerson,
    decided: dict[str, Any],
    rows: Iterable[dict[str, Any]],
    scenario: dict[str, Any],
    **turn: Any,
) -> tuple[str, bool, str | None]:
    """The native runner's routed turn through the person: ``turn`` carries ``scripted`` (codes
    the script just answered), ``ignore`` (pending markers that are no question), ``words`` (a
    code's words), ``answered``, ``offered``, ``session`` and ``acted``. A product stall ends the
    turn with no question."""
    if decided["terminal"] and decided.get("product_stall"):
        return decided["prompt"], True, None
    replies, codes = scenario["replies"], scenario.get("question_codes", {})
    said = {
        code: (turn["words"](code), str(replies.get(codes.get(code)))) for code in turn["scripted"]
    }
    state = dict(
        prompt=decided["prompt"],
        terminal=decided["terminal"],
        coded=[p for p in decided["unanswered"] if p not in turn["ignore"]],
        unavailable=decided["unavailable"],
        acted=turn["acted"],
    )
    return person.after_turn(
        state,
        rows,
        scripted=said,
        words=turn["words"],
        answered=turn["answered"],
        offered=turn["offered"],
        session=turn["session"],
    )
