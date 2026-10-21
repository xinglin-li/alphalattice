"""Walk every way on an answer offers, as offered, against a served Host; no model.

From each start the explorer sends every request document an answer offers back as it stands,
through the CLI it is given, runs every offered command as typed, and follows each answer to its
end. It grades what an agent would read: an offer its owner refuses, the same failing step again,
a stop with no way on, a wait on a stopped Task, a way on only the Workbench gives, an untyped
failure and the scenario's forbidden acts and route. Each step is one trace event that keeps the
whole answer, for the feedback test that reads the answers an agent decides on.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Iterator
from copy import deepcopy
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
_ENDING = ("CANCEL", "_ABANDON", "_STOP")


def data_decision_findings(
    feed: dict[str, Any], projection: dict[str, Any]
) -> list[dict[str, str]]:
    """The visible Task and rule attribution agree with their data owner, without a successor."""
    shown = projection.get("data_decision", {})
    task_id = feed["task_id"]
    rows = [r for r in shown.get("rows", []) if r.get("id") == task_id]
    runs = [r for r in shown.get("runs", []) if r.get("id") == task_id]
    rules = set()
    if len(rows) != 1 or len(runs) != 1:
        rules.add("data.task_row")
    waiting = (
        feed.get("lifecycle") == "BLOCKED"
        and feed.get("failure_code") == "data.truth_review_required"
    )
    expected = (
        ("data_decision_pending", "warning", "Waiting for a decision")
        if waiting
        else ("succeeded", "good", "Completed")
        if feed.get("lifecycle") == "SUCCEEDED"
        else None
    )
    if expected and (
        any((r.get("status"), r.get("tone"), r.get("word")) != expected for r in rows)
        or any(str(r.get("state")).lower() != expected[0] for r in runs)
    ):
        rules.add("data.decision_state")
    by_case = {r["case_token"]: r for r in shown.get("rule_labels", [])}
    if any(
        (
            by_case.get(r["case_token"], {}).get("label"),
            by_case.get(r["case_token"], {}).get("word"),
        )
        != ("", "Decided by rule")
        for r in feed.get("rule_decisions", [])
    ):
        rules.add("data.rule_disclosure")
    return [
        {
            "rule": rule,
            "loop_part": "feedback",
            "owner": "assets/workbench-source/js/app/live-workspace.js"
            if rule == "data.rule_disclosure"
            else "assets/workbench-source/js/app/data.js",
        }
        for rule in sorted(rules)
    ]


def evidence_reuse_findings(store: Path, before: frozenset[str]) -> list[dict[str, str]]:
    """A later day reads no filing an earlier analysis read, nor scores a selection over one.

    The filing is the unit of reuse (W3). `store` is the Evidence artifact root and `before`
    the records it held before the day; the day's own records are the rest.
    """

    def load(record: str) -> dict[str, Any]:
        return dict(json.loads((store / record).read_text(encoding="utf-8")))

    def filings(document_set: str) -> set[tuple[str, str]]:
        held = load(f"document-sets/{document_set}.json")["documents"]
        return {(d["entity_id"], d["revision_label"]) for d in held}

    def read(records: Any) -> set[tuple[str, str]]:
        sets = [load(r)["document_set_hash"] for r in records if r.startswith("analysis-pub")]
        return set().union(*map(filings, sets))

    today = {p.relative_to(store).as_posix() for p in store.rglob("*.json")} - before
    earlier = read(before)
    receipts = [load(r) for r in today if r.startswith("retrieval-access-receipts/")]
    failures = {
        "daily.evidence_reread": bool(read(today) & earlier),
        "daily.evidence_rescored": any(
            r.get("reused_from_receipt_hash") is None and filings(r["document_set_hash"]) <= earlier
            for r in receipts
        ),
    }
    owner = "evidence/alternative_evidence/runtime/task_adapter.py"
    return [
        {"rule": rule, "loop_part": "reuse", "owner": owner} for rule, f in failures.items() if f
    ]


def daily_findings(feed: dict[str, Any], projection: dict[str, Any]) -> list[dict[str, str]]:
    """Today's rendered outcome, members and decisions preserve the previous sealed day."""
    from .session import session_findings

    current, home = feed["current"], projection.get("home", {})
    selected, value = home.get("selection") or {}, home.get("value") or {}
    sessions, previous = projection.get("session", {}), projection.get("previous") or {}
    shown = {m["id"]: m for s in sessions.get("sessions", []) for m in s.get("members", [])}
    sealed = (
        "task_id",
        "publication",
        "review_selector",
        "position_rows",
        "exit_rows",
        "schedule",
        "date_risk",
    )
    failures = {
        "daily.home_stale": selected.get("task_id") != current["task_id"],
        "daily.home_publication": value.get("review_selector") != current["review_selector"],
        "daily.home_date": value.get("schedule") != current.get("schedule"),
        "daily.home_positions": not any(
            k.startswith("positions:" + current["task_id"] + ":") for k in home.get("rows", [])
        ),
        "daily.previous_changed": any(previous.get(k) != feed["previous"].get(k) for k in sealed),
        "daily.previous_report_changed": bool(feed.get("previous_report"))
        and (projection.get("previous_report") or {}).get("export_hash")
        != feed["previous_report"].get("export_hash"),
        "daily.data_disclosure": set(projection.get("data_cases", []))
        != set(feed.get("rule_cases", [])),
        "daily.member_missing": bool(set(feed.get("members", [])) - shown.keys()),
        "daily.usage_missing": any(
            not shown.get(sid, {}).get("usage_visible") or not shown.get(sid, {}).get("usage")
            for sid in feed.get("usage_members", [])
        ),
    }
    owners = {
        "daily.data_disclosure": "live-data.js",
        "daily.member_missing": "team-members.js",
        "daily.usage_missing": "team-members.js",
    }
    findings = [
        {
            "rule": rule,
            "loop_part": "feedback",
            "owner": "assets/workbench-source/js/app/" + owners.get(rule, "data.js"),
        }
        for rule, failed in failures.items()
        if failed
    ]
    if feed.get("session"):
        findings += session_findings(feed["session"], sessions)
    return findings


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


def _way(answer: dict[str, Any]) -> str:
    """How a step stood: its outcome, stop and the names of the ways on it offered."""
    names = sorted(name for name, _ in offers(answer)) + sorted(n for n, _ in commands(answer))
    code = answer.get("failure_code") or _data(answer).get("failure_code")
    return json.dumps([answer.get("outcome"), code, _state(answer), names])


def _unhashed(request: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in request.items() if not key.endswith("_hash")}


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


def mutations(
    request: dict[str, Any], required: list[str]
) -> Iterator[tuple[str, dict[str, Any], str]]:
    """Single deterministic edits of a live offer; no authority is widened."""
    for key in sorted(k for k in request if k != "operation" and request[k] is not None):
        value = request[key]
        if key in required:
            yield "drop-field", {k: v for k, v in request.items() if k != key}, key
        if isinstance(value, str) and re.search(r"\b[HTM]\d+\b", value):
            alias = re.sub(r"\b[HTM](\d+)\b", r"\1", value, count=1)
            yield "alias-number", {**request, key: alias}, key
        if not isinstance(value, str) or key.endswith(("_id", "_hash", "_session")):
            yield "value-change", {**request, key: "?"}, key
        if key.endswith("_id"):
            yield "wrong-id", {**request, key: "00000000-0000-0000-0000-000000000000"}, key
    if isinstance(message := request.get("committee_message"), dict):
        for kind, values, rule in (
            # A message that answers none, with its reply_to sent as an empty list.
            *[("empty-reply", {"reply_to": []}, "reply_to")] * (message.get("reply_to", 0) is None),
            ("shape", {"outcome": "ADOPT", "reply_to": None}, "SHAPE"),
            ("target", {"targets": ["H9999"]}, "TARGET"),
            ("reply", {"reply_to": "M9999"}, "REPLY_TO"),
            (
                "alias-number",
                json.loads(
                    re.sub(
                        r"\b[HTM](\d+)\b", r"\1", json.dumps(request["committee_message"]), count=1
                    )
                ),
                "NUMBER",
            ),
        ):
            changed = deepcopy(request)
            changed["committee_message"].update(values)
            yield kind, changed, rule


def _refusal_rule(answer: dict[str, Any], expected: str) -> tuple[bool, str, bool]:
    """Name the actual typed rule separately from the words explaining it."""
    data = _data(answer)
    code = str(answer.get("failure_code") or data.get("failure_code") or "")
    reasons = json.dumps(answer.get("reasons") or data.get("reasons") or {}, ensure_ascii=False)
    fields = (answer.get(k) or data.get(k) for k in ("detail", "message", "reasons", "usage_error"))
    words = json.dumps(list(fields), ensure_ascii=False)
    generic = re.search(
        r"request_invalid|invalid_request|usage_invalid|\.refused|cli_failed|handler_failed|unexpected_failure",
        code,
    )
    located = re.search(r"(?:\w+\.)+\w+(?::[A-Z_]+)?", reasons)
    typed = (located.group() if located else "") if generic else code
    suffix = typed.partition(":")[2]
    actual = suffix if suffix == expected or re.fullmatch(r"[A-Z_]+", suffix) else typed or expected
    named = bool(typed) or expected in (reasons if "request_invalid" in code else words)
    stated = actual.casefold() in words.casefold() or (
        (answer.get("rule") or data.get("rule")) == actual
        and bool(answer.get("detail") or data.get("detail"))
    )
    return named, actual, stated


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
    steps: int = 400
    """The walk's step budget: past it, every path ends as a horizon."""
    events: list[TraceEvent] = field(default_factory=list)
    observer: RoundTripObserver = field(default_factory=RoundTripObserver)
    sent: set[str] = field(default_factory=set)
    """Every request and command this walk has sent: a step's subtree is walked once."""
    again: set[str] = field(default_factory=set)
    """Each failed step sent once more where it was offered again, to see whether it moved."""
    request_fields: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    mutation_owners: dict[str, str] = field(default_factory=dict)
    read_operations: set[str] = field(default_factory=set)
    mutation_budget: float = 0.0
    mutation_limit: int | None = None
    mutation_seconds: float = 0.0
    mutation_checks: list[dict[str, Any]] = field(default_factory=list)
    mutation_kinds: set[str] | None = None
    findings: list[dict[str, Any]] = field(default_factory=list)
    mutated: set[str] = field(default_factory=set)
    observe: Callable[[dict[str, Any]], None] = lambda _: None
    terminal: Callable[[dict[str, Any]], bool] = lambda _: False
    terminal_reached: bool = False

    @property
    def _mutating(self) -> bool:
        return bool(self.mutation_budget if self.mutation_limit is None else self.mutation_limit)

    def _mutation_room(self, elapsed: float = 0.0) -> bool:
        if self.mutation_limit is not None:
            return len(self.mutation_checks) < self.mutation_limit
        return self.mutation_seconds + elapsed < self.mutation_budget

    def inject(
        self,
        request: dict[str, Any],
        where: str,
        baseline: dict[str, Any] | None = None,
        kinds: set[str] | list[str] | None = None,
        fields: set[str] | None = None,
    ) -> dict[str, Any] | None:
        """Probe offered-request mistakes; cancelled writes are findings before any CLI call."""
        if not self._mutation_room():
            return None
        started = time.monotonic()
        operation = str(request["operation"])
        if self._cancelled_write(request, baseline) and (kinds is None or "duplicate" in kinds):
            row, _ = self._mistake_answer(
                request, baseline, "duplicate", "CANCELLED", where, baseline
            )
            row["reason"] = "automatic resend after CANCELLED"
            self.mutation_checks.append(row)
            self.findings.append({**row, "loop_part": "agent"})
            return None
        required = self.request_fields.get(operation, {}).get("required", [])
        variants = [v for v in mutations(request, required) if v[1] != request]
        variants += [("misspell-flag", request, "--flie")]
        if baseline is not None:
            variants = [("duplicate", request, "same admission")]
        elif kinds is None and variants:
            state = re.sub(r"\b[\da-f]{8}(?:-[\da-f]{4}){3}-[\da-f]{12}\b", "<id>", where)
            key = f"{self.scenario.id}|{state}|{operation}|{','.join(sorted(required))}"
            index = int(hashlib.sha256(key.encode()).hexdigest(), 16) % len(variants)
            variants = [variants[index]]
        for kind, changed, rule in (
            v
            for v in variants
            if (kinds is None or v[0] in kinds) and (fields is None or v[2] in fields)
        ):
            if not self._mutation_room(time.monotonic() - started):
                break
            document = self.work / f"mistake-{len(self.events)}.json"
            document.write_text(json.dumps(changed), encoding="utf-8")
            flag = "--flie" if kind == "misspell-flag" else "--file"
            answer = self._send(["request", flag, str(document)], where + " > " + kind)
            self.events[-1].attributes["alphalattice.mutation"] = kind
            row, corrected = self._mistake_answer(request, answer, kind, rule, where, baseline)
            self.mutation_checks.append(row)
            if row["reason"]:
                self.findings.append({**row, "loop_part": "feedback"})
            if corrected is not None:
                self.mutation_seconds += time.monotonic() - started
                return corrected
        self.mutation_seconds += time.monotonic() - started
        return None

    def _cancelled_write(self, request: dict[str, Any], answer: dict[str, Any] | None) -> bool:
        """A duplicate write would contradict cancellation; reads and cancelling again do not."""
        return (
            answer is not None
            and _state(answer) == "CANCELLED"
            and request["operation"] not in self.read_operations
            and request["operation"] != "CANCEL"
        )

    def _mistake_answer(
        self,
        request: dict[str, Any],
        answer: dict[str, Any],
        kind: str,
        rule: str,
        where: str,
        baseline: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """A refusal names its rule and recovery; a duplicate preserves its admission."""
        operation = str(request["operation"])
        named, actual, words_state_rule = _refusal_rule(answer, rule)
        refused = answer.get("outcome") not in {"OK", "PENDING"}
        reason, corrected = None, None
        if not refused:
            if kind != "duplicate":
                reason = "changed request accepted without equivalence proof"
        elif not named:
            reason = "refusal loses the broken rule"
        else:
            content = kind in {"drop-field", "alias-number", "value-change", "wrong-id", "target"}
            reason, corrected = self._correction(
                request, answer, where, content and words_state_rule, baseline
            )
        if kind == "duplicate" and (not refused or corrected is not None):
            before = _data(baseline or {})
            keys = ("task_id", "message_id", "plan_hash", "update_plan_hash", "result_hash")
            identities = {k: before[k] for k in keys if before.get(k) is not None}
            repeated = _data(corrected or answer)
            same = identities and all(repeated.get(k) == v for k, v in identities.items())
            if operation not in self.read_operations and not same:
                reason, corrected = "duplicate admission differs or has no identity proof", None
        row = {
            "state": where,
            "mutation": kind,
            "answer": answer,
            "rule_named": named,
            "actual_rule": actual,
            "words_state_rule": words_state_rule,
            "owner": self.mutation_owners.get(
                "syntax" if kind == "misspell-flag" else operation, operation
            ),
            "reason": reason,
        }
        return row, corrected

    def _discovered_binding(
        self, original: dict[str, Any], request: dict[str, Any], sent: dict[str, Any]
    ) -> bool:
        """Discovery proves an exact returned plan or completed update, never a latest guess."""
        if request["operation"] == "WORKSPACE_SHOW" and original.get("strategy_package_id"):
            return sent.get("operation") == "WORKSPACE_SHOW" and any(
                item.get("strategy_id") == original["strategy_package_id"]
                for item in _data(sent).get("installed_strategies", [])
                if isinstance(item, dict)
            )
        if request["operation"] == "ACTIVITY_LIST" and original["operation"] in {
            "DATA_UPDATE_RUN",
            "RESEARCH_UPDATE_RUN",
        }:
            return any(
                payload.get("phase") == "RETURNED"
                and payload.get("status") == "PLANNED"
                and payload.get("operation") == original["operation"].removesuffix("_RUN") + "_PLAN"
                and original.get("update_plan_hash")
                and original["update_plan_hash"]
                in {subject.get("plan_hash"), subject.get("update_plan_hash")}
                for item in _data(sent).get("items", [])
                if isinstance(item, dict)
                and isinstance(payload := item.get("payload"), dict)
                and isinstance(subject := payload.get("subject"), dict)
            )
        return request["operation"] == "TASKS" and self._listed_task_binding(original, sent)

    def _listed_task_binding(self, original: dict[str, Any], sent: dict[str, Any]) -> bool:
        """Task discovery preserves the completed committee source or recovery identity."""
        if str(original["operation"]).startswith("COMMITTEE_"):
            return any(
                task.get("task_id") == original.get("update_task_id")
                and original.get("update_task_id")
                and task.get("task_kind") == "portfolio_public_watermark_advancement"
                and task.get("lifecycle") == "SUCCEEDED"
                for task in _data(sent).get("tasks", [])
                if isinstance(task, dict)
            )
        if original["operation"] in {"TASK_RECOVERY", "RECOVER"} or original.get(
            "recovery_task_id"
        ):
            task_id = original.get("recovery_task_id") or original.get("task_id")
            return any(
                task.get("task_id") == task_id
                and task_id
                and (
                    not original.get("recovery_task_hash")
                    or task.get("task_record_hash") == original["recovery_task_hash"]
                )
                for task in _data(sent).get("tasks", [])
                if isinstance(task, dict)
            )
        return False

    def _bound_read(
        self,
        original: dict[str, Any],
        request: dict[str, Any],
        sent: dict[str, Any],
        baseline: dict[str, Any] | None = None,
    ) -> bool:
        """A recovery read exposes the same owner's original selector context."""
        if request["operation"] == "OPERATION_LIST":
            schema = _data(sent).get("operation_schema") or {}
            return all(
                (
                    baseline is None,
                    request.get("schema_operation") == original["operation"],
                    schema.get("properties", {}).get("operation", {}).get("const")
                    == original["operation"],
                    set(original) <= set(schema.get("properties", {})),
                )
            )
        if not baseline and self._discovered_binding(original, request, sent):
            return True
        if self._recovery_source(original, request, sent):
            return (
                any(offer == original for _, offer in offers(sent)) or _state(sent) == "SUCCEEDED"
            )
        bindings = {
            k: v
            for k, v in original.items()
            if k.endswith(("_id", "_hash")) and request.get(k) == v
        }
        owner = self.mutation_owners.get(str(original["operation"]))
        bound = bool(
            owner
            and bindings
            and owner == self.mutation_owners.get(str(request["operation"]))
            and all(_data(sent).get(k) == v for k, v in bindings.items())
        )
        if baseline:
            if request["operation"] == "TASK_RECOVERY":
                return bound and _data(sent).get("task_id") == _data(baseline).get("task_id")
            return bool(
                bound
                and request["operation"] == "COMMITTEE_READ"
                and any(
                    item.get("id") == _data(baseline).get("message_id")
                    for item in _data(sent).get("messages", [])
                )
            )
        return bound

    @staticmethod
    def _recovery_source(
        original: dict[str, Any], request: dict[str, Any], sent: dict[str, Any]
    ) -> bool:
        """A recovery read proves its source only when both IDs match the original offer."""
        task_id = original.get("recovery_task_id")
        return bool(
            task_id
            and request["operation"] == "TASK_RECOVERY"
            and request.get("task_id") == task_id
            and _data(sent).get("task_id") == task_id
        )

    @staticmethod
    def _correction_offer(
        original: dict[str, Any], answer: dict[str, Any], edited: bool
    ) -> tuple[str, dict[str, Any]] | None:
        """Prefer the original request, then task discovery over unrelated history reads."""
        candidates = [(n, r) for n, r in offers(answer) if not _CHOICE.search(json.dumps(r))]
        if edited:
            return "agent-edit", original
        if not candidates:
            return None
        discovery = {"TASKS"} if any(k.endswith("task_id") for k in original) else set()
        return min(
            candidates,
            key=lambda p: (
                0
                if p[1] == original
                else 1
                if p[1]["operation"] in discovery or p[0] == "valid_values"
                else 2
            ),
        )

    def _accepted_correction(
        self, original: dict[str, Any], request: dict[str, Any], sent: dict[str, Any]
    ) -> bool:
        """Reading the exact stopped Task succeeds as a read even when its work was refused."""
        if sent.get("outcome") in {"OK", "PENDING"}:
            return True
        return (
            request == original
            and request["operation"] in self.read_operations
            and sent.get("operation") == request["operation"]
            and _data(sent).get("read_request") == request
            and _state(sent) in STOPPED
            and self._bound_read(original, request, sent)
        )

    def _correction(
        self,
        original: dict[str, Any],
        answer: dict[str, Any],
        where: str,
        edit: bool = False,
        baseline: dict[str, Any] | None = None,
    ) -> tuple[str | None, dict[str, Any] | None]:
        """An offered read is an aid; only the corrected bound request ends recovery."""
        seen: set[str] = set()
        read = False
        for _ in range(2):
            offer = self._correction_offer(original, answer, edit and read)
            if offer is None:
                return "refusal has no filled way back", None
            name, request = offer
            if request != original and request["operation"] not in self.read_operations:
                return "correction re-admits wrong work", None
            key = json.dumps(request, sort_keys=True)
            if key in seen:
                return "offered correction loops", None
            seen.add(key)
            correct = request == original
            document = self.work / f"correction-{len(self.events)}.json"
            document.write_text(json.dumps(request), encoding="utf-8")
            sent = self._send(["request", "--file", str(document)], where + " > correct:" + name)
            if request["operation"] == "TASK_RECOVERY" and original.get("recovery_task_id"):
                sent = self._settled(sent, where + " > source") or sent
            self.events[-1].attributes["alphalattice.mutation"] = "correction"
            if self._accepted_correction(original, request, sent):
                if correct:
                    return None, sent
                if baseline and self._bound_read(original, request, sent, baseline):
                    return None, baseline
                if sent.get("outcome") == "PENDING" and not self._bound_read(
                    original, request, sent
                ):
                    return "correction re-admits wrong work", None
                if edit and not self._bound_read(original, request, sent):
                    return "read does not expose the original bound context", None
                if self._recovery_source(original, request, sent) and _state(sent) == "SUCCEEDED":
                    self.events[-1].attributes["alphalattice.retired_offer"] = True
                    return None, sent
                read = True
            elif _way(sent) == _way(answer):
                return "offered correction repeats refusal", None
            answer = sent
        return "way back never reaches the corrected bound request", None

    def explore(self, starts: list[list[str]]) -> tuple[Grade, Trace]:
        """Walk every path from each start; the grade names each broken path and its part."""
        for argv in starts:
            if self.terminal_reached:
                break
            self._follow(self._send(argv, " ".join(argv)), (" ".join(argv),), {})
        trace = Trace(tuple(self.events))
        text = "\n".join(
            json.dumps(e.attributes, ensure_ascii=False)
            for e in self.events
            if "alphalattice.mutation" not in e.attributes
        )
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

    def _follow(self, answer: dict[str, Any], path: tuple[str, ...], seen: dict[str, str]) -> None:
        where = " > ".join(path)
        settled = self._settled(answer, where)
        if settled is None:
            return  # the Task outlived the scenario's time box: a horizon, not a failure
        self.observe(settled)
        code = str(settled.get("failure_code") or "")
        stops = self._grade(settled, where, code)
        if self._horizon(path, code, settled):
            return
        if stops and (heal := self.heal) is not None:
            self.heal = None
            heal()
        # A fresh hash is the same step and another Task's is not; a choice to fill or an end
        # of work is no way on.
        steps: list[tuple[str, str, dict[str, Any] | str]] = [
            (name, f"{name}|{json.dumps(_unhashed(request), sort_keys=True)}", request)
            for name, request in offers(settled)
            if not _CHOICE.search(json.dumps(request))
            and "choices" not in request
            and not str(request["operation"]).endswith(_ENDING)
        ]
        steps += [(name, line, line) for name, line in commands(settled)]
        decided: set[str] = set()
        for name, key, step in steps:
            # One answer per decision (`<kind>:<id>:<choice>`), and each request walked once.
            decision, sent_as = name.rpartition(":")[0], json.dumps(step, sort_keys=True)
            if key in seen:
                if (sent := self._again(name, step, where, seen[key])) is None:
                    continue
            elif (
                (decision.count(":") and decision in decided)
                or sent_as in self.sent
                or (sent := self._step(name, step, where)) is None
            ):
                continue
            decided.add(decision)
            self.sent.add(sent_as)
            self._confirm_probe(settled, sent, step, where)
            failed = sent.get("outcome") in {"REFUSED", "FAILED"} or _state(sent) in STOPPED
            # A decision taken moves the state: the same Task stopping again after it is no loop.
            after = {} if decision.count(":") else seen
            self._follow(sent, (*path, name), {**seen, key: _way(sent)} if failed else after)

    def _confirm_probe(
        self,
        answer: dict[str, Any],
        sent: dict[str, Any],
        step: dict[str, Any] | str,
        where: str,
    ) -> None:
        """A refused plan cannot offer a confirmation that re-admits its work."""
        if (
            self._mutating
            and answer.get("outcome") == "REFUSED"
            and sent.get("outcome") == "PENDING"
            and isinstance(step, dict)
            and str(step["operation"]).endswith("_CONFIRM")
        ):
            operation = str(step["operation"])
            named, actual, words_state_rule = _refusal_rule(answer, operation)
            self.findings.append(
                {
                    "state": where,
                    "mutation": "refused-confirm",
                    "answer": answer,
                    "correction": sent,
                    "rule_named": named,
                    "actual_rule": actual,
                    "words_state_rule": words_state_rule,
                    "owner": self.mutation_owners.get(operation, operation),
                    "reason": "refused plan offers a confirmation that re-admits",
                    "loop_part": "feedback",
                }
            )

    def _again(
        self, name: str, step: dict[str, Any] | str, where: str, failed: str
    ) -> dict[str, Any] | None:
        """A failed step offered again, sent once more: a loop when it fails the same way, and
        followed when it moved since (another stop, another way on)."""
        sent_as = json.dumps(step, sort_keys=True)
        if sent_as in self.again or (sent := self._step(name, step, where)) is None:
            return None
        self.again.add(sent_as)
        if _way(sent) != failed:
            return sent
        self._problem(f"`{name}` offered again after it failed the same way at {where}", "feedback")
        return None

    def _horizon(self, path: tuple[str, ...], code: str, answer: dict[str, Any]) -> bool:
        """A cancellation, step limit, the walk's budget, or a way on waiting for its time."""
        lead = _data(answer).get("next_action") or answer.get("next_action")
        if _state(answer) == "CANCELLED" and lead != "READ_THE_TASK_WITH_TASK_SHOW":
            if lead not in (None, "REPORT_THE_STOP"):
                self._problem(
                    f"cancelled Task directs automatic continuation at {' > '.join(path)}",
                    "feedback",
                )
            return True
        timed = "not_due" in code and "retry_after_at" in json.dumps(answer)
        self.terminal_reached |= self.terminal(answer)
        return self.terminal_reached or len(path) >= DEPTH or self._spent() or timed

    def _spent(self) -> bool:
        """Whether the walk's step budget is spent: every step but a mutation counts."""
        return sum("alphalattice.mutation" not in e.attributes for e in self.events) >= self.steps

    def _settled(self, answer: dict[str, Any], where: str) -> dict[str, Any] | None:
        """The answer once its admitted Task has ended or stopped; None past the time box.

        A wait that reaches its time box while the Task goes on is waited again, as an agent
        does, until the walk's step budget is spent.
        """
        task = _data(answer).get("task_id")
        if answer.get("outcome") != "PENDING" or not task or _state(answer) in STOPPED:
            return answer
        waited = ["activity", "wait", "--task", str(task), "--max-wait", str(self.time_box)]
        while True:
            ended = self._send(waited, where + " > wait")
            if ended.get("outcome") != "PENDING" or ended.get("failure_code"):
                return ended
            event = (_data(ended).get("wait_event") or {}).get("event")
            if event != "MAX_WAIT_REACHED" or self._spent():
                return None

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
        if self.terminal_reached:
            return None
        if isinstance(step, str):
            return self._send([step], where + f" > {name}", run=lambda argv: self.shell(argv[0]))
        document = self.work / f"offer-{len(self.events)}.json"
        document.write_text(json.dumps(step), encoding="utf-8")
        corrected = None
        path = where + " > " + name
        if self._mutating and json.dumps(step, sort_keys=True) not in self.mutated:
            self.mutated.add(json.dumps(step, sort_keys=True))
            corrected = self.inject(step, path, kinds=self.mutation_kinds)
        if corrected is not None:
            self.events[-1].attributes.pop("alphalattice.mutation", None)
            if not self.events[-1].attributes.get("alphalattice.retired_offer"):
                self.events[-1].attributes["alphalattice.corrected_original"] = True
        sent = corrected or self._send(["request", "--file", str(document)], path)
        if (
            self._mutating
            and sent.get("outcome") not in {"REFUSED", "FAILED"}
            and not self._cancelled_write(step, sent)
            and not self.events[-1].attributes.get("alphalattice.retired_offer")
        ):
            self.inject(step, path, baseline=sent)
        refused = str(sent.get("failure_code") or "").partition(":")[0]
        if any(mark in refused for mark in ("not_offered", "_invalid", "operation_field")):
            self._problem(
                f"`{name}` offered at {where} is refused as it stands: {refused}", "feedback"
            )
            return None
        return sent
