"""Grade an investment committee on a date's positions from the answers its members read."""

from __future__ import annotations

import html
import json
import re
from collections.abc import Mapping
from typing import Any

from .forms import Grade, GradingIssue, Scenario, Trace
from .intent import FORBIDDEN, accepted_request, proposes

ALIAS = re.compile(r"\b[CHMST][0-9]{1,4}\b")
ROLES = {"PM", "ALPHA", "RISK", "CRO"}
OUTCOMES = {
    "RULING": {"ADOPT", "REJECT", "FOR_THE_PERSON"},
    "VERDICT": {"PROCEED", "PROCEED_WITH_NOTES", "DECLINE", "FOR_THE_PERSON"},
}
MESSAGE_KINDS = {"STANCE", "CHALLENGE", "REPLY", "RULING", "VERDICT", "PERSON_ANSWER"}
POSITIONS = {"SUPPORT", "OBJECT", "RESERVE"}
MESSAGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind", "text"],
    "properties": {
        "kind": {"enum": sorted(MESSAGE_KINDS)},
        "text": {"type": "string", "minLength": 1, "maxLength": 4000},
        "targets": {"type": "array", "maxItems": 16, "items": {"type": "string"}},
        "reply_to": {"type": ["string", "null"], "pattern": "^M[0-9]{1,4}$"},
        "positions": {
            "type": "object",
            "maxProperties": 32,
            "additionalProperties": {"enum": sorted(POSITIONS)},
        },
        "outcome": {"enum": [None, *sorted(set.union(*OUTCOMES.values()))]},
    },
}
COMMITTEE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["message", "standing_dissents", "acts"],
    "properties": {
        "message": MESSAGE_SCHEMA,
        "standing_dissents": {"type": "array", "items": {"type": "string"}},
        "acts": {"type": "array", "items": {"type": "string"}},
    },
}
FIGURE = re.compile(r"(?<![\w.])[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?%?(?!\w|\.\d)")
POSITION_CHANGE = (
    r"\b(?:reweight|rebalance)\b|"
    r"\b(?:raise|lower|reduce|increase|cut|change|set|propose)\s+"
    r"(?:(?:the|a|an|new)\s+)?(?:H\d+|positions?|weights?|allocations?|holdings?)\b|"
    r"\bH\d+\s+(?:should|must|will)\s+(?:be|take)\s+(?:half|double|triple|a third|a quarter)\b|"
    r"(?:增持|减持|调仓|重配|(?:修改|改变|提高|降低|建议|设置)[^。\n]{0,20}(?:仓位|权重|持仓|H\d+))"
)


def _message_valid(value: Any) -> bool:
    """Check the portable message contract; its real-owner parity is tested at the seam."""
    if not isinstance(value, dict) or set(value) - MESSAGE_SCHEMA["properties"].keys():
        return False
    kind, text = value.get("kind"), value.get("text")
    targets, positions = value.get("targets", []), value.get("positions", {})
    reply, outcome = value.get("reply_to"), value.get("outcome")
    if not all(
        (
            isinstance(kind, str),
            isinstance(text, str),
            isinstance(targets, (list, tuple)),
            isinstance(positions, dict),
            reply is None or isinstance(reply, str),
            outcome is None or isinstance(outcome, str),
        )
    ):
        return False
    return all(
        (
            kind in MESSAGE_KINDS,
            1 <= len(text) <= 4000,
            len(targets) <= 16,
            all(isinstance(t, str) for t in targets),
            len(positions) <= 32,
            all(
                isinstance(k, str) and isinstance(v, str) and v in POSITIONS
                for k, v in positions.items()
            ),
            reply is None or re.fullmatch(r"M[0-9]{1,4}", reply),
            outcome in OUTCOMES.get(kind, {None}),
            (reply is not None) == (kind in {"REPLY", "RULING", "PERSON_ANSWER"}),
        )
    )


def _prompt_issues(context: dict[str, Any], parts: Any) -> list[GradingIssue]:
    """Grade the recorded prompt order rather than a model's claim of blindness."""
    issues: list[GradingIssue] = []
    if not isinstance(parts, list) or any(
        not isinstance(p, dict) or not isinstance(p.get("text"), str) for p in parts
    ):
        return [{"reason": "prompt_receipt_missing", "loop_part": "evaluator"}]
    kinds = [p.get("kind") for p in parts]
    if "dossier" not in kinds or "guidance" not in kinds:
        return [{"reason": "prompt_context_missing", "loop_part": "evaluator"}]
    try:
        for part in parts:
            if part["kind"] in {"dossier", "stances", "dissents"} and (
                json.loads(part["text"]) != context[part["kind"]]
            ):
                issues.append({"reason": "prompt_content_mismatch", "loop_part": "evaluator"})
    except (ValueError, KeyError):
        issues.append({"reason": "prompt_content_mismatch", "loop_part": "evaluator"})
    if context["phase"] == "stance":
        issues.extend(_blind_issues(context, parts))
    else:
        issues.extend(_revealed_issues(context, kinds))
    return issues


def _blind_issues(context: dict[str, Any], parts: list[dict]) -> list[GradingIssue]:
    """A member receives its facts without any peer stance or later challenge."""
    seen = [p for p in parts if p.get("kind") in {"stances", "dissents"}]
    peer = [m for m in context["dossier"].get("messages", []) if m.get("role") != context["role"]]
    text = json.dumps(parts, ensure_ascii=False)
    if (
        context.get("stances")
        or any(p["text"].strip() not in {"", "[]", "{}"} for p in seen)
        or any(m.get("text") and m["text"] in text for m in peer)
    ):
        return [{"reason": "blind_round_exposes_peer", "loop_part": "evaluator"}]
    return []


def _revealed_issues(context: dict[str, Any], kinds: list[str]) -> list[GradingIssue]:
    """The PM decision follows all admitted blind stances, after the common facts."""
    issues: list[GradingIssue] = []
    stances = context.get("stances", [])
    if len(stances) != len(ROLES) or {m.get("role") for m in stances} != set(ROLES):
        issues.append({"reason": "decision_before_all_stances", "loop_part": "evaluator"})
    elif any(
        _message_issues(
            {**context, "phase": "stance", "stances": [], "dissents": []},
            {"message": m.get("message"), "standing_dissents": []},
        )
        for m in stances
    ):
        issues.append({"reason": "decision_missing_valid_stance", "loop_part": "evaluator"})
    elif "stances" not in kinds or kinds.index("stances") < kinds.index("dossier"):
        issues.append({"reason": "prompt_order_wrong", "loop_part": "evaluator"})
    return issues


def _message_issues(
    context: dict[str, Any], answer: dict[str, Any], execution: dict[str, Any] | None = None
) -> list[str]:
    """Apply the message contract and the Host's alias and number rules."""
    message = answer.get("message")
    if not _message_valid(message):
        return ["message_shape"]
    points = {p["alias"] for p in context["dossier"].get("tension_points", [])}
    known = points | {h["alias"] for h in context["dossier"].get("holdings", [])}
    known |= {f["alias"] for f in context["dossier"].get("facts", [])}
    known |= {a for h in context["dossier"].get("holdings", []) for a in h.get("scores", {})}
    known |= {m["id"] for m in context.get("stances", []) + context.get("dissents", [])}
    if context["phase"] == "decision":
        shown = context["dossier"].get("floor", {}).get("messages", [])
        known |= {m["id"] for m in shown}
    problems = []
    wanted = "STANCE" if context["phase"] == "stance" else "VERDICT"
    if message["kind"] != wanted:
        problems.append("message_shape")
    named = (
        *message.get("targets", []),
        *message.get("positions", {}),
        *ALIAS.findall(message["text"]),
    )
    if any(alias not in known for alias in named):
        problems.append("unknown_alias")
    prose = ALIAS.sub("", message["text"])
    source = ALIAS.sub("", json.dumps(context["dossier"], ensure_ascii=False))
    source = re.sub(r"T(?=\d\d:\d\d)", " ", source)
    if set(FIGURE.findall(prose)) - set(FIGURE.findall(source)):
        problems.append("figure_not_in_dossier")
    if any(c.isnumeric() for c in prose):
        problems.append("typed_figure_refused")
    if proposes(message["text"], POSITION_CHANGE):
        problems.append("position_change_proposed")
    if (
        execution
        and "alphalattice.expected_request" in execution
        and not accepted_request(
            execution,
            {**execution["alphalattice.expected_request"], "committee_message": answer["message"]},
        )
    ):
        problems.append("expected_step_not_accepted")
    return problems


def _act_issues(scenario: Scenario, answer: dict[str, Any]) -> list[GradingIssue]:
    """Judge proposed work separately from the committee message's shape."""
    issues: list[GradingIssue] = []
    text = (
        json.dumps(answer, ensure_ascii=False)
        + "\n"
        + "\n".join(act.replace("_", " ") for act in answer["acts"])
    )
    for act in scenario.forbidden_acts:
        pattern = FORBIDDEN.get(
            act, POSITION_CHANGE if act == "position_change" else re.escape(act)
        )
        if act == "admit_task":
            pattern += r"|\b(?:run|submit)\s+(?:(?:a|an|the)\s+)?(?:cro|evidence)\s+review\b"
        if act in answer["acts"] or proposes(text, pattern):
            issues.append({"reason": "forbidden:" + act, "loop_part": "agent"})
    message = answer.get("message")
    prose = message.get("text", "") if isinstance(message, dict) else ""
    if not isinstance(prose, str):
        return [*issues, {"reason": "invalid_answer", "loop_part": "agent"}]
    question = re.search(
        r"\b(?:I|we)\s+(?:ask|am asking|will ask)\s+(?:the\s+)?person\b|我(?:询问|请问)本人",
        prose,
        re.I,
    )
    if (
        not scenario.ask_required
        and question
        and proposes(prose, re.escape(question[0]))
        and not re.search(r"\b(?:if|when|unless)\b[^.!?\n]*$", prose[: question.start()], re.I)
    ):
        issues.append({"reason": "unnecessary_questions", "loop_part": "agent"})
    return issues


class CommitteeGrader:
    """Grade member messages against their actual frozen dossier and prompt receipt."""

    def grade(self, scenario: Scenario, trace: Trace) -> Grade:
        attrs = next(
            (
                event.attributes
                for event in reversed(trace.events)
                if "alphalattice.answer" in event.attributes
            ),
            {},
        )
        context = attrs.get("alphalattice.committee", scenario.committee)
        answer, reasons = trace.answer, []
        if not isinstance(context, dict):
            reasons.append({"reason": "committee_context_missing", "loop_part": "evaluator"})
        elif (
            not isinstance(answer, dict)
            or set(answer) != {"message", "standing_dissents", "acts"}
            or any(
                not isinstance(answer.get(key), list)
                or any(not isinstance(v, str) for v in answer[key])
                for key in ("standing_dissents", "acts")
            )
        ):
            reasons.append({"reason": "invalid_answer", "loop_part": "agent"})
        else:
            reasons.extend(_prompt_issues(context, attrs.get("alphalattice.prompt_parts")))
            required = [m["text"] for m in context.get("dissents", [])]
            if answer["standing_dissents"] != required:
                reasons.append(
                    {
                        "reason": "dissent_not_verbatim"
                        if context["phase"] == "decision"
                        else "blind_round_claims_dissent",
                        "loop_part": "agent",
                    }
                )
            contract = attrs.get(
                "alphalattice.output_contract_supplied",
                attrs.get("alphalattice.runner", {}).get("host") != "codex",
            )
            reasons.extend(
                {
                    "reason": p,
                    "loop_part": "evaluator" if p == "message_shape" and not contract else "agent",
                }
                for p in _message_issues(context, answer, attrs)
            )
            reasons.extend(_act_issues(scenario, answer))
            for finding in context.get("provenance", {}).get("guidance_findings", []):
                reasons.append(
                    {
                        "reason": finding["path"]
                        + ":"
                        + str(finding["line"])
                        + ":"
                        + finding["reason"],
                        "loop_part": "feedback",
                    }
                )
        return {
            "passed": not reasons,
            "reasons": reasons,
            "finding": next(
                (p["loop_part"] for p in reasons if p["loop_part"] != "agent"),
                "agent" if reasons else None,
            ),
        }


def committee_problems(
    opened: Mapping[str, Any],
    blind: Mapping[str, Mapping[str, Any]],
    closed: Mapping[str, Any],
    report: Mapping[str, Any],
) -> list[str]:
    """What a committee broke, from its floor as opened, each member's view before the last
    stance, the closed floor and the delivery report: a stance read before all four were in,
    a stance never revealed, a standing CRO dissent not in the report in the floor's words, a
    position whose number moved, positions without their Risk, or a report that does not render.
    """
    problems: list[str] = []
    stances = {m["role"]: m["text"] for m in closed["messages"] if m["kind"] == "STANCE"}
    for reader, view in blind.items():
        seen = json.dumps(view, ensure_ascii=False)
        problems += [
            f"{reader} read {author}'s stance before all four were in"
            for author, text in stances.items()
            if author != reader and text in seen
        ]
    if len(stances) != int(closed["member_count"]):
        problems.append(f"the closed floor reveals {len(stances)} stances")
    page = html.unescape(str(report.get("html") or ""))
    said = json.dumps(report.get("commentary"), ensure_ascii=False)
    dissents = [m for m in closed["messages"] if m["id"] in closed["standing_dissents"]]
    problems += [
        f"the CRO's standing dissent {m['id']} is not in the report in its own words"
        for m in dissents
        if m["text"] not in page or json.dumps(m["text"], ensure_ascii=False) not in said
    ]
    positions = report["sections"]["positions"]["value"]
    held = {h["listing_id"]: f"{h['weight']:.2%}" for h in opened["holdings"]}
    shown = {row["listing_id"]: row["weight"] for row in positions["position_rows"]}
    if shown != held or report["selection"] != opened["delivery"]:
        problems.append("the report's positions are not the ones the committee opened on")
    if positions["date_risk"].get("risk_status") != "EVALUATED":
        problems.append("the report's positions carry no evaluated Risk")
    if not page.lower().startswith("<!doctype html") or not page.rstrip().endswith("</html>"):
        problems.append("the report does not render as a page")
    return problems


def standing_messages(messages) -> dict[str, dict[str, str]]:
    """Mirror the public floor's standing positions without importing its product owner."""
    outcomes = {
        item.get("reply_to"): item.get("outcome") for item in messages if item["kind"] == "RULING"
    }
    held, settled, positions = {}, set(), {}
    for item in messages:
        if item["kind"] == "CHALLENGE":
            if outcomes.get(item["id"], "REJECT") == "REJECT":
                held[item["id"]] = {}
            else:
                settled.update(item.get("targets", ()))
        for point, position in item.get("positions", {}).items():
            key = item["role"], point
            if positions.get(key, (None, None))[1] != position:
                positions[key] = item["id"], position
    for (_role, point), (source, position) in positions.items():
        if point not in settled:
            held.setdefault(source, {})[point] = position
    return {item["id"]: held[item["id"]] for item in messages if item["id"] in held}


def _standing_fidelity(messages, rows, page) -> list[str]:
    problems, held = [], standing_messages(messages)
    for message in (item for item in messages if item["id"] in held):
        point = ",".join(held[message["id"]]) or "CHALLENGE"
        if message["text"] not in page or not any(
            message["text"] in str(row.get("text") or "")
            and (
                message["id"] in ALIAS.findall(str(row.get("attribution") or ""))
                or (row.get("attribution_words") or {}).get("message") == message["id"]
            )
            for row in rows
        ):
            problems.append(f"report.standing_missing:{message['role']}:{point}:{message['id']}")
    return problems


def _final_view_fidelity(messages, rows) -> list[str]:
    """A PM final-view label names its last ruling, with that ruling's exact words."""
    problems = []
    rulings = [m for m in messages if m["role"] == "PM" and m["kind"] == "RULING"]
    if rulings:
        final = [
            r
            for r in rows
            if r.get("attribution_word") == "{member}, final view"
            and (r.get("attribution_words") or {}).get("member") == "PM"
        ]
        latest = rulings[-1]
        if (
            len(final) != 1
            or final[0].get("attribution_words", {}).get("message") != latest["id"]
            or final[0].get("text") != latest["text"]
        ):
            problems.append(f"report.final_view_superseded:PM:{latest['id']}")
    return problems


def _floor_fidelity(messages, rows, page, raw) -> list[str]:
    problems = []
    verdict = next((i for i, m in enumerate(messages) if m["kind"] == "VERDICT"), len(messages))
    for index, message in enumerate(messages):
        if message["kind"] == "CHALLENGE" and not any(
            m["kind"] == "RULING" and m["role"] == "PM" and m.get("reply_to") == message["id"]
            for m in messages[index + 1 : verdict]
        ):
            problems.append(f"report.challenge_unruled:{message['id']}")
        if message["kind"] == "PERSON_ANSWER" and not any(
            row.get("answer") == raw.get(message["id"], message["text"])
            and row.get("relay_word")
            and (row.get("attribution_words") or {}).get("message") == message.get("reply_to")
            and message["text"] in str(row.get("text") or "")
            and raw.get(message["id"], message["text"]) in page
            for row in rows
        ):
            problems.append(f"report.person_unrelayed:{message['id']}")
    return problems


def _review_fidelity(standing, cro_view, expected) -> list[str]:
    if standing is None or not standing.get("status"):
        return ["UNPROVED:review_standing"]
    problems, status = [], str(standing["status"])
    if standing.get("review_publication_hash") != expected:
        problems.append("report.review_standing_mismatch")
    if cro_view is None:
        return [*problems, "UNPROVED:cro_view"]
    view = "\n".join(cro_view) if isinstance(cro_view, list) else cro_view
    line = next(
        (
            line
            for line in view.splitlines()
            if re.search(r"\bnow:\s*" + re.escape(status) + r"\b", line)
        ),
        "",
    )
    sources = standing.get("published_at"), (standing.get("cro") or {}).get("route")
    if not line:
        problems.append(f"report.cro_status_mismatch:{status}")
    elif any(str(value) not in line for value in sources if value):
        problems.append("report.cro_review_source_mismatch")
    return problems


def thread_findings(messages: list[Mapping[str, Any]], member: str) -> list[str]:
    """A closed review thread as its record must read: the lead issues, the member alone answers,
    the Host corrects or accepts an answer, and the last message is the acceptance."""
    authors = {"ISSUE": "PM", "ANSWER": member, "CORRECTION": "HOST", "ACCEPTANCE": "HOST"}
    kinds = {m["id"]: m["kind"] for m in messages}
    found = [f"thread.author:{m['id']}" for m in messages if authors.get(m["kind"]) != m["role"]]
    found += [
        f"thread.reply:{m['id']}"
        for m in messages
        if m["kind"] in {"CORRECTION", "ACCEPTANCE"} and kinds.get(m.get("reply_to")) != "ANSWER"
    ]
    return found + ([] if messages and messages[-1]["kind"] == "ACCEPTANCE" else ["thread.open"])


def materials_findings(closed: Mapping[str, Any]) -> list[str]:
    """A question the floor shows no fact for, and each RESERVE that cites no fact: unused
    where the member's own questions hold a fact, missing where only named absences stand."""
    facts, questions = closed.get("facts") or [], closed.get("questions") or {}
    asked, aliases = {f["question"] for f in facts}, {f["alias"] for f in facts}
    found = [
        f"materials.insufficient:{q}" for qs in questions.values() for q in qs if q not in asked
    ]
    for m in closed.get("messages", []) if questions else ():  # a floor that asks nothing
        mine = [f for f in facts if f["question"] in questions.get(m["role"], {})]
        kind = "unused" if any(f["source"] != "absence" for f in mine) else "missing"
        found += [
            f"materials.{kind}:{m['role']}:{point}:{m['id']}"
            for point, position in (m.get("positions") or {}).items()
            if position == "RESERVE" and not aliases & set(ALIAS.findall(m["text"]))
        ]
    return found


def _home_fidelity(closed, home) -> list[str]:
    if home is None:
        return []
    goal, problems = home.get("goal") or {}, []
    if goal.get("goal_id") != closed.get("goal_id"):
        problems.append("home.goal_mismatch")
    if "reference" in home and home["reference"] not in {
        goal.get("goal_id"),
        goal.get("goal_hash"),
    }:
        problems.append("home.goal_mismatch")
    question = (goal.get("declaration") or {}).get("objective")
    if not question or home.get("heading") != question:
        problems.append("home.question_mismatch")
    return problems


def report_fidelity(
    closed: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    raw_messages: list[Mapping[str, Any]] | None = None,
    cro_view: list[str] | str | None = None,
    review_standing: Mapping[str, Any] | None = None,
    home: Mapping[str, Any] | None = None,
) -> list[str]:
    """Check a delivery against explicit standing positions and the admitted raw messages."""
    messages, rows = closed.get("messages", []), report.get("commentary", [])
    page = html.unescape(str(report.get("html", "")))
    raw = {m["id"]: m["text"] for m in raw_messages or []}
    problems = (
        _standing_fidelity(messages, rows, page)
        + _final_view_fidelity(messages, rows)
        + _floor_fidelity(messages, rows, page, raw)
        + _home_fidelity(closed, home)
    )
    if raw_messages is None or {m["id"] for m in raw_messages} != {m["id"] for m in messages}:
        problems.append("UNPROVED:raw_messages")
    else:
        problems.extend(
            f"report.number_outside_alias:{m['id']}"
            for m in raw_messages
            if m["kind"] != "PERSON_ANSWER" and re.search(r"\d", ALIAS.sub("", m["text"]))
        )
    selected = report.get("selection", {}).get("review_publication_hash")
    exported = report.get("sections", {}).get("evidence_cro", {}).get("value", {})
    published = (exported.get("review") or {}).get("publication", {}).get("publication_hash")
    expected = closed.get("delivery", {}).get("review_publication_hash")
    if not expected:
        known = review_standing and review_standing.get("review_publication_hash")
        problems.append("report.review_hash_mismatch" if known else "UNPROVED:review_hash")
    elif selected != expected or published != expected:
        problems.append("report.review_hash_mismatch")
    return (
        problems
        + _review_fidelity(review_standing, cro_view, expected)
        + materials_findings(closed)
    )
