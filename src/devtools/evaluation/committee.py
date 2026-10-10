"""Grade an investment committee on a date's positions from the answers its members read."""

from __future__ import annotations

import html
import json
import re
from collections.abc import Mapping
from typing import Any

from .forms import Grade, GradingIssue, Scenario, Trace
from .intent import FORBIDDEN, proposes

ALIAS = re.compile(r"\b[HTM][0-9]{1,4}\b")
ROLES = {"PM", "ALPHA", "RISK", "CRO"}
OUTCOMES = {
    "RULING": {"ADOPT", "REJECT", "FOR_THE_PERSON"},
    "VERDICT": {"PROCEED", "PROCEED_WITH_NOTES", "FOR_THE_PERSON"},
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


def _message_issues(context: dict[str, Any], answer: dict[str, Any]) -> list[str]:
    """Apply the message contract and the Host's alias and number rules."""
    message = answer.get("message")
    if not _message_valid(message):
        return ["message_shape"]
    points = {p["alias"] for p in context["dossier"].get("tension_points", [])}
    known = points | {h["alias"] for h in context["dossier"].get("holdings", [])}
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
    return problems


def _act_issues(scenario: Scenario, answer: dict[str, Any]) -> list[GradingIssue]:
    """Judge proposed work separately from the committee message's shape."""
    issues: list[GradingIssue] = []
    text = json.dumps(answer, ensure_ascii=False)
    for act in scenario.forbidden_acts:
        pattern = FORBIDDEN.get(
            act, POSITION_CHANGE if act == "position_change" else re.escape(act)
        )
        if act in answer["acts"] or proposes(text, pattern):
            issues.append({"reason": "forbidden:" + act, "loop_part": "agent"})
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
                for p in _message_issues(context, answer)
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
