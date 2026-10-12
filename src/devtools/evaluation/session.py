"""Grade the Session page's projection against its admitted observation feed."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime
from html import unescape

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
HOST_ROLE = "alphalattice_host"


def _visible(shown, cells):
    rules = set()
    host = shown["role_words"].get(HOST_ROLE)
    if host and any(host in r[0] for r in cells if r):
        rules.add("session.host_agent")
    generic = sum(
        int(n.replace(",", ""))
        for r in cells
        if len(r) > 1
        for n in re.findall(r"\b([\d,]+) messages?\b", r[1])
    )
    undeclared = sum(
        e.get("kind") == "message" and not e.get("message_kind") for e in shown["entries"]
    )
    if generic > undeclared:
        rules.add("session.activities")
    names = [m["name"] for m in shown["members"] if not m.get("roles")]
    if len(names) != len(set(names)):
        rules.add("session.id_collision")
    return rules


def _roles(subjects, shown, cells):
    roles = defaultdict(set)
    for _, s in subjects:
        if s.get("message_kind") and s.get("role") not in {None, HOST_ROLE}:
            roles[s.get("native_agent_id")].add(s["role"])
    words = shown["role_words"]
    for admitted in roles.values():
        labels = {words[r] for r in admitted if r in words}
        if len(labels) > 1 and any(all(v in row[0] for v in labels) for row in cells if row):
            return {"session.members_collapsed"}
    return set()


def _recipient(subjects, shown):
    admitted = {i["observation_id"]: s for i, s in subjects}
    for entry in shown["entries"]:
        issue = admitted.get(entry["id"], {})
        if (
            issue.get("message_kind") == "issue"
            and issue.get("recipient_id")
            and entry.get("recipient_label") == shown["unidentified"]
        ):
            return {"session.recipient"}
    return set()


def _goal(subjects, shown, goals):
    bound = [
        s.get("goal_id")
        for _, s in subjects
        if s.get("message_kind") == "session_bound"
        and s.get("input_channel") == "PRODUCT_OPERATION"
        and s.get("native_agent_id") == shown["id"]
        and _UUID.fullmatch(s.get("goal_id") or "")
    ]
    delivered = [
        s.get("goal_id")
        for _, s in subjects
        if (
            s.get("message_kind") == "answer"
            and s.get("input_channel") == "PRODUCT_ACCEPTED_ANSWER"
        )
        or (
            s.get("message_kind") == "acceptance"
            and s.get("input_channel") == "PRODUCT_REVIEW"
            and s.get("recipient_id")
        )
        or s.get("message_kind") == "verdict"
    ]
    scope = bound[-1] if bound else delivered[-1] if delivered else None
    expected = goals.get(scope, {}).get("question")
    if expected and (shown.get("title") != expected or shown.get("question") != expected):
        return {"session.title"}
    return set()


def _span(subjects, shown, times):
    active = set()
    for item, sub in subjects:
        if sub.get("message_kind") or sub.get("native_hook_event"):
            active.add(sub.get("native_agent_id"))
        if sub.get("message_kind") or sub.get("native_hook_event"):
            times.append(item["occurred_at"])
    if not times:
        return set()
    axis = [datetime.fromisoformat(t) for t in times]
    start, end = min(axis), max(axis)
    for member in shown["members"]:
        first = member.get("first_at")
        if (
            first
            and member["id"] != shown["id"]
            and member["id"] not in active
            and not start <= datetime.fromisoformat(first) <= end
        ):
            return {"session.outside_span"}
    return set()


def _usage(feed, shown):
    latest = {}
    records = [
        item["payload"].get("subject", {})
        for item in feed["items"]
        if (item.get("payload") or {}).get("event_kind") == "NATIVE_AGENT_USAGE"
    ] + feed.get("native_usage", [])
    for record in records:
        if record.get("native_session_id") == shown["id"] and record.get("model"):
            latest[record.get("native_agent_id"), record["model"]] = record
    members = {member["id"]: member for member in shown["members"]}
    fields = (
        "responses",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
    )
    for (actor, model), expected in latest.items():
        if actor not in members:
            continue  # Session-wide readings do not manufacture participation in its span.
        actual = next((u for u in members[actor].get("usage", []) if u["model"] == model), None)
        if (
            actual is None
            or not members[actor].get("usage_visible")
            or any(
                str(actual.get(key)) != str(expected[key])
                for key in fields
                if expected.get(key) is not None
            )
        ):
            return {"session.usage"}
    return set()


def _references(shown):
    return (
        {"session.reference"}
        if any(
            not ref.get("resolved") or ref.get("kind") != ref.get("resolved_kind")
            for ref in shown.get("references", [])
        )
        else set()
    )


def _retained(subjects, shown):
    kinds = {
        "NATIVE_COORDINATION_MESSAGE",
        "NATIVE_SUBAGENT_START_HOOK",
        "NATIVE_SUBAGENT_STOP_HOOK",
    }
    actors, entries = set(), set()
    for item, sub in subjects:
        if (item.get("payload") or {}).get("event_kind") not in kinds:
            continue
        if item.get("availability") not in {None, "AVAILABLE"}:
            continue
        if shown.get("goal_scope") and sub.get("goal_id") != shown["goal_scope"]:
            continue
        entries.add(item["observation_id"])
        if sub.get("native_agent_id") and sub.get("role") != HOST_ROLE:
            actors.add(sub["native_agent_id"])
    rules = set()
    if not actors <= {member["id"] for member in shown["members"]}:
        rules.add("session.members_missing")
    if not entries <= {entry["id"] for entry in shown["entries"]}:
        rules.add("session.exchanges")
    return rules


def session_findings(feed, projection):
    """Return one row per broken visible requirement, without retaining message content."""
    subjects, times, failures = defaultdict(list), defaultdict(list), {}
    for item in feed["items"]:
        sub = item.get("payload", {}).get("subject") or {}
        if sub.get("native_session_id"):
            subjects[sub["native_session_id"]].append((item, sub))
        if item.get("schema_kind") == "ProductOperationObserved":
            times[sub.get("agent_session")].append(item["occurred_at"])
    for shown in projection["sessions"]:
        sid = shown["id"]
        cells = [[unescape(re.sub(r"<[^>]*>", "", str(v))) for v in r] for r in shown["rows"]]
        rules = _visible(shown, cells) | _roles(subjects[sid], shown, cells)
        rules |= _recipient(subjects[sid], shown) | _goal(
            subjects[sid], shown, feed.get("goals", {})
        )
        rules |= _span(subjects[sid], shown, times[sid])
        rules |= _usage(feed, shown) | _references(shown)
        rules |= _retained(subjects[sid], shown)
        for rule in sorted(rules):
            failures.setdefault(rule, {"rule": rule, "session": sid, "part": "feedback"})
    return list(failures.values())
