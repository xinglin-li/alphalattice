"""Grade an investment committee on a date's positions from the answers its members read."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from typing import Any


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
