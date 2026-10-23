"""Collect named owner offers for the CLI contract and its request readers."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

_ITEM_IDENTIFIERS = (
    "entry_id",
    "task_id",
    "case_token",
    "review_publication_hash",
    "feature_trial_id",
)
"""What names a listed item's requests apart when two items offer one name: a history row by
its entry, since one Task can have two reviews."""
_OFFER_DEPTH = 5
"""How deep the collector looks for offered requests: `workspace show`'s are three parts down
(`intents.0.flows.factor`), and a deeper part offers none."""


def offered_requests(
    body: dict[str, Any], operation: str | None = None
) -> dict[str, dict[str, Any]]:
    """Every next request an answer offers, wherever it sits.

    The top level's by their names; a nested part's (each of the pending decisions, each flow
    of `workspace show`'s intents) by its name, or, where two offer one name, by the name and
    the part's identifier (``recovery:<task_id>``), or its place (``decisions.2``,
    ``intents.0.flows.factor``) when it has none. The same named request offered twice is
    listed once; distinct printed aliases remain sendable;
    two items that name theirs alike keep both, each by its place.

    Args:
        body: The owner's answer.
        operation: A legacy operation-only continuation; explicit actions collect all.

    Returns:
        Each offered request by the name ``--action`` takes.
    """
    found: list[tuple[str, str | None, str | None, dict[str, Any]]] = []
    seen: set[str] = set()

    def walk(value: Any, place: str | None, depth: int) -> None:
        if isinstance(value, dict):
            offered = value.get("next_requests")
            if isinstance(offered, dict):
                named = next((str(value[f]) for f in _ITEM_IDENTIFIERS if value.get(f)), None)
                for name, request in offered.items():
                    key = json.dumps((name, request), sort_keys=True, default=str)
                    if isinstance(request, dict) and key not in seen:
                        seen.add(key)
                        found.append((name, named, place, request))
            parts = [(key, inner) for key, inner in value.items() if key != "next_requests"]
        elif isinstance(value, list):
            parts = [(str(index), inner) for index, inner in enumerate(value)]
        else:
            return
        if depth < _OFFER_DEPTH:
            for key, inner in parts:
                walk(inner, key if place is None else f"{place}.{key}", depth + 1)

    walk(body, None, 0)
    counts = Counter(name for name, _named, _place, _request in found)
    labels = [
        name if place is None or counts[name] == 1 else f"{name}:{named or place}"
        for name, named, place, _request in found
    ]
    # Two different requests under one label keep both, each by its place, never the first
    # alone: one Task's two reviews each offered their own export.
    clashes = Counter(labels)
    requests: dict[str, dict[str, Any]] = {}
    for label, (name, _named, place, request) in zip(labels, found, strict=True):
        requests[f"{name}:{place}" if clashes[label] > 1 and place is not None else label] = request
    if operation == "GOAL_SUBMIT" and "complete" in requests:
        peers = (q for name, q in requests.items() if name != "complete")
        if any(q.get("operation") == operation for q in peers):
            del requests["complete"]
    return requests
