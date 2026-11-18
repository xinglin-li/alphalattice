"""A review's note and the experience the Host makes of it (doc 09, how an experience is written).

An agent's note on a trial, a task or a goal (its hypothesis, method, conclusion and lessons) is
one `review.completed` event: its text the note, its attributes the verdict and the run's
aggregates, read in the same unit from the spans of its traces (tokens, cost and seconds) and, for
a trial, its headline measures' place in their approach's spread. Its projection closes what it
reviews; an ended trial with no note awaits review.

Off the agent's path, `render_pending` renders each note with its metadata into one markdown body,
kept forever as blobs/<hex> of its BLAKE3, and records `experience.recorded` under an experience_id
the Host mints once. Version 1 makes the experience and replaces its layer's one before; a new
`RENDERER_VERSION` renders every experience again as its next version, the id unchanged.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.chain import canonical, digest
from alphalattice.kernel.data_layer.log import HostLog, uuid7

RENDERER_VERSION: Final = 1
"""The renderer's release: a change to `render`'s output takes the next number."""
LAYERS: Final = ("trial", "task", "goal")
VERDICTS: Final = ("success", "failure", "inconclusive")
PLACES: Final = 16
"""The most headline measures a trial's note places in their spread."""
_ENDED: Final = "phase IN ('completed', 'failed', 'cancelled')"


def _refuse(code: str, message: str) -> DataError:
    return DataError(message, code=f"experience.{code}")


def note_text(hypothesis: str, method: str, conclusion: str, lessons: list[str]) -> str:
    """A note's one text form; refused by name unless each part is said."""
    parts = {"hypothesis": hypothesis, "method": method, "conclusion": conclusion}
    if any(not isinstance(v, str) or not v.strip() for v in parts.values()):
        raise _refuse("note_incomplete", "a note says its hypothesis, method and conclusion")
    if not isinstance(lessons, list) or any(
        not isinstance(v, str) or not v.strip() for v in lessons
    ):
        raise _refuse("note_incomplete", "each lesson is a sentence")
    return canonical(parts | {"lessons": lessons}).decode()


def record_note(
    log: HostLog,
    layer: str,
    unit: str,
    verdict: str,
    text: str,
    *,
    workspace: str,
    actor: tuple[str, str],
    trials: list[str] | None = None,
) -> int:
    """Record a review's note with its run's aggregates as one event in one unit; returns its seq.
    Refused by name for an unknown layer, verdict or unit, and (by the projection) a trial or task
    that has not ended. A review task's note names the trials it reviewed (`trials`), which its
    experience names in turn."""
    if layer not in LAYERS:
        raise _refuse("layer_unknown", f"a note reviews a {', '.join(LAYERS)}")
    if verdict not in VERDICTS:
        raise _refuse("verdict_unknown", f"a verdict is one of {', '.join(VERDICTS)}")
    if trials is not None and (
        not isinstance(trials, list) or not trials or any(not isinstance(t, str) for t in trials)
    ):
        raise _refuse("trials_malformed", "a review names its trials as a list of trial ids")
    with log.unit() as con:
        attrs = {"verdict": verdict, **aggregates(con, layer, unit)}
        attrs |= {"trials": sorted(set(trials))} if trials else {}
        return log.record(
            "review.completed", (layer, unit), attrs, text=text, workspace=workspace, actor=actor
        )


def aggregates(con: Any, layer: str, unit: str) -> dict[str, Any]:
    """The run a note reviews, summed over the spans of its traces: tokens, cost and seconds (each
    trace's first start to its last end), and for a trial its headline measures' place."""
    traces = _traces(con, layer, unit)
    marks = ",".join("?" * len(traces))
    row = con.execute(
        "SELECT coalesce(sum(CAST(json_extract(attrs, '$.\"gen_ai.usage.input_tokens\"')"
        " AS INTEGER)), 0), coalesce(sum(CAST(json_extract(attrs,"
        " '$.\"gen_ai.usage.output_tokens\"') AS INTEGER)), 0),"
        " coalesce(sum(json_extract(attrs, '$.cost_usd')), 0.0)"
        f" FROM event WHERE trace_id IN ({marks}) AND {_ENDED}",
        traces,
    ).fetchone()
    spans = con.execute(
        "SELECT max(occurred_us) - min(started_us) FROM event"
        f" WHERE trace_id IN ({marks}) AND {_ENDED}"
        " GROUP BY trace_id",
        traces,
    ).fetchall()
    found = {
        "gen_ai.usage.input_tokens": row[0],
        "gen_ai.usage.output_tokens": row[1],
        "cost_usd": round(float(row[2]), 6),
        "seconds": round(sum(span[0] or 0 for span in spans) / 1e6, 6),
    }
    return found | ({"place": _place(con, unit)} if layer == "trial" else {})


def _traces(con: Any, layer: str, unit: str) -> list[str]:
    """The traces a reviewed unit ran in: a trial's run, a task's, a goal's tasks' and trials'."""
    if layer == "trial":
        sql = (
            "SELECT e.trace_id FROM trial t JOIN event e ON e.seq = t.ended_seq"
            " WHERE t.trial_id = ?"
        )
        known = "SELECT 1 FROM trial WHERE trial_id = ?"
    elif layer == "task":
        sql = known = "SELECT trace_id FROM task WHERE task_id = ?"
    else:
        sql = (
            "SELECT t.trace_id FROM task t JOIN task_scope s USING (task_id) WHERE s.goal_id = ?1"
            " UNION SELECT e.trace_id FROM trial t JOIN trial_scope s USING (trial_id)"
            " JOIN event e ON e.seq = t.ended_seq WHERE s.goal_id = ?1"
        )
        known = "SELECT 1 FROM goal WHERE goal_id = ?"
    if con.execute(known, (unit,)).fetchone() is None:
        raise _refuse("unit_unknown", f"no {layer} {unit} is on the log")
    return [row[0] for row in con.execute(sql, (unit,)).fetchall()]


def _place(con: Any, trial: str) -> list[dict[str, Any]]:
    """Where a trial's headline measures (whole-artifact ones) fall in its recipe's spread: the
    trials it is read against and its z-score, None while the spread has no width."""
    rows = con.execute(
        "SELECT l.name, m.metric_version, m.value, count(o.value), avg(o.value),"
        " avg(o.value * o.value)"
        " FROM trial t JOIN measure m ON m.artifact_id = t.result_artifact AND m.dims = ''"
        " JOIN definition_lineage l ON l.definition_id = m.metric_id"
        " JOIN trial ot ON ot.recipe_key = t.recipe_key"
        " JOIN measure o ON o.artifact_id = ot.result_artifact AND o.metric_id = m.metric_id"
        " AND o.metric_version = m.metric_version AND o.dims = ''"
        " WHERE t.trial_id = ? AND m.value IS NOT NULL"
        " GROUP BY m.metric_id, m.metric_version ORDER BY l.name, m.metric_version LIMIT ?",
        (trial, PLACES),
    ).fetchall()
    placed = []
    for metric, version, value, n, mean, square in rows:
        variance = (square - mean * mean) * n / (n - 1) if n > 1 else 0.0
        sd = math.sqrt(max(variance, 0.0))
        z = round((value - mean) / sd, 6) if sd > 1e-12 else None
        placed.append({"metric": metric, "version": version, "n": n, "z": z})
    return placed


def render(note: Mapping[str, Any], text: str) -> bytes:
    """One experience's markdown: the note under its metadata. `note` is the review event's row
    with its layer's scope; the same note always renders the same bytes."""
    written = json.loads(text)
    attrs = json.loads(note["attrs"])
    known = datetime.fromtimestamp(note["occurred_us"] / 1e6, UTC).isoformat()
    lines = [
        f"# Experience of {note['subject_kind']} {note['subject_id']}",
        "",
        f"- Verdict: {attrs['verdict']}",
        f"- Workspace: {note['workspace_id']}",
        f"- Goal: {note['goal_id'] or 'none'}",
        f"- Task: {note['task_id'] or 'none'}",
        f"- Known: {known}",
        f"- Reviewed by: {note['actor_kind']} {note['actor_id']}",
        f"- Run: {attrs['seconds']} s, {attrs['gen_ai.usage.input_tokens']} tokens in,"
        f" {attrs['gen_ai.usage.output_tokens']} out, {attrs['cost_usd']} USD",
    ]
    for place in attrs.get("place", []):
        z = "no spread yet" if place["z"] is None else f"z {place['z']}"
        lines.append(
            f"- Place: {place['metric']} v{place['version']}, {z} among {place['n']} trials"
        )
    for title, key in (
        ("Hypothesis", "hypothesis"),
        ("Method", "method"),
        ("Conclusion", "conclusion"),
    ):
        lines += ["", f"## {title}", "", written[key]]
    lines += ["", "## Lessons", ""] + (
        [f"- {lesson}" for lesson in written["lessons"]] or ["None recorded."]
    )
    return ("\n".join(lines) + "\n").encode()


_DUE: Final = (
    "SELECT e.seq, NULL AS experience_id, 0 AS version FROM event e"
    " WHERE e.type = 'review.completed' AND e.subject_kind IN ('trial', 'task', 'goal')"
    " AND NOT EXISTS (SELECT 1 FROM experience_version v WHERE v.note_seq = e.seq)"
    " UNION ALL SELECT min(note_seq), experience_id, max(version) FROM experience_version"
    " GROUP BY experience_id HAVING max(renderer_version) < ? ORDER BY 1"
)
_NOTE: Final = (
    "SELECT e.seq, e.subject_kind, e.subject_id, e.workspace_id, e.actor_kind, e.actor_id,"
    " e.occurred_us, e.attrs, x.text, CASE e.subject_kind"
    " WHEN 'trial' THEN (SELECT task_id FROM trial WHERE trial_id ="
    " e.subject_id) WHEN 'task' THEN e.subject_id END AS task_id, CASE e.subject_kind"
    " WHEN 'trial' THEN (SELECT goal_id FROM trial_scope WHERE trial_id = e.subject_id)"
    " WHEN 'task' THEN (SELECT goal_id FROM task_scope WHERE task_id = e.subject_id)"
    " ELSE e.subject_id END AS goal_id FROM event e JOIN event_text x ON x.seq = e.seq"
    " WHERE e.seq = ?"
)


def render_pending(log: HostLog, blobs: Path) -> list[str]:
    """Render each note with no experience yet, in log order (so a later note on a layer replaces
    an earlier one), and each experience an older renderer wrote as its next version. Each body is
    written to `blobs` before its event; returns the experiences recorded."""
    made = []
    for note_seq, experience, version in log.read(_DUE, (RENDERER_VERSION,)):
        note = dict(zip(_COLUMNS, log.read(_NOTE, (note_seq,))[0], strict=True))
        body = render(note, note["text"])
        named = digest(body)
        _keep(blobs, named, body)
        with log.unit() as con:  # a note another renderer took meanwhile is not rendered twice
            if (
                version == 0
                and con.execute(
                    "SELECT 1 FROM experience_version WHERE note_seq = ?", (note_seq,)
                ).fetchone()
            ):
                continue
            experience = experience or str(uuid7())
            attrs = {
                "version": version + 1,
                "note_seq": note_seq,
                "body": named,
                "size": len(body),
                "renderer_version": RENDERER_VERSION,
                "known_us": note["occurred_us"],
            }
            if note["task_id"] is not None:
                attrs["task_id"] = note["task_id"]
            log.record(
                "experience.recorded",
                ("experience", experience),
                attrs,
                workspace=note["workspace_id"],
            )
        made.append(experience)
    return made


_COLUMNS: Final = (
    "seq",
    "subject_kind",
    "subject_id",
    "workspace_id",
    "actor_kind",
    "actor_id",
    "occurred_us",
    "attrs",
    "text",
    "task_id",
    "goal_id",
)


def _keep(blobs: Path, named: str, body: bytes) -> None:
    """Write a body as blobs/<hex> once; a file already there holds the same bytes by its name."""
    path = blobs / named.removeprefix("blake3:")
    if path.exists():
        return
    blobs.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    partial.write_bytes(body)
    os.replace(partial, path)
