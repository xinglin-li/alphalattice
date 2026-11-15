"""Each event type's projection: the rows of log.sqlite that derive from the event log.

A projection reads the event's own fields, its attributes and, for a definition, its text; never a
clock, a file or another store. So `log.replay`, which copies the events into an empty log through
the same write, rebuilds every projected table ALL EQUAL, and `differences` names any table that
does not. The meaning of each type is the 0.2.8 reference's (tools/records/2026-11/2026-11-14/
schemas/project.py); a refusal here rolls the event back with everything else of its unit.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterable
from typing import Any, Final

from alphalattice.kernel.data.errors import DataError
from alphalattice.kernel.data_layer.chain import CHAIN_VERSION, canonical, digest

ENDED: Final = ("completed", "failed", "cancelled")
PROJECTED: Final = (
    "definition",
    "workspace",
    "goal",
    "task",
    "task_step",
    "wait",
    "recipe",
    "selection",
    "hypothesis",
    "trial",
    "prediction",
    "artifact",
    "measure",
    "event_artifact",
    "tag",
    "live_ref",
    "book",
    "ledger_entry",
    "setting",
    "chain_checkpoint",
    "chain_head",
    "daily_span",
)
"""Every table replay rebuilds from the events alone."""
MACHINE_TEXT: Final = frozenset({"definition.registered"})
"""The types whose text a projection reads, so it is never forgotten."""

Con = sqlite3.Connection
Row = sqlite3.Row
Attrs = dict[str, Any]
Projection = Callable[[Con, Row, Attrs, str | None], None]
HANDLERS: Final[dict[str, Projection]] = {}
"""Each projected event type's handler; it reads the event, its attributes and its text only."""


def on(*types: str) -> Callable[[Projection], Projection]:
    """Register a handler as the projection of `types`."""

    def register(handler: Projection) -> Projection:
        for name in types:
            HANDLERS[name] = handler
        return handler

    return register


def project(con: sqlite3.Connection, event: sqlite3.Row, text: str | None) -> None:
    """Apply one written event's projection, in the transaction that wrote it."""
    a = json.loads(event["attrs"])
    handler = HANDLERS.get(event["type"])
    if handler is not None:
        handler(con, event, a, text)
    for role in ("used", "produced"):  # any step names what it used and produced
        for artifact in a.get(role, ()):
            con.execute(
                "INSERT OR IGNORE INTO event_artifact VALUES (?, ?, ?)",
                (event["seq"], role, artifact),
            )
    for key, values in a.get("tags", {}).items():  # any event may tag its subject
        version = _tag_version(con, key, None)
        for value in values if isinstance(values, list) else [values]:
            _tag(con, event, key, version, value)
    if event["phase"] in ENDED:  # the day's totals, in the step's own transaction
        _daily(con, event, a)


def recipe_key(method: str, version: int, params: Any, upstream: Iterable[str] = ()) -> str:
    """An action's identity: its method and version, its parameters, the recipes it reads."""
    return digest(canonical([method, version, params, sorted(upstream)]))


def _refuse(code: str, message: str) -> DataError:
    return DataError(message, code=f"data_layer.{code}")


def _json(value: Any) -> str | None:
    return None if value is None else canonical(value).decode()


def _one(cursor: sqlite3.Cursor, code: str, message: str) -> None:
    if cursor.rowcount != 1:
        raise _refuse(code, message)


def _tag_version(con: sqlite3.Connection, key: str, version: int | None) -> int:
    """The vocabulary version a tag is given under: as named, else the key's current one."""
    if version is None:
        found = con.execute(
            "SELECT version FROM definition_current WHERE family = 'tag' AND name = ?", (key,)
        ).fetchone()
    else:
        found = con.execute(
            "SELECT version FROM definition WHERE family = 'tag' AND name = ? AND version = ?",
            (key, version),
        ).fetchone()
    if found is None:
        raise _refuse("tag_unregistered", f"no tag definition registers {key}")
    return int(found[0])


def _tag(con: sqlite3.Connection, e: sqlite3.Row, key: str, version: int, value: Any) -> None:
    con.execute(
        "INSERT OR IGNORE INTO tag (subject_kind, subject_id, key, tag_version, value, seq)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (e["subject_kind"], e["subject_id"], key, version, str(value), e["seq"]),
    )


def _daily(con: sqlite3.Connection, e: sqlite3.Row, a: dict[str, Any]) -> None:
    cost = a.get("cost_usd")
    con.execute(
        "INSERT INTO daily_span VALUES (date(? / 1000000, 'unixepoch'), ?, ?, ?, ?, ?, 1, ?, ?, ?,"
        " ?, ?, ?) ON CONFLICT (day, actor_kind, actor_id, node, type, outcome) DO UPDATE SET"
        " n = n + 1, micros = micros + excluded.micros, tokens_in = tokens_in + excluded.tokens_in,"
        " tokens_out = tokens_out + excluded.tokens_out,"
        " cost_microusd = cost_microusd + excluded.cost_microusd,"
        " bytes_read = bytes_read + excluded.bytes_read,"
        " bytes_written = bytes_written + excluded.bytes_written",
        (
            e["occurred_us"],
            e["actor_kind"],
            e["actor_id"],
            e["node"],
            e["type"],
            e["phase"],
            e["occurred_us"] - e["started_us"],
            int(a.get("gen_ai.usage.input_tokens", 0)),
            int(a.get("gen_ai.usage.output_tokens", 0)),
            0 if cost is None else round(cost * 1_000_000),
            int(a.get("bytes_read", 0)),
            int(a.get("bytes_written", 0)),
        ),
    )


@on("definition.registered")
def _definition(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    d = json.loads(text or "{}")
    values = [d["family"], d["name"], d["version"], d.get("status", "experimental")]
    values += [d.get("relation"), _json(d.get("conversion")), _json(d.get("spec", {})), d["text"]]
    values += [d.get("code_ref"), _json(d.get("golden")), d.get("owner", e["node"])]
    con.execute(
        f"INSERT INTO definition VALUES ({', '.join('?' * 13)}, NULL)",
        (*values, e["seq"], e["seq"]),
    )


@on("definition.status_changed")
def _definition_status(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    key = (a["family"], a["name"], a["version"])
    where = "WHERE family = ? AND name = ? AND version = ?"
    owner = con.execute(f"SELECT owner FROM definition {where}", key).fetchone()
    if owner is None or (owner[0] != e["node"] and e["actor_kind"] not in ("person", "host")):
        raise _refuse(
            "definition_not_owner",
            "a definition's status is moved by its owner, the person or the host",
        )
    con.execute(
        f"UPDATE definition SET status = ?, status_seq = ?, status_reason = ? {where}",
        (a["status"], e["seq"], a.get("reason"), *key),
    )


@on("workspace.created")
def _workspace(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    con.execute(
        "INSERT INTO workspace VALUES (?, ?, ?, 'active', ?, NULL)",
        (e["subject_id"], a["name"], a.get("folder"), e["seq"]),
    )


@on("workspace.opened")
def _workspace_opened(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    con.execute(
        "UPDATE workspace SET opened_seq = ? WHERE workspace_id = ?", (e["seq"], e["subject_id"])
    )


@on("workspace.status_changed")
def _workspace_status(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    con.execute(
        "UPDATE workspace SET status = ? WHERE workspace_id = ?", (a["status"], e["subject_id"])
    )


@on("setting.changed")
def _setting(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    con.execute(
        "INSERT OR REPLACE INTO setting VALUES (?, ?, ?, ?, ?)",
        (a["scope"], a["key"], _json(a.get("value")), a.get("secret_ref"), e["seq"]),
    )


@on("goal.set")
def _goal(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    con.execute(
        "INSERT INTO goal VALUES (?, ?, ?, 'open', ?, ?, NULL)",
        (e["subject_id"], e["workspace_id"], a["title"], a.get("deadline_us"), e["seq"]),
    )


@on("goal.closed")
def _goal_closed(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    con.execute(
        "UPDATE goal SET state = ?, closed_seq = ? WHERE goal_id = ?",
        (a["state"], e["seq"], e["subject_id"]),
    )


@on("task.requested")
def _task_requested(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    con.execute(
        "INSERT INTO task (task_id, workspace_id, goal_id, type, state, trace_id, product_version,"
        " requested_seq) VALUES (?, ?, ?, ?, 'requested', ?, ?, ?)",
        (
            e["subject_id"],
            e["workspace_id"],
            a.get("goal_id"),
            a["type"],
            e["trace_id"],
            a["product_version"],
            e["seq"],
        ),
    )


@on("task")
def _task(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    if e["phase"] == "started":
        con.execute(
            "UPDATE task SET state = 'running', started_seq = ?, attempts = attempts + 1"
            " WHERE task_id = ?",
            (e["seq"], e["subject_id"]),
        )
    else:
        con.execute(
            "UPDATE task SET state = ?, ended_seq = ? WHERE task_id = ?",
            (e["phase"], e["seq"], e["subject_id"]),
        )


@on("step.completed")
def _step(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    con.execute(
        "INSERT OR REPLACE INTO task_step VALUES (?, ?, ?, ?)",
        (e["subject_id"], a["step"], _json(a.get("output")), e["seq"]),
    )


@on("wait.started")
def _wait(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    con.execute(
        "INSERT INTO wait VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)",
        (
            e["subject_id"],
            a["task_id"],
            a["kind"],
            a.get("event_type"),
            _json(a.get("match")),
            a.get("fires_us"),
            a["purpose"],
            e["seq"],
        ),
    )
    con.execute(
        "UPDATE task SET state = 'waiting' WHERE task_id = ? AND state = 'running'", (a["task_id"],)
    )


@on("wait.resolved")
def _wait_resolved(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    con.execute(
        "UPDATE wait SET resolved_seq = ?, resolved_by_seq = ? WHERE wait_id = ?",
        (e["seq"], a.get("by_seq"), e["subject_id"]),
    )
    con.execute(  # a task resumes only when its last wait resolves
        "UPDATE task SET state = 'running' WHERE state = 'waiting' AND task_id ="
        " (SELECT task_id FROM wait WHERE wait_id = ?) AND NOT EXISTS (SELECT 1 FROM wait w"
        " WHERE w.task_id = task.task_id AND w.resolved_seq IS NULL)",
        (e["subject_id"],),
    )


@on("selection.made")
def _selection(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    con.execute(
        "INSERT INTO selection VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            e["subject_id"],
            e["workspace_id"],
            a["cutoff_us"],
            a.get("universe"),
            a.get("valid_from"),
            a.get("valid_to"),
            _json(a.get("spec", {})),
            e["seq"],
        ),
    )


@on("recipe.defined")
def _recipe(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    upstream = a.get("upstream", ())
    if e["subject_id"] != recipe_key(a["method"], a["method_version"], a["params"], upstream):
        raise _refuse(
            "recipe_key_mismatch",
            "a recipe's key is the digest of its method, version, parameters and upstream",
        )
    con.execute(
        "INSERT OR IGNORE INTO recipe VALUES (?, ?, ?, ?, ?, ?)",
        (
            e["subject_id"],
            a["method"],
            a["method_version"],
            _json(a["params"]),
            _json(sorted(upstream)),
            e["seq"],
        ),
    )


@on("hypothesis.stated")
def _hypothesis(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    con.execute(
        "INSERT INTO hypothesis VALUES (?, ?, ?, 'open', ?, ?)",
        (e["subject_id"], e["workspace_id"], a.get("goal_id"), e["seq"], e["seq"]),
    )


@on("hypothesis.standing_changed")
def _hypothesis_standing(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    con.execute(
        "UPDATE hypothesis SET standing = ?, standing_seq = ? WHERE hypothesis_id = ?",
        (a["standing"], e["seq"], e["subject_id"]),
    )
    if a.get("trial_id"):  # the transition keeps the standing it produced
        _one(
            con.execute(
                "UPDATE trial SET standing_after = ? WHERE trial_id = ? AND hypothesis_id = ?",
                (a["standing"], a["trial_id"], e["subject_id"]),
            ),
            "hypothesis_trial_unknown",
            "the trial that moved a standing tests that hypothesis",
        )


@on("trial.registered")
def _trial_registered(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    con.execute(
        "INSERT INTO trial (trial_id, workspace_id, goal_id, hypothesis_id, selection_id,"
        " recipe_key, seeds, product_version, model, context, task_id, state, registered_seq)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'registered', ?)",
        (
            e["subject_id"],
            e["workspace_id"],
            a.get("goal_id"),
            a.get("hypothesis_id"),
            a["selection_id"],
            a["recipe_key"],
            _json(a["seeds"]),
            a["product_version"],
            a.get("model"),
            _json(a.get("context", {})),
            a.get("task_id"),
            e["seq"],
        ),
    )


@on("trial")
def _trial(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    unregistered = "a trial runs and ends after its registration"
    if e["phase"] == "started":
        _one(
            con.execute(
                "UPDATE trial SET state = 'running' WHERE trial_id = ?", (e["subject_id"],)
            ),
            "trial_unregistered",
            unregistered,
        )
        return
    headline = a.get("result_artifact")
    if headline is not None:
        owner = con.execute(
            "SELECT trial_id FROM artifact WHERE artifact_id = ?", (headline,)
        ).fetchone()
        if owner is None or owner[0] != e["subject_id"]:
            raise _refuse(
                "trial_headline_not_its_own", "a completion names an artifact this trial produced"
            )
    _one(
        con.execute(
            "UPDATE trial SET state = ?, ended_seq = ?, result_artifact = coalesce(?,"
            " result_artifact), candidates = coalesce(?, candidates) WHERE trial_id = ?",
            (e["phase"], e["seq"], headline, a.get("candidates"), e["subject_id"]),
        ),
        "trial_unregistered",
        unregistered,
    )


@on("artifact.produced")
def _artifact(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    """An artifact's profile: what it means, how it is identified and where its body is."""
    trial = a.get("trial_id")
    if trial and con.execute("SELECT 1 FROM trial WHERE trial_id = ?", (trial,)).fetchone() is None:
        raise _refuse(
            "artifact_trial_unregistered", "a trial's artifact follows the trial's registration"
        )
    values = [e["subject_id"], a["identity_kind"], a["type"], a.get("type_version", 1)]
    values += [a["meaning"], a.get("recipe_key"), a.get("selection_id"), trial]
    values += [_json(a.get("shape")), a.get("valid_from"), a.get("valid_to")]
    values += [a.get("row_count"), a.get("size_bytes"), a.get("content_digest")]
    state = "never_stored" if a["storage"] == "none" else "present"
    con.execute(
        "INSERT INTO artifact (artifact_id, identity_kind, type, type_version, meaning, recipe_key,"
        " selection_id, trial_id, shape, valid_from, valid_to, row_count, size_bytes,"
        f" content_digest, storage, body_state, produced_seq) VALUES ({', '.join('?' * 17)})",
        (*values, a["storage"], state, e["seq"]),
    )
    if a["storage"] != "none":  # a kept trial keeps what it produces later too
        con.execute(
            "INSERT OR IGNORE INTO live_ref SELECT ?, 'trial', trial_id, ? FROM trial"
            " WHERE trial_id = ? AND kept = 1",
            (e["subject_id"], e["seq"], trial),
        )


@on("result.recorded")
def _result(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    for metric, version, dims, value in a["measures"]:
        con.execute(
            "INSERT INTO measure VALUES (?, ?, ?, ?, ?, ?)",
            (e["subject_id"], metric, version, dims, value, e["seq"]),
        )
    owner = con.execute(
        "SELECT trial_id FROM artifact WHERE artifact_id = ?", (e["subject_id"],)
    ).fetchone()
    if owner is None:
        raise _refuse("result_artifact_unknown", "measures follow their artifact's profile")
    if owner[0]:  # the trial is the artifact's own, never a caller's claim
        con.execute(
            "UPDATE trial SET reads_legacy = max(reads_legacy, ?) WHERE trial_id = ?",
            (int(bool(a.get("reads_legacy"))), owner[0]),
        )


@on("artifact.collected")
def _collected(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    con.execute(
        "UPDATE artifact SET body_state = 'collected', collected_seq = ? WHERE artifact_id = ?"
        " AND body_state = 'present'",
        (e["seq"], e["subject_id"]),
    )
    con.execute("DELETE FROM artifact_body WHERE artifact_id = ?", (e["subject_id"],))
    con.execute("DELETE FROM live_ref WHERE artifact_id = ?", (e["subject_id"],))


@on("prediction.registered")
def _prediction(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    con.execute(
        "INSERT INTO prediction (prediction_id, workspace_id, trial_id, target, horizon_us,"
        " confidence, registered_seq) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            e["subject_id"],
            e["workspace_id"],
            a.get("trial_id"),
            _json(a["target"]),
            a["horizon_us"],
            a["confidence"],
            e["seq"],
        ),
    )


@on("outcome.observed")
def _outcome(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    if e["subject_kind"] == "prediction":
        con.execute(
            "UPDATE prediction SET outcome_seq = ? WHERE prediction_id = ? AND outcome_seq IS NULL",
            (e["seq"], e["subject_id"]),
        )


@on("prediction.scored")
def _scored(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    con.execute(
        "UPDATE prediction SET score = ?, score_metric = ?, score_metric_version = ?,"
        " outcome_seq = coalesce(outcome_seq, ?) WHERE prediction_id = ?",
        (a["score"], a["metric"], a["metric_version"], e["seq"], e["subject_id"]),
    )


@on("trial.kept")
def _kept(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    con.execute("UPDATE trial SET kept = 1 WHERE trial_id = ?", (e["subject_id"],))
    con.execute(
        "INSERT OR IGNORE INTO live_ref SELECT artifact_id, 'trial', trial_id, ? FROM artifact"
        " WHERE trial_id = ? AND body_state = 'present'",
        (e["seq"], e["subject_id"]),
    )


@on("book.activated")
def _book(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    con.execute(
        "INSERT INTO book VALUES (?, ?, ?, 'active', ?, ?, NULL)",
        (e["subject_id"], e["workspace_id"], a["mode"], a.get("holding_artifact"), e["seq"]),
    )
    if a.get("holding_artifact"):
        con.execute(
            "INSERT OR IGNORE INTO live_ref VALUES (?, 'book', ?, ?)",
            (a["holding_artifact"], e["subject_id"], e["seq"]),
        )


@on("book.rebalanced")
def _book_rebalanced(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    _one(
        con.execute(
            "UPDATE book SET holding_artifact = ? WHERE book_id = ? AND state = 'active'",
            (a["holding_artifact"], e["subject_id"]),
        ),
        "book_not_active",
        "only an active book follows new holdings",
    )
    con.execute("DELETE FROM live_ref WHERE root_kind = 'book' AND root_id = ?", (e["subject_id"],))
    con.execute(
        "INSERT INTO live_ref VALUES (?, 'book', ?, ?)",
        (a["holding_artifact"], e["subject_id"], e["seq"]),
    )


@on("book.retired")
def _book_retired(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    con.execute(
        "UPDATE book SET state = 'retired', retired_seq = ? WHERE book_id = ?",
        (e["seq"], e["subject_id"]),
    )
    con.execute("DELETE FROM live_ref WHERE root_kind = 'book' AND root_id = ?", (e["subject_id"],))


@on("publication.released")
def _publication(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    con.execute(
        "INSERT OR IGNORE INTO live_ref VALUES (?, 'publication', ?, ?)",
        (e["subject_id"], e["event_id"], e["seq"]),
    )


@on("ledger.posted")
def _ledger(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    for line, (account, asset, quantity_n) in enumerate(a["lines"], 1):
        con.execute(
            "INSERT INTO ledger_entry VALUES (?, ?, ?, ?, ?, ?)",
            (e["seq"], line, e["subject_id"], account, asset, quantity_n),
        )
    if con.execute("SELECT 1 FROM ledger_unbalanced WHERE seq = ?", (e["seq"],)).fetchone():
        raise _refuse("ledger_unbalanced", "the lines of one posting sum to zero for each asset")


@on("tag.added")
def _tag_added(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    _tag(con, e, a["key"], _tag_version(con, a["key"], a.get("version")), a["value"])


@on("tag.removed")
def _tag_removed(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    con.execute(
        "DELETE FROM tag WHERE subject_kind = ? AND subject_id = ? AND key = ? AND tag_version = ?"
        " AND value = ?",
        (
            e["subject_kind"],
            e["subject_id"],
            a["key"],
            _tag_version(con, a["key"], a.get("version")),
            str(a["value"]),
        ),
    )


@on("record.forgotten")
def _forgotten(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    target = con.execute("SELECT type FROM event WHERE seq = ?", (a["seq"],)).fetchone()
    if target is None or target[0] in MACHINE_TEXT:
        raise _refuse(
            "record_unforgettable",
            "a definition's text is what replay rebuilds the registry from",
        )
    con.execute("DELETE FROM event_text WHERE seq = ?", (a["seq"],))  # the event and digest stay


@on("chain.checkpointed")
def _checkpoint(con: Con, e: Row, a: Attrs, text: str | None) -> None:
    con.execute(
        "INSERT INTO chain_checkpoint VALUES (?, ?, ?, ?, ?, ?)",
        (
            a["seq"],
            a["kind"],
            a["head"],
            e["occurred_us"],
            a.get("note"),
            a.get("chain_version", CHAIN_VERSION),
        ),
    )


def differences(a: sqlite3.Connection, b: sqlite3.Connection) -> list[str]:
    """The projected tables whose rows differ between two logs; ALL EQUAL is an empty list."""

    def rows(con: sqlite3.Connection, table: str) -> list[tuple[Any, ...]]:
        return sorted((tuple(row) for row in con.execute(f"SELECT * FROM {table}")), key=repr)

    return [table for table in PROJECTED if rows(a, table) != rows(b, table)]
