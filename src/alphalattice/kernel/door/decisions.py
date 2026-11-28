"""Durable questions and person decisions for a single Host workspace."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal
from uuid import UUID

from .port import DoorRefusal

if TYPE_CHECKING:
    from collections.abc import Iterator
    from sqlite3 import Connection

DECISIONS_DATABASE: Final = Path("runtime/decisions.sqlite3")
_PLUGIN_LAPSE: Final = (
    "UPDATE decision SET answer = CASE WHEN answer = 'CONFIRMED' THEN 'LAPSED' "
    "ELSE answer END, consumed = 1, external_key = NULL "
    "WHERE kind = 'plugin_activation' AND operation = ? "
)

type DecisionKind = Literal[
    "question", "confirmation", "paid_run", "activation", "plugin_activation"
]


@dataclass(frozen=True, slots=True)
class Decision:
    """A request the person can answer, with its unique admission key."""

    id: UUID
    kind: DecisionKind
    operation: str


class Decisions:
    """The Host-owned workspace store for person decisions and their current validity."""

    def __init__(self, path: Path, mint: Callable[[], UUID]) -> None:
        """Bind the workspace decision file and the Host allocator for decision IDs."""
        self.path = path
        self._mint = mint

    @contextmanager
    def _writing(self, operation: str) -> Iterator[Connection]:
        from sqlite3 import Error, connect  # lazy: no decisions means no database open

        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with closing(connect(self.path, timeout=1)) as connection, connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS decision ("
                    "id TEXT PRIMARY KEY, kind TEXT NOT NULL, operation TEXT NOT NULL, "
                    "request TEXT NOT NULL, answer TEXT, consumed INTEGER NOT NULL DEFAULT 0, "
                    "external_key TEXT UNIQUE, nonce TEXT UNIQUE)"
                )
                yield connection
        except (Error, OSError) as error:
            raise DoorRefusal("door.decisions_unavailable", operation) from error

    def ask(
        self,
        kind: DecisionKind,
        operation: str,
        request: Mapping[str, object],
        *,
        attended: bool,
        key: str | None = None,
    ) -> Decision:
        """Queue the exact question for a person, or give the immediate named refusal."""
        if not attended:
            raise DoorRefusal(
                "door.paid_run_approval_required"
                if kind == "paid_run"
                else "door.person_unavailable",
                operation,
            )
        decision = Decision(self._mint(), kind, operation)
        with self._writing(operation) as connection:
            row = connection.execute(
                "INSERT INTO decision (id, kind, operation, request, external_key) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(external_key) DO UPDATE "
                "SET id = CASE WHEN decision.consumed = 1 THEN excluded.id ELSE decision.id END, "
                "operation = CASE WHEN decision.consumed = 1 THEN excluded.operation "
                "ELSE decision.operation END, "
                "request = CASE WHEN decision.consumed = 1 THEN excluded.request "
                "ELSE decision.request END, "
                "answer = CASE WHEN decision.consumed = 1 THEN NULL ELSE decision.answer END, "
                "consumed = 0 RETURNING id, operation",
                (
                    str(decision.id),
                    kind,
                    operation,
                    json.dumps(dict(request), sort_keys=True),
                    key if kind == "question" else None,
                ),
            ).fetchone()
        assert row is not None
        return Decision(UUID(row[0]), kind, str(row[1]))

    def answer(self, decision: Decision, answer: str, *, by_person: bool) -> None:
        """Store one answer under the person authority the Host has already checked."""
        if not by_person:
            raise DoorRefusal("door.person_required", decision.operation)
        if decision.kind == "plugin_activation":
            if answer != "approve":
                raise DoorRefusal("door.decision_answer_invalid", decision.operation)
            answer = "CONFIRMED"
        with self._writing(decision.operation) as connection:
            changed = connection.execute(
                "UPDATE decision SET answer = ? WHERE id = ? AND kind = ? "
                "AND operation = ? AND answer IS NULL AND consumed = 0",
                (answer, str(decision.id), decision.kind, decision.operation),
            )
            if changed.rowcount != 1:
                raise DoorRefusal("door.decision_not_pending", decision.operation)

    def plugin_confirmation(
        self, name: str, fingerprint: str, effects: tuple[str, ...], *, attended: bool
    ) -> tuple[bool, Decision | None]:
        """Lapse older fingerprints and read or ask for this exact workspace confirmation."""
        if not (effects and attended) and not self._has_database(name):
            return not effects, None

        key = _plugin_key(name, fingerprint)
        with self._writing(name) as connection:
            # Keep lapsed decisions by id; a previously seen fingerprint needs a fresh answer.
            connection.execute(
                _PLUGIN_LAPSE + "AND external_key <> ?",
                (name, key),
            )
            if not effects:
                return True, None
            row = connection.execute(
                "SELECT id, answer FROM decision WHERE external_key = ?", (key,)
            ).fetchone()
            if row is not None and row[1] == "CONFIRMED":
                return True, None
            if not attended:
                return False, None
            if row is not None:
                return False, Decision(UUID(row[0]), "plugin_activation", name)
            question = Decision(self._mint(), "plugin_activation", name)
            request = {"plugin_name": name, "fingerprint": fingerprint, "effects": effects}
            connection.execute(
                "INSERT INTO decision (id, kind, operation, request, external_key) "
                "VALUES (?, ?, ?, ?, ?)",
                (str(question.id), question.kind, name, json.dumps(request, sort_keys=True), key),
            )
            return False, question

    def lapse_plugin_confirmation(self, name: str, fingerprint: str) -> None:
        """Retire the exact pinned decision when activation refuses a changed closure."""
        if self._has_database(name):
            with self._writing(name) as connection:
                connection.execute(
                    _PLUGIN_LAPSE + "AND external_key = ?", (name, _plugin_key(name, fingerprint))
                )

    def _has_database(self, operation: str) -> bool:
        """Distinguish an absent workspace store from one whose state cannot be read."""
        try:
            self.path.stat()
        except FileNotFoundError:
            return False
        except OSError as error:
            raise DoorRefusal("door.decisions_unavailable", operation) from error
        return True

    def take(self, decision: Decision) -> str | None:
        """Read a saved answer once. Approval cannot authorize a second run."""
        with self._writing(decision.operation) as connection:
            row = connection.execute(
                "UPDATE decision SET consumed = 1 WHERE id = ? AND kind = ? AND operation = ? "
                "AND consumed = 0 AND answer IS NOT NULL RETURNING answer",
                (str(decision.id), decision.kind, decision.operation),
            ).fetchone()
        return None if row is None else str(row[0])

    def hold_relay(
        self, key: str, operation: str, record: Mapping[str, object], *, activation: bool
    ) -> dict[str, object] | None:
        """Reserve a checked relayed answer. A repeated request reads the first use."""
        kind: DecisionKind = "activation" if activation else "confirmation"
        with self._writing(operation) as connection:
            written = connection.execute(
                "INSERT INTO decision (id, kind, operation, request, answer, consumed, "
                "external_key, nonce) VALUES (?, ?, ?, ?, ?, 1, ?, ?) "
                "ON CONFLICT(external_key) DO NOTHING",
                (
                    str(self._mint()),
                    kind,
                    operation,
                    json.dumps(dict(record)),
                    record["words"],
                    key,
                    record["nonce"],
                ),
            )
            if written.rowcount == 1:
                return None
            row = connection.execute(
                "SELECT request FROM decision WHERE external_key = ?", (key,)
            ).fetchone()
        assert row is not None
        result: dict[str, object] = json.loads(row[0])
        return result

    def release_relay(self, key: str) -> None:
        """Return the person's answer after its bound decision was refused."""
        with self._writing("release_relay") as connection:
            connection.execute(
                "DELETE FROM decision WHERE external_key = ? AND nonce IS NOT NULL", (key,)
            )

    def relay_nonce(self, nonce: str) -> dict[str, object] | None:
        """Look up the prior answer the person's next confirmation refers to."""
        if not self.path.is_file():
            return None
        from sqlite3 import connect  # lazy: most workspaces never open a decision file

        with closing(connect(self.path, timeout=1)) as connection:
            row = connection.execute(
                "SELECT request FROM decision WHERE nonce = ?", (nonce,)
            ).fetchone()
        result: dict[str, object] | None = None if row is None else json.loads(row[0])
        return result

    def spend_approval(self, decision: Decision, request: Mapping[str, object]) -> None:
        """Consume exactly one explicit paid-run approval for its admitted request."""
        with self._writing(decision.operation) as connection:
            granted = connection.execute(
                "UPDATE decision SET consumed = 1 WHERE id = ? AND kind = 'paid_run' "
                "AND operation = ? AND request = ? AND answer = 'approve' AND consumed = 0",
                (str(decision.id), decision.operation, json.dumps(dict(request), sort_keys=True)),
            )
            if granted.rowcount != 1:
                raise DoorRefusal("door.paid_run_approval_required", decision.operation)

    def approve_person(
        self, kind: Literal["confirmation", "activation"], operation: str, binding: str
    ) -> None:
        """Admit a direct person's answer in one transaction before its operation can run."""
        with self._writing(operation) as connection:
            connection.execute(
                "INSERT INTO decision (id, kind, operation, request, answer, consumed) "
                "VALUES (?, ?, ?, ?, 'approve', 1)",
                (str(self._mint()), kind, operation, json.dumps({"decision_hash": binding})),
            )

    def settle_question(self, key: str, operation: str, answer: str) -> None:
        """Complete the question a checked confirmation answered in its own session."""
        if not self.path.is_file():
            return
        with self._writing(operation) as connection:
            connection.execute(
                "UPDATE decision SET answer = ?, consumed = 1 "
                "WHERE kind = 'question' AND external_key = ? AND answer IS NULL",
                (answer, key),
            )


def question_key(code: str, goal: str, session: str) -> str:
    """One pending catalog question for a particular goal and session."""
    return json.dumps(("question", code, goal, session), separators=(",", ":"))


def _plugin_key(name: str, fingerprint: str) -> str:
    """Identify the current decision for one plugin fingerprint within this workspace."""
    return json.dumps(("plugin_activation", name, fingerprint))
