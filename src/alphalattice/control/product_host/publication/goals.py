"""Revisioned goals and the sessions bound to them, on the content store and the Host's lease."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from secrets import token_hex
from threading import RLock
from typing import cast
from uuid import UUID

from alphalattice.control.data_platform.data_decision_rule import approved_audits
from alphalattice.control.workspace_runtime.content_store import (
    ContentAddressedStore,
    ContentAddressedStoreError,
)
from alphalattice.interface.local_application.cli_contract import refusal_words
from alphalattice.interface.local_application.failure_codes import public_failure
from alphalattice.interface.local_application.goals import (
    Goal,
    GoalAcceptedAnswerContext,
    GoalDeclaration,
    GoalSession,
)
from alphalattice.interface.local_application.operations import GRAMMAR

EXPERIENCE_RECORD_WORDS = (
    "Start alphalattice_maintainer in RECORD mode with this goal (goal show {subject}), "
    "the person's corrections in their own words, and the Tasks' readbacks; then tell the "
    "person in one line what was added, merged or deleted."
)


def session_work_declaration(operation: str, vendor: str) -> dict[str, object]:
    """The Host's automatic declaration, also identifying exact historical declarations."""
    command = GRAMMAR[operation]
    name = "Claude Code" if vendor == "claude-code" else "Codex"
    return {
        "title": f"Session work: {command.noun} {command.verb}",
        "objective": (
            f"The work this {name} session began with `{command.noun} {command.verb}`: "
            f"{command.purpose}"
        )[:2400],
        "kind": "RESEARCH",
        "scope": "Opened by the Host at the session's first research request, which held no "
        "goal; the session may open its own goal to replace it.",
        "criteria": [
            {
                "criterion_id": "recorded",
                "text": "The session's requests, Tasks and results are recorded here.",
            }
        ],
    }


def _deliverables(goal: Goal) -> list[dict[str, object]] | None:
    """A complete goal's deliverables, each with what its submission put in it (U54).

    None before the Host saved a submission, which it does only once it found the record
    complete; each deliverable names its kind and the declaration's description.
    """
    if goal.submission is None:
        return None
    declared = {item.deliverable_id: item for item in goal.declaration.deliverables}
    return [
        {
            "deliverable_id": answer.deliverable_id,
            "kind": declared[answer.deliverable_id].kind
            if answer.deliverable_id in declared
            else None,
            "description": declared[answer.deliverable_id].description
            if answer.deliverable_id in declared
            else None,
            "reference_count": len(answer.references),
        }
        for answer in goal.submission.deliverables
    ]


def _metadata(value: Goal) -> dict[str, object]:
    """The same owner facts in discovery rows and the selected question."""
    return {
        "goal_id": str(value.goal_id),
        "goal_hash": value.goal_hash,
        "title": value.declaration.title,
        "objective": value.declaration.objective,
        "target_date": value.declaration.model_dump(mode="json", include={"target_date"}).get(
            "target_date"
        ),
        "kind": value.declaration.kind,
        "research_purpose": value.declaration.research.purpose
        if value.declaration.research
        else None,
        "state": value.state,
        "outcome": value.submission.outcome if value.submission else None,
        "revision": value.revision,
        "recorded_at": value.recorded_at.isoformat(),
        "reference_count": len(value.references),
        "statement_count": len(value.statements),
        "deliverables": _deliverables(value),
        "completion": None
        if value.completion is None
        else {"checked_at": value.completion.checked_at.isoformat()},
    }


def sessions_of(entries: Iterable[Mapping[str, object]]) -> list[dict[str, str]]:
    """The agent sessions a goal's record names, each once, by vendor and session.

    Args:
        entries: The goal's recorded requests and events.

    Returns:
        `{vendor, session_id}` in vendor and session order.
    """
    held = sorted(
        {
            (str(e["agent_vendor"]), str(e["agent_session"]))
            for e in entries
            if e.get("agent_vendor") and e.get("agent_session")
        }
    )
    return [{"vendor": vendor, "session_id": session} for vendor, session in held]


def _goal_item_refusal(goal_id: str, error: BaseException, *, item: str) -> dict[str, object]:
    """Name one unreadable goal record part with its typed recovery words."""
    code = public_failure(error, "goal.head_invalid")
    words = refusal_words(code)
    head = item == "HEAD"
    detail = words.get("detail") or (
        f"Goal {goal_id}'s current revision cannot be read."
        if head
        else f"Goal {goal_id}'s saved session attribution cannot be read."
    )
    way_on = (
        "Read workspace show and the kept backups; use goal schema before deliberately authoring "
        "a new goal. No prior revision is inferred."
        if head
        else "Read workspace show and the kept backups; use goal list to find other readable "
        "records. No session association is inferred."
    )
    return {
        "status": "REFUSED",
        "goal_id": goal_id,
        "item": item,
        "failure_code": code,
        "detail": f"{detail} {way_on}",
        "next_requests": {
            "workspace": {"operation": "WORKSPACE_SHOW"},
            "backups": {"operation": "WORKSPACE_BACKUPS"},
            "goals": {"operation": "GOAL_LIST"},
            **({"schema": {"operation": "GOAL_SCHEMA"}} if head else {}),
        },
    }


class GoalStore:
    """Goal revisions by content, each goal's head, its sessions and its record."""

    def __init__(self, artifact_root: Path, workspace_id: str) -> None:
        """Open the workspace's goal store under its artifacts."""
        self.content = ContentAddressedStore(
            artifact_root / "product-host" / "goals",
            uri_prefix="playpen://product-host/goals",
        )
        self.workspace_id = workspace_id
        # HTTP handlers share this one store under the existing process writer lease.
        self.lock = RLock()

    def load(self, goal_hash: str) -> Goal:
        """One exact revision, verified by its content hash."""
        value = self.content.load_model(
            category="revisions",
            content_hash=goal_hash,
            model=Goal,
            identity_field="goal_hash",
        )
        if value.workspace_id != self.workspace_id:
            raise ValueError("goal.workspace_mismatch")
        return value

    def head(self, goal_id: UUID) -> Goal | None:
        """A goal's current revision, or nothing when no such goal was opened."""
        path = self.content.root / "heads" / f"{goal_id}.json"
        if not path.is_file():
            return None
        try:
            pointer = json.loads(path.read_bytes())
            value = self.load(pointer["goal_hash"])
            if value.goal_id != goal_id:
                raise ValueError
            return value
        except ContentAddressedStoreError:
            # The pointer is valid; its sealed revision keeps absence, corruption or hash refusal.
            raise
        except (KeyError, TypeError, ValueError, OSError) as error:
            raise ValueError("goal.head_invalid") from error

    def publish(self, value: Goal, expected: str | None) -> Goal:
        """Publish the next revision over the head it was written against."""
        with self.lock:
            head = self.head(value.goal_id)
            if head is not None and head.goal_hash == value.goal_hash:
                return head
            ignored = {"goal_hash", "recorded_at", "intent_registered_at", "completion"}
            if head and head.model_dump(exclude=ignored) == value.model_dump(exclude=ignored):
                return head  # lost-response retry: first recorded revision wins
            if (head.goal_hash if head else None) != expected or value.parent_hash != expected:
                raise ValueError("goal.revision_conflict_read_latest")
            if value.workspace_id != self.workspace_id or value.revision != (
                1 if head is None else head.revision + 1
            ):
                raise ValueError("goal.revision_invalid")
            if head is not None and head.state != "OPEN":
                raise ValueError("goal.closed_open_a_follow_up")
            # An interrupted content-first write may leave small unreferenced metadata,
            # never a head naming absent bytes. Old revisions are not overwritten.
            self.content.publish_model(
                category="revisions", value=value, identity_field="goal_hash"
            )
            self.content.atomic_write(
                self.content.root / "heads" / f"{value.goal_id}.json",
                json.dumps({"goal_hash": value.goal_hash}, sort_keys=True).encode(),
            )
            return value

    def _session_path(self, session: GoalSession) -> Path:
        return self.content.root / "sessions" / session.vendor / f"{session.session_id}.json"

    def bind(self, session: GoalSession, goal_id: UUID) -> None:
        """Bind an agent session to one goal, replacing any earlier binding."""
        with self.lock:
            self.content.atomic_write(
                self._session_path(session),
                json.dumps({"goal_id": str(goal_id)}, sort_keys=True).encode(),
            )

    def bound(self, session: GoalSession) -> Goal | None:
        """The open goal a session is bound to, if any; a closed goal binds no more work."""
        path = self._session_path(session)
        if not path.is_file():
            return None
        try:
            goal_id = UUID(json.loads(path.read_bytes())["goal_id"])
        except (KeyError, ValueError, OSError) as error:
            raise ValueError("goal.session_binding_invalid") from error
        head = self.head(goal_id)
        return head if head is not None and head.state == "OPEN" else None

    def event_goal(self, key: str, decide: Callable[[], UUID | None]) -> UUID | None:
        """The goal an external event was filed under at its first receipt, or none.

        The event's row records this, and the activity ledger answers a replay of the same
        producer sequence only with the same content, so a replay reads the first receipt's
        answer rather than deciding again after its session took another goal.
        """
        path = self.content.root / "events" / f"{key}.json"
        decided = None if path.is_file() else decide()
        with self.lock:
            if not path.is_file():
                value = None if decided is None else str(decided)
                self.content.atomic_write(path, json.dumps({"goal_id": value}).encode())
                return decided
            try:
                recorded = json.loads(path.read_bytes())["goal_id"]
                return None if recorded is None else UUID(recorded)
            except (KeyError, TypeError, ValueError, OSError) as error:
                raise ValueError("goal.event_filing_invalid") from error

    def reserve_relay(self, key: str, record: Mapping[str, object]) -> dict[str, object] | None:
        """Hold one relayed yes by its spend key before its decision runs, atomically.

        Returns None when this request holds it now, else the record of the use that holds it,
        whose `nonce` lets the person's next yes to the same decision be told apart.
        """
        path = self.content.root / "relays" / f"{key}.json"
        with self.lock:
            if path.is_file():
                held: dict[str, object] = json.loads(path.read_bytes())
                return held
            self.content.atomic_write(path, json.dumps(dict(record), sort_keys=True).encode())
            return None

    def release_relay(self, key: str) -> None:
        """Give back a held yes whose decision was refused: it was not used."""
        with self.lock:
            (self.content.root / "relays" / f"{key}.json").unlink(missing_ok=True)

    def relay_nonce(self, nonce: str) -> dict[str, object] | None:
        """The used yes that issued this nonce, if the Host issued it."""
        folder = self.content.root / "relays"
        for path in sorted(folder.glob("*.json")) if folder.is_dir() else ():
            record: dict[str, object] = json.loads(path.read_bytes())
            if record.get("nonce") == nonce:
                return record
        return None

    def committee(self, task: str, goal_id: UUID | None = None) -> dict[str, str] | None:
        """The goal whose ledger holds an update's committee floor, and the floor's key seed.

        Given `goal_id`, it is kept for the update unless one already is. The seed stays here,
        outside every goal record, so no read returns it.
        """
        path = self.content.root / "committees" / f"{UUID(task)}.json"
        if goal_id is not None:
            with self.lock:
                if not path.is_file():
                    record = {"goal_id": str(goal_id), "seed": token_hex(32)}
                    self.content.atomic_write(path, json.dumps(record, sort_keys=True).encode())
        if not path.is_file():
            return None
        held: dict[str, str] = json.loads(path.read_bytes())
        return held

    def committee_tasks(self) -> tuple[str, ...]:
        """The updates whose committee floor this store holds, by Task id."""
        folder = self.content.root / "committees"
        return tuple(sorted(path.stem for path in folder.glob("*.json")))

    def attribute(self, goal_id: UUID, entry: Mapping[str, object]) -> None:
        """Keep one request or event a goal's work made: one immutable file each.

        Each file is named by its place in the record, so the record reads back in the order
        it was kept.
        """
        folder = self.content.root / "attribution" / str(goal_id)
        with self.lock:
            place = len(list(folder.glob("*.json"))) if folder.is_dir() else 0
            self.content.atomic_write(
                folder / f"{place:08d}.json", json.dumps(dict(entry), sort_keys=True).encode()
            )

    def accepted_answer_context(
        self, key: str, decide: Callable[[], GoalAcceptedAnswerContext]
    ) -> GoalAcceptedAnswerContext:
        """Freeze acceptance's Goal and prior exact packet in the existing event receipt.

        The scientific owner supplies this callback before optional observation. Existing
        Goal-only receipts stay readable with an unknown packet; a retry never backfills
        them from later assignments or a new Session binding.
        """
        path = self.content.root / "events" / f"{key}.json"
        with self.lock:
            if not path.is_file():
                context = decide()
                self.content.atomic_write(
                    path, json.dumps(context.model_dump(mode="json"), sort_keys=True).encode()
                )
                return context
            try:
                validated = GoalAcceptedAnswerContext.model_validate_json(path.read_bytes())
                if not isinstance(validated, GoalAcceptedAnswerContext):
                    raise ValueError("goal.event_filing_invalid")
                return validated
            except (TypeError, ValueError, OSError) as error:
                raise ValueError("goal.event_filing_invalid") from error

    def goal_ids(self) -> tuple[UUID, ...]:
        """Every goal this workspace holds, by its ID."""
        heads = self.content.root / "heads"
        return (
            tuple(sorted(UUID(path.stem) for path in heads.glob("*.json")))
            if heads.is_dir()
            else ()
        )

    def attributed(self, goal_id: UUID) -> tuple[dict[str, object], ...]:
        """Every request and event recorded under a goal, oldest first."""
        folder = self.content.root / "attribution" / str(goal_id)
        if not folder.is_dir():
            return ()
        try:
            entries = tuple(json.loads(p.read_bytes()) for p in sorted(folder.glob("*.json")))
            if any(not isinstance(entry, Mapping) for entry in entries):
                raise ValueError("goal.attribution_invalid")
            return tuple(dict(entry) for entry in entries)
        except (TypeError, ValueError, OSError) as error:
            raise ValueError("goal.attribution_invalid") from error

    def question(self, goal: Goal, entries: tuple[dict[str, object], ...]) -> str | None:
        """Authored intent, never an automatic session label; old records remain unchanged."""
        if goal.revision == 1 and any(e.get("goal_origin") == "SESSION_WORK" for e in entries):
            return None
        first = next((e for e in entries if e.get("operation")), {})
        operation, vendor = first.get("operation"), first.get("agent_vendor")
        if (
            isinstance(operation, str)
            and operation in GRAMMAR
            and vendor in {"codex", "claude-code"}
            and goal.declaration
            == (
                GoalDeclaration.model_validate(
                    session_work_declaration(str(operation), str(vendor))
                )
            )
        ):
            return None
        return goal.declaration.objective

    def current_question(
        self,
        heads: list[tuple[Goal, dict[str, object]]],
        records: dict[str, tuple[dict[str, object], ...]],
        sessions: dict[str, list[dict[str, str]]],
        task_id: UUID | str | None = None,
        refused: list[dict[str, object]] | None = None,
    ) -> dict[str, object] | None:
        """The dated delivery's bound question, otherwise the newest authored Goal."""
        questions, dated = {}, []
        for goal, metadata in heads:
            key = str(goal.goal_id)
            try:
                if key not in records:
                    records[key] = self.attributed(goal.goal_id)
                entries = records[key]
                if key not in sessions:
                    sessions[key] = sessions_of(entries)
                question = self.question(goal, entries)
            except (ValueError, OSError) as error:
                if refused is not None:
                    refused.append(_goal_item_refusal(key, error, item="SESSION_ATTRIBUTION"))
                continue
            if question is None:
                continue
            questions[key] = {**metadata, "question": question}
            for entry in entries:
                floor = entry.get("committee_open")
                if isinstance(floor, dict) and (
                    date := (floor.get("schedule") or {}).get("entry_session")
                ):
                    dated.append((date, str(floor["update_task_id"]), key))
        if task_id:
            task = str(task_id)
            binding = self.committee(task)
            if binding:
                return questions.get(binding["goal_id"])
            admitted = {
                str(g.goal_id)
                for g, _metadata in heads
                if any(
                    e.get("task_id") == task and e.get("operation") == "RESEARCH_UPDATE_RUN"
                    for e in records.get(str(g.goal_id), ())
                )
            }
            return questions.get(next(iter(admitted))) if len(admitted) == 1 else None
        if dated:
            _date, _task, key = max(dated)
            return questions[key]
        return max(
            questions.values(),
            key=lambda row: (str(row["recorded_at"]), str(row["goal_id"])),
            default=None,
        )

    @staticmethod
    def _ruled(task: str | None, entry: Mapping[str, object]) -> bool:
        """A rule's entry for this dated Task (any, for None), plus its Goal's first-use scope."""
        return entry.get("decided_by") == "RULE" and (
            task in (None, entry.get("task_id"))
            or str(entry.get("delegation", "")).startswith("first-use-goal:")
        )

    @staticmethod
    def rule_cases(task: str, entries: Sequence[Mapping[str, object]]) -> frozenset[str]:
        """The cases the rule chose for this dated Task; its audits are read apart."""
        return frozenset(
            str(entry["case_token"])
            for entry in entries
            if GoalStore._ruled(task, entry) and entry.get("operation") == "DATA_ISSUE_CONFIRM"
        )

    @staticmethod
    def rule_held(task: str | None, entries: Iterable[Mapping[str, object]]) -> frozenset[str]:
        """The cases that held the rule's earlier decision on their anomaly, as the Goal says."""
        return frozenset(
            str(entry["case_token"])
            for entry in entries
            if GoalStore._ruled(task, entry)
            and entry.get("operation") == "DATA_ISSUE_CONFIRM"
            and entry.get("status") == "HELD_BY_RULE"
        )

    @staticmethod
    def rule_audits(
        task: str | None, entries: Iterable[Mapping[str, object]]
    ) -> tuple[tuple[str, str], ...]:
        """Each listing whose full-history audit the rule approved and its cycle admitted."""
        ruled = (entry for entry in entries if GoalStore._ruled(task, entry))
        return tuple(
            sorted(
                {
                    (str(listing), str(entry["rule_hash"]))
                    for entry in approved_audits(ruled)
                    for listing in cast(Sequence[str], entry["listing_ids"])
                }
            )
        )

    def listing(
        self,
        *,
        limit: int,
        cursor: str | None,
        agent_session: str | None = None,
        task_id: UUID | str | None = None,
        records: dict[str, tuple[dict[str, object], ...]] | None = None,
    ) -> dict[str, object]:
        """List scoped Goal metadata and share its verified records within this read."""
        rows = []
        refused: list[dict[str, object]] = []
        sessions: dict[str, list[dict[str, str]]] = {}
        heads: list[tuple[Goal, dict[str, object]]] = []
        records = {} if records is None else records
        for path in (self.content.root / "heads").glob("*.json"):
            try:
                goal_id = UUID(path.stem)
            except ValueError as error:
                # A session-scoped listing cannot safely associate an invalid identifier with
                # the requested session, so it omits that refusal rather than disclose it.
                if agent_session is None:
                    refused.append(_goal_item_refusal(path.stem, error, item="HEAD"))
                continue
            try:
                value = self.head(goal_id)
                if value is None:
                    raise ValueError("goal.head_invalid")
            except (ValueError, OSError) as error:
                if agent_session is not None:
                    # A damaged head is shown only when its still-readable record proves the
                    # requested session belongs to it.
                    try:
                        named = sessions_of(self.attributed(goal_id))
                    except (ValueError, OSError):
                        continue
                    if not any(item["session_id"] == agent_session for item in named):
                        continue
                    sessions[str(goal_id)] = named
                refused.append(_goal_item_refusal(str(goal_id), error, item="HEAD"))
                continue
            if agent_session is not None:
                # A session's goals: read once for every goal, kept for its row (U23, U54).
                try:
                    records[str(goal_id)] = self.attributed(goal_id)
                    named = sessions_of(records[str(goal_id)])
                except (ValueError, OSError):
                    # A scoped result can include only records whose session is established.
                    continue
                if not any(item["session_id"] == agent_session for item in named):
                    continue
                sessions[str(value.goal_id)] = named
            metadata = _metadata(value)
            heads.append((value, metadata))
            rows.append(metadata)
        rows.sort(key=lambda r: (str(r["recorded_at"]), str(r["goal_id"])), reverse=True)
        current = self.current_question(heads, records, sessions, task_id, refused)
        if cursor:
            position = next((i for i, r in enumerate(rows) if r["goal_hash"] == cursor), None)
            if position is None:
                raise ValueError("goal.cursor_moved_reload")
            rows = rows[position + 1 :]
        # Each row's sessions (U23): read for the page alone, from the record that holds them.
        # A bad attribution refuses only that goal item and does not assert an empty session list.
        page_candidates = rows[:limit]
        page = []
        for row in page_candidates:
            page_goal_id = str(row["goal_id"])
            if page_goal_id not in sessions:
                continue
            page.append({**row, "sessions": sessions[page_goal_id]})
        answer: dict[str, object] = {
            "status": "AVAILABLE",
            "current_goal": current,
            "goals": page,
            "next_cursor": (
                page_candidates[-1]["goal_hash"]
                if all((page_candidates, len(rows) > limit))
                else None
            ),
            "claim": "METADATA_DISCOVERY_NOT_EVIDENCE_VERIFICATION",
        }
        if refused:
            answer["refused"] = refused
        return answer
