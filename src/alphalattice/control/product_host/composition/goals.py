"""The Host's revisioned Goal ledger and exact-read submission check (LAWS OP13).

Records declarations, attributed activity and submissions; never judges their truth or
computes research.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, fields, replace
from dataclasses import field as dataclass_field
from datetime import date, datetime, timedelta
from functools import partial
from pathlib import Path
from secrets import token_hex
from typing import Any, Self
from uuid import UUID, uuid5

from pydantic import ValidationError

from alphalattice.control.data_platform.contracts import (
    DataRemediationActorSubmission,
    DataRemediationExecutionReceipt,
)
from alphalattice.control.data_platform.data_decision_rule import (
    ESCALATE,
    RULED_RUNS,
    DataDecisionRule,
    held_decisions,
)
from alphalattice.control.data_platform.data_decision_rule import attach as attach_rule
from alphalattice.control.product_host.composition.first_use_road import (
    daily_road,
    first_use_command,
    undated,
)
from alphalattice.control.product_host.composition.goal_check import (
    goal_schema,
    missing_items,
    offer_completion,
    offered_attach,
    reference_request,
    resubmit,
)
from alphalattice.control.product_host.composition.plain_refusals import explain
from alphalattice.control.product_host.composition.research_workspace import (
    read_research_workspace_manifest,
)
from alphalattice.control.product_host.publication.goals import (
    EXPERIENCE_RECORD_WORDS,
    GoalRefusal,
    GoalStore,
    sessions_of,
)
from alphalattice.foundation.feature_engine.inputs.gateway import FeatureInputAgentCase
from alphalattice.foundation.feature_engine.storage.repositories import PanelStateRepository
from alphalattice.interface.local_application.activity import ExternalActivityEventDocument
from alphalattice.interface.local_application.cli_contract import (
    REQUEST_PROVENANCE,
    RequestProvenance,
    join,
    refusal_words,
    shell,
)
from alphalattice.interface.local_application.failure_codes import (
    located_failure,
    public_failure,
)
from alphalattice.interface.local_application.goals import (
    FIRST_USE_HOURS,
    Goal,
    GoalAcceptedAnswerContext,
    GoalAcceptedAnswerReceipt,
    GoalCompletion,
    GoalDeclaration,
    GoalReference,
    GoalReferenceRequest,
    GoalSession,
    GoalStatement,
    GoalSubmission,
    GoalTaskFact,
    opened_goal,
    relayed_decision,
    target_sessions,
)
from alphalattice.interface.local_application.native_bridge import joined_members
from alphalattice.interface.local_application.operations import FIRST_USE_STEPS
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchOperationRequest,
    PortfolioResearchRequestDocument,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.actor_execution import ActorKind
from alphalattice.protocols.actor_execution.bundles import (
    bundle_directory_key,
    bundle_slot,
    moment_text,
)

GOAL_NAMESPACE = UUID("8f1d6c63-3c3e-4f5f-9f0e-5c4b2a7e8d10")
"""A goal opened without an id is named from its declaration and who opened it, so a retried
open answers the goal it made rather than a second one."""
_SOURCE_DATE_REFUSALS = (
    "workspace_preparation.local_inputs_after_requested_date",
    "data.current_universe_no_feature_ready_listings",
)

TaskFacts = Callable[[UUID], tuple[Any, ...] | None]
"""A Task's kind, lifecycle, update time and succeeded successor, or nothing when absent."""

# Only exact read selections can be evidence. Plans, lists, newest-result defaults,
# actor requests and current activation are deliberately not evidence references.
REFERENCE_OPERATIONS = {
    "EXPERIMENT_CONTROLS": ("DATA_FEATURES",),
    "DATA_UPDATE_READBACK": ("DATA_FEATURES",),
    "EXPERIMENT_READBACK": ("FACTOR_FOUNDATION", "ALPHA", "RISK", "PORTFOLIO"),
    "EXPERIMENT_ALPHA_COMPARE": ("ALPHA",),
    "EXPERIMENT_CURATION": ("FACTOR_FOUNDATION",),
    "EXPERIMENT_FOUNDATION_READBACK": ("FACTOR_FOUNDATION",),
    "EXPERIMENT_COMPARE": ("PORTFOLIO",),
    "EXPERIMENT_RISK_EXPORT": ("RISK",),
    "REPORT": ("PORTFOLIO",),
    "COMPARE": ("PORTFOLIO",),
    "EVIDENCE_CRO_EXPORT": ("EVIDENCE_CRO",),
    "RESEARCH_UPDATE_READBACK": ("PORTFOLIO",),
    # A formula factor's trial and its review packet, each read exactly (V371).
    "FEATURE_TRIAL_READBACK": ("DATA_FEATURES",),
    "FEATURE_REVIEW": ("DATA_FEATURES",),
}
COMPARISON_REFUSALS = frozenset(
    {
        "portfolio_research.comparison_completed_books_required",
        "portfolio_research.comparison_input_or_support_mismatch",
        "portfolio_research.comparison_requires_two_results",
        "alpha_research.saved_comparison_input_binding_hash_mismatch",
        "alpha_research.saved_comparison_target_recipe_binding_hash_mismatch",
        "alpha_research.saved_comparison_target_materialization_binding_hash_mismatch",
        "alpha_research.saved_comparison_ordered_feature_ids_mismatch",
        "alpha_research.saved_comparison_target_values_mismatch",
        "alpha_research.saved_comparison_metric_policy_hash_mismatch",
        "alpha_research.saved_comparison_split_policy_hash_mismatch",
        "alpha_research.saved_comparison_fold_commitment_mismatch",
        "alpha_research.saved_comparison_score_support_mismatch",
    }
)
CONVERSATION_SHOWN = 50
"""A shown record keeps the newest entries; the narrative used by waiters keeps them all."""
ANSWERED = frozenset({"ACCEPTED", "DONE"})
"""An answer submission that closes its bundle: the product accepted it."""

_DECLARATION_TEMPLATE: dict[str, object] = {
    "title": "What the goal is called",
    "objective": "What is to be achieved, in one sentence.",
    "kind": "RESEARCH",
    "criteria": [{"criterion_id": "done", "text": "What completes it, in one sentence."}],
    "deliverables": [
        {
            "deliverable_id": "result",
            "kind": "RESULT",
            "description": "The exact result it delivers.",
        }
    ],
}
"""The shortest goal declaration, valid as it stands, its words for the author to replace
(V378); `goal schema` prints every field."""

MESSAGE_EVENT = "NATIVE_COORDINATION_MESSAGE"
_EVENT_FIELDS = (
    ("agent_id", "native_agent_id"),
    ("role", "role"),
    ("message_kind", "message_kind"),
    ("message_id", "message_id"),
    ("message_sha256", "message_sha256"),
    ("recipient_id", "recipient_id"),
    ("reply_to", "reply_to"),
    ("reference", "reference"),
    ("input_channel", "input_channel"),
    ("source_time_kind", "source_time_kind"),
    ("submitted_by", "submitted_by"),
    ("bundle_reference", "bundle_reference"),
    ("answer_reference", "answer_reference"),
    ("authorship_basis", "authorship_basis"),
    ("bundle_role", "bundle_role"),
)
USAGE_EVENT = "NATIVE_AGENT_USAGE"
_USAGE_COUNTS = (
    "responses",
    "input_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "output_tokens",
)
_USAGE_FIELDS = (
    "model",
    "efforts",
    "pin_differs",
    *_USAGE_COUNTS,
    "last_at",
    "source_kind",
    "sample_time_kind",
    "sampled_at",
)
_NATIVE_USAGE_CHANNELS = frozenset({"CODEX_SESSION_FILE", "CLAUDE_CODE_SESSION_FILE"})


def _usage_metadata(entry: Mapping[str, object]) -> dict[str, object]:
    """Project the recorded source once and retain the actual latest usage-record time."""
    metadata = {
        name: entry[name]
        for name in ("source_kind", "sample_time_kind", "sampled_at")
        if name in entry
    }
    channel = entry.get("input_channel")
    if (
        "source_kind" not in metadata
        and isinstance(channel, str)
        and channel in _NATIVE_USAGE_CHANNELS
    ):
        metadata["source_kind"] = channel
    if (
        metadata.get("sample_time_kind") == "LATEST_USAGE_RECORD_AT"
        and isinstance(entry.get("last_at"), str)
        and entry["last_at"]
    ):
        metadata["sampled_at"] = entry["last_at"]
    return metadata


def assignment_records(entries: Sequence[Mapping[str, Any]], goal: Goal) -> list[dict[str, Any]]:
    """Acceptance completes an exact bundle; closing the Goal archives unanswered work."""
    answered = {
        entry.get("bundle_reference"): entry
        for entry in entries
        if (entry.get("operation") == "AGENT_ANSWER_SUBMIT" and entry.get("status") in ANSWERED)
        or (
            entry.get("event_kind") == MESSAGE_EVENT
            and (
                (entry.get("input_channel"), entry.get("message_kind"))
                == ("PRODUCT_ACCEPTED_ANSWER", "answer")
                or (
                    entry.get("input_channel"),
                    entry.get("message_kind"),
                    entry.get("authorship_basis"),
                )
                == ("PRODUCT_REVIEW", "acceptance", "HOST_ACCEPTED")
            )
        )
    }
    prepared: dict[str, dict[str, Any]] = {}
    for entry in entries:
        reference = entry.get("bundle_reference")
        if (
            entry.get("operation") == "AGENT_BUNDLE_PREPARE"
            and entry.get("status") == "AGENT_BUNDLE_READY"
            and isinstance(reference, str)
        ):
            fields = {key: entry.get(key) for key in ("bundle_reference", "agent_role", "task_id")}
            prepared.setdefault(reference, {**fields, "prepared_at": entry.get("recorded_at")})
    for reference, row in prepared.items():
        answer = answered.get(reference)
        row["state"] = "COMPLETED" if answer else "ARCHIVED" if goal.state != "OPEN" else "OPEN"
        if answer:
            row["receipt"] = {
                key: answer[key]
                for key in ("recorded_at", "status", "answer_reference", "observation_id")
                if key in answer
            }
        elif goal.state != "OPEN":
            row["receipt"] = {"goal_id": str(goal.goal_id), "goal_hash": goal.goal_hash}
        else:
            row.update(
                reason="No accepted answer is recorded.",
                next_action="Ask the assigned specialist to submit the prepared answer.",
            )
    return list(prepared.values())


def conversation_entries(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The goal's conversation: what the product recorded of its Sessions' work, in time order.

    Each request a bound Session sent under the goal is a product fact, and so is an accepted
    answer filed beside it; messages filed before FLOW-1 stay as recorded.
    """
    rows: list[dict[str, Any]] = []
    for number, entry in enumerate(entries):
        if entry.get("event_kind") == MESSAGE_EVENT:
            rows.append(dict(entry))
        elif "operation" in entry and entry.get("agent_session"):
            rows.append(
                {
                    "recorded_at": entry.get("recorded_at"),
                    "message_kind": entry["operation"],
                    "message_id": f"request-{number}",
                    "input_channel": "PRODUCT_OPERATION",
                    "summary": str(entry.get("status") or ""),
                    "agent_vendor": entry.get("agent_vendor"),
                    "agent_session": entry.get("agent_session"),
                    "agent_id": entry.get("agent_session"),
                    **{
                        name: entry[name]
                        for name in ("task_id", "bundle_reference", "agent_role")
                        if entry.get(name)
                    },
                }
            )
    return sorted(rows, key=lambda row: str(row.get("recorded_at") or ""))


def session_usage(entries: Sequence[Mapping[str, object]]) -> list[dict[str, Any]]:
    """Each session's agents with their latest reading by model, without an unproved sum (AU).

    A reading is one participant's cumulative count from its admitted session file, the latest
    replacing the previous; parent and child readings are never added. One that differs from
    its role card says so (``pin_differs``).
    """
    readings: dict[tuple[str, str, str, str], Mapping[str, object]] = {}
    for entry in entries:
        key = (
            str(entry.get("agent_vendor")),
            str(entry.get("agent_session")),
            str(entry.get("agent_id")),
        )
        if entry.get("event_kind") == USAGE_EVENT and entry.get("model"):
            readings[(*key, str(entry["model"]))] = entry
    sessions: dict[tuple[str, str], dict[str, Any]] = {}
    for (vendor, session, agent, model), entry in sorted(readings.items()):
        texts = {name: str(entry.get(name, "")) for name in _USAGE_COUNTS}
        counts = {
            name: int(text) for name, text in texts.items() if text.isascii() and text.isdigit()
        }
        held = sessions.setdefault(
            (vendor, session),
            {"vendor": vendor, "session_id": session, "participants": {}},
        )
        participant = held["participants"].setdefault(
            agent,
            {"agent_id": agent, "role": entry.get("role"), "models": [], "pin_differs": []},
        )
        participant["models"].append(
            {
                "model": model,
                "efforts": [e for e in str(entry.get("efforts") or "").split(",") if e],
                **counts,
                "last_at": entry.get("last_at"),
                **_usage_metadata(entry),
            }
        )
        differs = {*participant["pin_differs"], *str(entry.get("pin_differs") or "").split(",")}
        participant["pin_differs"] = sorted(differs - {""})
    return [
        {
            "vendor": held["vendor"],
            "session_id": held["session_id"],
            "participants": list(held["participants"].values()),
            "aggregation": "NOT_COMBINED",
        }
        for held in sessions.values()
    ]


@dataclass(slots=True)
class GoalReferenceReadScope:
    """One goal request's verified owner reads, never a cross-request cache.

    References naming one owner operation reuse its response here, each checking its own
    sealed identity. As a context manager it bounds the Alpha owner's verified chunk tables to
    this request: a task cited twice is read once, the next request afresh.
    """

    _responses: dict[str, dict[str, Any]] = dataclass_field(default_factory=dict)
    _artifacts: ExitStack | None = None
    owner_read_count: int = 0
    owner_read_cache_hits: int = 0
    reference_verification_count: int = 0

    def __enter__(self) -> Self:
        """Open this request's scope for the owners' verified reads."""
        stack = ExitStack()
        self._artifacts = stack
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close the scope and release the reads it held."""
        stack, self._artifacts = self._artifacts, None
        if stack is not None:
            stack.close()

    def read(
        self,
        *,
        document: Mapping[str, Any],
        request: PortfolioResearchOperationRequest,
        reader: Callable[[PortfolioResearchOperationRequest, str], dict[str, Any]],
        caller: str,
    ) -> dict[str, Any]:
        """One owner read, answered once per request for the same exact document."""
        key = canonical_hash(dict(document))
        cached = self._responses.get(key)
        if cached is not None:
            self.owner_read_cache_hits += 1
            return cached
        body = reader(request, caller)
        self._responses[key] = body
        self.owner_read_count += 1
        return body

    def counts(self) -> dict[str, int | bool]:
        """How many owner reads this request made, cached and verified."""
        return {
            "request_scoped": True,
            "reference_verification_count": self.reference_verification_count,
            "owner_read_count": self.owner_read_count,
            "owner_read_cache_hits": self.owner_read_cache_hits,
        }


_SESSION_GOAL_OPERATIONS = frozenset(
    {
        "GOAL_SHOW",
        "GOAL_NARRATIVE",
        "GOAL_PROFILE",
        "GOAL_EXPORT",
        "GOAL_ATTACH",
        "GOAL_NOTE",
        "GOAL_REVISE",
        "GOAL_SUBMIT",
        "GOAL_ABANDON",
    }
)
"""The operations that take the session's own goal when the request names none (V391); a
write without its hash applies to the goal's current revision."""


class GoalApplication:
    """The goal ledger's owner: its operations, its record and its check (LAWS OP13)."""

    def __init__(
        self,
        store: GoalStore,
        clock: Callable[[], datetime],
        read: Callable[[PortfolioResearchOperationRequest, str], dict[str, Any]],
        admitted_at: Callable[[UUID], datetime],
        task_facts: TaskFacts,
        *,
        workspace: Path,
        preparation_completed: Callable[[], bool] = lambda: False,
    ) -> None:
        """Bind the store, clock, owners' reader, Task facts; handoffs name `workspace` (V492)."""
        self.store, self.clock, self.read = store, clock, read
        self.admitted_at, self.task_facts = admitted_at, task_facts
        self.workspace, self.preparation_completed = workspace, preparation_completed
        self.profile: Callable[[Goal], dict[str, Any]] = lambda _: {
            "status": "REFUSED",
            "failure_code": "activity.storage_unavailable",
        }

    # ------------------------------------------------------------------ attribution

    def attributed_goal(self, provenance: RequestProvenance | Goal | None) -> Goal | None:
        """The current Goal snapshot or the request's attributed Goal; neither grants authority.

        A named goal that is absent or closed is refused by name, so no work is recorded
        under a goal it cannot count toward (OP13).
        """
        if isinstance(provenance, Goal):
            return provenance
        if provenance is None:
            return None
        if provenance.goal_id is not None:
            head = self.store.head(UUID(str(provenance.goal_id)))
            if head is None:
                raise ValueError("goal.not_found")
            if head.state != "OPEN":
                raise ValueError("goal.closed_open_a_follow_up")
            return head
        session = self._session(provenance)
        return None if session is None else self.store.bound(session)

    def first_use(self) -> Goal | None:
        """The workspace's one first-use goal, if opened (V452); its sentence and date reuse it."""
        heads = (self.store.head(goal_id) for goal_id in self.store.goal_ids())
        return next((h for h in heads if h and h.declaration.kind == "FIRST_USE"), None)

    def network_delegation(
        self, subject: Goal | RequestProvenance | None, caller: str
    ) -> str | None:
        """The attributed Goal's current preparation network authority, shared by both entries."""
        try:
            goal = self.attributed_goal(subject)
        except ValueError:
            return None
        return None if goal is None else self.delegation(goal, "NETWORK_ACCESS_SET", caller)

    def prepare(
        self,
        request: PortfolioResearchOperationRequest,
        caller: str,
        plan: Callable[..., dict[str, Any]],
    ) -> dict[str, Any]:
        """Plan within the first use's authority, cutoff and immutable-date recovery."""
        goal = self.first_use()
        try:
            answer = plan(
                network_delegation=self.network_delegation(REQUEST_PROVENANCE.get(), caller),
                recovery_task_id=request.recovery_task_id,
                **self.preparation_scope(goal),
            )
        except ValueError as error:
            if str(error) not in _SOURCE_DATE_REFUSALS:
                raise
            answer = {
                "status": "REFUSED",
                "failure_code": public_failure(error, "workspace_preparation.refused"),
            }
        return self.preparation_recovery(answer, goal)

    def preparation_scope(self, goal: Goal | None = None) -> dict[str, Any]:
        """An open workspace first use caps preparation, independently of caller authority."""
        goal = goal or self.first_use()
        named = goal.declaration.target_date if goal and goal.state == "OPEN" else None
        dated = {} if named is None else target_sessions(named, self.clock())
        formation, cutoff = dated.get("formation_session"), dated.get("information_cutoff_at")
        return {
            "requested_formation_session": date.fromisoformat(str(formation))
            if formation
            else None,
            "requested_information_cutoff_at": datetime.fromisoformat(str(cutoff))
            if cutoff
            else None,
        }

    def preparation_recovery(
        self,
        answer: dict[str, Any],
        subject: Goal | RequestProvenance | None,
        code: str | None = None,
    ) -> dict[str, Any]:
        """Keep the original Goal, or explicitly choose a different date in a new workspace."""
        code = code or answer.get("failure_code") or answer.get("latest_failure_code")
        if code not in _SOURCE_DATE_REFUSALS:
            return answer
        subject = subject or self.first_use()
        try:
            subject = self.attributed_goal(subject)
        except ValueError:
            return answer
        if (
            subject is None
            or subject.declaration.kind != "FIRST_USE"
            or subject.state != "OPEN"
            or subject.declaration.target_date is None
        ):
            return answer
        nearest = None
        source = answer.get("source_inputs") or {}
        if source.get("readiness_status") == "RESEARCH_READY" and source.get(
            "panel_through"
        ) == answer.get("source_through"):
            through = date.fromisoformat(source["panel_through"])
            nearest = target_sessions(through + timedelta(days=1))["entry_session"]
        gid = str(subject.goal_id)
        objective, named = subject.declaration.objective, str(nearest or "<supported-date>")
        requests = dict(answer.get("next_requests") or {})
        if code == "data.current_universe_no_feature_ready_listings":
            requests.clear()
        requests.setdefault("readback", {"operation": "WORKSPACE_PREPARE_READBACK"})
        requests["goal"] = {"operation": "GOAL_SHOW", "goal_id": gid}
        requests["abandon"] = {"operation": "GOAL_ABANDON", "goal_id": gid, "change_reason": code}
        return {
            **answer,
            **refusal_words(code),
            **(
                {
                    "detail": "The acquired source admits no listing with sufficient research "
                    "history and quality for the requested date. Features and fits did not begin. "
                    "Source qualification is terminal; read the offered preparation state. "
                    "This acquired source supports no lawful date nearby. Any existing Goal keeps "
                    "its original date, and each workspace has one first use; after any offered "
                    "abandonment, ask the person to choose a new workspace and a supported date "
                    "for its first-use command.",
                    "next_action": "READ_THE_SOURCE_SUPPORT_AND_KEEP_THE_GOAL_DATE",
                }
                if code == "data.current_universe_no_feature_ready_listings"
                else {}
            ),
            "nearest_date": nearest,
            "nearest_date_support": "QUALIFIED_LOCAL_INPUTS" if nearest else "UNAVAILABLE",
            "new_workspace": {
                "choose": ["workspace", *([] if nearest else ["supported_date"])],
                "command_template": first_use_command("<new-workspace>", objective, named),
            },
            "next_requests": requests,
        }

    @staticmethod
    def delegation_ends(goal: Goal) -> datetime:
        """When a first-use goal's delegation ends: its hours from its opening (V452)."""
        return goal.intent_registered_at + timedelta(hours=FIRST_USE_HOURS)

    def first_use_open(self, goal: Goal) -> bool:
        """Whether a first-use goal still delegates: open, and within its hours."""
        return (
            goal.declaration.kind == "FIRST_USE"
            and goal.state == "OPEN"
            and self.clock() < self.delegation_ends(goal)
        )

    def first_use_delegation(self, goal: Goal) -> dict[str, Any]:
        """The delegation's authority and readable deadline."""
        end = self.delegation_ends(goal)
        return {
            "steps": sorted(FIRST_USE_STEPS),
            "ends_at": end.isoformat(),
            "ends_at_reading": moment_text(end),
            "active": self.first_use_open(goal),
        }

    def delegation(self, goal: Goal, operation: str, caller: str) -> str | None:
        """The person's delegation one of the goal's requests acts under, if any (V452, OP19).

        An open first-use goal within its hours delegates its steps (`FIRST_USE_STEPS`) to its
        agent; every other step, and every step after, stays the person's.

        Args:
            goal: The goal the request works for.
            operation: The request's operation.
            caller: Who sent it.

        Returns:
            The delegation's name, or None when the step is not delegated.
        """
        if caller == "HUMAN" or operation not in FIRST_USE_STEPS or not self.first_use_open(goal):
            return None
        if operation == "DATA_ISSUE_CONFIRM" and not self._own_preparation_stopped(goal):
            return None
        return f"first-use-goal:{goal.goal_id}"

    def data_decisions(self, goal: Goal | None, caller: str) -> str | None:
        """The delegation under which this caller decides its first use's data issues, if any.

        The issue list and the preview name it so that the agent confirms the case itself, under
        the goal, rather than asking the person for a grant.
        """
        return None if goal is None else self.delegation(goal, "DATA_ISSUE_CONFIRM", caller)

    @staticmethod
    def _data_index(
        entries: Sequence[Mapping[str, object]],
    ) -> dict[tuple[object, ...], list[Mapping[str, object]]]:
        index: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
        for entry in entries:
            if entry.get("operation") == "DATA_ISSUE_CONFIRM":
                key = tuple(entry.get(k) for k in ("case_token", "option_id", "rule_hash"))
                index.setdefault(key, []).append(entry)
        return index

    def _data_rule_scope(
        self, task_id: UUID, goal_hash: str | None = None, goal_id: UUID | None = None
    ) -> tuple[Goal, tuple[dict[str, object], ...], str] | None:
        """The exact open Goal owns this Task; a dated book uses workspace policy."""
        kind = (self.task_facts(task_id) or (None,))[0]
        goals = (
            [self.store.load(goal_hash)]
            if goal_hash
            else [self.store.head(g) for g in self.store.goal_ids()]
        )
        found = []
        for goal in goals:
            if (
                goal is None
                or goal.state != "OPEN"
                or (goal_id is not None and goal.goal_id != goal_id)
            ):
                continue
            entries = self.store.attributed(goal.goal_id)
            own = [e for e in entries if e.get("task_id") == str(task_id)]
            if own and self.first_use_open(goal):
                found.append((goal, entries, "first-use-goal"))
            elif kind in RULED_RUNS and any(e.get("operation") == RULED_RUNS[kind] for e in own):
                found.append((goal, entries, "workspace-rule-goal"))
        return found[0] if len(found) == 1 else None

    def data_rule(self, task_id: UUID) -> Callable[..., Any] | None:
        """Keep each bounded rule choice in its owning open Goal, with readable authority."""
        scope_owner = self._data_rule_scope(task_id)
        if scope_owner is None:
            return None
        goal, entries, authority = scope_owner
        manifest = read_research_workspace_manifest(self.workspace)
        rule = manifest.data_decision_rule or DataDecisionRule()
        if rule.max_cases == 0 or rule.max_population_share == 0:
            return None
        delegation = f"{authority}:{goal.goal_id}"
        panel = PanelStateRepository(self.workspace)
        index = self._data_index(entries)
        recorded = set(index)
        scope_evidence: Any = None
        scope: dict[str, Any] = {}
        held: dict[Any, str] = {}

        def live() -> Any:
            head = self.store.head(goal.goal_id)
            held = head is not None and (
                self.first_use_open(head) if authority == "first-use-goal" else head.state == "OPEN"
            )
            return head if held else None

        def decide(case: Any, evidence: Any, _impact: Any, _budget: Any) -> Any:
            nonlocal scope_evidence, scope, held
            if (current := live()) is None:
                return rule.submission(case, ESCALATE, str(goal.goal_id), authority=authority)
            if evidence is not scope_evidence:
                cases = tuple(
                    map(
                        FeatureInputAgentCase.read_document,
                        panel.feature_input_case_documents(case.manifest_revision),
                    )
                )
                scope_evidence = evidence
                active = panel.active_listing_quarantines(
                    case.manifest_revision, include_profile_history=True
                )
                held = held_decisions(
                    panel.feature_input_decisions(), rule.rule_hash, {q.listing_id for q in active}
                )
                scope = rule.scope(
                    cases,
                    len({item.listing_id for item in evidence}),
                    (
                        e
                        for e in self.store.attributed(goal.goal_id)
                        if e.get("task_id") == str(task_id)
                    ),
                    held,
                )
            available = case.case_token in scope["case_tokens"]
            option, status = rule.ruling(case, evidence, scope, held)
            choice = (case.case_token, option, rule.rule_hash)
            if choice not in recorded:
                self.store.attribute(
                    goal.goal_id,
                    {
                        "recorded_at": self.clock().isoformat(),
                        "operation": "DATA_ISSUE_CONFIRM",
                        "decided_by": "RULE",
                        "status": status,
                        "task_id": str(task_id),
                        "delegation": delegation,
                        "goal_hash": current.goal_hash,
                        "authority": authority,
                        "case_token": case.case_token,
                        "option_id": option,
                        "listing_ids": list(case.listing_ids),
                        "rule_hash": rule.rule_hash,
                        **{
                            k: scope[k]
                            for k in ("case_count", "affected_listing_count", "population_count")
                        },
                        "escalation_reason": None
                        if status != "ESCALATED"
                        else "CASE_CATALOG_UNAVAILABLE"
                        if not available
                        else "RULE_BOUND_EXCEEDED"
                        if scope["capped"]
                        else None,
                    },
                )
                recorded.add(choice)
            return rule.submission(case, option, str(goal.goal_id), authority=authority)

        own = {"task_id": str(task_id), "delegation": delegation, "authority": authority}
        return attach_rule(
            decide,
            rule,
            live,
            partial(self.store.attributed, goal.goal_id),
            lambda entry: self.store.attribute(
                goal.goal_id, {"recorded_at": self.clock().isoformat(), **own, **entry}
            ),
            str(task_id),
        )

    def resume_decided(
        self,
        goal: Goal,
        request: PortfolioResearchOperationRequest,
        body: dict[str, object],
        send: Callable[[PortfolioResearchOperationRequest], dict[str, object]],
    ) -> dict[str, object]:
        """The last owed decision resumes the first use's own stopped preparation in the same act.

        The same Task continues at its stage under the goal's delegation: no plan,
        run or wait turn. Any other answer passes as it is.
        """
        requests = body.get("next_requests")
        if (
            request.operation != "DATA_ISSUE_CONFIRM"
            or body.get("decisions_complete") is not True
            or not self.first_use_open(goal)
            or not isinstance(requests, dict)
        ):
            return body
        own = {e.get("task_id") for e in self.store.attributed(goal.goal_id)}
        for key, value in requests.items():
            task = key.removeprefix("continue:")
            facts = self.task_facts(UUID(task)) if key != task and task in own else None
            if facts is not None and facts[:2] == ("workspace_preparation", "BLOCKED"):
                try:
                    continued = send(PortfolioResearchOperationRequest(**value))
                except ValueError as error:
                    code = public_failure(error, "workspace_preparation.refused")
                    continued = {"failure_code": code, **refusal_words(code), **explain(code)}
                if not continued.get("task_id"):
                    return {
                        **body,
                        "continuation_problem": {
                            k: v
                            for k, v in continued.items()
                            if k not in {"status", "outcome", "next_requests"}
                        },
                        "next_requests": continued.get("next_requests") or requests,
                    }
                return {
                    **body,
                    "continued": continued,
                    "task_id": continued["task_id"],
                    "lifecycle": continued.get("lifecycle"),
                    "next_action": "WAIT_FOR_TASK",
                    "next_requests": {
                        "issues": {"operation": "DATA_ISSUES"},
                        "status": {"operation": "STATUS", "task_id": continued["task_id"]},
                    },
                }
        return body

    @staticmethod
    def _recorded_data_actor(
        entry: Mapping[str, object], actor: DataRemediationActorSubmission, rule: str
    ) -> bool:
        """A saved choice names either its exact rule actor or its actual agent session."""
        if actor.actor_kind is ActorKind.HOST_FALLBACK:
            if (entry.get("decided_by"), entry.get("rule_hash"), actor.actor_id) == (
                "RULE",
                rule,
                f"host-rule:{rule[:16]}",
            ):
                return True
        elif (
            entry.get("status") == "CONFIRMED_PENDING_REVALIDATION"
            and isinstance(vendor := entry.get("agent_vendor"), str)
            and isinstance(session := entry.get("agent_session"), str)
            and vendor
            and session
            and actor.actor_id == f"{vendor}:{session}"
        ):
            return True
        return False

    def recorded_data_confirmation(
        self, receipt: DataRemediationExecutionReceipt, authority: dict[UUID, Any] | None = None
    ) -> bool:
        """Verify original rule or delegated authority for this exact retained decision."""
        actor = receipt.actor_submission
        rationale, _, rule = receipt.submission.proposal.rationale.partition("; rule:")
        if not all(
            (
                actor.actor_kind in {ActorKind.EXTERNAL_AUTOMATION, ActorKind.HOST_FALLBACK},
                rationale.startswith(("first-use-goal:", "workspace-rule-goal:")),
            )
        ):
            return False
        workspace_rule = rationale.startswith("workspace-rule-goal:")
        if workspace_rule and actor.actor_kind is not ActorKind.HOST_FALLBACK:
            return False
        try:
            goal_id = UUID(rationale.split(":", 1)[1])
            memo = authority if authority is not None else {}
            if goal_id not in memo:
                head = self.store.head(goal_id)
                memo[goal_id] = (
                    head,
                    self._data_index(self.store.attributed(goal_id)) if head else {},
                )
            goal, index = memo[goal_id]
            if goal is None:
                return False
            if not all(
                (
                    workspace_rule or goal.declaration.kind == "FIRST_USE",
                    rationale
                    == ("workspace-rule-goal:" if workspace_rule else "first-use-goal:")
                    + str(goal.goal_id),
                )
            ):
                return False
            key = (
                receipt.submission.proposal.case_token,
                receipt.submission.proposal.option_id,
                rule if actor.actor_kind is ActorKind.HOST_FALLBACK else None,
            )
            for entry in index.get(key, ()):
                # The index already binds operation, case, option and rule.
                if entry.get("delegation") != rationale:
                    continue
                if workspace_rule:
                    scoped = self._data_rule_scope(
                        UUID(str(entry["task_id"])), str(entry["goal_hash"]), goal_id
                    )
                    if not all(
                        (scoped is not None, entry.get("authority") == "workspace-rule-goal")
                    ):
                        continue
                recorded_at = entry.get("recorded_at")
                if not isinstance(recorded_at, str):
                    continue
                stamped = datetime.fromisoformat(recorded_at)
                if not all(
                    (
                        goal.intent_registered_at <= stamped,
                        workspace_rule or stamped < self.delegation_ends(goal),
                    )
                ):
                    continue
                if self._recorded_data_actor(entry, actor, rule):
                    return True
        except (KeyError, OSError, TypeError, ValueError):
            return False
        return False

    def _own_preparation_stopped(self, goal: Goal) -> bool:
        """Whether a preparation this first use admitted is stopped, its data decisions owed.

        The delegation decides only its own preparation's issues, and none once it ends (OP19).
        """
        for entry in self.store.attributed(goal.goal_id):
            task = entry.get("task_id")
            facts = self.task_facts(UUID(str(task))) if task else None
            if facts is not None and facts[:2] == ("workspace_preparation", "BLOCKED"):
                return True
        return False

    def network_left_open(self, goal: Goal) -> bool:
        """Whether a first-use goal that has ended left open the network its delegation opened.

        Args:
            goal: The workspace's first-use goal.

        Returns:
            True when its delegation's last network step opened it and the goal is over.
        """
        if goal.declaration.kind != "FIRST_USE" or self.first_use_open(goal):
            return False
        opened = False
        for entry in self.store.attributed(goal.goal_id):
            if entry.get("operation") == "NETWORK_ACCESS_SET" and entry.get("delegation"):
                opened = entry.get("network_enabled") is True
        return opened

    def relayed(
        self,
        request: PortfolioResearchOperationRequest,
        caller: str,
        goal: Goal | None,
        provenance: RequestProvenance | None,
    ) -> dict[str, object] | None:
        """The person's yes an agent's request relays, held for it before it runs.

        It answers only its bound decision, from the session whose goal the request works for,
        once: the same words to the same question are refused while held or used, naming the
        nonce for the person's next yes. The Host's clock dates it; `release_relay` gives back
        one whose decision was refused.
        """
        confirmation = request.person_confirmation
        if confirmation is None or caller == "HUMAN":
            return None
        session = self._session(provenance)
        held = None if session is None else self.store.bound(session)
        if goal is None or held is None or held.goal_id != goal.goal_id:
            raise ValueError("person_confirmation.session_required")
        decision = relayed_decision(
            {field.name: getattr(request, field.name) for field in fields(request)}, confirmation
        )
        if confirmation.nonce is not None:
            earlier = self.store.relay_nonce(confirmation.nonce)
            if earlier is None or earlier.get("decision_hash") != decision:
                raise ValueError("person_confirmation.repeat_unknown")
        assert session is not None
        key = canonical_hash(
            [decision, confirmation.words, confirmation.question, confirmation.nonce]
        )
        record: dict[str, object] = {
            "question": confirmation.question,
            "words": confirmation.words,
            "decision_hash": decision,
            "repeat_of": confirmation.nonce,
            "relayed_at": self.clock().isoformat(),
            "agent_vendor": session.vendor,
            "agent_session": session.session_id,
            "nonce": token_hex(16),
        }
        used = self.store.reserve_relay(key, record)
        if used is not None:
            raise ValueError(f"person_confirmation.already_used:{used['nonce']}")
        return {**record, "key": key}

    def attribute(
        self,
        goal: Goal,
        request: PortfolioResearchOperationRequest,
        body: Mapping[str, Any],
        provenance: RequestProvenance | None,
        delegation: str | None = None,
        relayed: Mapping[str, object] | None = None,
    ) -> None:
        """Record one request the goal's work made, and the Task it started, if any.

        A step the person delegated is recorded as theirs, by the delegation that carried it
        (V452), and one they confirmed by the yes the agent relayed, whole; a network step
        records what it set.
        """
        receipt = body.get("receipt")
        task = (
            body.get("task_id")
            or body.get("publication_task_id")
            or (receipt.get("task_id") if isinstance(receipt, Mapping) else None)
        )
        bundle = body.get("bundle_reference")
        if (
            bundle is None
            and request.operation == "AGENT_ANSWER_SUBMIT"
            and request.bundle_directory
        ):
            # An Analyst's or CRO's answer names its bundle by directory; the record is its slot.
            bundle = bundle_slot(bundle_directory_key(request.bundle_directory))
        self.store.attribute(
            goal.goal_id,
            {
                "recorded_at": self.clock().isoformat(),
                "operation": request.operation,
                "status": str(body.get("status", "")),
                "task_id": str(task) if task else None,
                **{
                    key: value
                    for key, value in (
                        (
                            "plan_hash",
                            body.get("plan_hash")
                            if request.operation == "EXPERIMENT_PLAN"
                            else None,
                        ),
                        ("review_publication_hash", body.get("review_publication_hash")),
                        ("case_token", request.data_issue_case_token),
                        ("option_id", request.data_issue_option_id),
                    )
                    if value is not None
                },
                **(
                    {"feature_trial_id": str(body["feature_trial_id"])}
                    if body.get("feature_trial_id")
                    else {}
                ),
                **(
                    {"bundle_reference": str(bundle)}
                    if bundle
                    and request.operation in {"AGENT_BUNDLE_PREPARE", "AGENT_ANSWER_SUBMIT"}
                    else {}
                ),
                **(
                    {"agent_role": str(body["agent_role"])}
                    if request.operation == "AGENT_BUNDLE_PREPARE" and body.get("agent_role")
                    else {}
                ),
                "agent_vendor": provenance.vendor if provenance else None,
                "agent_session": provenance.session if provenance else None,
                **({"delegation": delegation} if delegation else {}),
                **({"relayed": dict(relayed)} if relayed else {}),
                **(
                    {"network_enabled": request.network_enabled}
                    if request.operation == "NETWORK_ACCESS_SET"
                    else {}
                ),
            },
        )

    def decision_attribution(self) -> dict[tuple[str, str], set[str]]:
        """Exact retained request selectors and the Goals they counted toward.

        Legacy absent selectors stay unknown; reading adds no attribution.
        """
        indexed: dict[tuple[str, str], set[str]] = {}
        for goal_id in self.store.goal_ids():
            for entry in self.store.attributed(goal_id):
                if "operation" not in entry:
                    continue
                for key in ("task_id", "plan_hash", "review_publication_hash", "case_token"):
                    value = entry.get(key)
                    if value is not None:
                        if not isinstance(value, str) or not value:
                            raise ValueError("goal.attribution_invalid")
                        indexed.setdefault((key, value), set()).add(str(goal_id))
        return indexed

    def accepted_answer_context(
        self,
        *,
        bundle_reference: str,
        answer_reference: str,
        session: GoalSession,
        first_submission: bool,
        provenance: RequestProvenance | None,
    ) -> GoalAcceptedAnswerContext:
        """Keep the first accepted receipt's Goal before optional filing.

        An older answer without this metadata records no Goal, rather than borrowing present
        work. The existing write-once event receipt holds this context.
        """
        key = canonical_hash(
            ["PRODUCT_ACCEPTED_ANSWER", bundle_reference, answer_reference, session.model_dump()]
        )

        def decide() -> GoalAcceptedAnswerContext:
            if not first_submission:
                return GoalAcceptedAnswerContext(goal_id=None)
            goal = self.attributed_goal(provenance)
            return GoalAcceptedAnswerContext(goal_id=None if goal is None else goal.goal_id)

        return self.store.accepted_answer_context(key, decide)

    def file_event(
        self,
        document: ExternalActivityEventDocument,
        provenance: RequestProvenance | None,
        *,
        accepted_receipt: GoalAcceptedAnswerReceipt | None = None,
    ) -> tuple[ExternalActivityEventDocument, Goal | None, GoalSession | None]:
        """The event with the goal its session holds, looked up once at its first receipt.

        The session is the one the event declares (a native Team's), else the request's own; a
        goal the request names must be open. The Host's goal replaces any the producer wrote.
        """
        subject = dict(document.subject)
        subject.pop("goal_id", None)
        session = self._event_session(subject, provenance)
        # Only the accepted scientific owner passes a receipt; it must name this very event.
        if accepted_receipt is not None and (
            document.event_kind != MESSAGE_EVENT
            or subject.get("input_channel") != "PRODUCT_ACCEPTED_ANSWER"
            or subject.get("message_kind") != "answer"
            or session != accepted_receipt.session
            or subject.get("reference") != str(accepted_receipt.task_id)
            or subject.get("bundle_reference") != accepted_receipt.bundle_reference
            or subject.get("answer_reference") != accepted_receipt.answer_reference
            or subject.get("bundle_role") != accepted_receipt.bundle_role
        ):
            raise ValueError("goal.event_not_filed")

        def decide() -> UUID | None:
            if accepted_receipt is not None:
                return accepted_receipt.goal_id
            if provenance is not None and provenance.goal_id is not None:
                named = self.attributed_goal(RequestProvenance(goal_id=provenance.goal_id))
                return None if named is None else named.goal_id
            held = None if session is None else self.store.bound(session)
            return None if held is None else held.goal_id

        key = canonical_hash(
            [document.producer_id, document.producer_session, document.producer_sequence]
        )
        goal_id = self.store.event_goal(key, decide)
        if accepted_receipt is not None and goal_id != accepted_receipt.goal_id:
            raise ValueError("goal.event_not_filed")
        goal = None if goal_id is None else self.store.head(goal_id)
        if goal is not None:
            subject["goal_id"] = str(goal.goal_id)
        return document.model_copy(update={"subject": subject}), goal, session

    def record_event(
        self,
        goal: Goal,
        document: ExternalActivityEventDocument,
        observation_id: str,
        session: GoalSession | None,
    ) -> dict[str, Any]:
        """Keep one admitted Team event in the goal's record, as the Host filed it."""
        subject = document.subject
        entry: dict[str, Any] = {
            "recorded_at": self.clock().isoformat(),
            "event_kind": document.event_kind,
            "occurred_at": document.occurred_at.isoformat(),
            "observation_id": observation_id,
            "agent_vendor": session.vendor if session else None,
            "agent_session": session.session_id if session else None,
            "goal_hash": goal.goal_hash,
            "producer_id": document.producer_id,
            "producer_session": document.producer_session,
            "producer_sequence": document.producer_sequence,
            **{name: subject.get(key) for name, key in _EVENT_FIELDS},
            "summary": document.retained_summary()[0],
        }
        if document.event_kind == USAGE_EVENT:  # a reading keeps its counts (AU)
            entry.update({name: subject[name] for name in _USAGE_FIELDS if name in subject})
            entry.update(_usage_metadata(entry))
        self.store.attribute(goal.goal_id, entry)
        return {"goal_id": str(goal.goal_id)}

    def _event_session(
        self, subject: Mapping[str, str], provenance: RequestProvenance | None
    ) -> GoalSession | None:
        vendor, session_id = subject.get("native_host"), subject.get("native_session_id")
        if vendor is None or session_id is None:
            return self._session(provenance)
        try:
            session: GoalSession = GoalSession.model_validate(
                {"vendor": vendor, "session_id": session_id}
            )
        except ValidationError:
            return None
        return session

    @staticmethod
    def _session(provenance: RequestProvenance | None) -> GoalSession | None:
        if provenance is None or provenance.vendor is None or provenance.session is None:
            return None
        session: GoalSession = GoalSession.model_validate(
            {"vendor": provenance.vendor, "session_id": provenance.session}
        )
        return session

    def record(
        self, goal: Goal, *, whole_conversation: bool = False, head: Goal | None = None
    ) -> dict[str, Any]:
        """The Host's facts; a waiter reads every message, never only the shown tail."""
        entries = joined_members(self.store.attributed(goal.goal_id))
        requests = [e for e in entries if "operation" in e]
        messages = conversation_entries(entries)
        assignments = assignment_records(entries, head or goal)
        tasks: list[GoalTaskFact] = []
        for task_id in dict.fromkeys(str(e["task_id"]) for e in requests if e.get("task_id")):
            facts = self.task_facts(UUID(task_id))
            if facts is not None:
                tasks.append(
                    GoalTaskFact(
                        task_id=UUID(task_id),
                        kind=facts[0],
                        state=facts[1],
                        updated_at=facts[2],
                        continued_by=(*facts, None)[3],
                    )
                )
        delegated = [
            {
                key: e[key]
                # A delegated data decision keeps its choice and who took it.
                for key in (
                    "recorded_at",
                    "operation",
                    "status",
                    "network_enabled",
                    "case_token",
                    "option_id",
                    "symbols",
                    "delegation",
                    "agent_session",
                )
                if key in e
            }
            for e in requests
            if e.get("delegation")
        ]
        return {
            "sessions": sessions_of(entries),
            "question": self.store.question(goal, entries),
            "tasks": [t.model_dump(mode="json") for t in tasks],
            # The person's delegation and each step it took, as the ledger keeps it (V452).
            **(
                {"delegation": self.first_use_delegation(goal)}
                if goal.declaration.kind == "FIRST_USE"
                else {}
            ),
            **({"delegated_steps": delegated} if delegated else {}),
            "request_count": len(requests),
            "event_count": len(entries) - len(requests),
            "message_count": len(messages),
            "conversation": messages if whole_conversation else messages[-CONVERSATION_SHOWN:],
            "open_assignments": [a for a in assignments if a["state"] == "OPEN"],
            "closed_assignments": [a for a in assignments if a["state"] != "OPEN"],
            "session_usage": session_usage(entries),
        }

    # ------------------------------------------------------------------ operations

    def operate(
        self,
        request: PortfolioResearchOperationRequest,
        caller: str,
        provenance: RequestProvenance | None = None,
    ) -> dict[str, Any]:
        """Run one goal operation for its caller and the provenance its request carried."""
        if request.operation == "GOAL_SCHEMA":
            return goal_schema(REFERENCE_OPERATIONS, _DECLARATION_TEMPLATE)
        if request.operation == "GOAL_LIST":
            return self.store.listing(
                limit=request.history_limit or 25,
                cursor=request.history_cursor,
                agent_session=request.agent_session,
                task_id=request.task_id,
            )
        if request.operation == "GOAL_OPEN":
            return opened_goal(self._open(request, caller, provenance), request.goal_declaration)
        if (
            request.operation in _SESSION_GOAL_OPERATIONS
            and request.goal_id is None
            and request.goal_hash is None
        ):
            bound = self.attributed_goal(provenance)
            if bound is None:
                raise ValueError("goal.goal_id_required")
            request = replace(request, goal_id=bound.goal_id)
        if request.operation == "GOAL_REVISE":
            assert request.goal_declaration is not None
            prior = self._selected(request)
            declaration = GoalDeclaration.model_validate(request.goal_declaration)
            if "FIRST_USE" in {prior.declaration.kind, declaration.kind}:
                # A first use keeps its sentence, kind and opening time (V452).
                raise ValueError("goal.first_use_is_not_revised")
            if prior.declaration.intent() != declaration.intent():
                # The previous objective's pre-execution timing must not be attributed
                # to a newly worded one. Its original revision remains intact.
                retained = tuple(
                    ref.model_copy(update={"intent_relation": "POST_HOC"})
                    for ref in prior.references
                )
                return self._save(
                    request, caller, prior, declaration=declaration, references=retained
                )
            return self._save(request, caller, prior, declaration=declaration)
        if request.operation == "GOAL_TAKE":
            return self._take(request, provenance)
        if request.operation == "GOAL_SUBMIT":
            return self._submit(request, caller)
        if request.operation == "GOAL_ABANDON":
            goal = self._head(request)
            if goal.state != "OPEN":
                raise ValueError("goal.closed_open_a_follow_up")
            return self._save(request, caller, goal, state="ABANDONED")
        goal = self._selected(request)
        readers = {"GOAL_NARRATIVE": self.narrative, "GOAL_PROFILE": self.profile}
        if request.operation in readers:
            return readers[request.operation](goal)
        if request.operation in {"GOAL_SHOW", "GOAL_EXPORT", "GOAL_REFERENCE"}:
            with GoalReferenceReadScope() as scope:
                if request.operation == "GOAL_REFERENCE":
                    reference_id = request.goal_reference_id
                    assert reference_id is not None
                    return self.reference_readback(goal, reference_id, caller, scope=scope)
                body = self.readback(goal, caller, scope=scope)
                if request.operation == "GOAL_EXPORT":
                    return self.export(goal, body, caller, scope=scope)
                if goal.state == "OPEN":
                    offer_completion(goal, body, partial(self._resolve, caller=caller, scope=scope))
                body["verification_counts"] = scope.counts()
                return body
        if request.operation == "GOAL_CONTINUE":
            if not any(
                str(request.task_id) == ref.request.get("task_id") for ref in goal.references
            ):
                raise ValueError("goal.task_not_linked")
            draft = self.read(
                PortfolioResearchOperationRequest(
                    operation="EXPERIMENT_DRAFT",
                    task_id=request.task_id,
                    research_input_id=request.research_input_id,
                    input_binding_hash=request.input_binding_hash,
                ),
                caller,
            )
            return {
                **draft,
                "goal_hash": goal.goal_hash,
                "goal_id": str(goal.goal_id),
                "goal_continuation": "DRAFT_ONLY_EXPLICIT_REPLAN_REQUIRED",
                "objective": goal.declaration.objective,
                "claim": "No new input selection, Task, fit or current authority is implicit.",
            }
        if goal.state != "OPEN":
            raise ValueError("goal.closed_open_a_follow_up")
        if request.operation == "GOAL_ATTACH":
            draft_ref = GoalReferenceRequest.model_validate(request.goal_reference)
            if any(v.reference_id == draft_ref.reference_id for v in goal.references):
                return {
                    "status": "REFUSED",
                    "failure_code": "goal.reference_id_already_used",
                    "fields": [["goal_reference", "reference_id"]],
                    "expected": {
                        "attached_reference_ids": sorted(r.reference_id for r in goal.references)
                    },
                    **explain("goal.reference_id_already_used"),
                }
            with GoalReferenceReadScope() as scope:
                ref, _ = self._sealed_reference(goal, draft_ref, caller, scope=scope)
            return self._save(request, caller, goal, references=(*goal.references, ref))
        if request.operation == "GOAL_NOTE":
            statement = GoalStatement.model_validate(request.goal_statement)
            self._require_references(goal, statement.evidence)
            if statement.disposition in {"SUPPORTS", "DOES_NOT_SUPPORT"} and not statement.evidence:
                raise GoalRefusal("goal.conclusion_requires_evidence", goal)
            if statement.responds_to and not any(
                s.statement_id == statement.responds_to for s in goal.statements
            ):
                raise ValueError("goal.response_target_absent")
            with GoalReferenceReadScope() as scope:
                # Re-read cited owners before accepting a statement; its text stays
                # attributed interpretation, never a correctness certificate.
                for ref in goal.references:
                    if ref.reference_id in statement.evidence:
                        self._verified(ref, caller, scope=scope)
            return self._save(request, caller, goal, statements=(*goal.statements, statement))
        raise ValueError("goal.operation_not_supported")

    def _open(
        self,
        request: PortfolioResearchOperationRequest,
        caller: str,
        provenance: RequestProvenance | None,
    ) -> dict[str, Any]:
        assert request.goal_declaration is not None
        declaration = GoalDeclaration.model_validate(request.goal_declaration)
        purpose = getattr(declaration.research, "purpose", None)
        if purpose == "CONTINUATION" and not declaration.parent_goal_id:
            raise ValueError("goal.continuation_requires_parent_goal")
        if (road := daily_road(self.workspace, declaration)) is not None:
            return dict(road)  # a completed first use: its daily road, never a second one
        session = self._session(provenance)
        goal_id = request.goal_id or uuid5(
            GOAL_NAMESPACE,
            canonical_hash(
                {
                    "declaration": declaration.model_dump(mode="json"),
                    "opened_by": session.model_dump(mode="json") if session else caller,
                }
            ),
        )
        prior = self.store.head(goal_id)
        if prior is None and declaration.kind == "FIRST_USE":
            prior = self.first_use()
            if prior and prior.declaration != declaration:
                raise ValueError("goal.first_use_already_opened")
        if prior is not None:
            if prior.revision == 1 and prior.declaration == declaration:
                answer = {"status": "REUSED_EXACT", **self._identity_of(prior)}
                return {**answer, **self._bind(prior, session)}
            raise ValueError("goal.already_open_revise_it")
        if (dated := undated(self.workspace, declaration, self.clock())) is not None:
            return dict(dated)
        if declaration.kind == "FIRST_USE" and self.preparation_completed():
            raise ValueError("goal.first_use_after_preparation")
        if (
            declaration.parent_goal_id is not None
            and self.store.head(declaration.parent_goal_id) is None
        ):
            raise ValueError("goal.parent_absent")
        answer = self._save(request, caller, None, declaration=declaration, goal_id=goal_id)
        saved = self.store.load(str(answer["goal_hash"]))
        bound = self._bind(saved, session)
        return {
            **answer,
            **bound,
            "goal_prompt": goal_prompt(saved, self.workspace),
            # The goal's commands take it from the session: no id, no hash (V391).
            **(
                {
                    "claim": "This session is bound to the goal: every request it sends is "
                    "recorded under it, and goal show, attach, note, revise and submit take it "
                    "from the session, so name neither the goal nor its hash. `goal_prompt` "
                    "hands the goal to another agent."
                }
                if bound["bound_session"] is not None
                else {}
            ),
        }

    def _bind(self, goal: Goal, session: GoalSession | None) -> dict[str, Any]:
        if session is None:
            return {"bound_session": None}
        self.store.bind(session, goal.goal_id)
        return {"bound_session": session.model_dump(mode="json")}

    def _take(
        self, request: PortfolioResearchOperationRequest, provenance: RequestProvenance | None
    ) -> dict[str, Any]:
        goal = self._head(request)
        if goal.state != "OPEN":
            raise ValueError("goal.closed_open_a_follow_up")
        session = self._session(provenance)
        if session is None:
            raise ValueError("goal.agent_session_required_name_the_goal_instead")
        return {
            "status": "GOAL_TAKEN",
            **self._identity_of(goal),
            **self._bind(goal, session),
            "claim": "Every request this session sends is recorded under the goal until it is "
            "submitted, abandoned or another goal is taken.",
        }

    @staticmethod
    def _reference_clash(goal: Goal, submission: GoalSubmission) -> dict[str, Any] | None:
        """Refuse clashing ids; identical held references keep a sendable retry (V385)."""
        held = {r.reference_id: r for r in goal.references}
        declared = [
            ("references", index, draft.reference_id)
            for index, draft in enumerate(submission.references)
        ] + [("files", index, file.reference_id) for index, file in enumerate(submission.files)]
        ids = [reference_id for _section, _index, reference_id in declared]
        clashing = {i for i in ids if i in held or ids.count(i) > 1}
        if not clashing:
            return None
        repeated = {
            draft.reference_id
            for draft in submission.references
            if draft.reference_id in held
            and draft.request == held[draft.reference_id].request
            and ids.count(draft.reference_id) == 1
        }
        answer: dict[str, Any] = {
            "status": "REFUSED",
            "failure_code": "goal.reference_id_already_used",
            "fields": [
                ["goal_submission", section, index, "reference_id"]
                for section, index, reference_id in declared
                if reference_id in clashing
            ],
            "expected": {"attached_reference_ids": sorted(held)},
            **explain("goal.reference_id_already_used"),
        }
        if repeated == clashing:
            document = submission.model_dump(mode="json")
            document["references"] = [
                ref for ref in document["references"] if ref["reference_id"] not in repeated
            ]
            answer["next_requests"] = {
                "submit": {
                    "operation": "GOAL_SUBMIT",
                    "goal_id": str(goal.goal_id),
                    "goal_hash": goal.goal_hash,
                    "goal_submission": document,
                }
            }
        return answer

    def _submit(self, request: PortfolioResearchOperationRequest, caller: str) -> dict[str, Any]:
        """Seal once; a failed render still discloses the completed record."""
        assert request.goal_submission is not None
        goal = self._head(request)
        if request.goal_hash is not None and request.goal_hash != goal.goal_hash:
            raise ValueError("goal.revision_conflict_read_latest")
        if goal.state != "OPEN":
            raise ValueError("goal.closed_open_a_follow_up")
        submission = GoalSubmission.model_validate(request.goal_submission)
        clash = self._reference_clash(goal, submission)
        if clash is not None:
            return clash
        known = {r.reference_id for r in goal.references}
        declared = {r.reference_id for r in submission.references} | {
            f.reference_id for f in submission.files
        }
        record = self.record(goal)
        tasks = tuple(GoalTaskFact.model_validate(t) for t in record["tasks"])
        with GoalReferenceReadScope() as scope:
            added: list[GoalReference] = []
            failures: dict[str, str] = {}
            checked: dict[str, tuple[GoalReference, dict[str, Any]]] = {}
            for draft in submission.references:
                try:
                    ref, body = self._sealed_reference(goal, draft, caller, scope=scope)
                    added.append(ref)
                    checked[ref.reference_id] = (ref, body)
                except (ValueError, KeyError, OSError) as error:
                    failures[draft.reference_id] = public_failure(
                        error, "goal.reference_unavailable"
                    )
            held = {r.reference_id: r for r in goal.references}

            def verify(reference_id: str) -> str | None:
                if reference_id in failures:
                    return failures[reference_id]
                ref = held.get(reference_id)
                if ref is None:
                    return None  # sealed just now, or a file the goal keeps by content
                try:
                    body = self._verified(ref, caller, scope=scope)
                    checked[reference_id] = (ref, body)
                except (ValueError, KeyError, OSError) as error:
                    return public_failure(error, "goal.reference_unavailable")
                return None

            missing = missing_items(
                goal,
                submission,
                tasks=tasks,
                verify=verify,
                cited_known=frozenset(known | declared),
            )
            if missing:
                offered = offered_attach(
                    goal, missing, tasks, lambda r: self._resolve(r, caller, scope=scope)
                )
                return {
                    "status": "INCOMPLETE",
                    **self._identity_of(goal),
                    "missing": missing,
                    "record": record,
                    "next_requests": {
                        "resubmit": resubmit(goal, submission, missing, offered),
                        "complete": resubmit(goal, submission, missing, offered, inline=True),
                        "show": {"operation": "GOAL_SHOW", "goal_id": str(goal.goal_id)},
                        **offered,
                    },
                    "claim": "Nothing was sealed. Each item names what the record lacks; the "
                    "Host does not judge whether the summary is true.",
                }
            completion = GoalCompletion(
                checked_at=self.clock(),
                sessions=tuple(GoalSession.model_validate(s) for s in record["sessions"]),
                tasks=tasks,
                request_count=int(record["request_count"]),
            )
            answer = self._save(
                request,
                caller,
                goal,
                reason="Submitted; the Host found the record complete.",
                state="COMPLETE",
                submission=submission,
                completion=completion,
                references=(*goal.references, *added),
            )
            result = {
                **answer,
                "status": "COMPLETE",
                "outcome": submission.outcome,
                "open_assignments": [],
                "claim": "Complete means the record is complete and its evidence verified, not "
                "that the objective was met.",
            }
            try:
                sealed = self.store.load(str(answer["goal_hash"]))
                body = self.readback(sealed, caller, scope=scope, verified=checked, head=sealed)
                result["report"] = self.export(sealed, body, caller, scope=scope)
                result["closure_message_id"] = "goal.closure.complete"
            except Exception as error:
                result["closure_message_id"] = "goal.closure.report_unavailable"
                result["report_error"] = {
                    "failure_code": "goal.closure.report_unavailable",
                    "cause_failure_code": public_failure(error, "goal.closure.report_unavailable"),
                }
            return result

    def _sealed_reference(
        self,
        goal: Goal,
        draft_ref: GoalReferenceRequest,
        caller: str,
        *,
        scope: GoalReferenceReadScope,
    ) -> tuple[GoalReference, dict[str, Any]]:
        """Seal an owner read and pass that verified body to this request's report."""
        body = self._resolve(draft_ref, caller, scope=scope)
        relation = "POST_HOC"
        research = goal.declaration.research
        if research is None or research.purpose != "EXISTING_RESULTS":
            task = draft_ref.request.get("task_id")
            relation = (
                "NO_EXECUTION_TIME_PROOF"
                if not task
                else (
                    "QUESTION_RECORDED_BEFORE_TASK_ADMISSION"
                    if goal.intent_registered_at < self.admitted_at(UUID(task))
                    else "POST_HOC"
                )
            )
        identity = self._identity(body)
        return GoalReference(
            **draft_ref.model_dump(),
            identity=identity,
            intent_relation=relation,
            reference_hash=canonical_hash({"request": draft_ref.request, "identity": identity}),
        ), body

    def _head(self, request: PortfolioResearchOperationRequest) -> Goal:
        if request.goal_id is None:
            raise ValueError("goal.goal_id_required")
        head = self.store.head(request.goal_id)
        if head is None:
            raise ValueError("goal.not_found")
        return head

    def _exact(self, request: PortfolioResearchOperationRequest) -> Goal:
        assert request.goal_hash is not None
        goal = self.store.load(request.goal_hash)
        if request.goal_id is not None and goal.goal_id != request.goal_id:
            raise ValueError("goal.goal_id_mismatch")
        return goal

    def _selected(self, request: PortfolioResearchOperationRequest) -> Goal:
        """An exact revision by its hash, else the goal's current head."""
        return self._exact(request) if request.goal_hash else self._head(request)

    @staticmethod
    def _require_references(goal: Goal, references: tuple[str, ...]) -> None:
        if not set(references) <= {r.reference_id for r in goal.references}:
            raise ValueError("goal.reference_absent")

    @staticmethod
    def _identity_of(goal: Goal) -> dict[str, Any]:
        return {
            "goal_id": str(goal.goal_id),
            "goal_hash": goal.goal_hash,
            "revision": goal.revision,
            "state": goal.state,
        }

    def _save(
        self,
        request: PortfolioResearchOperationRequest,
        caller: str,
        prior: Goal | None,
        *,
        goal_id: UUID | None = None,
        reason: str | None = None,
        **changes: Any,
    ) -> dict[str, Any]:
        now = self.clock()
        fields = (
            {k: getattr(prior, k) for k in type(prior).model_fields if k != "goal_hash"}
            if prior
            else {
                "goal_id": goal_id,
                "workspace_id": self.store.workspace_id,
                "references": (),
                "statements": (),
            }
        )
        fields.update(changes)
        declaration = GoalDeclaration.model_validate(fields["declaration"])
        # A title-only change or attached evidence does not pretend to register a new design.
        intent_at = (
            prior.intent_registered_at
            if prior and prior.declaration.intent() == declaration.intent()
            else now
        )
        reason = reason or request.change_reason or ("Goal opened." if prior is None else None)
        if reason is None:
            raise ValueError("goal.change_reason_required")
        fields.update(
            parent_hash=prior.goal_hash if prior else None,
            revision=prior.revision + 1 if prior else 1,
            recorded_at=now,
            intent_registered_at=intent_at,
            submitted_by=caller,
            change_reason=reason,
            declaration=declaration,
        )
        value = Goal.seal(**fields)
        saved = self.store.publish(value, prior.goal_hash if prior else None)
        answer: dict[str, Any] = {
            "status": "GOAL_SAVED",
            **self._identity_of(saved),
            "task_id": None,
            "numerical_work": "NONE",
            "next_requests": self._next(saved),
        }
        if request.operation == "GOAL_SUBMIT" and saved.completion is not None:
            answer.update(
                next_action="RECORD_THE_EXPERIENCE",
                detail=EXPERIENCE_RECORD_WORDS.format(subject=saved.goal_id),
            )
            answer["next_requests"].update(
                {
                    f"task:{task.task_id}": {"operation": "STATUS", "task_id": str(task.task_id)}
                    for task in saved.completion.tasks
                }
            )
        return answer

    def _resolve(
        self,
        ref: GoalReferenceRequest | GoalReference,
        caller: str,
        *,
        scope: GoalReferenceReadScope | None = None,
    ) -> dict[str, Any]:
        operation = str(ref.request.get("operation", ""))
        if ref.stage not in REFERENCE_OPERATIONS.get(operation, ()):
            raise ValueError("goal.reference_operation_not_admitted")
        request = reference_request(ref.request)
        if operation == "EXPERIMENT_CONTROLS" and not (
            request.input_binding_hash and request.research_input_id
        ):
            raise ValueError("goal.exact_input_required")
        if operation == "EVIDENCE_CRO_EXPORT":
            if not request.review_publication_hash or not any(
                (
                    request.result_hash,
                    request.handoff_hash,
                    request.update_task_id,
                    request.experiment_task_id,
                )
            ):
                raise ValueError("goal.exact_review_required")
            if request.experiment_task_id and not (
                request.experiment_receipt_hash and request.portfolio_session
            ):
                raise ValueError("goal.exact_review_required")
            if request.update_task_id and not (
                request.update_publication_hash and request.position_basis
            ):
                raise ValueError("goal.exact_review_required")
        try:
            body = (
                self.read(request, caller)
                if scope is None
                else scope.read(
                    document=ref.request, request=request, reader=self.read, caller=caller
                )
            )
        except ValueError as error:
            if (
                operation not in {"COMPARE", "EXPERIMENT_COMPARE", "EXPERIMENT_ALPHA_COMPARE"}
                or str(error) not in COMPARISON_REFUSALS
            ):
                raise
            body = {
                "status": "REFUSED",
                **located_failure(error, "goal.refused"),
            }
        if body.get("status") == "REFUSED" or body.get("refused"):
            if (
                operation in {"COMPARE", "EXPERIMENT_COMPARE", "EXPERIMENT_ALPHA_COMPARE"}
                and body.get("failure_code") in COMPARISON_REFUSALS
            ):
                return {
                    "status": "INCOMPATIBLE",
                    "owner_refusal": body,
                    "claim": "Owner refused this comparison; no metrics or common scope invented.",
                }
            raise ValueError(str(body.get("failure_code", body.get("refused"))))
        if operation == "EXPERIMENT_READBACK":
            kind = str(body.get("program", {}).get("kind", ""))
            stage = {
                "factor": "FACTOR_FOUNDATION",
                "alpha": "ALPHA",
                "risk": "RISK",
                "portfolio": "PORTFOLIO",
            }
            if kind and stage.get(kind.split(".")[0]) != ref.stage:
                raise ValueError("goal.experiment_stage_mismatch")
        return body

    @staticmethod
    def _identity(body: dict[str, Any]) -> dict[str, Any]:
        # Immutable scientific identities take precedence over volatile next-action text.
        identity = {
            k: body[k] for k in ("task_id", "input_binding_hash", "result_hash") if k in body
        }
        for key, field in (
            ("program", "program_hash"),
            ("receipt", "receipt_hash"),
            ("evidence", "evidence_hash"),
            ("admission", "admission_hash"),
            ("publication", "content_hash"),
        ):
            if isinstance(body.get(key), dict) and field in body[key]:
                identity[field] = body[key][field]
        if not any(k.endswith("hash") for k in identity):
            identity["readback_hash"] = canonical_hash(
                {k: v for k, v in body.items() if k not in {"next_requests", "html"}}
            )
        return identity

    def _verified(
        self,
        ref: GoalReference,
        caller: str,
        *,
        scope: GoalReferenceReadScope | None = None,
    ) -> dict[str, Any]:
        if scope is not None:
            scope.reference_verification_count += 1
        body = self._resolve(ref, caller, scope=scope)
        if self._identity(body) != ref.identity:
            raise ValueError("goal.reference_changed_reattach_explicitly")
        return body

    @staticmethod
    def _summary(body: dict[str, Any]) -> dict[str, Any]:
        # Do not ship series/arrays, weights or model payloads to every specialist.
        summary = {
            k: v
            for k, v in body.items()
            if k
            in {
                "status",
                "task_id",
                "kind",
                "receipt",
                "result",
                "limitations",
                "claim",
                "failure_code",
                "input_binding_hash",
                "research_input_id",
                "alpha_source",
                "portfolio_source",
                "coverage",
                "admission",
                "classification",
                "candidates",
                "authoring_options",
                "features",
                "sessions",
                "next_requests",
                "summary",
                "link",
                "risk_surface",
                "data_quality",
                "factor_options",
                "limits",
                "input_id",
                "curation",
                "decisions",
                "dimensions",
                "support",
                "disposition",
                "owner_refusal",
                "risk",
                "fold_results",
                "folds",
                "score_support",
                "relation",
                "declared_parameter_difference",
                "declared_feature_difference",
                "prerequisites",
            }
            and k not in {"receipt", "portfolio_source"}
        }
        template = body.get("template", {}).get("experiment")
        if template:
            summary["input_scope"] = {
                key: template[key]
                for key in (
                    "data_snapshot_handle",
                    "universe_handle",
                    "sessions",
                    "publication_intent",
                )
                if key in template
            }
        if "dimensions" in body:
            summary["subjects"] = {
                side: {k: body[side].get(k) for k in ("task_id", "document", "limitations")}
                for side in ("left", "right")
                if isinstance(body.get(side), dict)
            }
        if isinstance(body.get("review"), dict):
            summary["review"] = {
                k: body["review"][k]
                for k in ("publication", "recommendation", "dossier")
                if k in body["review"]
            }
        if "result_hash" in body and "readouts" in body:
            # Installed books keep their original report vocabulary. They are
            # not rewritten into authored-experiment receipts or invented Tasks.
            summary.update(
                {
                    key: body[key]
                    for key in (
                        "result_hash",
                        "report_hash",
                        "originating_task_id",
                        "readouts",
                        "window",
                        "schedule",
                        "controls",
                        "unit_rows",
                        "report_unit",
                    )
                    if key in body
                }
            )
        if isinstance(body.get("document"), dict):
            summary["declaration"] = body["document"]
        if isinstance(body.get("receipt"), dict) and isinstance(body.get("fold_results"), list):
            receipt = body["receipt"]
            summary["alpha_receipt"] = {
                key: receipt.get(key)
                for key in (
                    "receipt_hash",
                    "program_hash",
                    "development_program_hash",
                    "metric_policy_hash",
                    "research_recipe_hash",
                    "split_policy",
                )
            }
        if isinstance(body.get("execution_preview"), dict):
            summary["evaluation_scope"] = body["execution_preview"]
        if isinstance(body.get("evidence"), dict):
            sessions = body["evidence"].get("formation_sessions", [])
            summary["evaluated_interval"] = {
                "start": sessions[0] if sessions else None,
                "end": sessions[-1] if sessions else None,
                "formation_count": len(sessions),
                "claim": "Recorded evaluated axis; not requested data support or execution cost.",
            }
        return summary

    @staticmethod
    def _next(goal: Goal) -> dict[str, Any]:
        identity = {"goal_id": str(goal.goal_id)}
        base: dict[str, Any] = {
            "show": {"operation": "GOAL_SHOW", **identity},
            "narrative": {"operation": "GOAL_NARRATIVE", "goal_hash": goal.goal_hash},
            "export": {"operation": "GOAL_EXPORT", "goal_hash": goal.goal_hash},
        }
        if goal.state != "OPEN":
            return base
        # A first use keeps the person's sentence: nothing revises it (V452).
        revise = (
            {}
            if goal.declaration.kind == "FIRST_USE"
            else {
                "revise": {
                    "operation": "GOAL_REVISE",
                    **identity,
                    "goal_hash": goal.goal_hash,
                    "goal_declaration": None,
                    "change_reason": None,
                }
            }
        )
        return {
            **base,
            "take": {"operation": "GOAL_TAKE", **identity},
            "submit": {
                "operation": "GOAL_SUBMIT",
                **identity,
                "goal_hash": goal.goal_hash,
                # Each open field is None, a choice the client leaves to its reader; the
                # goal and its revision stay bound (V441). `goal show --save-declaration`
                # writes the completion, and the declaration reads in `show`.
                "goal_submission": None,
            },
            **revise,
        }

    def _saved(
        self, goal: Goal, *, whole_conversation: bool = False, head: Goal | None = None
    ) -> dict[str, Any]:
        """Project saved facts; a supplied head is this request's fresh publication.

        Normal reads resolve the current head here.
        """
        head = head if head is not None else self.store.head(goal.goal_id)
        statement_context = {}
        ancestor = goal
        while True:
            for statement in ancestor.statements:
                statement_context[statement.statement_id] = {
                    "goal_hash": ancestor.goal_hash,
                    "objective": ancestor.declaration.objective,
                    "design": "CURRENT_DESIGN"
                    if ancestor.declaration.intent() == goal.declaration.intent()
                    else "PRIOR_DESIGN",
                }
            if ancestor.parent_hash is None:
                break
            parent = self.store.load(ancestor.parent_hash)
            if parent.goal_id != goal.goal_id or parent.revision != ancestor.revision - 1:
                raise ValueError("goal.revision_chain_invalid")
            ancestor = parent
        return {
            "goal_id": str(goal.goal_id),
            "goal_hash": goal.goal_hash,
            "goal": goal.model_dump(mode="json"),
            "statement_context": statement_context,
            "head_hash": head.goal_hash if head else None,
            "head": {
                **self._identity_of(head),
                "outcome": head.submission.outcome if head.submission else None,
            }
            if head
            else None,
            "state": goal.state,
            "outcome": goal.outcome(),
            "open_choices": [
                s.model_dump(mode="json") for s in goal.statements if s.disposition == "OPEN"
            ],
            "record": self.record(goal, whole_conversation=whole_conversation, head=head),
            "next_requests": {
                **self._next(goal),
                **(
                    {
                        "prior_revision": {
                            "operation": "GOAL_SHOW",
                            "goal_hash": goal.parent_hash,
                        }
                    }
                    if goal.parent_hash
                    else {}
                ),
            },
        }

    def narrative(self, goal: Goal) -> dict[str, Any]:
        """The saved revision as recorded, without re-verifying its evidence.

        Its content hash was checked on load, so its recorded identities are exact;
        ``GOAL_SHOW`` re-checks every reference and ``GOAL_REFERENCE`` one. Its conversation is
        whole, so a waiter loses no message between reads.
        """
        return {
            "status": "GOAL_NARRATIVE",
            **self._saved(goal, whole_conversation=True),
            "references": [
                {
                    "reference": ref.model_dump(mode="json"),
                    "state": "SAVED_NOT_VERIFIED",
                    "next_requests": {
                        "open": ref.request,
                        "verify": {
                            "operation": "GOAL_REFERENCE",
                            "goal_hash": goal.goal_hash,
                            "goal_reference_id": ref.reference_id,
                        },
                    },
                }
                for ref in goal.references
            ],
            "evidence_verification": "NOT_PERFORMED",
            "claim": "Saved revision identity is verified; references are as recorded and "
            "not re-verified against their owners. GOAL_SHOW verifies all; "
            "GOAL_REFERENCE verifies one selected reference. "
            "No current/scientific approval or retention pin.",
        }

    def _reference_row(
        self,
        ref: GoalReference,
        caller: str,
        *,
        scope: GoalReferenceReadScope,
        verified: Mapping[str, tuple[GoalReference, dict[str, Any]]] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Use a proof only for this identical full reference and request/caller.
        Public reads pass no proof and always verify at the owner.
        """
        row: dict[str, Any] = {
            "reference": ref.model_dump(mode="json"),
            "next_requests": {"open": ref.request},
        }
        gap: dict[str, Any] | None = None
        try:
            prior = (verified or {}).get(ref.reference_id)
            body = (
                prior[1]
                if prior is not None and prior[0] == ref
                else self._verified(ref, caller, scope=scope)
            )
            incomplete = (
                ref.request["operation"] == "EXPERIMENT_READBACK"
                and body.get("status") != "EXPERIMENT_PUBLISHED"
            )
            row.update(
                state="VERIFIED_TASK_STATE"
                if incomplete
                else "VERIFIED_REFUSAL"
                if body.get("status") == "INCOMPATIBLE"
                else "VERIFIED_READBACK",
                summary=self._summary(body),
            )
            if incomplete:
                gap = {
                    "reference_id": ref.reference_id,
                    "failure_code": body.get("failure_code") or "goal.task_not_succeeded",
                }
            row["next_requests"].update(body.get("next_requests", {}))
        except (ValueError, KeyError, OSError) as error:
            code = public_failure(error, "goal.reference_unavailable")
            row.update(state="UNAVAILABLE", failure_code=code)
            gap = {"reference_id": ref.reference_id, "failure_code": code}
        return row, gap

    def reference_readback(
        self,
        goal: Goal,
        reference_id: str,
        caller: str,
        *,
        scope: GoalReferenceReadScope | None = None,
    ) -> dict[str, Any]:
        """Verify one named reference of the exact saved revision at its owner."""
        ref = next((r for r in goal.references if r.reference_id == reference_id), None)
        if ref is None:
            raise ValueError("goal.reference_not_found")
        scope = scope or GoalReferenceReadScope()
        row, _gap = self._reference_row(ref, caller, scope=scope)
        return {
            "status": "GOAL_REFERENCE_READBACK",
            "goal_hash": goal.goal_hash,
            "reference": row,
            "verification_counts": scope.counts(),
            "evidence_verification": "SELECTED_REFERENCE",
            "claim": "Only the selected reference is checked at its owner; its state names "
            "the verification outcome. Other references are not checked by this read. "
            "Interpretations and role names are caller-attributed. Registration timing "
            "is not independent preregistration. No current/scientific approval or retention pin.",
        }

    def readback(
        self,
        goal: Goal,
        caller: str,
        *,
        scope: GoalReferenceReadScope | None = None,
        verified: Mapping[str, tuple[GoalReference, dict[str, Any]]] | None = None,
        head: Goal | None = None,
    ) -> dict[str, Any]:
        """Read a revision; proofs must belong to this exact request and caller.

        Only completion passes its freshly published head and already verified reads.
        """
        scope = scope or GoalReferenceReadScope()
        rows, missing = [], []
        for ref in goal.references:
            row, gap = self._reference_row(ref, caller, scope=scope, verified=verified)
            if gap is not None:
                missing.append(gap)
            rows.append(row)
        available = {r["reference"]["stage"] for r in rows if r["state"] == "VERIFIED_READBACK"}
        research = goal.declaration.research
        missing.extend(
            {"stage": s, "failure_code": "goal.stage_not_evidenced"}
            for s in (research.required_stages if research else ())
            if s not in available
        )
        return {
            "status": "GOAL_SHOW",
            **self._saved(goal, head=head),
            "references": rows,
            "verification_counts": scope.counts(),
            "gaps": missing,
            "evidence_verification": "COMPLETE",
            "claim": "Reference integrity is product-verified; interpretations and role names "
            "are caller-attributed. Registration timing is not independent preregistration. "
            "No current/scientific approval or retention pin.",
        }

    def export(
        self,
        goal: Goal,
        body: dict[str, Any],
        caller: str,
        *,
        scope: GoalReferenceReadScope | None = None,
    ) -> dict[str, Any]:
        """The verified readback with the book deliveries it names, as a sealed page."""
        import yaml  # type: ignore[import-untyped]

        from alphalattice.interface.local_application.experiment_report import render_goal

        scope = scope or GoalReferenceReadScope()
        deliveries = []
        delivery_gaps = []
        for row in body["references"]:
            delivery = row["next_requests"].get("delivery")
            if row["state"] == "VERIFIED_READBACK" and delivery:
                delivery = dict(delivery)
                # A single exact selected link may join this book's export. Merely
                # citing an unrelated Risk study or review never associates it.
                for operation, source_field, target_field in (
                    ("EVIDENCE_CRO_EXPORT", "review_publication_hash", "review_publication_hash"),
                    ("EXPERIMENT_RISK_EXPORT", "risk_report_hash", "risk_report_hash"),
                ):
                    matches = []
                    for ref in goal.references:
                        q = ref.request
                        if q.get("operation") != operation:
                            continue
                        if operation == "EVIDENCE_CRO_EXPORT":
                            same = (
                                q.get("experiment_task_id") == delivery["task_id"]
                                and q.get("experiment_receipt_hash")
                                == delivery["experiment_receipt_hash"]
                                and q.get("portfolio_session") == delivery["portfolio_session"]
                            )
                        else:
                            same = q.get("task_id") == delivery["task_id"]
                        if same and q.get(source_field):
                            matches.append(q[source_field])
                    if len(set(matches)) == 1:
                        delivery[target_field] = matches[0]
                    elif matches:
                        delivery_gaps.append(
                            {
                                "task_id": delivery["task_id"],
                                "field": target_field,
                                "status": "EXPLICIT_SELECTION_REQUIRED",
                            }
                        )
                delivery_request = PortfolioResearchRequestDocument.model_validate(
                    delivery
                ).to_operation_request()
                deliveries.append(
                    scope.read(
                        document=delivery,
                        request=delivery_request,
                        reader=self.read,
                        caller=caller,
                    )
                )
        snapshot = {
            **body,
            "status": "GOAL_EXPORTED",
            "portfolio_deliveries": deliveries,
            "delivery_gaps": delivery_gaps,
            "verification_counts": scope.counts(),
        }
        rendered = render_goal(snapshot)
        return {
            **snapshot,
            "html": rendered,
            "yaml": yaml.safe_dump(goal.declaration.model_dump(mode="json"), sort_keys=False),
            "export_hash": canonical_hash({**snapshot, "html": rendered}),
        }


def goal_prompt(goal: Goal, workspace: Path) -> str:
    """The `/goal ` prompt that hands a goal to Claude or Codex, at most 4,000 characters.

    Its commands name the goal's workspace, which every command requires, quoted for the shell
    the CLI prints commands for, so the agent taking the goal runs them as written (V492).
    """

    def command(*parts: str) -> str:
        return join(["alphalattice", "--workspace", str(workspace), *parts], shell())

    criteria = "; ".join(f"{c.criterion_id}: {c.text}" for c in goal.declaration.criteria)
    goal_id = str(goal.goal_id)
    completion = str(workspace / "completion.yaml")
    text = (
        f"/goal {goal.declaration.objective} (Host goal {goal.goal_id}, {goal.declaration.kind}). "
        f"First run `{command('goal', 'take', goal_id)}`, so every request you send is recorded "
        "under it. Work through the alphalattice CLI on that workspace. Finish with "
        f"`{command('goal', 'submit', goal_id, '--file', completion)}` "
        f"(`{command('goal', 'schema')}` shows the document); you are done only when it answers "
        "COMPLETE. If it answers INCOMPLETE, supply each missing item and submit again. "
        f"Completion criteria: {criteria}"
    )
    return text[:4000]
