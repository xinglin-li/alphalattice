"""Goals: what an agent is asked to achieve, recorded by the Host and checked when submitted.

The Host keeps a goal's declaration and its revisions, its exact-read references and attributed
statements, what the agent sessions bound to it did and said (their requests, and the Team
events each session sent while it held the goal, among them assignments and their replies),
and its submission, which it checks against that record and seals, or answers with each
missing item and the request that supplies it
(LAWS OP13). It runs no loop, judges no summary's truth and computes no number: "never a second
research engine".
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, fields, replace
from dataclasses import field as dataclass_field
from datetime import datetime, timedelta
from pathlib import Path
from secrets import token_hex
from typing import Any, Self
from uuid import UUID, uuid5

from pydantic import ValidationError

from alphalattice.control.product_host.composition.goal_check import missing_items
from alphalattice.control.product_host.composition.plain_refusals import explain
from alphalattice.control.product_host.publication.goals import GoalStore, sessions_of
from alphalattice.interface.local_application.activity import ExternalActivityEventDocument
from alphalattice.interface.local_application.cli_contract import (
    RequestProvenance,
    join,
    shell,
)
from alphalattice.interface.local_application.failure_codes import located_failure, public_failure
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
)
from alphalattice.interface.local_application.operations import FIRST_USE_STEPS
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchOperationRequest,
    PortfolioResearchRequestDocument,
    decision_hash,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.actor_execution.bundles import bundle_directory_key, bundle_slot

GOAL_NAMESPACE = UUID("8f1d6c63-3c3e-4f5f-9f0e-5c4b2a7e8d10")
"""A goal opened without an id is named from its declaration and who opened it, so a retried
open answers the goal it made rather than a second one."""

TaskFacts = Callable[[UUID], tuple[str, str, datetime | None] | None]
"""A Task's kind, Task Control lifecycle and canonical update time, or nothing when absent."""

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


def open_assignments(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Each bundle prepared under the goal with no accepted answer yet: a reminder (FLOW-1).

    A bundle is the product's own assignment of a question to a specialist; the product
    accepting its answer closes it. Nothing here blocks the goal's completion.
    """
    answered = {
        entry.get("bundle_reference")
        for entry in entries
        if entry.get("operation") == "AGENT_ANSWER_SUBMIT" and entry.get("status") in ANSWERED
    }
    prepared: dict[str, dict[str, Any]] = {}
    for entry in entries:
        reference = entry.get("bundle_reference")
        if (
            entry.get("operation") == "AGENT_BUNDLE_PREPARE"
            and entry.get("status") == "AGENT_BUNDLE_READY"
            and isinstance(reference, str)
            and reference not in answered
        ):
            prepared.setdefault(
                reference,
                {
                    "bundle_reference": reference,
                    "agent_role": entry.get("agent_role"),
                    "task_id": entry.get("task_id"),
                    "prepared_at": entry.get("recorded_at"),
                },
            )
    return list(prepared.values())


def conversation_entries(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The goal's conversation: what the product recorded of its Sessions' work, in time order.

    Each request a bound agent Session sent under the goal (bundle preparations and answer
    submissions among them) is a product fact; so is an accepted answer filed beside it.
    Messages filed before FLOW-1 stay as they were recorded.
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

    A reading is the cumulative count from one participant's admitted session file; the
    latest reading replaces the previous one for that participant and model. Parent and
    child readings have no established disjointness, so they are never added. Where a reading
    differs from the agent's role card, the bridge that read both says so (``pin_differs``).
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

    A goal can retain two historical references that intentionally name the
    same owner operation.  Reusing that one response within this request still
    checks each reference's own sealed identity, so a stale reference remains
    visible rather than being masked by a cache.

    Entered as a context manager, the scope also bounds the Alpha owner's
    verified chunk tables to this request: an Alpha task cited directly and
    again through a saved comparison has its evidence graph read and hashed
    once per request, while every reference still verifies its own identity
    and the next request reads afresh.
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


def completion_template(goal: Goal) -> str:
    """The open goal's completion to fill: the Host's slots prefilled, the judgment left (V419).

    A whole GOAL_SUBMIT request bound to the goal and its revision, so it completes this goal
    whatever goal the session holds, and a later revision refuses it (V426). Each criterion and
    deliverable slot by its id, with its words as a comment; the goal's references listed as the
    evidence to cite. The judgment (the outcome, each criterion's answer, the summary) is a
    placeholder the contract refuses until it is written, so it is never sent as it stands.

    Args:
        goal: The goal's head revision.

    Returns:
        The request's YAML text, `request --file` sends it once filled.
    """

    def said(text: str) -> str:
        return " ".join(text.split())[:160]

    def quoted(value: str) -> str:
        return json.dumps(value, ensure_ascii=False)

    declared = goal.declaration
    lines = [
        f"# The completion of goal {quoted(said(declared.title))}, revision {goal.revision}: your",
        "# judgment only; the Host adds its own facts (sessions, Tasks, requests). Replace each",
        "# <...>, then `request --file <this file>`: it completes this goal's revision, and a",
        "# later revision refuses it (read the goal again). Evidence names a reference by its id:",
        *(f"#   {ref.reference_id}: {said(ref.label)} ({ref.stage})" for ref in goal.references),
        "# A read the goal does not hold yet goes under `references:` with its own id.",
        "operation: GOAL_SUBMIT",
        f"goal_id: {quoted(str(goal.goal_id))}",
        f"goal_hash: {quoted(goal.goal_hash)}",
        "goal_submission:",
        "  outcome: <ACHIEVED, PARTLY_ACHIEVED or NOT_ACHIEVED>",
        "  summary: <what the goal reached, in your words>",
        "  criteria:",
    ]
    for criterion in declared.criteria:
        lines += [
            f"  # {said(criterion.text)}",
            f"  - criterion_id: {quoted(criterion.criterion_id)}",
            "    answer: <MET, NOT_MET or NOT_ASSESSED>",
            "    evidence: []",
            '    note: ""',
        ]
    if declared.deliverables:
        lines.append("  deliverables:")
    for deliverable in declared.deliverables:
        need = "required" if deliverable.required else "optional"
        lines += [
            f"  # {deliverable.kind}, {need}: {said(deliverable.description)}",
            f"  - deliverable_id: {quoted(deliverable.deliverable_id)}",
            "    references: [<reference id>]",
        ]
    return "\n".join(lines) + "\n"


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
    ) -> None:
        """Bind the goal store, the clock, the owners' reader and Task Control's facts.

        `workspace` is the one a goal's handoff names in its commands (V492).
        """
        self.store, self.clock, self.read = store, clock, read
        self.admitted_at, self.task_facts = admitted_at, task_facts
        self.workspace = workspace

    # ------------------------------------------------------------------ attribution

    def attributed_goal(self, provenance: RequestProvenance | None) -> Goal | None:
        """The open goal a request works for: the one it names, else its session's.

        A named goal that is absent or closed is refused by name, so no work is recorded
        under a goal it cannot count toward (OP13).
        """
        if provenance is None:
            return None
        if provenance.goal_id is not None:
            head = self.store.head(UUID(provenance.goal_id))
            if head is None:
                raise ValueError("goal.not_found")
            if head.state != "OPEN":
                raise ValueError("goal.closed_open_a_follow_up")
            return head
        session = self._session(provenance)
        return None if session is None else self.store.bound(session)

    def first_use(self) -> Goal | None:
        """The workspace's first-use goal, if one was ever opened (V452)."""
        for goal_id in self.store.goal_ids():
            head = self.store.head(goal_id)
            if head is not None and head.declaration.kind == "FIRST_USE":
                return head
        return None

    @staticmethod
    def delegation_ends(goal: Goal) -> datetime:
        """When a first-use goal's delegation ends: its hours from its opening (V452)."""
        return goal.intent_registered_at + timedelta(hours=FIRST_USE_HOURS)

    def _first_use_open(self, goal: Goal) -> bool:
        return (
            goal.declaration.kind == "FIRST_USE"
            and goal.state == "OPEN"
            and self.clock() < self.delegation_ends(goal)
        )

    def first_use_delegation(self, goal: Goal) -> dict[str, Any]:
        """A first-use goal's delegation as its record and the person's page read it (U70)."""
        return {
            "steps": sorted(FIRST_USE_STEPS),
            "ends_at": self.delegation_ends(goal).isoformat(),
            "active": self._first_use_open(goal),
        }

    def delegation(self, goal: Goal, operation: str, caller: str) -> str | None:
        """The person's delegation one of the goal's requests acts under, if any (V452, OP19).

        A first-use goal, open and within its hours, delegates its steps (`FIRST_USE_STEPS`) to
        the agent that runs it; every other step stays the person's, as does every step after.

        Args:
            goal: The goal the request works for.
            operation: The request's operation.
            caller: Who sent it.

        Returns:
            The delegation's name, or None when the step is not delegated.
        """
        if caller == "HUMAN" or operation not in FIRST_USE_STEPS or not self._first_use_open(goal):
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

    def _own_preparation_stopped(self, goal: Goal) -> bool:
        """Whether a preparation this first use admitted is stopped, its data decisions owed.

        The delegation decides only its own preparation's data issues: no other preparation's,
        no standing grant, nothing once the goal ends (OP19).
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
        if goal.declaration.kind != "FIRST_USE" or self._first_use_open(goal):
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

        It answers only the decision it was bound to, from the agent session whose goal the
        request works for, and once: the same words to the same question for the same decision
        are refused while held or used, naming the nonce with which the person's next yes to it
        is sent. The Host's clock dates it; `release_relay` gives back one whose decision was
        refused.
        """
        confirmation = request.person_confirmation
        if confirmation is None or caller == "HUMAN":
            return None
        session = self._session(provenance)
        held = None if session is None else self.store.bound(session)
        if goal is None or held is None or held.goal_id != goal.goal_id:
            raise ValueError("person_confirmation.session_required")
        decision = decision_hash(
            {field.name: getattr(request, field.name) for field in fields(request)}
        )
        if confirmation.decision_hash != decision:
            raise ValueError("person_confirmation.decision_mismatch")
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

        The session is the one the event declares (a native Team's, where the lead and its
        children speak), else the request's own; a goal the request names must be open, as
        for any request. The goal is the Host's word: one the producer wrote is replaced.
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

    def record(self, goal: Goal, *, whole_conversation: bool = False) -> dict[str, Any]:
        """The Host's facts; a waiter reads every message, never only the shown tail."""
        entries = self.store.attributed(goal.goal_id)
        requests = [e for e in entries if "operation" in e]
        messages = conversation_entries(entries)
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
            "open_assignments": open_assignments(entries),
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
            return {
                "status": "AVAILABLE",
                "schemas": {
                    name: model.model_json_schema()
                    for name, model in (
                        ("declaration", GoalDeclaration),
                        ("reference", GoalReferenceRequest),
                        ("statement", GoalStatement),
                        ("submission", GoalSubmission),
                    )
                },
                "reference_operations": REFERENCE_OPERATIONS,
                # The shortest declaration, valid as it stands, its words to replace, so
                # `goal schema --save-declaration goal.yaml` writes one to edit (V378).
                "template": GoalDeclaration.model_validate(_DECLARATION_TEMPLATE).model_dump(
                    mode="json", exclude_defaults=True
                ),
                "claim": "Contracts and declared budgets are not permission grants; the Host "
                "checks a submission against its own record, never the truth of its summary.",
            }
        if request.operation == "GOAL_LIST":
            return self.store.listing(
                limit=request.history_limit or 25,
                cursor=request.history_cursor,
                agent_session=request.agent_session,
            )
        if request.operation == "GOAL_OPEN":
            return self._open(request, caller, provenance)
        if (
            request.operation in _SESSION_GOAL_OPERATIONS
            and request.goal_id is None
            and request.goal_hash is None
        ):
            # The session's own goal, so an agent never copies its id or hash (V391).
            bound = self.attributed_goal(provenance)
            if bound is None:
                raise ValueError("goal.goal_id_required")
            request = replace(request, goal_id=bound.goal_id)
        if request.operation == "GOAL_REVISE":
            assert request.goal_declaration is not None
            prior = self._selected(request)
            declaration = GoalDeclaration.model_validate(request.goal_declaration)
            if "FIRST_USE" in {prior.declaration.kind, declaration.kind}:
                # Its objective is the person's sentence and its hours run from its opening;
                # neither is revised, nor does another goal become one (V452).
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
        if request.operation == "GOAL_REFERENCE":
            assert request.goal_reference_id is not None
            with GoalReferenceReadScope() as scope:
                return self.reference_readback(goal, request.goal_reference_id, caller, scope=scope)
        if request.operation == "GOAL_NARRATIVE":
            return self.narrative(goal)
        if request.operation in {"GOAL_SHOW", "GOAL_EXPORT"}:
            with GoalReferenceReadScope() as scope:
                body = self.readback(goal, caller, scope=scope)
                if request.operation == "GOAL_EXPORT":
                    return self.export(goal, body, caller, scope=scope)
                if goal.state == "OPEN":
                    # Its completion to fill, which `--save-declaration` writes (V419).
                    body["yaml"] = completion_template(goal)
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
                ref = self._sealed_reference(goal, draft_ref, caller, scope=scope)
            return self._save(request, caller, goal, references=(*goal.references, ref))
        if request.operation == "GOAL_NOTE":
            statement = GoalStatement.model_validate(request.goal_statement)
            self._require_references(goal, statement.evidence)
            if statement.disposition in {"SUPPORTS", "DOES_NOT_SUPPORT"} and not statement.evidence:
                raise ValueError("goal.conclusion_requires_evidence")
            if statement.responds_to and not any(
                s.statement_id == statement.responds_to for s in goal.statements
            ):
                raise ValueError("goal.response_target_absent")
            with GoalReferenceReadScope() as scope:
                # Re-read cited owners before accepting a statement. Text remains
                # attributed interpretation, not a deterministic correctness or
                # host-identity certificate.
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
        research = declaration.research
        if research and research.purpose == "CONTINUATION" and not declaration.parent_goal_id:
            raise ValueError("goal.continuation_requires_parent_goal")
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
        if prior is None and declaration.kind == "FIRST_USE" and self.first_use() is not None:
            # A workspace has one first use, opened from the person's sentence (V452).
            raise ValueError("goal.first_use_already_opened")
        if prior is not None:
            if prior.revision == 1 and prior.declaration == declaration:
                answer = {"status": "REUSED_EXACT", **self._identity_of(prior)}
                return {**answer, **self._bind(prior, session)}
            raise ValueError("goal.already_open_revise_it")
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
        """A submission declaring an id the goal holds, or one id twice, refused with its way on.

        Each clashing entry is named by its field. An entry that only re-declares a reference the
        goal holds (the same request) is dropped from the offered resubmission, its citations
        kept; one naming another result under a held id is the agent's to rename (V385).
        """

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
            for draft in submission.references:
                try:
                    added.append(self._sealed_reference(goal, draft, caller, scope=scope))
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
                    self._verified(ref, caller, scope=scope)
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
            return {
                "status": "INCOMPLETE",
                **self._identity_of(goal),
                "missing": missing,
                "record": record,
                "next_requests": {
                    "resubmit": {
                        "operation": "GOAL_SUBMIT",
                        "goal_id": str(goal.goal_id),
                        "goal_hash": goal.goal_hash,
                        # Left open, the goal and its revision bound (V441).
                        "goal_submission": None,
                    },
                    "show": {"operation": "GOAL_SHOW", "goal_id": str(goal.goal_id)},
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
        return {
            **answer,
            "status": "COMPLETE",
            "outcome": submission.outcome,
            # A prepared bundle with no accepted answer is said, never a reason to refuse.
            **(
                {"open_assignments": record["open_assignments"]}
                if record["open_assignments"]
                else {}
            ),
            "claim": "Complete means the record is complete and its evidence verified, not "
            "that the objective was met.",
        }

    def _sealed_reference(
        self,
        goal: Goal,
        draft_ref: GoalReferenceRequest,
        caller: str,
        *,
        scope: GoalReferenceReadScope,
    ) -> GoalReference:
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
        )

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
        return {
            "status": "GOAL_SAVED",
            **self._identity_of(saved),
            "task_id": None,
            "numerical_work": "NONE",
            "next_requests": self._next(saved),
        }

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
        request = PortfolioResearchRequestDocument.model_validate(
            ref.request
        ).to_operation_request()
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

    def _saved(self, goal: Goal, *, whole_conversation: bool = False) -> dict[str, Any]:
        """The saved revision's own facts, shared by the narrative and the verified readback."""
        head = self.store.head(goal.goal_id)
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
            "state": goal.state,
            "outcome": goal.outcome(),
            "open_choices": [
                s.model_dump(mode="json") for s in goal.statements if s.disposition == "OPEN"
            ],
            "record": self.record(goal, whole_conversation=whole_conversation),
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

        The revision's own content hash was checked when the store loaded it, so the
        objective, statements and the references' recorded identities are
        exact. Whether each referenced owner still returns that identity is a separate,
        slower question: ``GOAL_SHOW`` checks every reference; ``GOAL_REFERENCE`` checks one.
        Its conversation is complete, so
        a waiter cannot lose a message when more than the shown window arrives between reads.
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
        self, ref: GoalReference, caller: str, *, scope: GoalReferenceReadScope
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """One reference's owner verification, shared by selected and whole revision reads."""
        row: dict[str, Any] = {
            "reference": ref.model_dump(mode="json"),
            "next_requests": {"open": ref.request},
        }
        gap: dict[str, Any] | None = None
        try:
            body = self._verified(ref, caller, scope=scope)
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
        self, goal: Goal, caller: str, *, scope: GoalReferenceReadScope | None = None
    ) -> dict[str, Any]:
        """The revision with every reference re-verified at its owner, and its gaps."""
        scope = scope or GoalReferenceReadScope()
        rows, missing = [], []
        for ref in goal.references:
            row, gap = self._reference_row(ref, caller, scope=scope)
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
            **self._saved(goal),
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
