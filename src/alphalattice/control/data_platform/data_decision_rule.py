"""Declared data choices and submissions, independent of local grant mechanics."""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from functools import partial
from typing import Any, Literal, cast

from pydantic import ConfigDict, Field, TypeAdapter
from pydantic.dataclasses import dataclass

from alphalattice.control.data_platform.contracts import (
    DataRemediationActorSubmission,
    DataRemediationExecutionSubmission,
)
from alphalattice.foundation.feature_engine.inputs.gateway import (
    FeatureInputAgentCase,
    ListingQualityEvidence,
)
from alphalattice.foundation.market_data_ops.runtime.remediation import canonical_hash
from alphalattice.foundation.market_data_ops.sources.contracts import DataRemediationProposal
from alphalattice.protocols.actor_execution.contracts import ActorKind

SHIPPED_PRECISION = 1e-6
"""The shipped `adjusted_precision`, outside a rule's identity: the release before it had none."""


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class DataDecisionRule:
    """The Host's declared workspace rule for data issues, never its results.

    An isolated verified move (one listing, one unexplained session, identity verified, no
    provider correction, retries spent) is retained with its caveat; any other case is
    quarantined, recoverably. Past the cap the rule decides none: many unexplained moves at
    once may be a source problem, a judgment it leaves to the agent. A changed rule is a new
    `rule_hash`, which every receipt it signs carries.

    Two kinds of decision keep their own bounds. Data cases (a move retained or quarantined)
    count against `max_cases` and the share of names they touch; full-history audits (a
    re-fetch a provider's back-adjustment asks for, which the action set then explains) count
    against the same share of names, apart. Mixed, an ordinary ex-dividend day would escalate.
    A re-raised case whose anomaly signature already carries this rule's decision holds it and
    counts toward no bound: the same evidence gets the same decision, not a new one.

    `adjusted_precision` is the relative difference within which a provider's re-sent adjusted
    close for a stored session leaves the stored value standing (raw closes unchanged): a
    later observation does not rewrite a published session. Past it the series restates.
    """

    max_cases: int = Field(default=10, ge=0, strict=True)
    max_population_share: float = Field(default=0.02, ge=0, le=1, allow_inf_nan=False, strict=True)
    adjusted_precision: float = Field(
        default=SHIPPED_PRECISION,
        ge=0,
        le=1e-3,
        allow_inf_nan=False,
        strict=True,
        exclude_if=lambda value: value == SHIPPED_PRECISION,
    )
    version: Literal["data-decision-rule/1"] = "data-decision-rule/1"

    @property
    def rule_hash(self) -> str:
        """The rule's identity: its declared fields but a shipped precision (old rules keep it)."""
        return str(canonical_hash(_DECLARED.dump_python(self, mode="json")))

    def capped(self, cases: int, population: int, *, affected_listings: int) -> bool:
        """Count cases and distinct affected listings against the declared bounds."""
        return cases > self.max_cases or affected_listings > self.max_population_share * population

    def audits_capped(self, audited: int, population: int) -> bool:
        """Count audited listings against the share of names, the audits' own bound."""
        return audited > self.max_population_share * population

    @staticmethod
    def taken(
        case: FeatureInputAgentCase, decided: Mapping[tuple[str, str | None], str]
    ) -> str | None:
        """The option this rule last took for every listing's same anomaly, if one."""
        taken = {
            decided.get((item.listing_id, item.anomaly_signature_hash))
            for item in case.evidence
            if item.listing_id in case.listing_ids
        }
        return taken.pop() if len(taken) == 1 else None

    def held(
        self, case: FeatureInputAgentCase, decided: Mapping[tuple[str, str | None], str]
    ) -> str | None:
        """The option this rule took for the same anomaly, while still its fresh choice.

        The rule's choice on this case's own evidence must equal it: a provider correction, say,
        makes a new decision.
        """
        option = self.taken(case, decided)
        return option if option is not None and option == self.choose(case, case.evidence) else None

    def scope(
        self,
        cases: Sequence[FeatureInputAgentCase],
        population: int,
        records: Iterable[Mapping[str, Any]] = (),
        decided: Mapping[tuple[str, str | None], str] | None = None,
    ) -> dict[str, Any]:
        """An escalated set keeps its original bound after partial choices or exclusion.

        A case this rule already decided on the same anomaly holds and counts toward no bound;
        the Task's audits keep their own.
        """
        held = {case.case_token for case in cases if decided and self.held(case, decided)}
        prior = next(
            (
                row
                for row in records
                if row.get("rule_hash") == self.rule_hash
                and row.get("operation") != FULL_HISTORY
                and row.get("escalation_reason") == "RULE_BOUND_EXCEEDED"
            ),
            None,
        )
        counted = [case for case in cases if case.case_token not in held]
        counts = (
            {
                key: prior[key]
                for key in ("case_count", "affected_listing_count", "population_count")
            }
            if prior
            else {
                "case_count": len(counted),
                "affected_listing_count": len(
                    {listing for case in counted for listing in case.listing_ids}
                ),
                "population_count": population,
            }
        )
        return {
            **counts,
            "case_tokens": frozenset(case.case_token for case in cases),
            "capped": bool(prior)
            or self.capped(
                counts["case_count"],
                counts["population_count"],
                affected_listings=counts["affected_listing_count"],
            ),
        }

    def decide(
        self,
        case: FeatureInputAgentCase,
        evidence: tuple[ListingQualityEvidence, ...],
        scope: Mapping[str, Any],
    ) -> str:
        """Only a current case inside the original declared bound may be decided by rule."""
        return (
            ESCALATE
            if scope["capped"] or case.case_token not in scope["case_tokens"]
            else self.choose(case, evidence)
        )

    def ruling(
        self,
        case: FeatureInputAgentCase,
        evidence: tuple[ListingQualityEvidence, ...],
        scope: Mapping[str, Any],
        decided: Mapping[tuple[str, str | None], str],
    ) -> tuple[str, str]:
        """This case's option and status: held, decided, changed, or escalated.

        A case holds this rule's decision when its anomaly carries it and the rule would decide
        the same now; a different fresh choice is a new decision, disclosed as changed.
        """
        held = self.held(case, decided) if case.case_token in scope["case_tokens"] else None
        option = held or self.decide(case, evidence, scope)
        before = self.taken(case, decided)
        status = (
            "HELD_BY_RULE"
            if held
            else "ESCALATED"
            if option == ESCALATE
            else "CHANGED_BY_RULE"
            if before is not None
            else "DECIDED_BY_RULE"
        )
        return option, status

    def choose(
        self, case: FeatureInputAgentCase, evidence: tuple[ListingQualityEvidence, ...]
    ) -> str:
        """The offered option this rule takes for `case`, over every listing's evidence."""
        own = [item for item in evidence if item.listing_id in case.listing_ids]
        isolated = (
            case.case_kind == "LISTING"
            and case.failure_code == "data.unexplained_raw_move"
            and len(own) == 1
            and all(
                item.reason_codes == ("UNEXPLAINED_RAW_MOVE",)
                and item.identity_verified
                and not item.provider_correction_observed
                and item.retry_exhausted
                and len(item.unexplained_sessions) == 1
                for item in own
            )
        )
        offered = {option.option_id for option in case.options}
        preferred = (RETAIN,) * isolated + (QUARANTINE, ESCALATE)
        return next((option for option in preferred if option in offered), ESCALATE)

    def submission(
        self,
        case: FeatureInputAgentCase,
        option_id: str,
        goal_id: str,
        *,
        authority: str = "first-use-goal",
    ) -> DataRemediationActorSubmission:
        """The Host's signed choice: HOST_FALLBACK, the goal's delegation and this rule."""
        proposal = DataRemediationProposal(
            run_id=case.run_id,
            case_token=case.case_token,
            evidence_hash=case.evidence_hash,
            option_id=option_id,
            rationale=f"{authority}:{goal_id}; rule:{self.rule_hash}",
        )
        payload = {
            "kind": "DataRemediationExecutionSubmission",
            "proposal": proposal.model_dump(mode="json"),
        }
        return DataRemediationActorSubmission(
            submission=DataRemediationExecutionSubmission(
                **payload, submission_hash=canonical_hash(payload)
            ),
            actor_kind=ActorKind.HOST_FALLBACK,
            actor_id=f"host-rule:{self.rule_hash[:16]}",
        )


RETAIN = "retain_isolated_raw_move_with_caveat"
QUARANTINE = "recoverable_quarantine"
ESCALATE = "escalate_data_truth_conflict"
RULED_RUNS = {
    "portfolio_public_watermark_advancement": "RESEARCH_UPDATE_RUN",
    "workspace_data_update": "DATA_UPDATE_RUN",
}
"""The Task kinds the workspace rule decides for the open Goal whose run admitted them: a dated
book's update and a data update, so the same evidence gets the same decision."""


_DECLARED = TypeAdapter(DataDecisionRule)

AUDIT = "full_history_audit"
"""A full-history audit the rule approved, as its disclosed row names it."""
KEPT = "adjusted_precision_kept"
"""Stored adjusted closes the rule's precision kept, as their disclosed row names them."""
FULL_HISTORY = "FULL_HISTORY_AUDIT"
"""The Goal entry of the rule's decision on a full-history audit, never a membership change."""
ADJUSTED_PRECISION = "ADJUSTED_PRECISION_KEPT"
"""The Goal entry recording listings whose stored adjusted closes stood under the precision."""


def rule_receipt(saved: Mapping[str, Any]) -> tuple[str, str] | None:
    """The option and rule hash of an applied choice a Host rule signed, else None."""
    receipt = saved.get("receipt")
    effect = saved.get("effect") or {}
    if not receipt or not effect or effect.get("failure_reasons"):
        return None
    actor, proposal = receipt["actor_submission"], receipt["submission"]["proposal"]
    rule = proposal["rationale"].partition("; rule:")[2]
    signed = (
        actor["actor_kind"] == "HOST_FALLBACK" and actor.get("actor_id") == f"host-rule:{rule[:16]}"
    )
    return (proposal["option_id"], rule) if rule and signed else None


def held_decisions(
    choices: Iterable[tuple[Any, Mapping[str, Any]]],
    rule_hash: str,
    quarantined: Collection[str] = (),
) -> dict[tuple[str, str | None], str]:
    """Each (listing, anomaly signature) whose newest decision is this rule's, with its option.

    The decisions read oldest first: a later one on the same anomaly by anyone else (the
    person's choice) ends the rule's hold, and a quarantine since released holds nothing.
    """
    held: dict[tuple[str, str | None], str] = {}
    for document, saved in choices:
        signed = rule_receipt(saved)
        case = FeatureInputAgentCase.read_document(document)
        for item in case.evidence:
            if item.listing_id not in case.listing_ids or not item.anomaly_signature_hash:
                continue
            key = (item.listing_id, item.anomaly_signature_hash)
            held.pop(key, None)
            if signed is not None and signed[1] == rule_hash:
                held[key] = signed[0]
    return {
        key: option for key, option in held.items() if option != QUARANTINE or key[0] in quarantined
    }


def kept_listings(entries: Iterable[Mapping[str, Any]]) -> dict[str, tuple[str, float]]:
    """Each listing whose stored adjusted closes stood under the rule's precision, latest first."""
    stood: dict[str, tuple[str, float]] = {}
    for e in entries:
        if e.get("operation") == ADJUSTED_PRECISION and e.get("decided_by") == "RULE":
            for listing, largest in zip(
                e["listing_ids"], e["largest_relative_differences"], strict=True
            ):
                stood[str(listing)] = (str(e["rule_hash"]), float(largest))
    return stood


def approved_audits(entries: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The rule's audit approvals among these entries, less any its cycle did not admit."""
    last: dict[Any, Mapping[str, Any]] = {}
    for e in entries:  # oldest first: a retried audit's later status stands
        if e.get("operation") == FULL_HISTORY and e.get("decided_by") == "RULE":
            last[e.get("audit_id")] = e
    return [e for e in last.values() if e.get("status") == "DECIDED_BY_RULE"]


def attach(
    decide: Any,
    rule: DataDecisionRule,
    live: Callable[[], Any],
    entries: Callable[[], Iterable[Mapping[str, Any]]],
    record: Callable[[dict[str, Any]], None],
    task: str,
) -> Callable[..., Any]:
    """Give a Task's decide its audits, its precision and its record of closes that stood."""
    granted: list[Any] = []
    decide.audit = partial(audit, rule, live, entries, record, task)
    decide.precision = partial(precision, rule, live, granted)
    decide.kept = partial(kept, rule, granted, record)
    return cast(Callable[..., Any], decide)


def precision(rule: DataDecisionRule, live: Callable[[], Any], granted: list[Any]) -> float:
    """The precision a maintenance cycle may keep stored closes under.

    The rule's while its Goal's authority holds, which then records the cycle's keeps; else
    none, and the series restates.
    """
    goal = live()
    granted[:] = [] if goal is None else [goal]
    return 0.0 if goal is None else rule.adjusted_precision


def kept(
    rule: DataDecisionRule,
    granted: list[Any],
    record: Callable[[dict[str, Any]], None],
    documents: Iterable[Mapping[str, Any]],
    manifest_listings: Iterable[Any],
) -> None:
    """Record the listings whose stored adjusted closes stood under the rule's precision.

    Each carries its symbol and the largest relative difference the provider sent again, as
    its maintenance change document records it.
    """
    symbols = {item.listing_id: item.symbol for item in manifest_listings}
    listings = {
        str(item["listing_id"]): (
            symbols.get(str(item["listing_id"]), str(item["listing_id"])),
            float(item["adjusted_precision_kept"]),
        )
        for item in documents
        if "adjusted_precision_kept" in item
    }
    if not listings:
        return
    if not granted:  # a keep needs a precision, which only a live Goal grants
        raise ValueError("workspace_maintenance.precision_kept_without_authority")
    goal, ordered = granted[0], sorted(listings)
    record(
        {
            "operation": ADJUSTED_PRECISION,
            "decided_by": "RULE",
            "status": "KEPT_BY_RULE",
            "listing_ids": ordered,
            "symbols": [listings[item][0] for item in ordered],
            "largest_relative_differences": [listings[item][1] for item in ordered],
            "adjusted_precision": rule.adjusted_precision,
            "rule_hash": rule.rule_hash,
            "goal_hash": goal.goal_hash,
        }
    )


def audit(
    rule: DataDecisionRule,
    live: Callable[[], Any],
    entries: Callable[[], Iterable[Mapping[str, Any]]],
    record: Callable[[dict[str, Any]], None],
    task: str,
    listings: Mapping[str, str],
    population: int,
    admit: Callable[[str], Any],
) -> Any:
    """Approve a full-history audit within the audits' own bound, written ahead of admission.

    A provider's back-adjustment after a dividend asks for one on ordinary days. The bound
    counts the listings the rule already audited for this Task against the share of names,
    apart from its data cases. Past it the refusal is recorded, so the Task's audits stay
    escalated; outside the Goal's authority the stop stays the person's or the agent's confirm.
    The decision is recorded before the cycle admits it, and an admission that fails records
    that it did not.

    Args:
        rule: The workspace's declared rule, or the shipped default.
        live: The owning Goal while its authority holds, else None.
        entries: The owning Goal's recorded entries.
        record: Writes one decision into the owning Goal.
        task: The Task whose cycle stopped.
        listings: The listings whose whole history the cycle must audit, with their symbols.
        population: The listings in the research population.
        admit: Admits the audit onto the cycle under this rule's hash.

    Returns:
        What `admit` returned, or None when the stop stands.
    """
    goal = live()
    own = [
        e
        for e in entries()
        if e.get("task_id") == task
        and e.get("rule_hash") == rule.rule_hash
        and e.get("decided_by") == "RULE"
    ]
    audited = {item for e in approved_audits(own) for item in e["listing_ids"]}
    new = sorted(set(listings) - audited)
    counts = {"audit_count": len(audited) + len(new), "population_count": population}
    escalated = any(
        e.get("operation") == FULL_HISTORY and e.get("status") == "ESCALATED" for e in own
    )
    if goal is None or not new or escalated:
        return None
    row = {
        "operation": FULL_HISTORY,
        "decided_by": "RULE",
        "listing_ids": new,
        "symbols": [listings[item] for item in new],
        "rule_hash": rule.rule_hash,
        "goal_hash": goal.goal_hash,
        "audit_id": canonical_hash([task, new, rule.rule_hash]),
        **counts,
    }
    if rule.audits_capped(counts["audit_count"], population):
        record({**row, "status": "ESCALATED", "escalation_reason": "RULE_BOUND_EXCEEDED"})
        return None
    record({**row, "status": "DECIDED_BY_RULE"})
    try:
        return admit(rule.rule_hash)
    except Exception:
        record({**row, "status": "NOT_ADMITTED"})
        raise
