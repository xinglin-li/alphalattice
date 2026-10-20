"""Host-sealed Data choices over existing pre-Factor and maintenance case owners.

Selections remain separate from deterministic execution; no raw-data mutation
or model-derived data-truth authority is granted by a confirmation.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from functools import partial
from typing import TYPE_CHECKING, Any, Final, Literal, cast
from uuid import UUID

from alphalattice.control.data_platform.contracts import (
    DataRemediationExecutionReceipt,
    DataRemediationExecutionSubmission,
)
from alphalattice.control.data_platform.data_decision_rule import (
    ADJUSTMENT_SEGMENTS,
    AUDIT,
    KEPT,
    QUARANTINE,
    RAW_REBASES,
    RETAIN,
    DataDecisionRule,
    KeptListing,
    KeptObservations,
    rule_receipt,
)
from alphalattice.control.data_platform.delegation import DataIssueDelegation
from alphalattice.control.guanyin.data.workspace_maintenance import maintenance_failure_detail
from alphalattice.control.product_host.composition.plain_refusals import task_record_refusal
from alphalattice.control.product_host.composition.research_workspace import (
    RESEARCH_WORKSPACE_MANIFEST_NAME,
    ResearchWorkspaceManifest,
    read_research_workspace_manifest,
)
from alphalattice.foundation.feature_engine.inputs.contracts import ListingQuarantine
from alphalattice.foundation.feature_engine.inputs.gateway import (
    FeatureInputAgentCase,
    FeatureInputGateway,
    FeatureInputGovernanceService,
    QuarantineContinuation,
    QuarantineContinuationRefusal,
)
from alphalattice.foundation.feature_engine.storage.repositories import PanelStateRepository
from alphalattice.foundation.market_data_ops.runtime.remediation import (
    PolicyDecision,
    RemediationAction,
    RemediationOption,
    canonical_hash,
)
from alphalattice.foundation.market_data_ops.sources.contracts import (
    DataRemediationProposal,
)
from alphalattice.foundation.market_data_ops.sources.manifest import UniverseManifest
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.interface.local_application.cli_contract import (
    REQUEST_PROVENANCE,
    refusal_words,
)
from alphalattice.interface.local_application.failure_codes import public_failure
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchOperationRequest,
)
from alphalattice.protocols.actor_execution import (
    ActorKind,
)

if TYPE_CHECKING:
    from pathlib import Path

    from alphalattice.control.product_host.composition.application_session import (
        WorkspaceApplicationSession,
    )
    from alphalattice.control.task_control.contracts import TaskRecord
    from alphalattice.control.task_control.registry import DuckDbTaskControlRegistry
from alphalattice.control.data_platform.remediation_case import (
    matching_raw_retention_proof,
    raw_retention_decision_proofs,
    seal_validated_data_remediation_execution,
)
from alphalattice.control.task_control.contracts import TaskLifecycle, WorkItemLifecycle


def superseded_preparations(
    registry: DuckDbTaskControlRegistry, tasks: Iterable[TaskRecord]
) -> frozenset[str]:
    """The preparation Tasks a successor supersedes, by id (V510).

    A successor supersedes its source while it can resume: still on its way, stopped, or
    cancelled after it verified a stage to resume from. One cancelled before any stage verified
    holds nothing, so its source stands again; counting it hid a still-valid original, and the
    next plan was refused `workspace_preparation.existing_data_requires_explicit_binding`.

    Args:
        registry: Task Control, whose work items say what a cancelled successor verified.
        tasks: The workspace's Tasks.

    Returns:
        The ids of the preparation Tasks a successor supersedes.
    """
    return frozenset(
        str(task.input.payload["source_task_id"])
        for task in tasks
        if task.task_kind == "workspace_preparation"
        and task.input.payload.get("source_task_id")
        and (task.lifecycle is not TaskLifecycle.CANCELLED or _holds_a_stage(registry, task))
    )


def _holds_a_stage(registry: DuckDbTaskControlRegistry, task: TaskRecord) -> bool:
    return any(
        item.lifecycle is WorkItemLifecycle.VERIFIED for item in registry.work_items(task.task_id)
    )


def preparation_replan_refusal(
    registry: DuckDbTaskControlRegistry, workspace: Path, task: TaskRecord
) -> str | None:
    """Why no plan can resume this stopped preparation, or None.

    The one rule its plan, its recovery view and its data decisions' continuation read, so none
    offers a resuming plan that the plan refuses or answers with another predecessor.
    """
    if task.lifecycle is TaskLifecycle.RECOVERY_REQUIRED:
        return "workspace_preparation.resume_in_place"
    if task.lifecycle not in {TaskLifecycle.BLOCKED, TaskLifecycle.CANCELLED}:
        return "workspace_preparation.resume_scope_changed"
    if (
        task.failure_code == "data.current_universe_no_feature_ready_listings"
        and task.input.payload["plan"].get("information_cutoff_at") is not None
    ):
        return task.failure_code
    if read_research_workspace_manifest(workspace).experiment_inputs:
        return "workspace_preparation.inputs_already_published"
    if str(task.task_id) in superseded_preparations(registry, registry.tasks()):
        return "workspace_preparation.resume_scope_changed"
    if task.lifecycle is TaskLifecycle.CANCELLED and not _holds_a_stage(registry, task):
        return "workspace_preparation.nothing_captured_to_resume"
    return None


def reread_data_issues(code: str) -> dict[str, object]:
    """A stale selection recovers through the current cases, without applying any effect."""
    return {
        "status": "REFUSED",
        "failure_code": code,
        "next_action": "DATA_ISSUES",
        "next_requests": {"issues": {"operation": "DATA_ISSUES"}},
        "effect_applied": False,
    }


def _choice_request(
    operation: Literal["DATA_ISSUE_PREVIEW", "DATA_ISSUE_CONFIRM", "DATA_ISSUE_DELEGATE"],
    case: FeatureInputAgentCase,
    option: RemediationOption,
) -> dict[str, str]:
    return {
        "operation": operation,
        "data_issue_case_token": case.case_token,
        "data_issue_evidence_hash": case.evidence_hash,
        "data_issue_option_id": option.option_id,
        "data_issue_option_hash": option.option_hash,
    }


RULE_WORDS: Final = (
    "Data issues decided by {subject}: {subject} decided within {subject} cases and {subject} of "
    "names, {subject} held from an earlier decision; full-history audits: {subject} within "
    "{subject} of names. {subject} of {subject} names remain in the research population."
)
"""The rule's disclosure in one sentence, each slot filled in turn, so a reader's Chinese table
words it as the page does; held cases count toward no bound and are named apart."""
_RULE_CAVEATS: Final = {
    RETAIN: "An unexplained one-day move is kept as reported; it may be a data error.",
    QUARANTINE: "Set aside until its next recheck; the original inputs are kept.",
    AUDIT: "Its provider back-adjusted its history; its whole history was audited before use.",
    KEPT: "Its provider re-sent stored adjusted closes within the rule's precision; they stand.",
}
"""What each choice of the Host's rule means for the research, said beside the listing."""
_FACTOR_CAVEAT: Final = (
    "Its provider back-adjusted its history; the stored closes were scaled by its factors, "
    "so their returns stand."
)
"""A kept back-adjustment's caveat, beside its segments: each first and last session and factor."""
_REBASE_CAVEAT: Final = (
    "Its provider reported a split and rescaled its history; the stored bars were scaled by the "
    "split's factor, so their returns and Features stand."
)
"""A kept split's caveat, beside its rebases: each ex-date, first and last session rebased, price
and volume factor and receipt."""
_KEPT_OBSERVATIONS_CAVEAT: Final = (
    "After a split, these stored observation rows and blocks kept their bytes: their recomputed "
    "values differed only within the rule's agreement."
)
"""The caveat beside an update's observations kept as stored."""


def kept_observation_rows(
    retained: Sequence[KeptObservations], symbols: Mapping[str, str]
) -> dict[str, list[dict[str, Any]]]:
    """Each update's observations kept as stored, named as its delivery shows them; none, none."""
    return {"kept_observations": rows} if (rows := _kept_rows(retained, symbols)) else {}


def _kept_rows(
    retained: Sequence[KeptObservations], symbols: Mapping[str, str]
) -> list[dict[str, Any]]:
    return [
        {
            "observation_snapshot_hash": snapshot,
            "kept_rows": [
                {"listing_id": listing, "symbol": symbols.get(listing), "receipt_hash": receipt}
                | {"rows": count, "largest_relative_difference": largest}
                for listing, receipt, count, largest in rows
            ],
            "kept_blocks": list(blocks),
            "rebase_agreement": list(agreement),
            "caveat": _KEPT_OBSERVATIONS_CAVEAT,
        }
        for snapshot, rows, blocks, agreement in retained
    ]


class WorkspaceDataIssueApplication:
    """Human choice over the existing persisted Feature Input catalog.

    Confirmation seals intent only. The maintenance owner re-evaluates source
    evidence before applying it; the response never claims an unexecuted effect.
    """

    def __init__(
        self,
        session: WorkspaceApplicationSession,
        clock: Callable[[], datetime],
        *,
        recorded_data_confirmation: Callable[..., bool] | None = None,
        preparation_current: Callable[[TaskRecord], bool] | None = None,
    ) -> None:
        """Wire retained mutation authority, Data/Panel readers and exact issue delegation.

        Args:
            session: Retained workspace writer/task session.
            clock: Explicit observed-time source.
            recorded_data_confirmation: Original authority for retained rule decisions.
            preparation_current: Owner predicate for exact continuation; absent fails closed.
        """
        self.session = session
        self.clock = clock
        self.preparation_current = preparation_current
        self.market = MarketDataRepository(session.workspace)
        self.panel = PanelStateRepository(self.market.database, market_data=self.market)
        self.retention_proofs = partial(
            raw_retention_decision_proofs,
            self.panel,
            self.market,
            historical_authority=recorded_data_confirmation,
        )
        self.delegation = DataIssueDelegation(session.workspace)

    def operate(
        self, request: PortfolioResearchOperationRequest, caller: str, delegated_by: str | None
    ) -> dict[str, object]:
        """Route the admitted data choice and recover a stale choice through its current read."""
        assert request.data_issue_case_token is not None
        assert request.data_issue_evidence_hash is not None
        assert request.data_issue_option_id is not None
        assert request.data_issue_option_hash is not None
        if request.operation == "DATA_ISSUE_DELEGATE":
            assert request.task_id is not None
        choice = dict(
            case_token=request.data_issue_case_token,
            evidence_hash=request.data_issue_evidence_hash,
            option_id=request.data_issue_option_id,
            option_hash=request.data_issue_option_hash,
        )
        if request.operation == "DATA_ISSUE_PREVIEW":
            return cast(
                dict[str, object],
                self.preview(**choice, delegated_by=delegated_by),
            )
        if request.operation == "DATA_ISSUE_DELEGATE":
            return cast(
                dict[str, object],
                self.delegate(**choice, task_id=request.task_id, caller=caller),
            )
        try:
            return cast(
                dict[str, object],
                self.confirm(**choice, caller=caller, grant_hash=request.data_issue_grant_hash),
            )
        except ValueError as error:
            if not str(error).startswith(
                (
                    "feature_input.resolution_conflict",
                    "feature_input.case_no_longer_pending",
                    "feature_input.option_changed",
                )
            ):
                raise
            return reread_data_issues(public_failure(error, "feature_input.refused"))

    def _cases(
        self,
        *,
        limit: int | None = None,
        cursor: str | None = None,
        manifest_revision: str | None = None,
    ) -> tuple[FeatureInputAgentCase, ...]:
        if manifest_revision is None:
            if not self.market.path.is_file():
                return ()
            readiness = self.market.readiness.load("us-current-index-research")
            if readiness is None or readiness.active_manifest_revision is None:
                return ()
            manifest_revision = readiness.active_manifest_revision
        return tuple(
            FeatureInputAgentCase.read_document(document)
            for document in self.panel.feature_input_case_documents(
                manifest_revision,
                limit=limit,
                after_token=cursor,
            )
        )

    def _case_index(
        self, manifest: UniverseManifest | None, *, limit: int = 25, cursor: str | None = None
    ) -> tuple[
        tuple[FeatureInputAgentCase, ...], tuple[FeatureInputAgentCase, ...], dict[str, object]
    ]:
        """Every current case named in a line, beside the page of whole ones (V236, V172).

        A whole page of cases is longer than an agent's tool return (AX1d cut it five times),
        so the answer names every case and its count first; ``--section case_index`` reads just
        that, and ``next_page`` the next whole cases.
        """
        if manifest is None:
            return (), (), {"case_count": 0, "case_index": []}
        every = self._cases(manifest_revision=manifest.revision_sha256)
        selected = tuple(case for case in every if cursor is None or case.case_token > cursor)[
            : limit + 1
        ]
        return (
            every,
            selected,
            {
                "case_count": len(every),
                "case_index": [
                    {
                        "case_token": case.case_token,
                        "failure_code": case.failure_code,
                        "listing_count": len(case.listing_ids),
                    }
                    for case in every
                ],
            },
        )

    def current_decisions_ready(self, *, proofs=None, resolutions=None, cases=None) -> bool:
        """All active cases have a standing choice for preparation to revalidate."""
        manifest = self._active_manifest()
        if manifest is None:
            return False
        now = self.clock()
        with self.market.database.retain(read_only=True):
            cases = (
                cases
                if cases is not None
                else self._cases(manifest_revision=manifest.revision_sha256)
            )
            if not cases:
                return False
            proofs = proofs if proofs is not None else self.retention_proofs()
            if resolutions is None:
                resolutions = {
                    FeatureInputAgentCase.read_document(d).case_token: r
                    for d, r in self.panel.feature_input_decisions()
                }
            for case in cases:
                resolution = resolutions.get(case.case_token)
                if resolution is None:
                    continued = matching_raw_retention_proof(
                        panel_state=self.panel,
                        market_data=self.market,
                        candidate_manifest=manifest,
                        case=case,
                        proofs=proofs,
                    )
                    if continued is not None:
                        continue
                if (
                    resolution is None
                    or resolution.get("effect", {}).get("failure_reasons")
                    or self._wait_elapsed(resolution, now)
                ):
                    return False
        return True

    def _continuation(self, task: TaskRecord) -> tuple[str, str, dict[str, str]] | None:
        """The request that continues one stopped Task, or None where no plan can resume it."""
        plan = task.input.payload["plan"]
        if task.task_kind != "workspace_preparation":
            run = {"update_plan_hash": plan["content_hash"]}
            return "DATA_UPDATE_RUN", "/api/data-update/run", run
        if self.preparation_current is None or not self.preparation_current(task):
            return (
                "WORKSPACE_PREPARE_PLAN",
                "/api/workspace/preparation/plan",
                {"recovery_task_id": str(task.task_id)},
            )
        registry = self.session.task_control_registry
        if task.lifecycle.value == "BLOCKED" and preparation_replan_refusal(
            registry, self.session.workspace, task
        ):
            return None
        # A decided stop resumes the same Task at its retained stage.
        confirm = {"preparation_plan_hash": plan["plan_hash"]}
        return "WORKSPACE_PREPARE_CONFIRM", "/api/workspace/preparation/confirm", confirm

    def _governance(self) -> FeatureInputGovernanceService:
        """The same judgement owner the maintenance cycle uses, over the same stores."""
        return FeatureInputGovernanceService(
            market_data=self.market,
            panel_state=self.panel,
            mutation_gate=self.session.mutation_gate,
            gateway=FeatureInputGateway(),
        )

    @staticmethod
    def _ruled_choice(saved: Any) -> tuple[str, str] | None:
        """Disclose only an applied choice whose rule hash names its Host signer."""
        signed = rule_receipt(saved)
        return signed if signed is not None and signed[0] in _RULE_CAVEATS else None

    @staticmethod
    def _rule_rows(
        case: FeatureInputAgentCase,
        choice: tuple[str, str],
        symbols: dict[str, str],
        active: set[str],
        seen: dict[Any, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """One row per listing a ruled case names; the same anomaly re-raised over a longer
        window (its signature, never its range) under the same rule folds into its row."""
        moves = {item.listing_id: item.anomaly_signature_hash for item in case.evidence}
        option, rows = choice[0], []
        for listing in case.listing_ids:
            if (key := (listing, *choice, moves.get(listing))) in seen:
                seen[key]["_cases"].add(case.case_token)
                continue
            seen[key] = {
                "_cases": {case.case_token},
                "listing_id": listing,
                "symbol": symbols.get(listing),
                "option_id": option,
                "caveat": _RULE_CAVEATS[option]
                if option != QUARANTINE or listing in active
                else "The earlier quarantine is no longer active; this listing was readmitted.",
            }
            rows.append(seen[key])
        return rows

    def rule_decisions(
        self,
        manifest: UniverseManifest | None = None,
        every: tuple[FeatureInputAgentCase, ...] | None = None,
        *,
        input_available: bool = True,
        choices=None,
        case_tokens: set[str] | None = None,
        audits: Sequence[tuple[str, str]] | None = None,
        held: Collection[str] | None = None,
        kept: Sequence[KeptListing] = (),
        retained: Sequence[KeptObservations] = (),
    ) -> dict[str, Any] | None:
        """The data issues the Host's rule decided, in one line.

        Each listing with its choice and caveat, the population left after its quarantines, and
        the rule's identity; None when it decided none. Host-signed choices include first-use
        preparation and dated books' workspace policy, among open and resolved cases. The same
        move re-raised (history re-acquired, so a new case) folds into its one row. Each
        full-history audit the rule approved is a row too, and the words name whose rule decided.
        An update's observations kept as stored under the rebase agreement are named beside them.
        """
        if not input_available:
            return None
        manifest = manifest or self._active_manifest()
        if manifest is None:
            return None
        if every is None:
            every = self._cases(manifest_revision=manifest.revision_sha256)
        choices = choices if choices is not None else self.panel.feature_input_decisions()
        active = {
            q.listing_id
            for q in self.panel.active_listing_quarantines(
                manifest.revision_sha256, include_profile_history=True
            )
        }
        manifests = {manifest.revision_sha256: manifest}
        symbols = {item.listing_id: item.symbol for item in manifest.listings}
        decided, rules, aside, seen = [], set(), set(), dict[Any, dict[str, Any]]()
        for document, saved in choices:
            case = FeatureInputAgentCase.read_document(document)
            if case_tokens is not None and case.case_token not in case_tokens:
                continue
            choice = self._ruled_choice(saved)
            if choice is None:
                continue
            option, rule = choice
            rules.add(rule)
            if option == QUARANTINE:
                aside.update(set(case.listing_ids) & active)
            if not set(case.listing_ids) <= symbols.keys():  # set aside: its own manifest names it
                if case.manifest_revision not in manifests:
                    manifests[case.manifest_revision] = self.market.load_universe_manifest_revision(
                        case.manifest_revision
                    )
                symbols |= {
                    item.listing_id: item.symbol
                    for item in manifests[case.manifest_revision].listings
                }
            decided += self._rule_rows(case, choice, symbols, active, seen)
        configured = self._configured()
        audit_rows, audit_rules, held, unreadable = self._audit_rows(
            choices, (audits, held, kept), symbols, configured
        )
        decided, rules = decided + audit_rows, rules | audit_rules
        if not (decided or unreadable):  # a split's kept observations come with its kept row
            return None
        names = {item.listing_id for item in manifest.listings}
        source, left = len(names | aside), len(names - aside)
        whose, origins = self._rule_origins(rules, configured)
        bounds = self._rule_bounds(decided, configured, held)
        parts = (whose, *bounds["parts"], left, source)
        return {
            "words": functools.reduce(
                lambda text, part: text.replace("{subject}", str(part), 1), parts, RULE_WORDS
            ),
            "bounds": bounds["kinds"],
            "decided": sorted(decided, key=lambda d: (d["listing_id"], d["option_id"])),
            "source_population": source,
            "effective_population": left,
            "rule_hashes": sorted(rules),
            "rules": origins,
            # A Goal whose record cannot be read is named, never a crash of the data issues.
            **({"unreadable_goals": unreadable} if unreadable else {}),
            **kept_observation_rows(retained, symbols),
        }

    def _audit_rows(
        self,
        choices: Any,
        given: tuple[
            Sequence[tuple[str, str]] | None,
            Collection[str] | None,
            Sequence[KeptListing],
        ],
        symbols: dict[str, str],
        configured: ResearchWorkspaceManifest | None,
    ) -> tuple[list[dict[str, Any]], set[str], Collection[str], list[dict[str, str]]]:
        """The rule's approved audits as disclosed rows, the rules that approved them, the cases
        that held its decision, and any Goal whose record could not be read: as given (a dated
        update's own Goal), else read from the Goals that hold the rule's decisions."""
        audits, held, unreadable, kept = (
            self._rule_audits(choices, configured)
            if given[0] is None
            else (given[0], given[1] or (), [], given[2])
        )
        rows = [
            {"listing_id": item, "symbol": symbols.get(item), "option_id": AUDIT}
            | {"caveat": _RULE_CAVEATS[AUDIT]}
            for item, _rule in audits
        ] + [
            {"listing_id": item, "symbol": symbols.get(item), "option_id": KEPT}
            | {"largest_relative_difference": largest, "caveat": _RULE_CAVEATS[KEPT]}
            | ({ADJUSTMENT_SEGMENTS: segments, "caveat": _FACTOR_CAVEAT} if segments else {})
            | ({RAW_REBASES: rebases, "caveat": _REBASE_CAVEAT} if rebases else {})
            for item, _rule, largest, segments, rebases in kept
        ]
        rules = {rule for _item, rule in audits} | {item[1] for item in kept}
        return rows, rules, held, unreadable

    @staticmethod
    def _rule_bounds(
        decided: list[dict[str, Any]],
        configured: ResearchWorkspaceManifest | None,
        held_cases: Collection[str],
    ) -> dict[str, Any]:
        """Each kind's count against its own bound: data cases (a row reads held when a case
        folded into it held the rule's decision, as its Goal recorded), and full-history audits."""
        rule = (configured.data_decision_rule if configured else None) or DataDecisionRule()
        folded: set[str] = set()
        listings = 0  # a listing counts while one of its cases is decided, not held
        for row in decided:
            if "_cases" in row:
                tokens = row.pop("_cases")
                folded |= tokens
                row["held"] = bool(tokens & set(held_cases))
                listings += bool(tokens - set(held_cases))
        audits = sum(row["option_id"] == AUDIT for row in decided)
        held = len(folded & set(held_cases))
        cases, share = len(folded) - held, f"{rule.max_population_share:.0%}"
        return {
            # A held case is the same decision again and counts toward no bound (OP19).
            "parts": (cases, rule.max_cases, share, held, audits, share),
            "kinds": {
                "cases": {
                    "count": cases,
                    "listings": listings,
                    "held": held,
                    "max_cases": rule.max_cases,
                    "max_population_share": rule.max_population_share,
                },
                "audits": {"count": audits, "max_population_share": rule.max_population_share},
            },
        }

    def _configured(self) -> ResearchWorkspaceManifest | None:
        """The research workspace's manifest; a data-only workspace has none, so its rule is the
        shipped default and it holds no Goal."""
        if not (self.session.workspace / RESEARCH_WORKSPACE_MANIFEST_NAME).exists():
            return None
        return read_research_workspace_manifest(self.session.workspace)

    def _rule_origins(
        self, rules: set[str], configured: ResearchWorkspaceManifest | None
    ) -> tuple[str, list[dict[str, str]]]:
        """Whose rule decided: this workspace's declared one, the shipped default, or earlier."""
        declared = configured.data_decision_rule if configured else None
        origin = {
            DataDecisionRule().rule_hash: "SHIPPED_DEFAULT",
            **({declared.rule_hash: "WORKSPACE"} if declared else {}),
        }
        sources = {origin.get(rule, "EARLIER") for rule in rules}
        words = {
            "WORKSPACE": "this workspace's rule",
            "SHIPPED_DEFAULT": "the shipped default rule",
        }
        whose = (
            words.get(next(iter(sources)), "an earlier rule")
            if len(sources) == 1
            else "several rules"
        )
        rows = [
            {"rule_hash": rule, "source": origin.get(rule, "EARLIER")} for rule in sorted(rules)
        ]
        return whose, rows

    def _rule_audits(
        self, choices: Any, configured: ResearchWorkspaceManifest | None
    ) -> tuple[
        tuple[tuple[str, str], ...],
        frozenset[str],
        list[dict[str, str]],
        tuple[KeptListing, ...],
    ]:
        """The full-history audits the rule approved, the cases that held its decision and the
        listings whose stored closes stood under its precision, read once from the Goals that
        hold its decisions: those its signed choices name, and the open ones."""
        from alphalattice.control.product_host.publication.goals import GoalStore

        if configured is None:
            return (), frozenset(), [], ()
        store = GoalStore(self.session.workspace / "artifacts", configured.workspace_id)
        named = {
            saved["receipt"]["submission"]["proposal"]["rationale"]
            .partition(";")[0]
            .partition(":")[2]
            for _document, saved in choices
            if self._ruled_choice(saved) is not None
        }
        pairs: set[tuple[str, str]] = set()
        held: set[str] = set()
        stood: set[KeptListing] = set()
        unreadable = []
        for goal in store.goal_ids():
            try:
                if str(goal) in named or getattr(store.head(goal), "state", None) == "OPEN":
                    entries = store.attributed(goal)
                    pairs |= set(store.rule_audits(None, entries))
                    held |= store.rule_held(None, entries)
                    stood |= set(store.rule_kept(None, entries))
            except ValueError as error:
                failure = public_failure(error, "goal.attribution_invalid")
                unreadable.append({"goal_id": str(goal), "failure_code": failure})
        return tuple(sorted(pairs)), frozenset(held), unreadable, tuple(sorted(stood))

    def _escalation(
        self,
        delegated_by: str | None,
        manifest: UniverseManifest | None,
        every: tuple[FeatureInputAgentCase, ...],
        task_ids: set[str],
    ) -> dict[str, object]:
        """Why the Host's rule left a first use's issues to its agent: past its cap (jidoka)."""
        if delegated_by is None or manifest is None or not every:
            return {}
        from alphalattice.control.product_host.publication.goals import GoalStore

        configured = read_research_workspace_manifest(self.session.workspace)
        rule = configured.data_decision_rule or DataDecisionRule()
        records = GoalStore(
            self.session.workspace / "artifacts", configured.workspace_id
        ).attributed(UUID(delegated_by.removeprefix("first-use-goal:")))
        scope = rule.scope(
            every, len(manifest.listings), (r for r in records if r.get("task_id") in task_ids)
        )
        if not scope["capped"]:
            return {}
        case_count, listing_count = scope["case_count"], scope["affected_listing_count"]
        return {
            "escalated_by_rule": {
                "words": f"{case_count} cases affect {listing_count} listings, past the "
                f"rule's cap of {rule.max_cases} cases or {rule.max_population_share:.0%} of "
                "listings. Many at once may be a source problem, "
                "so the whole set is yours to judge.",
                "listing_count": listing_count,
                "case_count": case_count,
                "rule_hash": rule.rule_hash,
            }
        }

    def _active_manifest(self) -> UniverseManifest | None:
        if not self.market.path.is_file():
            return None
        readiness = self.market.readiness.load("us-current-index-research")
        if readiness is None or readiness.active_manifest_revision is None:
            return None
        return self.market.load_universe_manifest_revision(readiness.active_manifest_revision)

    def _continued_dispositions(self, manifest: UniverseManifest) -> list[dict[str, object]]:
        """Standing quarantines carried past a recheck, read back as a chain.

        Each chain: the original decision (case, option, execution receipt,
        first row), then every continuation in order (the row, the rule, the
        evidence the recheck examined), each verified by its record hash.
        Nothing here is a clearance: the chain ends when the listing is
        requalified by the existing owner.
        """
        chains: list[dict[str, object]] = []
        # A quarantined listing may sit outside the admitted manifest; the
        # profile's source admission names every listing of the lineage.
        source = self.market.source_admission_manifest(
            market_profile_id=manifest.profile.market_profile_id
        )
        symbols = {
            item.listing_id: item.symbol
            for candidate in (source, manifest)
            if candidate is not None
            for item in candidate.listings
        }
        for standing in self.panel.active_listing_quarantines(
            manifest.revision_sha256,
            include_profile_history=True,
            qualification_domain="MARKET_DATA",
        ):
            if standing.continued_from_quarantine_hash is None:
                continue
            lineage = {
                row.quarantine_hash: (row, lifecycle, document)
                for row, lifecycle, document in self.panel.quarantine_lineage(
                    market_profile_id=manifest.profile.market_profile_id,
                    listing_id=standing.listing_id,
                )
            }
            chain: list[QuarantineContinuation] = []
            cursor = standing.quarantine_hash
            while cursor in lineage and lineage[cursor][2] is not None:
                record = QuarantineContinuation.read_document(str(lineage[cursor][2]))
                # The stored row is the authority the record must describe.
                if record.quarantine != lineage[cursor][0]:
                    raise ValueError("feature_input.continuation_identity_mismatch")
                chain.insert(0, record)
                cursor = record.continued_from_quarantine_hash
            if cursor not in lineage or not chain:
                continue  # A chain whose root is not in this profile is not readable here.
            root = lineage[cursor][0]
            decision = self.panel.feature_input_decision_for_receipt(root.execution_receipt_hash)
            original: dict[str, object] = {
                "quarantine_hash": root.quarantine_hash,
                "execution_receipt_hash": root.execution_receipt_hash,
                "evidence_hash": root.evidence_hash,
                "reason_codes": list(root.reason_codes),
                "recheck_after_at": root.recheck_after_at.isoformat(),
                "case_token": None,
                "option_id": None,
            }
            if decision is not None:
                case_document, resolution = decision
                case = FeatureInputAgentCase.read_document(case_document)
                receipt = resolution.get("receipt")
                policy_decision = (
                    receipt.get("policy_decision") if isinstance(receipt, dict) else {}
                )
                original["case_token"] = case.case_token
                original["issue_hash"] = case.issue_hash
                original["option_id"] = (
                    policy_decision.get("option_id") if isinstance(policy_decision, dict) else None
                )
                original["receipt_hash"] = (
                    receipt.get("receipt_hash") if isinstance(receipt, dict) else None
                )
            chains.append(
                {
                    "listing_id": standing.listing_id,
                    "symbol": symbols.get(standing.listing_id),
                    "original": original,
                    "continuations": [
                        {
                            "quarantine_hash": item.quarantine.quarantine_hash,
                            "continued_from_quarantine_hash": item.continued_from_quarantine_hash,
                            "rule": item.rule,
                            "continuation_hash": item.continuation_hash,
                            "observed_at": item.observed_at.isoformat(),
                            "recheck_after_at": item.recheck_after_at.isoformat(),
                            "current_evidence_hash": item.evidence.evidence_hash,
                            "anomaly_signature_hash": item.evidence.anomaly_signature_hash,
                            "unexplained_sessions": [
                                session.isoformat()
                                for session in item.evidence.unexplained_sessions
                            ],
                            "range": [
                                item.evidence.range_start.isoformat(),
                                item.evidence.range_end.isoformat(),
                            ],
                        }
                        for item in chain
                    ],
                    "recheck_after_at": standing.recheck_after_at.isoformat(),
                    "next_check": "RECHECK_AT_DUE_TIME_NOT_AUTOMATIC_RECOVERY",
                    "claim_limit": "CONTINUED_DISPOSITION_NOT_CURRENT_CLEARANCE",
                }
            )
        return chains

    def _standing_quarantine_note(
        self,
        manifest: UniverseManifest,
        case: FeatureInputAgentCase,
        now: datetime,
        standing: dict[str, ListingQuarantine],
    ) -> dict[str, object] | None:
        """Why a pending case was raised over a listing that already carries a decision.

        The same judgement the maintenance cycle made, over the same frozen
        inputs (the standing row, the case's evidence): the reasons a
        continuation was refused, so the page says why it asks again.
        """
        notes = []
        governance = self._governance()
        for evidence in case.evidence:
            quarantine = standing.get(evidence.listing_id)
            if quarantine is None:
                continue
            judged = governance.judge_standing_quarantine(
                candidate_manifest=manifest,
                quarantine=quarantine,
                current=evidence,
                observed_at=now.astimezone(UTC),
            )
            notes.append(
                {
                    "listing_id": evidence.listing_id,
                    "quarantine_hash": quarantine.quarantine_hash,
                    "execution_receipt_hash": quarantine.execution_receipt_hash,
                    "continuation_refused": list(judged.reasons)
                    if isinstance(judged, QuarantineContinuationRefusal)
                    else [],
                }
            )
        return {"standing": notes} if notes else None

    def _continuations(self, tasks):
        """Only stopped Tasks with an admitted owner way on are offered for continuation."""
        continuations = []
        next_requests: dict[str, dict[str, str]] = {}
        for task in tasks:
            if (way := self._continuation(task)) is None:
                continue
            operation, endpoint, payload = way
            next_requests[f"continue:{task.task_id}"] = {"operation": operation, **payload}
            continuations.append(
                {
                    "task_id": str(task.task_id),
                    "task_kind": task.task_kind,
                    "lifecycle": task.lifecycle.value,
                    "failure_code": task.failure_code,
                    "failure_reason": self._failure_reason(task.failure_code, task.task_kind),
                    "operation": operation,
                    "endpoint": endpoint,
                    "payload": payload,
                }
            )
        return continuations, next_requests

    def readback(
        self, *, limit: int = 25, cursor: str | None = None, delegated_by: str | None = None
    ) -> dict[str, object]:
        """Read bounded current issue choices, recorded decisions and exact continuation grants.

        Args:
            limit: Explicit page size from 1 through 50.
            cursor: Optional exact case-token cursor.
            delegated_by: First-use confirmation authority; no person's grant is then offered.

        Returns:
            Cases, choices, effects, grants and next requests; no current-readiness claim.

        Raises:
            ValueError: Case page bound or cursor is invalid.
        """
        if not 1 <= limit <= 50 or (
            cursor is not None
            and (len(cursor) != 64 or any(c not in "0123456789abcdef" for c in cursor))
        ):
            raise ValueError("feature_input.case_page_invalid")
        active_manifest = self._active_manifest()
        every, selected, case_index = self._case_index(active_manifest, limit=limit, cursor=cursor)
        cases = selected[:limit]
        now = self.clock()
        batch = self.session.task_control_registry.record_collection()
        # Supersession and continuation need the complete Task authority. Cases remain readable.
        recorded_tasks = batch.records if not batch.refused_task_ids else ()
        superseded = superseded_preparations(self.session.task_control_registry, recorded_tasks)
        tasks = [
            task
            for task in recorded_tasks
            if task.task_kind in {"workspace_preparation", "workspace_data_update"}
            and task.lifecycle.value in {"DEFERRED", "BLOCKED", "REVIEW_PENDING"}
            and (
                task.task_kind == "workspace_data_update"
                or str(task.failure_code or "").startswith(("data.", "feature_input."))
            )
            and str(task.task_id) not in superseded
        ]
        continuations, next_requests = self._continuations(tasks)
        issues = []
        refused_cases: list[dict[str, object]] = []
        manifests: dict[str, UniverseManifest] = {}
        unavailable_revisions: set[str] = set()
        for case in cases:
            revision = case.manifest_revision
            if revision in manifests or revision in unavailable_revisions:
                continue
            try:
                manifests[revision] = self.market.load_universe_manifest_revision(revision)
            except (OSError, ValueError):
                # A readable case that names one unavailable immutable manifest is an item
                # refusal. The selected active manifest and case catalog were read above.
                unavailable_revisions.add(revision)
        standing_by_revision: dict[str, dict[str, ListingQuarantine]] = {}
        choices = self.panel.feature_input_decisions()
        retention_proofs = self.retention_proofs(decisions=choices)
        resolutions = {FeatureInputAgentCase.read_document(d).case_token: r for d, r in choices}
        for case in cases:
            if case.manifest_revision in unavailable_revisions:
                code = "feature_input.case_manifest_unavailable"
                words = refusal_words(f"{code}:{case.case_token}")
                issues_request: dict[str, str | int] = {"operation": "DATA_ISSUES"}
                if limit is not None:
                    issues_request["history_limit"] = limit
                if cursor is not None:
                    issues_request["history_cursor"] = cursor
                refused_cases.append(
                    {
                        "status": "REFUSED",
                        "case_token": case.case_token,
                        "failure_code": code,
                        "detail": words.get("detail")
                        or (
                            f"Data issue case {case.case_token} names a market manifest that "
                            "cannot be read; its subjects and choices are omitted. Read the "
                            "workspace and kept backups, then read Data Issues in a restored copy."
                        ),
                        "next_requests": {
                            "issues": issues_request,
                            "workspace": {"operation": "WORKSPACE_SHOW"},
                            "backups": {"operation": "WORKSPACE_BACKUPS"},
                        },
                    }
                )
                continue
            resolution = resolutions.get(case.case_token)
            continuation = (
                matching_raw_retention_proof(
                    panel_state=self.panel,
                    market_data=self.market,
                    candidate_manifest=manifests[case.manifest_revision],
                    case=case,
                    proofs=retention_proofs,
                )
                if resolution is None
                else None
            )
            refused = bool(resolution and resolution.get("effect", {}).get("failure_reasons"))
            # A wait that elapsed is a decision that ran its course: the
            # case is a choice again, and the elapsed decision reads back
            # among the prior ones rather than as the standing one.
            elapsed = self._wait_elapsed(resolution, now)
            prior_decisions: list[object] = []
            if resolution is not None:
                archived = resolution.get("prior_decisions")
                prior_decisions.extend(archived if isinstance(archived, list) else [])
                if elapsed:
                    prior_decisions.append(
                        {"receipt": resolution["receipt"], "effect": resolution["effect"]}
                    )
            if continuation is not None:
                prior_decisions.append(
                    {
                        "receipt": json.loads(continuation.receipt_document),
                        "effect": json.loads(continuation.effect_document),
                        "continued_from_manifest_revision": continuation.case.manifest_revision,
                        "continuation_receipt_hash": continuation.receipt_hash,
                    }
                )
            if (
                continuation is None
                and case.options_current_at(now)
                and not self._decided(resolution, now)
            ):
                for option in case.options:
                    if self._confirmable(option):
                        next_requests[f"preview:{case.case_token}:{option.option_id}"] = (
                            _choice_request("DATA_ISSUE_PREVIEW", case, option)
                        )
                        if delegated_by is not None:
                            # The first use's agent decides each case itself: its confirm is
                            # offered beside the preview, which only shows what it applies.
                            next_requests[f"confirm:{case.case_token}:{option.option_id}"] = (
                                _choice_request("DATA_ISSUE_CONFIRM", case, option)
                            )
                        for task in () if delegated_by is not None else tasks:
                            if (
                                task.task_kind == "workspace_preparation"
                                and task.lifecycle.value == "BLOCKED"
                                and task.failure_code == "data.truth_review_required"
                            ):
                                next_requests[
                                    f"delegate:{task.task_id}:{case.case_token}:{option.option_id}"
                                ] = {
                                    **_choice_request("DATA_ISSUE_DELEGATE", case, option),
                                    "task_id": str(task.task_id),
                                }
            effect = resolution.get("effect") if resolution else None
            waiting = (
                isinstance(effect, dict)
                and effect.get("retry_after_at") is not None
                and not elapsed
            )
            standing_note = None
            if resolution is None or elapsed:
                if case.manifest_revision not in standing_by_revision:
                    standing_by_revision[case.manifest_revision] = {
                        item.listing_id: item
                        for item in self.panel.active_listing_quarantines(
                            case.manifest_revision,
                            include_profile_history=True,
                            qualification_domain="MARKET_DATA",
                        )
                    }
                standing_note = self._standing_quarantine_note(
                    manifests[case.manifest_revision],
                    case,
                    now,
                    standing_by_revision[case.manifest_revision],
                )
            issues.append(
                {
                    "issue_hash": case.issue_hash,
                    "case": case.document(),
                    "subjects": {
                        item.listing_id: item.symbol
                        for item in manifests[case.manifest_revision].listings
                        if item.listing_id in case.listing_ids
                    },
                    "standing_quarantine": standing_note,
                    "status": "OPTION_REFUSED"
                    if refused
                    else "WAITING_FOR_RETRY"
                    if waiting
                    else "CONFIRMED_PENDING_REVALIDATION"
                    if (resolution and not elapsed) or continuation is not None
                    else "AWAITING_CHOICE",
                    "options_current": case.options_current_at(now),
                    "confirmation": "HUMAN",
                    "confirmable_option_ids": [
                        option.option_id for option in case.options if self._confirmable(option)
                    ],
                    "resolution": None if elapsed else resolution,
                    "prior_decisions": prior_decisions,
                    "continued_decision": continuation is not None,
                    "next_action": "RESUME_APPROVED_WORK"
                    if (resolution and not elapsed) or continuation is not None
                    else "SELECT_PERMITTED_OPTION",
                }
            )
        history = []
        if self.market.path.is_file() and self.market.readiness.load("us-current-index-research"):
            for document in self.panel.feature_input_case_documents(
                None, lifecycle="RESOLVED", limit=10
            ):
                case = FeatureInputAgentCase.read_document(document)
                history.append(
                    {
                        "case_token": case.case_token,
                        "issue_hash": case.issue_hash,
                        "listing_ids": case.listing_ids,
                        "failure_code": case.failure_code,
                        "resolution": self.panel.feature_input_resolution(case.case_token),
                        "claim_limit": "HISTORICAL_EFFECT_NOT_CURRENT_CLEARANCE",
                    }
                )
        following = cases[-1].case_token if len(selected) > limit else None
        if following is not None:
            # The next page, offered as the request that reads it (V172).
            next_requests["next_page"] = {
                "operation": "DATA_ISSUES",
                "history_cursor": following,
                "history_limit": limit,
            }
        # What waits on a person, all owners' in one read, before the pages (V237).
        next_requests["pending"] = {"operation": "PENDING_DECISIONS"}
        grants = self.delegation.active_grants(now=now)
        for grant in grants:
            next_requests[f"confirm:{grant['grant_hash']}"] = {
                "operation": "DATA_ISSUE_CONFIRM",
                "data_issue_case_token": grant["case_token"],
                "data_issue_evidence_hash": grant["evidence_hash"],
                "data_issue_option_id": grant["option_id"],
                "data_issue_option_hash": grant["option_hash"],
                "data_issue_grant_hash": grant["grant_hash"],
            }
        answer: dict[str, object] = {
            "status": "ISSUES_PENDING"
            if cases
            else "DATA_TASK_WAITING_OR_BLOCKED"
            if tasks
            else "NO_PENDING_CASE_CATALOG",
            "issues": issues,
            **case_index,
            "continuations": continuations,
            "decided_by_rule": self.rule_decisions(active_manifest, every, choices=choices),
            **self._escalation(
                delegated_by, active_manifest, every, {str(t.task_id) for t in tasks}
            ),
            "next_cursor": following,
            "recorded_decisions": history,
            "delegations": [
                {
                    "grant_hash": grant["grant_hash"],
                    "case_token": grant["case_token"],
                    "evidence_hash": grant["evidence_hash"],
                    "option_id": grant["option_id"],
                    "option_hash": grant["option_hash"],
                    "target_listing_ids": grant["target_listing_ids"],
                    "preparation_task_id": grant["preparation_task_id"],
                    "expires_at": grant["expires_at"],
                    "issued_by": grant["issued_by"],
                    "actual_executor": grant["actual_executor"],
                    "next_requests": {
                        "confirm": next_requests[f"confirm:{grant['grant_hash']}"],
                        "revoke": {
                            "operation": "DATA_ISSUE_REVOKE",
                            "data_issue_grant_hash": grant["grant_hash"],
                        },
                        "preparation_plan": {"operation": "WORKSPACE_PREPARE_PLAN"},
                    },
                }
                for grant in grants
            ],
            "continued_dispositions": (
                self._continued_dispositions(active_manifest) if active_manifest is not None else []
            ),
            "next_requests": next_requests,
            "decisions_complete": self.current_decisions_ready(
                proofs=retention_proofs, resolutions=resolutions, cases=every
            ),
            **({"delegated_by": delegated_by} if delegated_by is not None else {}),
            "claim_limit": "NO_DATA_TRUTH_OR_CURRENT_READINESS_CLAIM",
        }
        if batch.refused_task_ids:
            answer["task_refusals"] = [
                task_record_refusal(task_id) for task_id in batch.refused_task_ids
            ]
        if refused_cases:
            answer["refused_cases"] = refused_cases
        return answer

    def _failure_reason(self, code: str | None, task_kind: str) -> dict[str, object] | None:
        """Why the Task stopped, in words the page may show, and whether a retry can help.

        The stable code is the Task's; the explanation is the recorded
        remediation failure's bounded text when the newest one carries this
        code, else the code's own user-safe detail. Nothing here is raw
        exception, Provider or model text.
        """

        if code is None:
            return None
        # The maintenance package composes the coordinator, which uses this
        # module's sealing; the registry is reached at call time.
        from alphalattice.control.data_platform.maintenance.registry import (
            DuckDbWorkspaceMaintenanceRegistry,
        )

        recorded = None
        if self.market.path.is_file():
            recorded = DuckDbWorkspaceMaintenanceRegistry(
                self.market.path, gate=self.session.mutation_gate
            ).latest_data_remediation_failure()
        matching = recorded if recorded is not None and recorded.failure_code == code else None
        preparation_truth_review = (
            code == "data.truth_review_required" and task_kind == "workspace_preparation"
        )
        return {
            "code": code,
            "explanation": (
                "Listings need a data decision before preparation can continue. "
                "Open Data issues, choose a permitted option for each case, then continue the Task."
                if preparation_truth_review
                else matching.explanation
                if matching is not None and matching.explanation
                else maintenance_failure_detail(code)
            ),
            "retryable": matching.retryable if matching is not None else None,
            "recorded_at": (
                matching.observed_at.astimezone(UTC).isoformat() if matching is not None else None
            ),
        }

    @staticmethod
    def _wait_elapsed(resolution: dict[str, object] | None, now: datetime) -> bool:
        """Whether the standing decision is a wait whose time has passed."""
        effect = (resolution or {}).get("effect")
        retry_after = effect.get("retry_after_at") if isinstance(effect, dict) else None
        return retry_after is not None and datetime.fromisoformat(str(retry_after)) <= (
            now.astimezone(UTC)
        )

    @classmethod
    def _decided(cls, resolution: dict[str, object] | None, now: datetime) -> bool:
        """Whether a decision stands for the case, so it takes no other choice now: one is
        recorded, its effect was not refused, and it is no wait whose time has passed. The issues
        list offers it no choice and its preview refuses, as a different confirm conflicts."""
        effect = (resolution or {}).get("effect")
        refused = isinstance(effect, dict) and bool(effect.get("failure_reasons"))
        return resolution is not None and not refused and not cls._wait_elapsed(resolution, now)

    @staticmethod
    def _confirmable(option: RemediationOption) -> bool:
        return option.policy_args.action in {
            RemediationAction.RETAIN_RAW_VALUE_WITH_CAVEAT,
            RemediationAction.QUARANTINE_LISTING,
            RemediationAction.EXCLUDE_FROM_NEXT_MANIFEST,
            RemediationAction.WAIT_THEN_RETRY,
        }

    def _selection(
        self, *, case_token: str, evidence_hash: str, option_id: str, option_hash: str
    ) -> tuple[FeatureInputAgentCase, RemediationOption]:
        case = next((item for item in self._cases() if item.case_token == case_token), None)
        if case is None:
            raise ValueError("feature_input.case_no_longer_pending")
        gateway = FeatureInputGateway()
        if case.policy_hash != gateway.policy.policy_hash:
            raise ValueError("feature_input.resolution_policy_changed")
        if not case.options_current_at(self.clock().astimezone(UTC)):
            raise ValueError("feature_input.case_expired_reassess_required")
        try:
            option = gateway.validate_agent_selection(
                case,
                run_id=case.run_id,
                case_token=case_token,
                evidence_hash=evidence_hash,
                option_id=option_id,
                current_evidence_hash=case.evidence_hash,
                rediagnosis_count=case.rediagnosis_count,
            )
        except ValueError as error:
            raise ValueError("feature_input.option_changed") from error
        if option.option_hash != option_hash:
            raise ValueError("feature_input.option_changed")
        if not self._confirmable(option):
            raise ValueError(
                "feature_input.option_requires_data_owner_action:" + option.policy_args.action.value
            )
        return case, option

    def preview(
        self,
        *,
        case_token: str,
        evidence_hash: str,
        option_id: str,
        option_hash: str,
        delegated_by: str | None = None,
    ) -> dict[str, object]:
        """Preview consequences of an exact stored issue choice without applying its effect.

        Args:
            case_token: Exact pending case identity.
            evidence_hash: Exact case evidence identity.
            option_id: Explicit installed choice identifier.
            option_hash: Exact selected option identity.
            delegated_by: The first-use delegation under which the reader confirms it itself.

        Returns:
            Human confirmation preview or case-no-longer-pending refusal; execution revalidates the
            source.
        """
        if self._decided(self.panel.feature_input_resolution(case_token), self.clock()):
            return reread_data_issues("feature_input.case_already_decided")
        try:
            case, option = self._selection(
                case_token=case_token,
                evidence_hash=evidence_hash,
                option_id=option_id,
                option_hash=option_hash,
            )
        except ValueError as error:
            if str(error) not in {
                "feature_input.case_no_longer_pending",
                "feature_input.option_changed",
            }:
                raise
            return reread_data_issues(str(error))
        return {
            "status": "CONFIRMATION_REQUIRED",
            "confirmation": "HUMAN" if delegated_by is None else "FIRST_USE_DELEGATION",
            "case_token": case.case_token,
            "option": option.model_dump(mode="json"),
            "consequences": {
                "action": option.policy_args.action.value,
                "target_listing_ids": list(option.target_listing_ids),
                "raw_values": "PRESERVED",
                "membership_history": "PRESERVED",
                "qualified_input": "SUBJECT_TO_REVALIDATED_EFFECT",
                "effect_applied": False,
            },
            "executors": {
                "direct_confirmation": "HUMAN",
                "delegated_confirmation": "EXTERNAL_AUTOMATION_WITH_EXACT_HUMAN_GRANT"
                if delegated_by is None
                else delegated_by,
            },
            "next_action": "DATA_ISSUE_CONFIRM",
            "next_requests": {"confirm": _choice_request("DATA_ISSUE_CONFIRM", case, option)},
            "effect_applied": False,
            "claim_limit": "STORED_CATALOG_PREVIEW_SOURCE_REVALIDATED_AT_EXECUTION",
        }

    def delegate(
        self,
        *,
        case_token: str,
        evidence_hash: str,
        option_id: str,
        option_hash: str,
        task_id: UUID,
        caller: str,
    ) -> dict[str, object]:
        """Issue an exact human grant for one blocked data-truth preparation task.

        Args:
            case_token: Exact pending case identity.
            evidence_hash: Exact case evidence identity.
            option_id: Explicit installed choice identifier.
            option_hash: Exact selected option identity.
            task_id: Exact preparation task blocked for data truth review.
            caller: Explicit caller required to be HUMAN.

        Returns:
            Retained grant and exact confirmation/preview requests; no effect is applied.

        Raises:
            ValueError: Caller, exact choice or blocked preparation scope is invalid.
        """
        if caller != "HUMAN":
            raise ValueError("feature_input.human_delegation_required")

        def issue() -> dict[str, object]:
            case, option = self._selection(
                case_token=case_token,
                evidence_hash=evidence_hash,
                option_id=option_id,
                option_hash=option_hash,
            )
            task = self.session.task_control_registry.task(task_id)
            if (
                task.task_kind != "workspace_preparation"
                or task.lifecycle.value != "BLOCKED"
                or task.failure_code != "data.truth_review_required"
            ):
                raise ValueError("feature_input.delegation_task_not_blocked_for_data_truth")
            grant = self.delegation.issue(
                case=case,
                option=option,
                task_id=str(task_id),
                plan_hash=task.input.payload["plan"]["plan_hash"],
                now=self.clock(),
            )
            return {
                "status": "DELEGATED",
                "grant": grant,
                "effect_applied": False,
                "next_requests": {
                    "confirm": {
                        **_choice_request("DATA_ISSUE_CONFIRM", case, option),
                        "data_issue_grant_hash": grant["grant_hash"],
                    },
                    "preparation_plan": {"operation": "WORKSPACE_PREPARE_PLAN"},
                },
                "claim_limit": "HUMAN_GRANTED_EXACT_SCOPE_EXECUTOR_STILL_REVALIDATES",
            }

        return self.session.mutation_gate.run(issue)

    def revoke(self, *, grant_hash: str, caller: str) -> dict[str, object]:
        """Revoke one human-issued delegation under the shared mutation gate.

        Args:
            grant_hash: Exact delegation grant identity.
            caller: Explicit caller required to be HUMAN.

        Returns:
            Deterministic revocation result.

        Raises:
            ValueError: Caller is not human or grant is invalid.
        """
        if caller != "HUMAN":
            raise ValueError("feature_input.human_delegation_required")
        return self.session.mutation_gate.run(
            lambda: self.delegation.revoke(grant_hash, now=self.clock())
        )

    def confirm(
        self,
        *,
        case_token: str,
        evidence_hash: str,
        option_id: str,
        option_hash: str,
        caller: str,
        grant_hash: str | None = None,
    ) -> dict[str, object]:
        """Revalidate exact choice and caller scope before recording a remediation receipt.

        Grant revocation and acceptance share the mutation gate. A recorded effect is historical
        evidence rather than current clearance; raw values and membership history remain governed by
        their owners.

        Args:
            case_token: Exact pending case identity.
            evidence_hash: Exact case evidence identity.
            option_id: Explicit installed choice identifier.
            option_hash: Exact selected option identity.
            caller: Human or exactly granted external automation.
            grant_hash: Exact delegation required for external automation, absent for human
                confirmation.

        Returns:
            Exact prior effect readback or confirmed receipt awaiting effect revalidation and
            permitted continuation.

        Raises:
            ValueError: Caller/grant/task scope, selected choice or prior resolution conflicts.
        """
        if caller not in {"HUMAN", "EXTERNAL_AUTOMATION"} or (
            caller == "EXTERNAL_AUTOMATION" and grant_hash is None
        ):
            raise ValueError("feature_input.human_confirmation_required")
        if caller == "HUMAN" and grant_hash is not None:
            raise ValueError("feature_input.human_confirmation_does_not_use_delegation")
        # The first-use goal supplies authority; its agent remains the executor. The receipt
        # carries that authority in the rationale beside the actual session's actor binding.
        provenance = REQUEST_PROVENANCE.get()
        delegated = getattr(provenance, "delegation", None)
        actor_kind = (
            ActorKind.HUMAN
            if caller == "HUMAN" and delegated is None
            else ActorKind.EXTERNAL_AUTOMATION
        )
        actor_id = (
            ":".join(value for value in (provenance.vendor, provenance.session) if value)
            or "external-automation"
            if delegated is not None and provenance is not None
            else "local-web-human"
            if caller == "HUMAN"
            else self.delegation.actor_id(grant_hash or "")
        )

        def accept() -> dict[str, object]:
            grant = (
                self.delegation.read(grant_hash, now=self.clock())
                if grant_hash is not None
                else None
            )
            prior = self.panel.feature_input_resolution(case_token)
            if (
                prior is not None
                and prior.get("effect") is not None
                and not self._wait_elapsed(prior, self.clock())
            ):
                receipt = DataRemediationExecutionReceipt.model_validate(prior["receipt"])
                decision = receipt.policy_decision
                recorded_actor = receipt.actor_submission
                authority = (
                    receipt.submission.proposal.rationale
                    if recorded_actor.actor_kind is ActorKind.EXTERNAL_AUTOMATION
                    else recorded_actor.actor_id
                    if recorded_actor.actor_kind is ActorKind.HUMAN
                    else None
                )
                matches = (
                    decision.case_token,
                    decision.evidence_hash,
                    decision.option_id,
                    decision.option_hash,
                ) == (
                    case_token,
                    evidence_hash,
                    option_id,
                    option_hash,
                ) and (
                    authority == delegated
                    if delegated is not None
                    else recorded_actor.actor_kind is actor_kind
                    and recorded_actor.actor_id == actor_id
                )
                failed = bool(prior["effect"].get("failure_reasons"))
                if not matches and not failed:
                    raise ValueError("feature_input.resolution_conflict")
                if matches:
                    return {
                        "status": "REFUSED" if failed else "ALREADY_APPLIED",
                        "failure_code": "feature_input.option_preflight_refused"
                        if failed
                        else None,
                        "receipt_hash": receipt.receipt_hash,
                        "case_token": case_token,
                        "effect": prior["effect"],
                        "next_action": "DATA_ISSUES",
                        "next_requests": {"issues": {"operation": "DATA_ISSUES"}},
                        "claim_limit": "HISTORICAL_EFFECT_NOT_CURRENT_CLEARANCE",
                    }
            case, option = self._selection(
                case_token=case_token,
                evidence_hash=evidence_hash,
                option_id=option_id,
                option_hash=option_hash,
            )
            if caller == "EXTERNAL_AUTOMATION":
                assert grant is not None
                if not self.delegation.matches_choice(grant, case, option):
                    raise ValueError("feature_input.delegation_scope_mismatch")
                try:
                    preparation = self.session.task_control_registry.task(
                        UUID(grant["preparation_task_id"])
                    )
                except (KeyError, ValueError) as error:
                    raise ValueError("feature_input.delegation_task_not_found") from error
                if (
                    preparation.task_kind != "workspace_preparation"
                    or not self.delegation.matches_resume(
                        grant,
                        task_id=str(preparation.task_id),
                        plan_hash=preparation.input.payload["plan"]["plan_hash"],
                        lifecycle=preparation.lifecycle.value,
                        failure_code=preparation.failure_code,
                    )
                ):
                    raise ValueError("feature_input.delegation_task_scope_changed")
            proposal = DataRemediationProposal(
                run_id=case.run_id,
                case_token=case_token,
                evidence_hash=evidence_hash,
                option_id=option_id,
                rationale=(
                    delegated
                    if delegated is not None
                    else "Human confirmed the displayed immutable data option."
                    if caller == "HUMAN"
                    else self.delegation.rationale(grant_hash or "")
                ),
            )
            payload = {
                "kind": "DataRemediationExecutionSubmission",
                "proposal": proposal.model_dump(mode="json"),
            }
            submission = DataRemediationExecutionSubmission(
                **payload, submission_hash=canonical_hash(payload)
            )
            receipt = seal_validated_data_remediation_execution(
                submission=submission,
                policy_decision=PolicyDecision(
                    case_token=case_token,
                    evidence_hash=evidence_hash,
                    option_id=option_id,
                    option_hash=option_hash,
                    policy_hash=canonical_hash(option.policy_args),
                    disposition=option.disposition,
                ),
                actor_kind=actor_kind,
                actor_id=actor_id,
                execution_policy_hash=case.policy_hash,
            )
            self.panel.record_feature_input_resolution(case_token, receipt.model_dump(mode="json"))
            followup = self.readback()
            requests = cast(dict[str, dict[str, str]], followup.get("next_requests", {}))
            continued = any(key.startswith("continue:") for key in requests)
            return {
                "status": "CONFIRMED_PENDING_REVALIDATION",
                "receipt_hash": receipt.receipt_hash,
                "case_token": case_token,
                "next_action": "RESUME_APPROVED_WORK",
                "decisions_complete": followup["decisions_complete"],
                "continuations": followup["continuations"],
                "next_requests": {
                    "issues": {"operation": "DATA_ISSUES"},
                    **{
                        key: (
                            {**value, "data_issue_grant_hash": grant_hash}
                            if grant_hash is not None
                            and value["operation"] == "WORKSPACE_PREPARE_CONFIRM"
                            else value
                        )
                        for key, value in requests.items()
                        if key.startswith("continue:")
                    },
                    **(
                        {}
                        if continued
                        else {"preparation_plan": {"operation": "WORKSPACE_PREPARE_PLAN"}}
                    ),
                },
            }

        return self.session.mutation_gate.run(accept)
