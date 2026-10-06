"""Host-sealed Data choices over existing pre-Factor and maintenance case owners.

Selections remain separate from deterministic execution; no raw-data mutation
or model-derived data-truth authority is granted by a confirmation.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, cast
from uuid import UUID

from alphalattice.control.data_platform.contracts import (
    DataRemediationExecutionReceipt,
    DataRemediationExecutionSubmission,
)
from alphalattice.control.data_platform.delegation import DataIssueDelegation
from alphalattice.control.guanyin.data.workspace_maintenance import maintenance_failure_detail
from alphalattice.control.product_host.composition.plain_refusals import task_record_refusal
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
from alphalattice.protocols.actor_execution import (
    ActorKind,
)

if TYPE_CHECKING:
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
        and (
            task.lifecycle is not TaskLifecycle.CANCELLED
            or any(
                item.lifecycle is WorkItemLifecycle.VERIFIED
                for item in registry.work_items(task.task_id)
            )
        )
    )


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


class WorkspaceDataIssueApplication:
    """Human choice over the existing persisted Feature Input catalog.

    Confirmation seals intent only. The maintenance owner re-evaluates source
    evidence before applying it; the response never claims an unexecuted effect.
    """

    def __init__(self, session: WorkspaceApplicationSession, clock: Callable[[], datetime]) -> None:
        """Wire retained mutation authority, Data/Panel readers and exact issue delegation.

        Args:
            session: Retained workspace writer/task session.
            clock: Explicit observed-time source.
        """
        self.session = session
        self.clock = clock
        self.market = MarketDataRepository(session.workspace)
        self.panel = PanelStateRepository(self.market.database, market_data=self.market)
        self.delegation = DataIssueDelegation(session.workspace)

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

    def _case_index(self, manifest: UniverseManifest | None) -> dict[str, object]:
        """Every current case named in a line, beside the page of whole ones (V236, V172).

        A whole page of cases is longer than an agent's tool return (AX1d cut it five times),
        so the answer names every case and its count first; ``--section case_index`` reads just
        that, and ``next_page`` the next whole cases.
        """
        if manifest is None:
            return {"case_count": 0, "case_index": []}
        every = self._cases(manifest_revision=manifest.revision_sha256)
        return {
            "case_count": len(every),
            "case_index": [
                {
                    "case_token": case.case_token,
                    "failure_code": case.failure_code,
                    "listing_count": len(case.listing_ids),
                }
                for case in every
            ],
        }

    def current_decisions_ready(self) -> bool:
        """All active cases have a standing choice for preparation to revalidate."""
        manifest = self._active_manifest()
        if manifest is None:
            return False
        now = self.clock()
        with self.market.database.retain(read_only=True):
            cases = self._cases(manifest_revision=manifest.revision_sha256)
            if not cases:
                return False
            proofs = raw_retention_decision_proofs(self.panel, self.market)
            for case in cases:
                resolution = self.panel.feature_input_resolution(case.case_token)
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

    def _governance(self) -> FeatureInputGovernanceService:
        """The same judgement owner the maintenance cycle uses, over the same stores."""
        return FeatureInputGovernanceService(
            market_data=self.market,
            panel_state=self.panel,
            mutation_gate=self.session.mutation_gate,
            gateway=FeatureInputGateway(),
        )

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

    def readback(self, *, limit: int = 25, cursor: str | None = None) -> dict[str, object]:
        """Read bounded current issue choices, recorded decisions and exact continuation grants.

        Args:
            limit: Explicit page size from 1 through 50.
            cursor: Optional exact case-token cursor.

        Returns:
            Cases, current choices, historical effects, active grants and permitted next requests;
            no data-truth/current-readiness claim.

        Raises:
            ValueError: Case page bound or cursor is invalid.
        """
        if not 1 <= limit <= 50 or (
            cursor is not None
            and (len(cursor) != 64 or any(c not in "0123456789abcdef" for c in cursor))
        ):
            raise ValueError("feature_input.case_page_invalid")
        active_manifest = self._active_manifest()
        selected = (
            self._cases(
                limit=limit + 1,
                cursor=cursor,
                manifest_revision=active_manifest.revision_sha256,
            )
            if active_manifest is not None
            else ()
        )
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
        continuations = []
        next_requests: dict[str, dict[str, str]] = {}
        for task in tasks:
            plan = task.input.payload["plan"]
            if (
                task.task_kind == "workspace_preparation"
                and task.lifecycle.value == "BLOCKED"
                and task.failure_code == "data.truth_review_required"
            ):
                operation, endpoint, payload = (
                    "WORKSPACE_PREPARE_PLAN",
                    "/api/workspace/preparation/plan",
                    {},
                )
            elif task.task_kind == "workspace_preparation":
                operation, endpoint, payload = (
                    "WORKSPACE_PREPARE_CONFIRM",
                    "/api/workspace/preparation/confirm",
                    {"preparation_plan_hash": plan["plan_hash"]},
                )
            else:
                operation, endpoint, payload = (
                    "DATA_UPDATE_RUN",
                    "/api/data-update/run",
                    {"update_plan_hash": plan["content_hash"]},
                )
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
        retention_proofs = raw_retention_decision_proofs(self.panel, self.market)
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
            resolution = self.panel.feature_input_resolution(case.case_token)
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
                and (resolution is None or refused or elapsed)
            ):
                for option in case.options:
                    if self._confirmable(option):
                        next_requests[f"preview:{case.case_token}:{option.option_id}"] = (
                            _choice_request("DATA_ISSUE_PREVIEW", case, option)
                        )
                        for task in tasks:
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
            **self._case_index(active_manifest),
            "continuations": continuations,
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
        option = gateway.validate_agent_selection(
            case,
            run_id=case.run_id,
            case_token=case_token,
            evidence_hash=evidence_hash,
            option_id=option_id,
            current_evidence_hash=case.evidence_hash,
            rediagnosis_count=case.rediagnosis_count,
        )
        if option.option_hash != option_hash:
            raise ValueError("feature_input.option_changed")
        if not self._confirmable(option):
            raise ValueError(
                "feature_input.option_requires_data_owner_action:" + option.policy_args.action.value
            )
        return case, option

    def preview(
        self, *, case_token: str, evidence_hash: str, option_id: str, option_hash: str
    ) -> dict[str, object]:
        """Preview consequences of an exact stored issue choice without applying its effect.

        Args:
            case_token: Exact pending case identity.
            evidence_hash: Exact case evidence identity.
            option_id: Explicit installed choice identifier.
            option_hash: Exact selected option identity.

        Returns:
            Human confirmation preview or case-no-longer-pending refusal; execution revalidates the
            source.
        """
        try:
            case, option = self._selection(
                case_token=case_token,
                evidence_hash=evidence_hash,
                option_id=option_id,
                option_hash=option_hash,
            )
        except ValueError as error:
            if str(error) != "feature_input.case_no_longer_pending":
                raise
            return {
                "status": "REFUSED",
                "failure_code": "feature_input.case_no_longer_pending",
                "next_action": "DATA_ISSUES",
                "next_requests": {"issues": {"operation": "DATA_ISSUES"}},
                "effect_applied": False,
            }
        return {
            "status": "CONFIRMATION_REQUIRED",
            "confirmation": "HUMAN",
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
                "delegated_confirmation": "EXTERNAL_AUTOMATION_WITH_EXACT_HUMAN_GRANT",
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
        actor_kind = ActorKind.HUMAN if caller == "HUMAN" else ActorKind.EXTERNAL_AUTOMATION
        # A person's decision their first-use goal delegated names that goal (V452).
        delegated = getattr(REQUEST_PROVENANCE.get(), "delegation", None)
        actor_id = (
            (delegated or "local-web-human")
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
                    receipt.actor_submission.actor_kind is actor_kind
                    and receipt.actor_submission.actor_id == actor_id
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
                    "Human confirmed the displayed immutable data option."
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
            return {
                "status": "CONFIRMED_PENDING_REVALIDATION",
                "receipt_hash": receipt.receipt_hash,
                "case_token": case_token,
                "next_action": "RESUME_APPROVED_WORK",
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
                    "preparation_plan": {"operation": "WORKSPACE_PREPARE_PLAN"},
                },
            }

        return self.session.mutation_gate.run(accept)
