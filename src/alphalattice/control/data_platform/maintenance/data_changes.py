"""Bound data-change proposals and retained valuation over existing Data owners.

Moved from the Product Host (O3): the proposal a data change makes, the prior members a
formation still owes, the valuation grants a book's held listings take and the evidence a
held valuation must show are the data platform's decisions; the Host's data update
application composes them with the session and the Task.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

from alphalattice.control.data_platform.maintenance.contracts import (
    RemovedMemberTail,
    WorkspaceDataChange,
    WorkspaceDataChangeExecution,
    WorkspaceDataUpdatePlan,
    WorkspaceInputStatus,
    WorkspaceMaintenanceRequest,
    WorkspaceValuationGrant,
    WorkspaceValuationReceipt,
    full_history_audit_requirement,
)
from alphalattice.control.data_platform.maintenance.registry import (
    DuckDbWorkspaceMaintenanceRegistry,
)
from alphalattice.control.data_platform.preflight import (
    resolve_trading_session_authority,
)
from alphalattice.control.data_platform.readiness import WorkspaceReadinessGate
from alphalattice.control.workspace_runtime.content_store import (
    CommittedIndex,
    CommittedKind,
    ContentAddressedStore,
)
from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate
from alphalattice.foundation.market_data_ops.runtime.universe_maintenance import (
    CurrentUniverseMaintenance,
    CurrentUniverseMaintenanceStatus,
    current_universe_maintenance_id,
)
from alphalattice.foundation.market_data_ops.sources.manifest import (
    UniverseManifest,
    build_quality_filtered_research_manifest,
)
from alphalattice.foundation.market_data_ops.sources.providers import MarketDataProvider
from alphalattice.foundation.market_data_ops.storage.duckdb import (
    CurrentUniverseMaintenanceListing,
    MarketDataRepository,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

type BookObligations = dict[str, tuple[str, tuple[str, ...]]]


_CHANGE_EXECUTIONS = CommittedKind(
    "data-change-executions", "data-change-executions", WorkspaceDataChangeExecution, "content_hash"
)
_VALUATION_RECEIPTS = CommittedKind(
    "data-valuation-receipts", "data-valuation-receipts", WorkspaceValuationReceipt, "content_hash"
)


def _rejection_hash(document: dict[str, object] | None) -> str:
    """The receipt of a listing's rejection: its failure cause and rejected history."""
    return str(
        canonical_hash(
            {
                k: v
                for k, v in (document or {}).items()
                if k in {"failure_cause", "rejected_history"}
            }
        )
    )


class WorkspaceDataChanges:
    """Metadata only until the Data Task executes a Human-confirmed request."""

    def __init__(self, root: Path, mutation_gate: WorkspaceMutationGate):
        """Resolve immutable data-change content and its committed execution index.

        Args:
            root: Workspace root for immutable data-change artifacts.
            mutation_gate: Workspace mutation owner used to serialize durable changes.
        """
        self.root = root
        self.gate = mutation_gate
        self.content = ContentAddressedStore(
            root / "runtime/artifacts", uri_prefix="playpen://workspace-data"
        )
        self.index = CommittedIndex(self.content.root, self.content)

    def save_plan(self, plan: WorkspaceDataUpdatePlan) -> None:
        """Publish the immutable plan under its canonical content identity.

        Args:
            plan: Immutable approved Data update plan.
        """
        self.content.publish_model(
            category="data-update-plans", value=plan, identity_field="content_hash"
        )

    def load_plan(self, identity: str) -> WorkspaceDataUpdatePlan:
        """Load and validate a content-addressed Data update plan.

        Args:
            identity: Canonical identity of a stored Data update plan.

        Returns:
            The stored plan validated against the requested content identity.
        """
        return self.content.load_model(
            category="data-update-plans",
            content_hash=identity,
            model=WorkspaceDataUpdatePlan,
            identity_field="content_hash",
        )

    def execution(self, plan: WorkspaceDataUpdatePlan) -> WorkspaceDataChangeExecution | None:
        """Open the committed execution and verify its plan/request binding.

        Args:
            plan: Immutable approved Data update plan.

        Returns:
            The bound execution, or None when no execution is committed.

        Raises:
            ValueError: Stored execution identity or reconstructed request binding differs.
        """
        value = self.index.open(_CHANGE_EXECUTIONS, plan.content_hash)
        if value is not None and (
            value.plan_hash != plan.content_hash
            or value.request != self._request(plan, value.request.membership_revision)
        ):
            raise ValueError("workspace_data_update.execution_binding_invalid")
        return value

    def _request(
        self, plan: WorkspaceDataUpdatePlan, manifest_revision: str
    ) -> WorkspaceMaintenanceRequest:
        active = MarketDataRepository(self.root).load_universe_manifest_revision(manifest_revision)
        ids = {v.listing_id for v in active.listings}
        return plan.request.with_changes(
            membership_revision=manifest_revision,
            full_history_listing_ids=tuple(
                v for v in plan.request.full_history_listing_ids if v in ids
            ),
        )

    def capture_execution(
        self, plan: WorkspaceDataUpdatePlan, manifest_revision: str
    ) -> WorkspaceDataChangeExecution:
        """Reuse a bound execution or capture the exact activated request once.

        Args:
            plan: Immutable approved Data update plan.
            manifest_revision: Activated membership revision used by execution.

        Returns:
            The immutable plan-bound execution record.
        """
        existing = self.execution(plan)
        if existing is not None:
            return existing
        request = self._request(plan, manifest_revision)
        value = WorkspaceDataChangeExecution.seal(plan_hash=plan.content_hash, request=request)
        self.index.commit(_CHANGE_EXECUTIONS, plan.content_hash, value)
        return value

    def proposal(
        self,
        *,
        state: WorkspaceInputStatus,
        readiness: WorkspaceReadinessGate,
        observed_at: datetime,
        obligations: BookObligations,
        valuation_grants: tuple[WorkspaceValuationGrant, ...] = (),
        target_session: date | None = None,
    ) -> WorkspaceDataChange | None:
        """Derive a universe change or verified full-history audit proposal from owner evidence.

        Args:
            state: Current workspace input status against which a change is proposed.
            readiness: Owner that assesses current membership/source readiness.
            observed_at: Operational observation clock; callers supply an aware instant.
            obligations: Book-head identities and listing obligations retained by each book root.
            valuation_grants: Existing held-listing valuation grants to preserve.
            target_session: Market session targeted by this operation.

        Returns:
            A sealed change proposal, or None when no change is required.

        Raises:
            ValueError: Recorded full-history requirements cannot be verified.
        """
        market = readiness.market_data
        decision = readiness.assess(observed_at=observed_at)
        manifest = market.load_universe_manifest_revision(state.manifest_revision)
        heads = tuple((root, value[0]) for root, value in sorted(obligations.items()))
        if decision.proposal is not None:
            proposal = decision.proposal
            record = market.readiness.load(manifest.profile.market_profile_id)
            assert record is not None
            held = {v for _, ids in obligations.values() for v in ids}
            retained = tuple(
                v.listing_id
                for v in manifest.listings
                if v.symbol in proposal.removals and v.listing_id in held
            )
            return WorkspaceDataChange.seal(
                action="UNIVERSE",
                transition_id=proposal.transition_id,
                candidate_document_hash=canonical_hash(record.pending_candidate_manifest_document),
                additions=proposal.additions,
                removals=proposal.removals,
                book_heads=heads,
                valuation_manifest=build_quality_filtered_research_manifest(
                    manifest, eligible_listing_ids=retained
                )
                if retained
                else None,
            )
        registry = DuckDbWorkspaceMaintenanceRegistry(
            market.database.path, gate=self.gate, initialize=False
        )
        cycle = registry.latest_cycle(manifest.profile.market_profile_id)
        required: set[str] = set()
        if cycle is not None and cycle.failure_code == "data.full_history_audit_approval_required":
            # The run this cycle keyed its listings under: a membership change's working child
            # or a granted audit gives it an id its sealed request does not.
            scope = registry.market_data_scope(cycle.cycle_id)
            maintenance_id = (
                scope["maintenance_id"]
                if scope is not None
                else current_universe_maintenance_id(
                    market.load_universe_manifest_revision(cycle.request.membership_revision),
                    as_of_session=cycle.request.target_market_session,
                    authorized_full_history_listing_ids=cycle.request.full_history_listing_ids,
                )
            )
            required.update(
                v.listing_id
                for v in market.current_universe_maintenance_listings(maintenance_id)
                if v.failure_code == "data.full_history_audit_approval_required"
            )
            evidence = full_history_audit_requirement(cycle.request, tuple(required))
            if not required or evidence not in cycle.effect_receipts:
                raise ValueError("workspace_data_update.audit_requirement_unverified")
        if target_session is not None:
            scopes = [
                build_quality_filtered_research_manifest(
                    g.manifest, eligible_listing_ids=g.listing_ids
                )
                for g in valuation_grants
            ]
            formation = self.formation_scope(state)
            if formation is not None:
                scopes.append(formation)
            for scope in scopes:
                identity = current_universe_maintenance_id(scope, as_of_session=target_session)
                required.update(
                    v.listing_id
                    for v in market.current_universe_maintenance_listings(identity)
                    if v.failure_code == "data.full_history_audit_approval_required"
                )
        if not required:
            return None
        return WorkspaceDataChange.seal(
            action="FULL_HISTORY_AUDIT",
            full_history_listing_ids=tuple(sorted(required)),
            book_heads=heads,
        )

    @staticmethod
    def valuation_scopes(plan: WorkspaceDataUpdatePlan) -> tuple[UniverseManifest, ...]:
        """Resolve distinct retained valuation manifests from an approved plan.

        Args:
            plan: Immutable approved Data update plan.

        Returns:
            Valuation grant/change manifests deduplicated by immutable manifest revision.
        """
        scopes = [
            build_quality_filtered_research_manifest(g.manifest, eligible_listing_ids=g.listing_ids)
            for g in plan.valuation_grants
        ]
        if plan.change is not None and plan.change.valuation_manifest is not None:
            scopes.append(plan.change.valuation_manifest)
        return tuple({m.revision_sha256: m for m in scopes}.values())

    def formation_scope(
        self, state: WorkspaceInputStatus, change: WorkspaceDataChange | None = None
    ) -> UniverseManifest | None:
        """Prior Panel members still owed catch-up, not renewed membership.

        A source change can activate before its EXIT is effective. Its final
        formation observations must reach Feature before the new Panel consumes
        the transition. Once that Panel is published this extra scope disappears;
        continuing held-position quotes remain the separate valuation contract.
        """
        removed = (
            set(change.removals) if change is not None and change.action == "UNIVERSE" else set()
        )
        prior_revision = state.panel_manifest_revision or state.manifest_revision
        if prior_revision == state.manifest_revision and not removed:
            return None
        market = MarketDataRepository(self.root)
        prior = market.load_universe_manifest_revision(prior_revision)
        active = market.load_universe_manifest_revision(state.manifest_revision)
        active_ids = {v.listing_id for v in active.listings if v.symbol not in removed}
        retained = tuple(v.listing_id for v in prior.listings if v.listing_id not in active_ids)
        return (
            build_quality_filtered_research_manifest(prior, eligible_listing_ids=retained)
            if retained
            else None
        )

    def maintain_formation(
        self,
        *,
        plan: WorkspaceDataUpdatePlan,
        provider: MarketDataProvider,
        observed_at: datetime,
        cancelled: Callable[[], bool],
    ) -> tuple[str, dict[str, object] | None] | None:
        """Acquire the bounded pre-exit source tail before Feature work starts."""
        scope = self.formation_scope(plan.before, plan.change)
        if scope is None:
            return None
        runner = self._quote_runner(plan, scope, provider)
        while True:
            if cancelled():
                return "workspace_data_update.formation_cancelled", None
            outcome = runner.run(observed_at=observed_at, work_budget=1)
            if outcome.status is not CurrentUniverseMaintenanceStatus.RUNNING:
                break
        if outcome.status is CurrentUniverseMaintenanceStatus.DEFERRED:
            return outcome.failure_code or "workspace_data_update.formation_source_deferred", None
        failures = tuple(
            row
            for row in runner.store.current_universe_maintenance_listings(runner.maintenance_id)
            if row.state == "FAILED"
        )
        # A confirmed empty response leaves the old bars untouched. The installed
        # partial-source Feature policy marks missing sessions unavailable and
        # applies its unchanged coverage floors; it is not a renewed membership
        # or a zero return. A genuinely invalid removed-member source can retain
        # its exactly verified prefix, with the unresolved tail recorded separately.
        # Source identity, action integrity and permission failures still block.
        retained: set[str] = set()
        for row in failures:
            tail = self._removed_member_tail(plan, runner, row, observed_at=observed_at)
            if tail is not None:
                self.gate.run(
                    runner.store.update_current_universe_maintenance_listing,
                    maintenance_id=row.maintenance_id,
                    listing_id=row.listing_id,
                    state=row.state,
                    failure_code=row.failure_code,
                    change_document={
                        **(row.change_document or {}),
                        "removed_member_tail": tail.model_dump(mode="json"),
                    },
                    observed_at=observed_at,
                )
                retained.add(row.listing_id)
        failed = next(
            (
                row
                for row in failures
                if row.failure_code == "data.full_history_audit_approval_required"
            ),
            next(
                (
                    row
                    for row in failures
                    if row.failure_code != "data.empty_payload" and row.listing_id not in retained
                ),
                None,
            ),
        )
        if failed is None:
            return None
        facts = (failed.change_document or {}).get("failure_cause")
        if not isinstance(facts, dict):
            facts = {
                "exception_type": "UNKNOWN",
                "detail": "The source failure cause was not recorded.",
                "step": "Provider price history",
                "unit": next(v.symbol for v in scope.listings if v.listing_id == failed.listing_id),
                "row_count": "UNKNOWN",
                "sanitizer_code": "UNKNOWN",
            }
        return failed.failure_code or "workspace_data_update.formation_source_incomplete", facts

    def _removed_member_tail(
        self,
        plan: WorkspaceDataUpdatePlan,
        runner: CurrentUniverseMaintenance,
        row: CurrentUniverseMaintenanceListing,
        *,
        observed_at: datetime,
    ) -> RemovedMemberTail | None:
        facts = (row.change_document or {}).get("failure_cause")
        if (
            not isinstance(facts, dict)
            or runner.trading_session_authority is None
            # A source that stopped short admitted nothing; its tail is disclosed alike.
            or (
                row.failure_code == "data.maintenance_stale_payload"
                and facts.get("sanitizer_code") != "STALE_PAYLOAD"
            )
            or (
                row.failure_code != "data.maintenance_stale_payload"
                and (
                    row.failure_code != "data.sanitizer.corrupted_payload"
                    or facts.get("exception_type") != "CorruptedPayload"
                    or facts.get("sanitizer_code")
                    not in {"INVALID_OHLC", "INVALID_VOLUME", "DUPLICATE_SESSION"}
                    or type(facts.get("row_count")) is not int
                    or facts["row_count"] <= 0
                )
            )
        ):
            return None
        market = runner.store
        active = market.load_universe_manifest_revision(plan.before.manifest_revision)
        removed = plan.change.removals if plan.change is not None else ()
        if row.listing_id in {v.listing_id for v in active.listings if v.symbol not in removed}:
            return None
        bars = market.raw_bars(row.listing_id, through=plan.request.target_market_session)
        if not bars:
            return None
        anchor = market.latest_action_audit_receipts(
            (row.listing_id,), provider=runner.provider.name
        ).get(row.listing_id)
        if anchor is None:
            return None
        witnesses = [anchor]
        if anchor.history_end < bars[-1].session_date:
            link = market.latest_action_audit_receipts(
                (row.listing_id,),
                provider=runner.provider.name,
                requested_as_of=bars[-1].session_date,
            ).get(row.listing_id)
            if link is None or link.mapping_revision != anchor.mapping_revision:
                return None
            witnesses.append(link)
        # The audit owner verifies the same recorded bytes, policy and mappings.
        # Historical truth does not acquire a new Provider receipt or observation.
        if any(
            market.verified_action_audit_receipt(
                runner.manifest,
                receipt_hash=receipt.receipt_hash,
                listing_id=row.listing_id,
                provider=runner.provider.name,
                requested_as_of=receipt.requested_as_of,
                now=observed_at,
                allow_historical=True,
            )
            is None
            for receipt in witnesses
        ) or any(
            not any(
                receipt.history_start <= bar.session_date <= receipt.history_end
                for receipt in witnesses
            )
            for bar in bars
        ):
            return None
        authority = runner.trading_session_authority
        membership = market.membership_schedule(
            active.profile.market_profile_id,
            sessions=authority.sessions,
            fallback_listing_ids=tuple(v.listing_id for v in active.listings),
        )
        missing = tuple(
            session
            for session in authority.sessions
            if bars[-1].session_date < session <= plan.request.target_market_session
            and row.listing_id in membership.members(session)
        )
        return RemovedMemberTail.seal(
            listing_id=row.listing_id,
            symbol=row.symbol,
            maintenance_id=row.maintenance_id,
            formation_manifest_revision=runner.manifest.revision_sha256,
            last_verified_session=bars[-1].session_date,
            target_session=plan.request.target_market_session,
            missing_sessions=missing,
            missing_session_count=len(missing),
            session_authority_hash=authority.authority_hash,
            retained_audit_receipt_hashes=tuple(receipt.receipt_hash for receipt in witnesses),
            rejection_receipt_hash=_rejection_hash(row.change_document),
            sanitizer_code=facts["sanitizer_code"],
        )

    def removed_member_tails(self, plan: WorkspaceDataUpdatePlan) -> tuple[RemovedMemberTail, ...]:
        """Read saved disclosures bound to this plan's original formation units, without work.

        The saved source scope, listing and target must agree with the original
        run; a typed disclosure from another unit never becomes this plan's fact.
        """
        scope = self.formation_scope(plan.before, plan.change)
        if scope is None:
            return ()
        authorized = tuple(
            sorted(
                set(plan.request.full_history_listing_ids) & {v.listing_id for v in scope.listings}
            )
        )
        maintenance_id = current_universe_maintenance_id(
            scope,
            as_of_session=plan.request.target_market_session,
            authorized_full_history_listing_ids=authorized,
        )
        rows = MarketDataRepository(self.root).current_universe_maintenance_listings(maintenance_id)
        tails = []
        for row in rows:
            if not row.change_document or "removed_member_tail" not in row.change_document:
                continue
            try:
                tail = RemovedMemberTail.model_validate(row.change_document["removed_member_tail"])
                if (
                    row.state != "FAILED"
                    or row.failure_code
                    not in {"data.sanitizer.corrupted_payload", "data.maintenance_stale_payload"}
                    or tail.sanitizer_code
                    != (row.change_document.get("failure_cause") or {}).get("sanitizer_code")
                    or tail.maintenance_id != maintenance_id
                    or tail.formation_manifest_revision != scope.revision_sha256
                    or tail.listing_id != row.listing_id
                    or tail.symbol != row.symbol
                    or tail.target_session != plan.request.target_market_session
                    or tail.rejection_receipt_hash != _rejection_hash(row.change_document)
                ):
                    raise ValueError("removed member tail names another formation unit")
            except ValueError as exc:
                raise ValueError("workspace_data_update.identity_invalid") from exc
            tails.append(tail)
        return tuple(tails)

    def maintain_valuation(
        self,
        *,
        plan: WorkspaceDataUpdatePlan,
        provider: MarketDataProvider,
        observed_at: datetime,
        cancelled: Callable[[], bool],
    ) -> tuple[str, ...]:
        """Maintain each held-listing scope and commit its verified completion receipt.

        Args:
            plan: Immutable approved Data update plan.
            provider: Bound market data provider for acquisition or receipt verification.
            observed_at: Operational observation clock; callers supply an aware instant.
            cancelled: Checkpoint callback reporting cancellation.

        Returns:
            Canonical identities of committed valuation receipts.

        Raises:
            ValueError: Cancellation, required full-history approval or unavailable valuation
                prevents completion.
        """
        market = MarketDataRepository(self.root)
        receipts = []
        for manifest in self.valuation_scopes(plan):
            runner = self._quote_runner(plan, manifest, provider)
            while True:
                if cancelled():
                    raise ValueError("workspace_data_update.valuation_cancelled")
                outcome = runner.run(observed_at=observed_at, work_budget=1)
                if outcome.status is not CurrentUniverseMaintenanceStatus.RUNNING:
                    break
            if outcome.status is not CurrentUniverseMaintenanceStatus.COMPLETED or outcome.failed:
                rows = market.current_universe_maintenance_listings(runner.maintenance_id)
                raise ValueError(
                    "data.full_history_audit_approval_required"
                    if any(
                        v.failure_code == "data.full_history_audit_approval_required" for v in rows
                    )
                    else "workspace_data_update.held_valuation_unavailable"
                )
            receipt = WorkspaceValuationReceipt.seal(
                plan_hash=plan.content_hash,
                manifest=manifest,
                maintenance_id=runner.maintenance_id,
                target_session=plan.request.target_market_session,
                outcome_hash=canonical_hash(asdict(outcome)),
            )
            self.index.commit(
                _VALUATION_RECEIPTS,
                canonical_hash([plan.content_hash, manifest.revision_sha256]),
                receipt,
            )
            receipts.append(receipt.content_hash)
        return tuple(receipts)

    def verified_valuation(
        self, plan: WorkspaceDataUpdatePlan, provider: MarketDataProvider
    ) -> tuple[str, ...]:
        """Verify committed valuation lineage, updated listings and action-audit evidence.

        Args:
            plan: Immutable approved Data update plan.
            provider: Bound market data provider for acquisition or receipt verification.

        Returns:
            Verified valuation receipt identities.

        Raises:
            ValueError: A receipt, scope, outcome, listing state or action-audit proof is invalid.
        """
        receipts = []
        market = MarketDataRepository(self.root)
        for manifest in self.valuation_scopes(plan):
            value = self.index.open(
                _VALUATION_RECEIPTS, canonical_hash([plan.content_hash, manifest.revision_sha256])
            )
            runner = self._quote_runner(plan, manifest, provider)
            rows = market.current_universe_maintenance_listings(runner.maintenance_id)
            if (
                value is None
                or value.plan_hash != plan.content_hash
                or value.manifest != manifest
                or value.maintenance_id != runner.maintenance_id
                or value.target_session != plan.request.target_market_session
                or not rows
                or {v.listing_id for v in rows} != {v.listing_id for v in manifest.listings}
                or any(v.state != "UPDATED" for v in rows)
            ):
                raise ValueError("workspace_data_update.valuation_evidence_invalid")
            if value.outcome_hash != canonical_hash(
                asdict(runner._outcome(CurrentUniverseMaintenanceStatus.COMPLETED))
            ):
                raise ValueError("workspace_data_update.valuation_evidence_invalid")
            if any(
                market.verified_action_audit_receipt(
                    manifest,
                    receipt_hash=str((v.change_document or {}).get("provider_receipt_hash", "")),
                    listing_id=v.listing_id,
                    provider=provider.name,
                    requested_as_of=plan.request.target_market_session,
                    now=plan.request.request_clock,
                )
                is None
                for v in rows
            ):
                raise ValueError("workspace_data_update.valuation_evidence_invalid")
            receipts.append(value.content_hash)
        return tuple(receipts)

    def _quote_runner(
        self,
        plan: WorkspaceDataUpdatePlan,
        manifest: UniverseManifest,
        provider: MarketDataProvider,
    ) -> CurrentUniverseMaintenance:
        approved = frozenset(plan.request.full_history_listing_ids) & {
            v.listing_id for v in manifest.listings
        }
        market = MarketDataRepository(self.root)
        covered = market.manifest_raw_range(manifest)
        if covered is None:
            # Formation subsets are not registered until maintenance starts.
            # Their retained listing bytes still bound the whole calendar axis.
            ranges = market.listing_raw_ranges(
                tuple(v.listing_id for v in manifest.listings),
                through=plan.request.target_market_session,
            )
            if ranges:
                covered = min(v[0] for v in ranges.values()), max(v[1] for v in ranges.values())
        return CurrentUniverseMaintenance(
            store=market,
            manifest=manifest,
            provider=provider,
            as_of_session=plan.request.target_market_session,
            max_workers=2,
            mutation_gate=self.gate,
            full_audit_listing_ids=approved,
            full_history_escalation_listing_ids=approved,
            trading_session_authority=resolve_trading_session_authority(
                start=covered[0] if covered else plan.request.target_market_session,
                end=plan.request.target_market_session,
                as_of_timestamp=plan.request.request_clock,
            ),
        )

    @staticmethod
    def active_grants(
        approved: tuple[WorkspaceDataUpdatePlan, ...],
        obligations: BookObligations,
    ) -> tuple[WorkspaceValuationGrant, ...]:
        """Derive disjoint required held-listing grants from the newest approved plans first.

        Args:
            approved: Ordered approved plans from which active valuation grants are derived.
            obligations: Book-head identities and listing obligations retained by each book root.

        Returns:
            Valuation grants covering still-obligated listings without duplicate assignment.
        """
        grants = []
        assigned: set[str] = set()
        for plan in reversed(approved):
            change = plan.change
            if change is None or change.valuation_manifest is None:
                continue
            roots = tuple(r for r, _ in change.book_heads if r in obligations)
            needed = {v for r in roots for v in obligations[r][1]}
            ids = tuple(
                sorted(
                    {v.listing_id for v in change.valuation_manifest.listings} & needed - assigned
                )
            )
            if ids:
                grants.append(
                    WorkspaceValuationGrant.seal(
                        approved_plan_hash=plan.content_hash,
                        manifest=change.valuation_manifest,
                        book_roots=roots,
                        listing_ids=ids,
                    )
                )
                assigned.update(ids)
        return tuple(grants)
