"""Composition for protected finalization: freeze, inject, continue, release.

Product Host does four things here and no fifth. It freezes a candidate from an
already-published development result; it constructs the Validation Gate and hands
it across Portfolio's port; it wires Portfolio's own protected continuation to
the executor the development path used; and it performs the atomic release that
binds one package to one receipt.

What it deliberately does not do: compute anything, decide a metric, own a second
task registry, or expose a protected report before a closed receipt exists. The
release is the only place a protected report becomes visible, and it refuses
unless the receipt in hand closes the exact package in hand.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Protocol
from uuid import UUID

import numpy as np
import numpy.typing as npt

from alphalattice.capabilities.portfolio_backtesting.contracts import PortfolioWalkForwardState
from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.product_host.storage.inventory import storage_capacity_scope
from alphalattice.control.task_control.contracts import TaskLifecycle
from alphalattice.control.task_control.registry import DuckDbTaskControlRegistry
from alphalattice.control.task_control.runner import TaskControlRunner
from alphalattice.investment.alpha_research.scores.product_recipe import (
    INSTALLED_ALPHA_PRODUCT_RECIPE,
)
from alphalattice.investment.alpha_research.scores.product_replay import (
    AlphaProductRecipeView,
)
from alphalattice.investment.portfolio_strategy_lab.application.advancement import (
    ordered_listing_axis_hash,
)
from alphalattice.investment.portfolio_strategy_lab.application.candidate_freeze import (
    PortfolioFinalizationCompositionError,
)
from alphalattice.investment.portfolio_strategy_lab.application.contracts import (
    PortfolioBenchmarkComparison,
    PortfolioDeclaredPathReport,
    PortfolioEconomicLedger,
    PortfolioExecutionLedger,
    PortfolioExecutionProgram,
    PortfolioResearchResult,
    PortfolioResearchSpec,
)
from alphalattice.investment.portfolio_strategy_lab.application.executor import (
    PortfolioContinuationCarry,
    PortfolioResearchExecutor,
    ResolvedPortfolioExecution,
)
from alphalattice.investment.portfolio_strategy_lab.application.finalization import (
    CompletedDevelopmentRun,
    FinalPortfolioEvaluationPackage,
    FrozenPortfolioCandidate,
    PortfolioFinalizationError,
    PortfolioValidationReceipt,
    ProtectedContinuationResult,
    ProtectedEvaluationPermit,
    ProtectedEvaluationPort,
    ProtectedFixture,
    ReleasedPortfolioArtifacts,
    ValidatedPortfolioHandoff,
)
from alphalattice.investment.portfolio_strategy_lab.application.finalization_task import (
    PortfolioFinalizationTaskAdapter,
    portfolio_finalization_task_contract,
)
from alphalattice.investment.portfolio_strategy_lab.application.package_inspection import (
    PendingPackageInspector,
)
from alphalattice.investment.portfolio_strategy_lab.application.readback import (
    OpenedUpstreamArtifact,
    exact_readback,
)
from alphalattice.investment.portfolio_strategy_lab.application.resolution import (
    PortfolioExecutionResolver,
    PortfolioResearchCompiler,
)
from alphalattice.investment.portfolio_strategy_lab.application.task import (
    PORTFOLIO_PUBLIC_TASK_KIND,
    portfolio_research_task_input,
)
from alphalattice.investment.portfolio_strategy_lab.publication.finalization_ledger import (
    PortfolioFinalizationStore,
)
from alphalattice.investment.portfolio_strategy_lab.publication.portfolio_ledger import (
    PENDING_FINALIZATION_NAMESPACE,
    PortfolioLedgerStore,
)
from alphalattice.investment.risk_research.surfaces.decomposition import (
    INSTALLED_RISK_DECOMPOSITION_RECIPE,
)
from alphalattice.investment.risk_research.surfaces.returns import (
    RiskReturnArtifactStore,
)
from alphalattice.oversight.model_validation.protected_gate import (
    ProtectedValidationGate,
    ProtectedValidationStore,
)


class AlphaEvidenceClosureReader(Protocol):
    """Verify one Alpha evidence manifest and name what it closes over.

    A protocol so the release path never learns which evidence format it is
    verifying: the installed package's owner supplies the reader, and a build
    without one simply cannot open that role.
    """

    def __call__(
        self, *, root: Path, expected_manifest_sha256: str
    ) -> tuple[str, str, int] | None: ...


@dataclass(frozen=True, slots=True)
class HostCompletedDevelopmentTasks:
    """Task Control, read-only, projected down to what freezing has to check.

    Host work: Portfolio declares the port and must not import the task registry,
    and the registry has no business knowing what a frozen candidate is. What
    crosses is a projection built from durable records -- the task's kind and
    lifecycle, its sealed input bindings, and the result identities its own
    *verified* stage receipts name.
    """

    registry: DuckDbTaskControlRegistry

    def open_completed_run(self, task_id: str) -> CompletedDevelopmentRun | None:
        """Project durable task kind/state and actually verified result identities for finalization.

        Args:
            task_id: Exact task identity to reopen.

        Returns:
            Completed-development projection when verified evidence exists, otherwise None; foreign
            task kinds remain distinguishable.
        """
        try:
            record = self.registry.task(UUID(task_id))
        except (KeyError, ValueError, LookupError):
            return None
        published = tuple(
            evidence.content_hash
            for receipt in self.registry.stage_receipts(record.task_id)
            if receipt.status == "VERIFIED"
            for evidence in receipt.evidence
        )
        if not published:
            # A task with no verified evidence has not published anything, and a
            # projection claiming otherwise would be the defect this replaces.
            return None
        if record.task_kind == PORTFOLIO_PUBLIC_TASK_KIND:
            try:
                durable = portfolio_research_task_input(record)
            except ValueError:
                return None
            spec_hash = durable.spec.spec_hash
            program_hash = durable.program.program_hash
            authorities_hash = durable.program.authorities_hash
        else:
            # Preserve the distinct wrong-kind refusal. These hashes are never
            # consumed for a non-Portfolio task: `_require_completed_development_run`
            # rejects its kind first. They merely satisfy the cross-package
            # projection without pretending the foreign input follows Portfolio's
            # typed schema.
            spec_hash = "0" * 64
            program_hash = "0" * 64
            authorities_hash = None
        return CompletedDevelopmentRun.create(
            task_id=str(record.task_id),
            task_kind=record.task_kind,
            lifecycle=record.lifecycle.value,
            spec_hash=spec_hash,
            program_hash=program_hash,
            authorities_hash=authorities_hash,
            published_result_hashes=published,
        )


@dataclass
class ExecutorProtectedContinuation:
    """Portfolio's protected run: the frozen book, carried forward, not restarted.

    A continuation is not "the same policy over a later window". It is the exact
    sealed state -- both books, both cash balances and the policy's sleeve array
    -- handed to the same walk-forward owner as its opening state, so the first
    protected formation is priced against the book the development path actually
    ended holding.

    Three refusals guard that. The protected axis must be exactly the permitted
    fixture; every session on it must fall after the development boundary; and
    the sealed boundary must be restorable in full onto the protected listing
    axis. A continuation that could not restore the state used to fall back to a
    flat start and copy the sealed hash onto the result, which produced a
    plausible protected path that had continued nothing.

    Everything is published into the pending namespace. Nothing a protected run
    computes reaches the public store until a receipt closes it.
    """

    workspace_id: str
    workspace: Path
    spec: PortfolioResearchSpec
    resolver: PortfolioExecutionResolver
    executor: PortfolioResearchExecutor
    compiler: PortfolioResearchCompiler
    development_ledger: PortfolioLedgerStore
    evaluations: int = 0

    def evaluate(
        self, *, permit: ProtectedEvaluationPermit, candidate: FrozenPortfolioCandidate
    ) -> ProtectedContinuationResult:
        """Require frozen authority and execute continuation solely in the pending namespace.

        Args:
            permit: Explicit owner-issued protected evaluation permit.
            candidate: Exact frozen candidate and pre-protected state.

        Returns:
            Continuation identities and counts only after exact boundary verification; reused
            execution does not count as a new evaluation.

        Raises:
            PortfolioFinalizationCompositionError: Store is public, frozen
                spec/holdings/workspace/support differs, support overlaps development or initial
                boundary differs.
        """
        if self.executor.store.is_public:
            # Checked here rather than trusted at construction: this is the one
            # method that writes, and a protected write into the public store is
            # the failure that cannot be walked back.
            raise PortfolioFinalizationCompositionError(
                "product_host.protected_continuation_must_not_write_the_public_store"
            )
        # Before resolving anything. The configuration a protected run executes
        # under has to be the one that was frozen and the one the permit names --
        # otherwise the run is a differently configured evaluation, and every
        # identity downstream of it would be about a question nobody permitted.
        if (
            self.spec.spec_hash != candidate.spec_hash
            or self.spec.spec_hash != permit.configuration_hash
        ):
            raise PortfolioFinalizationCompositionError(
                "product_host.protected_spec_is_not_the_frozen_configuration"
            )
        if self.spec.holdings_spec_hash != candidate.holdings_spec_hash:
            raise PortfolioFinalizationCompositionError(
                "product_host.protected_holdings_spec_is_not_the_frozen_one"
            )
        if self.workspace_id != candidate.workspace_id:
            raise PortfolioFinalizationCompositionError(
                "product_host.protected_workspace_is_not_the_frozen_one"
            )
        authorities = self.resolver.resolve_authorities(workspace=self.workspace, spec=self.spec)
        sessions = tuple(authorities.candidate_sessions)
        if sessions != tuple(permit.fixture.sessions):
            raise PortfolioFinalizationCompositionError(
                "product_host.protected_axis_is_not_the_permitted_fixture"
            )
        boundary = candidate.pre_protected_state.last_formation_session
        if any(value <= boundary for value in sessions):
            # A protected window that reaches back over the development path is
            # not a continuation, it is a second evaluation of sessions the
            # candidate was already scored on.
            raise PortfolioFinalizationCompositionError(
                "product_host.protected_axis_overlaps_the_development_path"
            )
        resolution = self.resolver.resolve(workspace=self.workspace, spec=self.spec)
        authorities.require_resolution(resolution)
        carry = self._restore(candidate=candidate, resolution=resolution)
        program = self.compiler.compile(
            spec=self.spec,
            authorities=authorities,
            resolved=resolution,
            continued_from_state_hash=carry.boundary.boundary_hash,
        )
        result = self.executor.execute(
            workspace_id=self.workspace_id,
            spec=self.spec,
            program=program,
            resolved=lambda: resolution,
            coverage=authorities.coverage,
            authorities_hash=authorities.authorities_hash,
            continuation=carry,
        )
        if result.action != "REUSED_EXACT":
            # Counted where the work actually happened. A resumed finalization
            # reaches this method again and the executor answers from its own
            # reuse index without walking the book a second time -- that is a
            # recovery, and counting it as an evaluation would make the
            # one-evaluation guarantee unmeasurable.
            self.evaluations += 1
        execution = self.executor.store.load_execution(result.execution_ledger_hash)
        if execution.initial_boundary != carry.boundary:
            raise PortfolioFinalizationCompositionError(
                "product_host.protected_path_did_not_open_on_the_sealed_state"
            )
        return ProtectedContinuationResult.create(
            permit_hash=permit.permit_hash,
            continued_from_state_hash=candidate.pre_protected_state.state_hash,
            protected_result_hash=result.result_hash,
            execution_ledger_hash=result.execution_ledger_hash,
            economic_ledger_hash=result.economic_ledger_hash,
            report_hash=result.report_hash,
            formation_count=len(execution.formation_sessions),
            listing_count=len(execution.ordered_listing_ids),
        )

    def _restore(
        self, *, candidate: FrozenPortfolioCandidate, resolution: ResolvedPortfolioExecution
    ) -> PortfolioContinuationCarry:
        """Rebuild the walk-forward state from the sealed boundary, or refuse.

        Reads the lanes back out of the development store by content hash, so a
        boundary whose lanes are missing or whose axis has moved fails here
        rather than becoming a silent restart. The listing axis check is the one
        that matters most: these vectors are positional, and continuing them onto
        a different axis would re-assign every holding to a different company.
        """

        sealed = candidate.pre_protected_state
        boundary = sealed.terminal_boundary
        listings = tuple(resolution.workspace.ordered_listing_ids)
        if ordered_listing_axis_hash(listings) != sealed.listing_axis_hash:
            raise PortfolioFinalizationCompositionError(
                "product_host.protected_listing_axis_does_not_match_the_sealed_state"
            )
        if boundary.listing_count != len(listings):
            raise PortfolioFinalizationCompositionError(
                "product_host.sealed_boundary_axis_mismatch"
            )
        pretrade = self._lane("boundary-weights", boundary.pretrade_weights_hash, len(listings))
        reference = self._lane("boundary-weights", boundary.optimizer_reference_hash, len(listings))
        sleeves: npt.NDArray[np.float64] | None = None
        if boundary.sleeve_weights_hash is not None:
            flat = self._lane(
                "boundary-sleeves",
                boundary.sleeve_weights_hash,
                boundary.sleeve_count * len(listings),
            )
            sleeves = flat.reshape((boundary.sleeve_count, len(listings)))
        return PortfolioContinuationCarry(
            boundary=boundary,
            state=PortfolioWalkForwardState(
                pretrade_weights=pretrade,
                pretrade_cash=boundary.pretrade_cash,
                optimizer_reference=reference,
                optimizer_reference_cash=boundary.optimizer_reference_cash,
            ),
            sleeve_weights=sleeves,
        )

    def _lane(self, category: str, content_hash: str, size: int) -> npt.NDArray[np.float64]:
        try:
            payload = self.development_ledger.load_lane(
                category=category, content_hash=content_hash
            )
        except Exception as error:
            raise PortfolioFinalizationCompositionError(
                "product_host.sealed_boundary_lane_unreadable"
            ) from error
        values: npt.NDArray[np.float64] = np.frombuffer(payload, dtype="<f8")
        if values.size != size:
            raise PortfolioFinalizationCompositionError(
                "product_host.sealed_boundary_lane_axis_invalid"
            )
        return values.astype(np.float64, copy=True)


@dataclass
class HostFinalizationRelease:
    """The atomic bind, and the only door out of the pending namespace.

    Every check here is about the pair rather than about either half: a valid
    receipt for another package, or a closed receipt whose candidate is not this
    one, would each release a report nobody validated.

    Only after all of them pass does the protected result move. Until then it
    exists solely in the pending root: no ordinary result, report, rendered page
    or program index in the public store can reach it, so "invisible until
    closure" is a property of where the bytes are rather than a flag somebody
    has to remember to check.
    """

    workspace_id: str
    pending: PortfolioLedgerStore | None = None
    public: PortfolioLedgerStore | None = None
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    releases: int = 0
    promotions: int = 0

    def release(
        self,
        *,
        package: FinalPortfolioEvaluationPackage,
        receipt: PortfolioValidationReceipt,
        candidate: FrozenPortfolioCandidate,
        finalization_task_id: str,
    ) -> ValidatedPortfolioHandoff:
        """Require a closed exactly bound validation receipt and seal the validated handoff.

        Args:
            package: Exact final evaluation package.
            receipt: Closed validation receipt for this package/candidate/state.
            candidate: Exact frozen candidate.
            finalization_task_id: Exact task receiving the handoff.

        Returns:
            Validated handoff binding released report and observed release time.

        Raises:
            PortfolioFinalizationCompositionError: Receipt is not closed or
                package/candidate/pre-protected state differs.
        """
        if not receipt.closed:
            raise PortfolioFinalizationCompositionError(
                "product_host.release_requires_a_closed_receipt:" + receipt.closure
            )
        if receipt.package_hash != package.package_hash:
            raise PortfolioFinalizationCompositionError(
                "product_host.release_receipt_is_for_another_package"
            )
        if receipt.candidate_hash != candidate.candidate_hash:
            raise PortfolioFinalizationCompositionError(
                "product_host.release_receipt_is_for_another_candidate"
            )
        if receipt.pre_protected_state_hash != candidate.pre_protected_state.state_hash:
            raise PortfolioFinalizationCompositionError(
                "product_host.release_receipt_state_mismatch"
            )
        self.releases += 1
        return ValidatedPortfolioHandoff.create(
            workspace_id=self.workspace_id,
            finalization_task_id=finalization_task_id,
            candidate_hash=candidate.candidate_hash,
            package_hash=package.package_hash,
            validation_receipt_hash=receipt.receipt_hash,
            released_report_hash=package.protected_report_hash,
            released_at=self.clock(),
        )

    def promote(self, package: FinalPortfolioEvaluationPackage) -> str | None:
        """Copy the validated artifacts into the public store, idempotently.

        Preparation, not release. Content-addressed all the way down, so running
        it twice after a crash publishes the same bytes to the same paths and
        changes nothing, and it writes no index -- so a tree half-copied by a
        crash discovers nothing and reads as exactly what it is.

        Deliberately *after* the handoff is durable. A handoff minted afterwards
        would get a new timestamp on every recovery attempt, and the identity a
        reader was promised would depend on when the machine happened to die.
        """
        if self.pending is None or self.public is None:
            return None
        adopted = self.public.adopt_result_from(self.pending, package.protected_result_hash)
        self.promotions += 1
        return adopted


@dataclass(frozen=True, slots=True)
class HostUpstreamEvidence:
    """Open the Alpha and Risk artifacts a Program names, at their own owners.

    A Program binds four identities it does not own. Readback has to prove those
    identities name something real, and the only honest way to do that is to go
    to each owner and open the thing.

    Composition's job here is to know *where* each owner lives -- the installed
    recipes, the admitted evidence root, the Risk return store -- and nothing
    else. It does not mint a receipt, and it cannot: every identity below comes
    back off the artifact that was opened, never off the request. That is the
    whole difference between readback and lineage, and it is what makes an
    arbitrary hash fail rather than pass.
    """

    artifact_root: Path
    alpha_evidence_root: Path | None = None
    alpha_evidence_closure: AlphaEvidenceClosureReader | None = None
    """Whoever can verify this build's Alpha evidence closure, if anybody can."""

    alpha_recipe: AlphaProductRecipeView = INSTALLED_ALPHA_PRODUCT_RECIPE

    def open_upstream(self, *, role: str, identity_hash: str) -> OpenedUpstreamArtifact | None:
        opened = self._open(role=role, identity_hash=identity_hash)
        if opened is None:
            return None
        opened_identity, closure_hash, child_count = opened
        return OpenedUpstreamArtifact.create(
            role=role,
            identity_hash=opened_identity,
            closure_hash=closure_hash,
            child_count=child_count,
            admitted_at=datetime.now(UTC),
        )

    def _open(self, *, role: str, identity_hash: str) -> tuple[str, str, int] | None:
        """Return the artifact's *own* identity, or `None` if it did not open."""

        if role == "ALPHA_RECIPE":
            # An installed object rather than a stored one: opening it means
            # resolving the recipe this build actually runs and reading the hash
            # it computes for itself.
            recipe_hash = str(self.alpha_recipe.recipe_hash)
            return recipe_hash, recipe_hash, 0
        if role == "RISK_RECIPE":
            recipe_hash = str(INSTALLED_RISK_DECOMPOSITION_RECIPE.recipe_hash)
            return recipe_hash, recipe_hash, 0
        if role == "ALPHA_EVIDENCE_MANIFEST":
            if self.alpha_evidence_root is None:
                return None
            if self.alpha_evidence_closure is None:
                # No installed reader for this build's Alpha evidence, so the
                # role does not open. A release cannot claim a manifest it has
                # no owner able to verify.
                return None
            # The reader hashes the manifest file and refuses unless it is this
            # identity, then cross-checks every definition field against the
            # installed recipe. Passing the requested identity in is not
            # circular: what comes back is the assertion that a file with that
            # sha256 exists and describes this build's Alpha.
            return self.alpha_evidence_closure(
                root=self.alpha_evidence_root, expected_manifest_sha256=identity_hash
            )
        if role == "RISK_RETURN_SURFACE":
            try:
                risk_closure = RiskReturnArtifactStore(self.artifact_root).verify_closure(
                    identity_hash
                )
            except Exception:
                return None
            return (
                risk_closure.surface_hash,
                risk_closure.closure_hash,
                risk_closure.chunk_count,
            )
        return None


@dataclass(frozen=True, slots=True)
class HostReleasedArtifacts:
    """The safe projection: what a public reader of a protected result may open.

    Every route to a released protected result goes through here, and here
    requires the committed release marker. Copied bytes are not an answer --
    a crash midway through adoption leaves a public store holding artifacts that
    were never released, and a reader that went straight to `load_result` could
    not tell that from a finished release.

    The marker is also what the projection is *checked against*: the result it
    hands back has to be the one the marker names, bound to the handoff and the
    receipt the marker names.
    """

    store: PortfolioFinalizationStore
    public: PortfolioLedgerStore

    def open_released(self, package_hash: str) -> tuple[ReleasedPortfolioArtifacts, object] | None:
        """The release marker and its result, or `None` if nothing was released."""
        marker = self.store.find_release_for_package(package_hash)
        if marker is None:
            return None
        result = self.public.load_result(marker.released_result_hash)
        if (
            result.result_hash != marker.released_result_hash
            or result.report_hash != marker.released_report_hash
        ):
            raise PortfolioFinalizationCompositionError(
                "product_host.released_marker_does_not_describe_its_result"
            )
        handoff = self.store.load_handoff(marker.handoff_hash)
        if (
            handoff.package_hash != marker.package_hash
            or handoff.validation_receipt_hash != marker.validation_receipt_hash
            or handoff.released_report_hash != marker.released_report_hash
        ):
            raise PortfolioFinalizationCompositionError(
                "product_host.released_marker_does_not_match_its_handoff"
            )
        return marker, result


@dataclass(frozen=True, slots=True)
class HostFinalizationArtifactOpener:
    """One read-only view over the two stores a finalization wrote into.

    Readback needs to *open* the package, the receipt and the handoff, and those
    live under two authorities: Portfolio published the package and the handoff,
    the Gate sealed the receipt. Composing them here is Host work -- it is the
    only place that legitimately knows both -- and it keeps the readback owner
    from importing the Gate whose answer it is checking.
    """

    store: PortfolioFinalizationStore
    validation: ProtectedValidationStore

    def open_package(self, package_hash: str) -> FinalPortfolioEvaluationPackage | None:
        """Reopen the exact final evaluation package through its retained deterministic store.

        Args:
            package_hash: Exact content identity.

        Returns:
            Validated exact artifact; storage refusals propagate.
        """
        return self.store.load_package(package_hash)

    def open_validation_receipt(self, receipt_hash: str) -> PortfolioValidationReceipt | None:
        """Reopen the exact validation receipt through its retained deterministic store.

        Args:
            receipt_hash: Exact content identity.

        Returns:
            Validated exact artifact; storage refusals propagate.
        """
        return self.validation.load_receipt(receipt_hash)

    def open_handoff(self, handoff_hash: str) -> ValidatedPortfolioHandoff | None:
        """Reopen the exact validated handoff through its retained deterministic store.

        Args:
            handoff_hash: Exact content identity.

        Returns:
            Validated exact artifact; storage refusals propagate.
        """
        return self.store.load_handoff(handoff_hash)

    def open_release(self, package_hash: str) -> ReleasedPortfolioArtifacts | None:
        """Read the committed release associated with one exact package.

        Args:
            package_hash: Exact package lookup key.

        Returns:
            Committed released-artifact declaration, or None before release.
        """
        return self.store.find_release_for_package(package_hash)


@dataclass(frozen=True, slots=True)
class ValidatedPortfolioAuthorityView:
    """The exact released Portfolio closure downstream capabilities may consume."""

    handoff: ValidatedPortfolioHandoff
    package: FinalPortfolioEvaluationPackage
    receipt: PortfolioValidationReceipt
    release: ReleasedPortfolioArtifacts
    result: PortfolioResearchResult
    program: PortfolioExecutionProgram
    execution: PortfolioExecutionLedger
    economics: PortfolioEconomicLedger
    comparison: PortfolioBenchmarkComparison
    report: PortfolioDeclaredPathReport


@dataclass(frozen=True, slots=True)
class HostValidatedPortfolioHandoffReader:
    """Reopen one immutable handoff without consulting a current pointer.

    Alternative Evidence and CRO receive this view, never a mutable Validation
    marker and never a Validation-owned metric bundle.  The read is accepted
    only after Portfolio's zero-work exact readback proves the complete released
    Portfolio/finalization subgraph.
    """

    artifact_root: Path

    def open(self, handoff_hash: str) -> ValidatedPortfolioAuthorityView:
        """Reopen released handoff lineage and require all declared closure layers verified.

        Args:
            handoff_hash: Exact validated handoff identity.

        Returns:
            Validated authority view of handoff/package/receipt/release and complete
            numerical/report evidence.

        Raises:
            PortfolioFinalizationCompositionError: Release is absent, artifacts are unreadable or a
                required closure layer is not VERIFIED.
        """
        root = self.artifact_root.resolve()
        finalization = PortfolioFinalizationStore(root)
        validation = ProtectedValidationStore(root)
        ledger = PortfolioLedgerStore(root)
        try:
            handoff = finalization.load_handoff(handoff_hash)
            package = finalization.load_package(handoff.package_hash)
            receipt = validation.load_receipt(handoff.validation_receipt_hash)
            release = finalization.find_release_for_package(package.package_hash)
            if release is None:
                raise PortfolioFinalizationCompositionError(
                    "product_host.validated_handoff_release_absent"
                )
            readback = exact_readback(
                ledger,
                workspace_id=handoff.workspace_id,
                result_hash=release.released_result_hash,
                package_hash=package.package_hash,
                validation_receipt_hash=receipt.receipt_hash,
                handoff_hash=handoff.handoff_hash,
                finalization=HostFinalizationArtifactOpener(
                    store=finalization,
                    validation=validation,
                ),
            )
        except PortfolioFinalizationCompositionError:
            raise
        except Exception as error:
            raise PortfolioFinalizationCompositionError(
                "product_host.validated_handoff_unreadable"
            ) from error

        required = (
            "portfolio_program",
            "numerical_input_assembly",
            "execution_ledger",
            "executed_weight_lanes",
            "economic_ledger",
            "benchmark_comparison",
            "controls_receipt",
            "schedule_and_window_guards",
            "report_facts",
            "rendered_artifact",
            "final_evaluation_package",
            "validation_receipt",
            "validated_handoff",
            "released_artifacts",
        )
        failed = tuple(
            layer_id for layer_id in required if readback.layer(layer_id).disposition != "VERIFIED"
        )
        if failed:
            raise PortfolioFinalizationCompositionError(
                "product_host.validated_handoff_closure_invalid:" + ",".join(failed)
            )

        result = ledger.load_result(release.released_result_hash)
        program = ledger.load_program(result.program_hash)
        execution = ledger.load_execution(result.execution_ledger_hash)
        economics = ledger.load_economics(result.economic_ledger_hash)
        report = ledger.load_report(result.report_hash)
        comparison = ledger.load_comparison(report.benchmark_comparison_hash)
        return ValidatedPortfolioAuthorityView(
            handoff=handoff,
            package=package,
            receipt=receipt,
            release=release,
            result=result,
            program=program,
            execution=execution,
            economics=economics,
            comparison=comparison,
            report=report,
        )


@dataclass(frozen=True, slots=True)
class CompletedFinalization:
    """What one finalization produced, with the task that produced all of it."""

    task_id: UUID
    candidate: FrozenPortfolioCandidate
    permit: ProtectedEvaluationPermit
    package: FinalPortfolioEvaluationPackage
    receipt: PortfolioValidationReceipt
    handoff: ValidatedPortfolioHandoff
    release: ReleasedPortfolioArtifacts
    """The marker that made this finalization observable, verified on return."""


class PortfolioFinalizationApplication:
    """Drive one finalization Task with the Gate injected across Portfolio's port."""

    def __init__(
        self,
        *,
        workspace_id: str,
        session: WorkspaceApplicationSession,
        continuation: ExecutorProtectedContinuation,
        gate: ProtectedEvaluationPort | None = None,
        alpha_evidence_root: Path | None = None,
        alpha_evidence_closure: AlphaEvidenceClosureReader | None = None,
        alpha_recipe: AlphaProductRecipeView = INSTALLED_ALPHA_PRODUCT_RECIPE,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        """Compose public/pending stores, validation, upstream evidence and release.

        Args:
            workspace_id: Explicit workspace identity.
            session: Retained workspace writer/task session.
            continuation: Deterministic protected continuation owner.
            gate: Optional installed protected evaluation port.
            alpha_evidence_root: Optional exact upstream evidence copy root.
            alpha_evidence_closure: Optional deterministic upstream closure reader.
            alpha_recipe: Exact installed Alpha recipe declaration.
            clock: Explicit observed-time source.
        """
        self.workspace_id = workspace_id
        self.session = session
        self.continuation = continuation
        self.clock = clock
        artifact_root = session.workspace / "runtime" / "artifacts"
        self.ledger = PortfolioLedgerStore(artifact_root)
        self.pending = PortfolioLedgerStore(artifact_root, namespace=PENDING_FINALIZATION_NAMESPACE)
        self.store = PortfolioFinalizationStore(artifact_root)
        # Composition constructs the Gate and hands it Portfolio's registry as a
        # read-only port; Portfolio only ever sees the evaluation port back.
        self.gate: ProtectedEvaluationPort = gate or ProtectedValidationGate(
            store=ProtectedValidationStore(artifact_root),
            registry=self.store,
            inspector=PendingPackageInspector(self.pending),
            clock=clock,
        )
        # Composed here, from the workspace, so a readback caller never has to
        # arrange anything first. There is nothing to "record": the opener goes
        # to each owner and opens the artifact.
        self.upstream = HostUpstreamEvidence(
            artifact_root=artifact_root,
            alpha_evidence_root=alpha_evidence_root,
            alpha_evidence_closure=alpha_evidence_closure,
            alpha_recipe=alpha_recipe,
        )
        self.release = HostFinalizationRelease(
            workspace_id=workspace_id,
            pending=self.pending,
            public=self.ledger,
            clock=clock,
        )
        self.adapter: PortfolioFinalizationTaskAdapter | None = None

    def finalize(
        self, *, candidate: FrozenPortfolioCandidate, fixture: ProtectedFixture
    ) -> CompletedFinalization:
        """Admit one finalization task and run it to a released handoff.

        Registers nothing. The candidate must already have been frozen and
        published by `freeze_candidate`, and this method refuses if it was not --
        registering it here would let a finalization create the very evidence the
        Gate is about to check it against, which is no check at all.
        """
        registered = self.store.open_candidate(candidate.candidate_hash)
        if registered is None or registered != candidate:
            raise PortfolioFinalizationCompositionError(
                "product_host.candidate_was_not_frozen_before_finalization"
            )
        envelope, goal, plan = portfolio_finalization_task_contract(
            workspace_id=self.workspace_id, candidate=candidate, fixture=fixture
        )
        admission = self.session.task_control_registry.admit(
            input_envelope=envelope, goal=goal, plan=plan, observed_at=self.clock()
        )
        return self._drive(task_id=admission.record.task_id, candidate=candidate, fixture=fixture)

    def recover(
        self, *, task_id: UUID, candidate: FrozenPortfolioCandidate, fixture: ProtectedFixture
    ) -> CompletedFinalization:
        """Resume the sealed finalization this task id was admitted for.

        The task's own input is the authority: a task of another kind, or one
        admitted over a different candidate or fixture, is refused rather than
        resumed under this candidate's permit.
        """
        task = self.session.task_control_registry.task(task_id)
        expected, _goal, _plan = portfolio_finalization_task_contract(
            workspace_id=self.workspace_id, candidate=candidate, fixture=fixture
        )
        if (
            task.task_kind != PortfolioFinalizationTaskAdapter.task_kind
            or task.input.payload != expected.payload
        ):
            raise PortfolioFinalizationCompositionError(
                "product_host.finalization_recovery_task_mismatch"
            )
        return self._drive(task_id=task_id, candidate=candidate, fixture=fixture)

    def _drive(
        self, *, task_id: UUID, candidate: FrozenPortfolioCandidate, fixture: ProtectedFixture
    ) -> CompletedFinalization:
        adapter = PortfolioFinalizationTaskAdapter(
            workspace_id=self.workspace_id,
            candidate=candidate,
            fixture=fixture,
            gate=self.gate,
            continuation=self.continuation,
            release=self.release,
            store=self.store,
        )
        self.adapter = adapter
        runner = TaskControlRunner(
            registry=self.session.task_control_registry,
            adapters={adapter.task_kind: adapter},
            runtime_path=str(self.session.runtime_path),
            clock=self.clock,
            stage_scope=storage_capacity_scope,
        )
        try:
            record = self.session.task_control_registry.task(task_id)
            if record.lifecycle is TaskLifecycle.RECOVERY_REQUIRED:
                record = runner.recover(task_id)
            elif record.lifecycle is TaskLifecycle.QUEUED:
                dispatched = runner.run_next()
                if dispatched is None:
                    raise PortfolioFinalizationCompositionError(
                        "product_host.finalization_task_not_dispatched"
                    )
                record = dispatched
        finally:
            runner.close()
        if record.lifecycle is not TaskLifecycle.SUCCEEDED:
            raise PortfolioFinalizationCompositionError(
                "product_host.finalization_task_not_succeeded:" + record.lifecycle.value
            )
        # A task that was already succeeded dispatches nothing, so nothing above
        # populated the adapter. Read its durable state before the completion
        # checks, or they would judge an empty object.
        adapter.rehydrate()
        if adapter.permit is None or adapter.package is None or adapter.receipt is None:
            raise PortfolioFinalizationCompositionError(
                "product_host.finalization_incomplete_despite_success"
            )
        handoff = self.store.find_handoff_for_package(adapter.package.package_hash)
        if handoff is None:
            raise PortfolioFinalizationCompositionError("product_host.finalization_handoff_absent")
        marker = self.store.find_release_for_package(adapter.package.package_hash)
        if marker is None:
            raise PortfolioFinalizationCompositionError("product_host.finalization_release_absent")
        if (
            marker.handoff_hash != handoff.handoff_hash
            or marker.validation_receipt_hash != adapter.receipt.receipt_hash
            or marker.released_result_hash != adapter.package.protected_result_hash
            or marker.released_report_hash != adapter.package.protected_report_hash
        ):
            # A completed finalization is a *released* one. Returning the handoff
            # without the marker that binds it would let a caller act on a
            # closure whose artifacts are still only adopted.
            raise PortfolioFinalizationCompositionError(
                "product_host.finalization_release_binding_invalid"
            )
        return CompletedFinalization(
            task_id=task_id,
            candidate=candidate,
            permit=adapter.permit,
            package=adapter.package,
            receipt=adapter.receipt,
            handoff=handoff,
            release=marker,
        )

    def read_back(self, result_hash: str, package: FinalPortfolioEvaluationPackage | None = None):  # type: ignore[no-untyped-def]
        """Exact readback with every opener this workspace can supply.

        The production readback seam. A caller asks for a result; composition
        supplies the upstream opener, the finalization opener and the Gate's
        receipt store, because knowing where those live is Host work and
        readback's job is to open what it is given and refuse what it cannot.
        """
        handoff = (
            None if package is None else self.store.find_handoff_for_package(package.package_hash)
        )
        receipt_hash = None if handoff is None else handoff.validation_receipt_hash
        return exact_readback(
            self.ledger,
            workspace_id=self.workspace_id,
            result_hash=result_hash,
            package_hash=None if package is None else package.package_hash,
            validation_receipt_hash=receipt_hash,
            handoff_hash=None if handoff is None else handoff.handoff_hash,
            upstream=self.upstream,
            finalization=HostFinalizationArtifactOpener(
                store=self.store,
                validation=ProtectedValidationStore(
                    self.session.workspace / "runtime" / "artifacts"
                ),
            ),
        )

    # ------------------------------------------------------------- firewall

    def refuse_protected_study_window(
        self, *, fixture: ProtectedFixture, start: date, end: date
    ) -> None:
        """A development study window may not reach the protected fixture.

        Checked against the fixture's own sessions rather than against a date
        range, because the firewall is about which evidence a window would open
        and two windows can share dates without sharing evidence.
        """
        if fixture.overlaps(start=start, end=end):
            raise PortfolioFinalizationError(
                "portfolio_finalization.study_window_reaches_the_protected_fixture"
            )


__all__ = [
    "CompletedFinalization",
    "ExecutorProtectedContinuation",
    "HostCompletedDevelopmentTasks",
    "HostFinalizationArtifactOpener",
    "HostFinalizationRelease",
    "HostReleasedArtifacts",
    "HostValidatedPortfolioHandoffReader",
    "PortfolioFinalizationApplication",
    "ValidatedPortfolioAuthorityView",
]
