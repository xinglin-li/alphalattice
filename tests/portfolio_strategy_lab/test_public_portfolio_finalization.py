"""Gate 9B successor core: finalization, readback, replay, and every refusal.

The acceptance matrix, run against the installed owners on isolated synthetic
fixtures. No real post-2024-08-12 evidence is located, listed, read or inferred
anywhere here; the only protected window in this file is a `ProtectedFixture`
whose disposition has exactly one admissible value.

The measurements are the point. Permits issued, protected continuations,
package publications, closures and handoffs are counters on the adapter, and the
readback and replay receipts carry their own work counters -- so "recovery did
not evaluate twice" and "exact readback computed nothing" are readings rather
than claims.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pytest

from alphalattice.capabilities.portfolio_backtesting.contracts import PortfolioWalkForwardState
from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.product_host.composition.portfolio_application import (
    PortfolioResearchApplication,
)
from alphalattice.control.product_host.composition.portfolio_finalization import (
    ExecutorProtectedContinuation,
    HostCompletedDevelopmentTasks,
    HostFinalizationArtifactOpener,
    HostFinalizationRelease,
    HostReleasedArtifacts,
    HostUpstreamEvidence,
    HostValidatedPortfolioHandoffReader,
    PortfolioFinalizationApplication,
)
from alphalattice.control.task_control.contracts import TaskEvidence, TaskLifecycle
from alphalattice.investment.alpha_research.scores.product_recipe import (
    INSTALLED_ALPHA_PRODUCT_RECIPE,
    AlphaProductRecipe,
)
from alphalattice.investment.alpha_research.scores.product_replay import (
    EVIDENCE_PACKAGE_ID,
)
from alphalattice.investment.portfolio_strategy_lab.application.advancement import (
    PortfolioLedgerCoverage,
)
from alphalattice.investment.portfolio_strategy_lab.application.candidate_freeze import (
    PortfolioFinalizationCompositionError,
    freeze_candidate,
)
from alphalattice.investment.portfolio_strategy_lab.application.contracts import (
    PortfolioDeclaredPathReport,
    PortfolioExecutionLedger,
    PortfolioExecutionProgram,
    PortfolioResearchSpec,
    PortfolioSupportCoverage,
    SealedPortfolioBoundaryState,
)
from alphalattice.investment.portfolio_strategy_lab.application.executor import (
    PortfolioContinuationCarry,
    PortfolioResearchExecutor,
)
from alphalattice.investment.portfolio_strategy_lab.application.finalization import (
    CLAIM_LIMIT_NO_RESELECTION,
    PROTECTED_FIXTURE_DISPOSITION,
    FinalPortfolioEvaluationPackage,
    FrozenPortfolioCandidate,
    PortfolioFinalizationError,
    PortfolioValidationReceipt,
    ProtectedEvaluationPermit,
    ProtectedFixture,
    ProtectedPackageInspection,
    SealedPreProtectedState,
)
from alphalattice.investment.portfolio_strategy_lab.application.finalization_task import (
    FINALIZATION_EVIDENCE_KIND,
    FINALIZATION_STAGES,
    PortfolioFinalizationTaskAdapter,
)
from alphalattice.investment.portfolio_strategy_lab.application.package_inspection import (
    PendingPackageInspector,
)
from alphalattice.investment.portfolio_strategy_lab.application.readback import (
    READBACK_LAYERS,
    REPLAY_LAYERS,
    ReadbackError,
    ReadbackLayer,
    ReadbackWork,
    StrongReplayReceipt,
    _upstream_layer,
    exact_readback,
    strong_replay,
)
from alphalattice.investment.portfolio_strategy_lab.application.resolution import (
    PortfolioResearchCompiler,
)
from alphalattice.investment.portfolio_strategy_lab.application.task import (
    PORTFOLIO_PUBLIC_TASK_KIND,
    portfolio_research_task_contract,
)
from alphalattice.investment.portfolio_strategy_lab.policies.installed_strategies import (
    product_evidence_closure_reader,
)
from alphalattice.investment.portfolio_strategy_lab.policies.tranche_book import (
    due_sleeves,
)
from alphalattice.investment.portfolio_strategy_lab.publication.finalization_ledger import (
    PortfolioFinalizationStore,
)
from alphalattice.investment.portfolio_strategy_lab.publication.portfolio_ledger import (
    PENDING_FINALIZATION_NAMESPACE,
    PortfolioLedgerStore,
)
from alphalattice.investment.risk_research.contracts import (
    CausalRiskReturnSurface,
    RiskUniverseEpoch,
)
from alphalattice.investment.risk_research.surfaces.decomposition import (
    INSTALLED_RISK_DECOMPOSITION_RECIPE,
)
from alphalattice.investment.risk_research.surfaces.returns import (
    RiskReturnArtifactStore,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.recipe_identity import recipe_identity
from alphalattice.oversight.model_validation.protected_gate import (
    ProtectedValidationGate,
    ProtectedValidationGateError,
    ProtectedValidationStore,
)
from tests.portfolio_strategy_lab.local_web_support import (
    TEST_PACKAGE,
    _resolved,
    _Resolver,
    _session_keyed_resolved,
)

_PACKAGES = {TEST_PACKAGE.package_hash: TEST_PACKAGE}
"""The one installed declaration these fixtures compile Programs under.

An executor is handed it for the same reason the application derives it from its
resolver: a report rebuilt from a stored ledger still has to say which strategy
produced it, and a protected release rebuilds exactly that way.
"""

DEVELOPMENT_SESSIONS = (date(2024, 1, 2), date(2024, 1, 3), date(2024, 2, 1))
# Strictly after the development boundary, and deliberately a different length:
# a fixture that matched the development axis would let a shape assumption hide.
PROTECTED_SESSIONS = (date(2024, 3, 1), date(2024, 3, 4), date(2024, 3, 5), date(2024, 3, 6))
LISTINGS = tuple(f"listing-{index:04d}" for index in range(80))


def _fixture(sessions: tuple[date, ...] = PROTECTED_SESSIONS) -> ProtectedFixture:
    return ProtectedFixture.of(
        fixture_id="qa-protected-window", sessions=sessions, listings=LISTINGS
    )


@dataclass
class _Finalization:
    """One wired finalization: workspace, development result, and the application."""

    session: WorkspaceApplicationSession
    development: PortfolioResearchApplication
    application: PortfolioFinalizationApplication
    continuation: ExecutorProtectedContinuation
    ledger: PortfolioLedgerStore
    pending: PortfolioLedgerStore
    store: PortfolioFinalizationStore
    coverage: PortfolioSupportCoverage
    spec: PortfolioResearchSpec
    result_hash: str
    development_task_id: str


def _wire(
    session: WorkspaceApplicationSession,
    *,
    workspace_id: str = "qa-final",
    development_resolution=None,  # type: ignore[no-untyped-def]
    protected_resolution=None,  # type: ignore[no-untyped-def]
    alpha_evidence_root: Path | None = None,
    alpha_evidence_closure=None,  # type: ignore[no-untyped-def]
    alpha_recipe=INSTALLED_ALPHA_PRODUCT_RECIPE,  # type: ignore[no-untyped-def]
) -> _Finalization:
    """Run one development path, then wire the finalization over its result."""

    workspace = session.workspace
    spec = PortfolioResearchSpec.default()
    development_inputs = development_resolution or _resolved(sessions=DEVELOPMENT_SESSIONS)
    protected_inputs = protected_resolution or _resolved(sessions=PROTECTED_SESSIONS)
    development = PortfolioResearchApplication(
        workspace_id=workspace_id,
        workspace=workspace,
        manifest_binding=lambda: "a" * 64,
        session=session,
        resolver=_Resolver(development_inputs, numerical=development_inputs),
    )
    completed = development.run(spec=spec)
    development_resolver = _Resolver(development_inputs, numerical=development_inputs)
    coverage = development_resolver.resolve_authorities(workspace=workspace, spec=spec).coverage
    artifact_root = workspace / "runtime" / "artifacts"
    # A *separate* executor over the pending namespace. The protected run must
    # not be able to write the public store at all, which is a different and
    # much stronger statement than "it publishes a record marked pending".
    pending = PortfolioLedgerStore(artifact_root, namespace=PENDING_FINALIZATION_NAMESPACE)
    continuation = ExecutorProtectedContinuation(
        workspace_id=workspace_id,
        workspace=workspace,
        spec=spec,
        # The protected resolver differs from the development one only in its
        # axis: the same policy, the same widths, a later window.
        resolver=_Resolver(protected_inputs, numerical=protected_inputs),
        executor=PortfolioResearchExecutor(pending, packages=_PACKAGES),
        compiler=PortfolioResearchCompiler(),
        development_ledger=development.ledger,
    )
    application = PortfolioFinalizationApplication(
        workspace_id=workspace_id,
        session=session,
        continuation=continuation,
        alpha_evidence_root=alpha_evidence_root,
        alpha_evidence_closure=(
            alpha_evidence_closure or product_evidence_closure_reader(recipe=alpha_recipe)
        ),
        alpha_recipe=alpha_recipe,
    )
    return _Finalization(
        session=session,
        development=development,
        application=application,
        continuation=continuation,
        ledger=development.ledger,
        pending=pending,
        store=application.store,
        coverage=coverage,
        spec=spec,
        result_hash=completed.result.result_hash,
        development_task_id=str(completed.pipeline_manifest.task_id),
    )


def _candidate(wired: _Finalization):  # type: ignore[no-untyped-def]
    """Freeze *and register*. Registration is a separate, earlier act than admission."""

    return freeze_candidate(
        wired.ledger,
        wired.store,
        workspace_id=wired.application.workspace_id,
        result_hash=wired.result_hash,
        development_task_id=wired.development_task_id,
        tasks=HostCompletedDevelopmentTasks(wired.session.task_control_registry),
    )


# ================================================ row 3: it works end to end


def test_a_synthetic_finalization_runs_once_under_one_task(tmp_path: Path) -> None:
    """Row 3: one task id across permit, continuation, package, closure, handoff."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        development_task = session.task_control_registry.tasks()[0]
        fixture = _fixture()

        finalized = wired.application.finalize(candidate=candidate, fixture=fixture)
        adapter = wired.application.adapter
        assert adapter is not None

        record = session.task_control_registry.task(finalized.task_id)
        assert record.lifecycle is TaskLifecycle.SUCCEEDED
        # One newly admitted task; the development task is untouched.
        assert finalized.task_id != development_task.task_id
        after = session.task_control_registry.task(development_task.task_id)
        assert after.record_hash == development_task.record_hash
        assert after.plan.work_items == development_task.plan.work_items
        # And exactly one finalization task exists.
        finalization_tasks = [
            value
            for value in session.task_control_registry.tasks()
            if value.task_kind == PortfolioFinalizationTaskAdapter.task_kind
        ]
        assert len(finalization_tasks) == 1

    # Measured work, not asserted disposition.
    assert adapter.permits_issued == 1
    assert adapter.protected_continuations == 1
    assert adapter.package_publications == 1
    assert adapter.closures == 1
    assert adapter.handoff_publications == 1
    assert wired.continuation.evaluations == 1
    assert wired.application.release.releases == 1

    assert finalized.permit.fixture.disposition == PROTECTED_FIXTURE_DISPOSITION
    assert finalized.package.visibility == "PENDING_CLOSURE"
    assert finalized.receipt.closed
    assert CLAIM_LIMIT_NO_RESELECTION in finalized.receipt.claim_limits
    assert finalized.handoff.package_hash == finalized.package.package_hash
    assert finalized.handoff.validation_receipt_hash == finalized.receipt.receipt_hash
    assert finalized.handoff.released_report_hash == finalized.package.protected_report_hash
    assert finalized.handoff.finalization_task_id == str(finalized.task_id)
    assert finalized.package.protected_formation_count == len(PROTECTED_SESSIONS)


def test_the_validation_receipt_carries_no_metric_and_the_handoff_copies_none() -> None:
    """The firewall the receipt exists to be: identities, limits, no number."""

    fields = set(PortfolioValidationReceipt.model_fields)
    for forbidden in ("return", "sharpe", "wealth", "turnover", "beta", "metric", "score"):
        assert not any(forbidden in name for name in fields), forbidden
    # The contract enforces it rather than trusting this test: a field added
    # later with a results-shaped name fails at construction.
    with pytest.raises(ValueError, match="validation_receipt_carries_a_metric"):

        class _Leaky(PortfolioValidationReceipt):
            cumulative_return: float = 0.0

        _Leaky.create(
            permit_hash="a" * 64,
            candidate_hash="b" * 64,
            package_hash="c" * 64,
            fixture_hash="d" * 64,
            pre_protected_state_hash="e" * 64,
            closure="CLOSED_EXACT",
            claim_limits=(CLAIM_LIMIT_NO_RESELECTION,),
            sealed_at=datetime.now(UTC),
        )


# ============================== rows 4 and 5: interruption and exact recovery


@pytest.mark.parametrize(
    ("interrupt_at", "expected_permits", "expected_continuations"),
    [
        # Interrupted before the continuation: the resumed run does it, once.
        ("continue_protected_path", 0, 1),
        # Interrupted after the package: the resumed run does neither again.
        ("verify_protected_closure", 0, 0),
    ],
)
def test_an_interrupted_finalization_resumes_without_repeating_a_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_at: str,
    expected_permits: int,
    expected_continuations: int,
) -> None:
    """Rows 4 and 5: resume after the permit, and after the pending package.

    The resumed adapter is a fresh object in a fresh attempt, so every count it
    reports is work *this* attempt did. Zero permits and zero continuations on
    the resumed run is the whole claim.
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        fixture = _fixture()
        original = PortfolioFinalizationTaskAdapter.execute_stage

        def _interrupt(
            self: PortfolioFinalizationTaskAdapter,
            *,
            task: object,
            execution: object,
            work_item: object,
        ) -> object:
            if work_item.stage_id == interrupt_at:  # type: ignore[attr-defined]
                raise RuntimeError("simulated interruption at " + interrupt_at)
            return original(self, task=task, execution=execution, work_item=work_item)  # type: ignore[arg-type]

        monkeypatch.setattr(PortfolioFinalizationTaskAdapter, "execute_stage", _interrupt)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            wired.application.finalize(candidate=candidate, fixture=fixture)

        interrupted = wired.application.adapter
        assert interrupted is not None
        permits_before = interrupted.permits_issued
        task = next(
            value
            for value in session.task_control_registry.tasks()
            if value.task_kind == PortfolioFinalizationTaskAdapter.task_kind
        )
        assert task.lifecycle is TaskLifecycle.RECOVERY_REQUIRED
        continuations_before = wired.continuation.evaluations

        monkeypatch.setattr(PortfolioFinalizationTaskAdapter, "execute_stage", original)
        finalized = wired.application.recover(
            task_id=task.task_id, candidate=candidate, fixture=fixture
        )
        resumed = wired.application.adapter
        assert resumed is not None and resumed is not interrupted

    # The same task id carried the whole protocol across the interruption.
    assert finalized.task_id == task.task_id
    # And the resumed attempt issued no second permit and ran no second
    # protected evaluation.
    assert resumed.permits_issued == expected_permits
    assert resumed.protected_continuations == expected_continuations
    # The claim that matters across both attempts: one permit, one protected
    # evaluation, one release -- whichever step the interruption landed on.
    assert permits_before + resumed.permits_issued == 1
    assert continuations_before + resumed.protected_continuations == 1
    assert wired.continuation.evaluations == 1
    assert wired.application.release.releases == 1
    assert finalized.receipt.closed
    assert finalized.handoff.finalization_task_id == str(task.task_id)


def test_recovery_refuses_a_task_admitted_over_another_candidate(tmp_path: Path) -> None:
    """A task id is not authority: the sealed input has to be this candidate's."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        finalized = wired.application.finalize(candidate=candidate, fixture=_fixture())
        # Same task id, a different fixture: the sealed input no longer matches.
        with pytest.raises(
            PortfolioFinalizationCompositionError, match="finalization_recovery_task_mismatch"
        ):
            wired.application.recover(
                task_id=finalized.task_id,
                candidate=candidate,
                fixture=_fixture(sessions=PROTECTED_SESSIONS[:2]),
            )


# ================================= row 6: duplicate permit, claim, publication


def test_a_second_permit_for_the_same_candidate_is_refused(tmp_path: Path) -> None:
    """Row 6: one evaluation of one frozen candidate, and no second chance."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        fixture = _fixture()
        gate = ProtectedValidationGate(
            store=ProtectedValidationStore(session.workspace / "runtime" / "artifacts"),
            registry=wired.store,
            inspector=PendingPackageInspector(wired.pending),
        )
        permit = gate.admit(candidate=candidate, fixture=fixture)
        with pytest.raises(ProtectedValidationGateError, match="permit_already_issued"):
            gate.admit(candidate=candidate, fixture=fixture)
        # A different fixture does not buy a second evaluation either.
        with pytest.raises(ProtectedValidationGateError, match="permit_already_issued"):
            gate.admit(candidate=candidate, fixture=_fixture(sessions=PROTECTED_SESSIONS[:2]))
        assert gate.reopen_permit(candidate=candidate) == permit


def test_a_second_closure_of_one_permit_is_refused(tmp_path: Path) -> None:
    """The permit is spent at closure; a package cannot be swapped afterwards."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        finalized = wired.application.finalize(candidate=candidate, fixture=_fixture())
        gate = wired.application.gate
        with pytest.raises(ProtectedValidationGateError, match="permit_already_claimed"):
            gate.close(permit=finalized.permit, candidate=candidate, package=finalized.package)
        assert gate.reopen_receipt(permit=finalized.permit) == finalized.receipt


def test_a_second_handoff_for_one_package_is_refused(tmp_path: Path) -> None:
    """One package, one handoff: downstream work cannot be given two answers."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        finalized = wired.application.finalize(candidate=candidate, fixture=_fixture())
        release = HostFinalizationRelease(workspace_id="qa-final")
        second = release.release(
            package=finalized.package,
            receipt=finalized.receipt,
            candidate=candidate,
            finalization_task_id="a-different-task",
        )
        assert second.handoff_hash != finalized.handoff.handoff_hash
        with pytest.raises(ValueError, match="committed_identity_reused"):
            wired.application.store.publish_handoff(second)


# ============================================= row 7: every tamper is refused


def test_a_mismatched_candidate_configuration_or_state_fails_closed(tmp_path: Path) -> None:
    """Row 7: the four ways closure can be wrong, told apart rather than merged."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        fixture = _fixture()
        gate = ProtectedValidationGate(
            store=ProtectedValidationStore(session.workspace / "runtime" / "artifacts"),
            registry=wired.store,
            inspector=PendingPackageInspector(wired.pending),
        )
        permit = gate.admit(candidate=candidate, fixture=fixture)
        package = FinalPortfolioEvaluationPackage.create(
            permit_hash=permit.permit_hash,
            candidate_hash=candidate.candidate_hash,
            continued_from_state_hash=candidate.pre_protected_state.state_hash,
            fixture_hash=fixture.fixture_hash,
            protected_result_hash="d" * 64,
            protected_execution_ledger_hash="a" * 64,
            protected_economic_ledger_hash="b" * 64,
            protected_report_hash="c" * 64,
            protected_formation_count=len(PROTECTED_SESSIONS),
            protected_listing_count=len(LISTINGS),
        )

        # A package that continued some other state.
        discontinuous = FinalPortfolioEvaluationPackage.create(
            **{
                **package.model_dump(mode="python", exclude={"package_hash"}),
                "continued_from_state_hash": "d" * 64,
            }
        )
        receipt = gate.close(permit=permit, candidate=candidate, package=discontinuous)
        assert receipt.closure == "REFUSED_STATE_DISCONTINUOUS"
        assert not receipt.closed
        assert CLAIM_LIMIT_NO_RESELECTION not in receipt.claim_limits
        # A refused closure still spends the permit: the protocol allows one
        # answer, not one *successful* answer.
        with pytest.raises(ProtectedValidationGateError, match="permit_already_claimed"):
            gate.close(permit=permit, candidate=candidate, package=package)


def test_a_permit_this_gate_never_issued_is_refused(tmp_path: Path) -> None:
    """A well-formed permit is not this Gate's permit."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        fixture = _fixture()
        elsewhere = ProtectedValidationGate(
            store=ProtectedValidationStore(tmp_path / "elsewhere"),
            registry=wired.store,
            inspector=PendingPackageInspector(wired.pending),
        )
        foreign = elsewhere.admit(candidate=candidate, fixture=fixture)
        package = FinalPortfolioEvaluationPackage.create(
            permit_hash=foreign.permit_hash,
            candidate_hash=candidate.candidate_hash,
            continued_from_state_hash=candidate.pre_protected_state.state_hash,
            fixture_hash=fixture.fixture_hash,
            protected_result_hash="d" * 64,
            protected_execution_ledger_hash="a" * 64,
            protected_economic_ledger_hash="b" * 64,
            protected_report_hash="c" * 64,
            protected_formation_count=4,
            protected_listing_count=len(LISTINGS),
        )
        with pytest.raises(ProtectedValidationGateError, match="permit_not_issued_here"):
            wired.application.gate.close(permit=foreign, candidate=candidate, package=package)


def test_the_release_refuses_a_receipt_for_another_package(tmp_path: Path) -> None:
    """Row 7: the atomic bind is about the pair, not about either half."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        finalized = wired.application.finalize(candidate=candidate, fixture=_fixture())
        release = HostFinalizationRelease(workspace_id="qa-final")
        tampered = FinalPortfolioEvaluationPackage.create(
            **{
                **finalized.package.model_dump(mode="python", exclude={"package_hash"}),
                "protected_report_hash": "e" * 64,
            }
        )
        with pytest.raises(
            PortfolioFinalizationCompositionError, match="receipt_is_for_another_package"
        ):
            release.release(
                package=tampered,
                receipt=finalized.receipt,
                candidate=candidate,
                finalization_task_id=str(finalized.task_id),
            )


def test_the_release_refuses_an_unclosed_receipt(tmp_path: Path) -> None:
    """Row 8's other half: no closed receipt, no released report."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        fixture = _fixture()
        gate = ProtectedValidationGate(
            store=ProtectedValidationStore(session.workspace / "runtime" / "artifacts"),
            registry=wired.store,
            inspector=PendingPackageInspector(wired.pending),
        )
        permit = gate.admit(candidate=candidate, fixture=fixture)
        package = FinalPortfolioEvaluationPackage.create(
            permit_hash=permit.permit_hash,
            candidate_hash="9" * 64,
            continued_from_state_hash=candidate.pre_protected_state.state_hash,
            fixture_hash=fixture.fixture_hash,
            protected_result_hash="d" * 64,
            protected_execution_ledger_hash="a" * 64,
            protected_economic_ledger_hash="b" * 64,
            protected_report_hash="c" * 64,
            protected_formation_count=4,
            protected_listing_count=len(LISTINGS),
        )
        refused = gate.close(permit=permit, candidate=candidate, package=package)
        assert refused.closure == "REFUSED_PACKAGE_MISMATCH"
        with pytest.raises(
            PortfolioFinalizationCompositionError, match="requires_a_closed_receipt"
        ):
            HostFinalizationRelease(workspace_id="qa-final").release(
                package=package,
                receipt=refused,
                candidate=candidate,
                finalization_task_id="t",
            )


# ================== row 8: the protected report is invisible before closure


def test_the_pending_package_is_not_visible_as_a_normal_result(tmp_path: Path) -> None:
    """Row 8: metrics exist and are unreachable until a receipt says otherwise."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        fixture = _fixture()
        original = PortfolioFinalizationTaskAdapter.execute_stage
        stopped_after = FINALIZATION_STAGES.index("seal_pending_package")

        def _stop(
            self: PortfolioFinalizationTaskAdapter,
            *,
            task: object,
            execution: object,
            work_item: object,
        ) -> object:
            if FINALIZATION_STAGES.index(work_item.stage_id) > stopped_after:  # type: ignore[attr-defined]
                raise RuntimeError("simulated stop before closure")
            return original(self, task=task, execution=execution, work_item=work_item)  # type: ignore[arg-type]

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(PortfolioFinalizationTaskAdapter, "execute_stage", _stop)
            with pytest.raises(RuntimeError, match="simulated stop"):
                wired.application.finalize(candidate=candidate, fixture=fixture)

        adapter = wired.application.adapter
        assert adapter is not None and adapter.package is not None
        package = wired.application.store.load_package(adapter.package.package_hash)
        assert package.visibility == "PENDING_CLOSURE"
        # No handoff exists, so nothing downstream can reach the report.
        assert wired.application.store.find_handoff_for_package(package.package_hash) is None
        # And the release refuses without a receipt at all.
        assert adapter.receipt is None


# =========================== row 9: a declared path has no winner to compare to


def test_the_default_declared_path_produces_its_own_report_with_no_winner(
    tmp_path: Path,
) -> None:
    """Row 9: one admitted product choice, no comparator, no selection receipt."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        report = wired.development.report(wired.result_hash)
        payload = report.model_dump(mode="json")

    assert report.window_guard.disposition == "DESCRIPTIVE_SUBWINDOW"
    assert not report.window_guard.may_select_or_promote
    for forbidden in ("winner", "selected_validation_path", "selection", "competitor", "rank"):
        assert not any(forbidden in key for key in payload), forbidden
    # A declared path is not scored against anything: the report carries one
    # path and its own guards.
    assert report.ledger_coverage.geometry_disposition == (
        "CONTINUOUS_ADMITTED_LEDGER_NO_SELECTION"
    )


# ==================== rows 1, 2, 10, 12: readback, replay and their refusals


def test_exact_readback_reopens_every_layer_and_computes_nothing(tmp_path: Path) -> None:
    """Row 1: every allowed work counter is zero, and every layer is named."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        finalized = wired.application.finalize(candidate=candidate, fixture=_fixture())
        development = exact_readback(
            wired.ledger, workspace_id="qa-final", result_hash=wired.result_hash
        )
        # With an opener the three finalization layers are proved by opening the
        # artifacts; without one they would report PARTIAL, which is the honest
        # answer to "here are three hashes, take my word for it".
        # The *released protected* result, which is what the finalization
        # identities are about. Reading back the development result while
        # quoting them describes two different paths.
        finalized_readback = exact_readback(
            wired.ledger,
            workspace_id="qa-final",
            result_hash=finalized.package.protected_result_hash,
            package_hash=finalized.package.package_hash,
            validation_receipt_hash=finalized.receipt.receipt_hash,
            handoff_hash=finalized.handoff.handoff_hash,
            finalization=HostFinalizationArtifactOpener(
                store=wired.store,
                validation=ProtectedValidationStore(session.workspace / "runtime" / "artifacts"),
            ),
        )
        unopened = exact_readback(
            wired.ledger,
            workspace_id="qa-final",
            result_hash=finalized.package.protected_result_hash,
            package_hash=finalized.package.package_hash,
            validation_receipt_hash=finalized.receipt.receipt_hash,
            handoff_hash=finalized.handoff.handoff_hash,
        )
        # And the counterexample the release exists to prevent: the same three
        # valid identities against the development result.
        crossed = exact_readback(
            wired.ledger,
            workspace_id="qa-final",
            result_hash=wired.result_hash,
            package_hash=finalized.package.package_hash,
            validation_receipt_hash=finalized.receipt.receipt_hash,
            handoff_hash=finalized.handoff.handoff_hash,
            finalization=HostFinalizationArtifactOpener(
                store=wired.store,
                validation=ProtectedValidationStore(session.workspace / "runtime" / "artifacts"),
            ),
        )
    assert crossed.layer("final_evaluation_package").disposition == "FAILED"
    assert "another result" in crossed.layer("final_evaluation_package").detail
    for layer_id in ("final_evaluation_package", "validation_receipt", "validated_handoff"):
        assert unopened.layer(layer_id).disposition == "PARTIAL", layer_id
        assert "not read" in unopened.layer(layer_id).detail

    assert development.work.total == 0
    assert finalized_readback.work.total == 0
    assert tuple(v.layer_id for v in development.layers) == READBACK_LAYERS
    # Row 12: the dispositions are distinguishable, and the receipt says which
    # layers it could not prove rather than omitting them.
    assert development.layer("portfolio_program").disposition == "VERIFIED"
    assert development.layer("execution_ledger").disposition == "VERIFIED"
    assert development.layer("executed_weight_lanes").disposition == "VERIFIED"
    assert development.layer("numerical_input_assembly").disposition == "VERIFIED"
    assert development.layer("final_evaluation_package").disposition == "NOT_APPLICABLE"
    assert "development path" in development.layer("validated_handoff").detail
    # Four upstream layers are bound but were not opened, because no opener was
    # supplied and they do not live in this workspace. They report `PARTIAL`
    # with the reason -- a hash copied off the Program is a binding, not a proof.
    assert set(development.unproved) == {
        "alpha_recipe",
        "alpha_evidence_package",
        "risk_recipe",
        "risk_return_surface",
        "final_evaluation_package",
        "validation_receipt",
        "validated_handoff",
        "released_artifacts",
    }
    for layer_id in ("alpha_recipe", "risk_recipe", "risk_return_surface"):
        assert development.layer(layer_id).disposition == "PARTIAL", layer_id
        assert "not read" in development.layer(layer_id).detail or (
            "no upstream opener" in development.layer(layer_id).detail
        )
    # The finalized readback proves the three layers the development one cannot,
    # and it proves them by *opening* each artifact from its store.
    for layer_id in ("final_evaluation_package", "validation_receipt", "validated_handoff"):
        assert finalized_readback.layer(layer_id).disposition == "VERIFIED"
        assert "opened" in finalized_readback.layer(layer_id).detail
    assert finalized_readback.layer("validation_receipt").identity_hash == (
        finalized.receipt.receipt_hash
    )


def test_strong_replay_rederives_the_ledger_and_matches_by_identity(tmp_path: Path) -> None:
    """Row 2: exact identities, and only the work a rederivation is allowed."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        receipt = strong_replay(
            wired.ledger,
            workspace_id="qa-final",
            result_hash=wired.result_hash,
            spec=wired.spec,
            coverage=wired.coverage,
        )
        sealed = wired.ledger.load_result(wired.result_hash)
        sealed_report = wired.ledger.load_report(sealed.report_hash)

    assert receipt.disposition == "REPLAYED_EXACT"
    assert tuple(v.layer_id for v in receipt.layers) == REPLAY_LAYERS
    # Exact means every layer, not "nothing failed". Turnover included: it is
    # charged against the drifted book, and the drift is rerun here from the
    # sealed per-name outcome matrix rather than guessed from executed books.
    for layer in receipt.layers:
        assert layer.disposition == "VERIFIED", (layer.layer_id, layer.detail)
    assert "drifted book" in receipt.layer("one_way_turnover").detail
    assert receipt.work.backtesting_transitions == len(PROTECTED_SESSIONS) - 1 or (
        receipt.work.backtesting_transitions == len(DEVELOPMENT_SESSIONS)
    )
    assert receipt.rederived_economic_ledger_hash == sealed.economic_ledger_hash
    assert receipt.sealed_economic_ledger_hash == sealed.economic_ledger_hash
    # The window-end book is inside the report identity, so a verified report
    # layer is a rederived book. Asserted with the book's own content so that a
    # projection which silently emitted nothing could not pass as verified.
    assert sealed_report.window_end_book.positions
    assert sealed_report.window_end_book.held_count > 0
    assert sealed_report.window_end_book.listing_axis_hash == (
        sealed_report.ledger_coverage.listing_axis_hash
    )
    # Permitted work happened; forbidden work did not.
    assert receipt.work.backtesting_transitions > 0
    assert receipt.work.metric_recomputations > 0
    assert receipt.work.forbidden_in_replay == 0
    assert receipt.work.alpha_fits == 0
    assert receipt.work.risk_estimations == 0
    assert receipt.work.provider_calls == 0
    assert receipt.work.protected_reads == 0
    assert receipt.work.pointer_mutations == 0
    assert receipt.work.publications == 0


def test_a_real_legacy_program_shape_loads_but_cannot_replay(tmp_path: Path) -> None:
    """Row 10: refused before any recomputation, not after a failed comparison.

    Built from a *serialized* predecessor Program rather than by patching the
    live class. That distinction is the test: the point is that an artifact
    written before the assembly existed still loads at the hash it was published
    under, and that this is what makes it unreplayable -- not that a property can
    be made to return `None`.
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        current = wired.ledger.load_program(
            wired.ledger.load_result(wired.result_hash).program_hash
        )
        assert current.numerical_input_assembly_hash is not None
        assert current.replayable

        # The predecessor shape: the identity with neither optional binding
        # present at all, hashed exactly as the pre-remediation writer hashed it.
        legacy_identity = current.model_dump(
            mode="json",
            exclude={
                "program_hash",
                "numerical_input_assembly_hash",
                "continued_from_state_hash",
            },
        )
        legacy_hash = canonical_hash(legacy_identity)
        legacy_path = wired.ledger.root / "programs" / f"{legacy_hash}.json"
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_path.write_bytes(
            json.dumps(
                {**legacy_identity, "program_hash": legacy_hash},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        reopened = wired.ledger.load_program(legacy_hash)
        assert reopened.program_hash == legacy_hash
        assert reopened.numerical_input_assembly_hash is None
        assert not reopened.replayable

        # Point a result at it and ask for a replay.
        sealed = wired.ledger.load_result(wired.result_hash)
        legacy_result = sealed.model_dump(mode="json", exclude={"result_hash"}, exclude_none=True)
        legacy_result["program_hash"] = legacy_hash
        # `action` is excluded from the result identity, so it is excluded here
        # too -- otherwise the artifact would not reopen at the hash it names.
        legacy_result_hash = canonical_hash(
            {k: v for k, v in legacy_result.items() if k != "action"}
        )
        result_path = wired.ledger.root / "results" / f"{legacy_result_hash}.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_bytes(
            json.dumps(
                {**legacy_result, "result_hash": legacy_result_hash},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        receipt = strong_replay(
            wired.ledger,
            workspace_id="qa-final",
            result_hash=legacy_result_hash,
            spec=wired.spec,
            coverage=wired.coverage,
        )

    assert receipt.disposition == "REFUSED_LEGACY_ASSEMBLY"
    assert receipt.numerical_input_assembly_hash is None
    assert receipt.work.total == 0
    assert all(v.disposition == "LEGACY" for v in receipt.layers)
    assert receipt.rederived_economic_ledger_hash is None


def test_a_sealed_output_that_does_not_follow_from_the_inputs_fails(tmp_path: Path) -> None:
    """Row 12: a failed replay is distinguishable from a refused one.

    Every child here is content-addressed, so a byte-level tamper does not
    produce a mismatch -- it produces an artifact that will not open. The case
    strong replay actually exists to catch is subtler and is the one built here:
    a ledger that is perfectly self-consistent, loads at its own hash, and
    records a turnover the sealed inputs do not produce.

    Under the *correct* spec, because a different spec is a different question
    and is refused before any work. And the failure localizes: the transition
    still reruns, so the layers upstream of the altered number stay VERIFIED.
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        result = wired.ledger.load_result(wired.result_hash)
        sealed = wired.ledger.load_execution(result.execution_ledger_hash)
        turnovers = list(sealed.one_way_turnovers)
        turnovers[1] = turnovers[1] + 0.01
        forged_identity = {
            **sealed.model_dump(mode="json", exclude={"ledger_hash"}),
            "one_way_turnovers": turnovers,
        }
        forged_ledger_hash = canonical_hash(forged_identity)
        (wired.ledger.root / "execution-ledgers" / f"{forged_ledger_hash}.json").write_bytes(
            json.dumps(
                {**forged_identity, "ledger_hash": forged_ledger_hash},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        forged_result_identity = {
            k: v
            for k, v in result.model_dump(
                mode="json", exclude={"result_hash"}, exclude_none=True
            ).items()
            if k != "action"
        } | {"execution_ledger_hash": forged_ledger_hash}
        forged_result_hash = canonical_hash(forged_result_identity)
        (wired.ledger.root / "results" / f"{forged_result_hash}.json").write_bytes(
            json.dumps(
                {
                    **forged_result_identity,
                    "action": "PUBLISHED",
                    "result_hash": forged_result_hash,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        receipt = strong_replay(
            wired.ledger,
            workspace_id="qa-final",
            result_hash=forged_result_hash,
            spec=wired.spec,
            coverage=wired.coverage,
        )

    assert receipt.disposition == "FAILED_MISMATCH"
    assert receipt.layer("one_way_turnover").disposition == "FAILED"
    assert receipt.layer("economic_metrics").disposition == "FAILED"
    # The rerun itself is sound; only the recorded number is wrong.
    for layer_id in ("portfolio_state", "drifted_pretrade_state", "fills_and_missed_fills"):
        assert receipt.layer(layer_id).disposition == "VERIFIED", layer_id


@pytest.mark.parametrize(
    "different",
    ["cost", "study_start", "study_end", "report_unit"],
)
def test_a_replay_of_a_different_request_is_refused_before_any_work(
    tmp_path: Path, different: str
) -> None:
    """A spec that is not the sealed one is a different question, not a failure.

    Including the two the reviewer named: a study window or a report unit changes
    nothing about the fills and everything about the report, so a replay under
    one would rebuild a perfectly correct report for a question nobody asked and
    then report the path broken. It is refused by name, and refused before a
    single Backtesting transition runs.
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    overrides: dict[str, object] = {
        "cost": {"cost_bps_per_side": "10"},
        "study_start": {"study_start": DEVELOPMENT_SESSIONS[1]},
        "study_end": {"study_end": DEVELOPMENT_SESSIONS[1]},
        "report_unit": {"report_unit": "SIMPLE_CUMULATIVE"},
    }[different]
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        receipt = strong_replay(
            wired.ledger,
            workspace_id="qa-final",
            result_hash=wired.result_hash,
            spec=PortfolioResearchSpec.create(**overrides),
            coverage=wired.coverage,
        )

    assert receipt.disposition == "REFUSED_SPEC_MISMATCH"
    assert receipt.work.total == 0
    assert receipt.work.backtesting_transitions == 0
    assert all(v.disposition == "LEGACY" for v in receipt.layers)
    assert "different request" in receipt.layer("report_metrics").detail


def test_a_missing_result_is_a_refusal_not_a_verified_readback(tmp_path: Path) -> None:
    """Row 12: absence is reported, never rendered as a pass."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        with pytest.raises(ValueError, match="result_not_reopenable"):
            exact_readback(wired.ledger, workspace_id="qa-final", result_hash="f" * 64)


# ================== row 11: a development window cannot reach the fixture


def test_a_development_study_window_cannot_reach_the_protected_fixture(
    tmp_path: Path,
) -> None:
    """Row 11: the firewall is about which evidence, not about a date range."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        fixture = _fixture()
        # A window entirely inside the development path is fine.
        wired.application.refuse_protected_study_window(
            fixture=fixture, start=DEVELOPMENT_SESSIONS[0], end=DEVELOPMENT_SESSIONS[-1]
        )
        # One that reaches the fixture is not.
        with pytest.raises(
            PortfolioFinalizationError, match="study_window_reaches_the_protected_fixture"
        ):
            wired.application.refuse_protected_study_window(
                fixture=fixture,
                start=DEVELOPMENT_SESSIONS[0],
                end=PROTECTED_SESSIONS[0],
            )


def test_a_protected_axis_that_overlaps_development_is_not_a_continuation(
    tmp_path: Path,
) -> None:
    """A continuation extends the book; it does not re-evaluate scored sessions."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        overlapping = DEVELOPMENT_SESSIONS[-1:] + PROTECTED_SESSIONS[:2]
        wired.continuation.resolver = _Resolver(_resolved(sessions=overlapping))
        with pytest.raises(
            PortfolioFinalizationCompositionError,
            match="protected_axis_overlaps_the_development_path",
        ):
            wired.application.finalize(candidate=candidate, fixture=_fixture(overlapping))


def test_the_candidate_freezes_only_what_the_development_run_published(
    tmp_path: Path,
) -> None:
    """Freezing reads; it never chooses a boundary that suits a later window."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        result = wired.ledger.load_result(wired.result_hash)
        ledger = wired.ledger.load_execution(result.execution_ledger_hash)

    assert candidate.development_result_hash == result.result_hash
    assert candidate.execution_ledger_hash == ledger.ledger_hash
    assert candidate.pre_protected_state.last_formation_session == ledger.formation_sessions[-1]
    assert candidate.pre_protected_state.final_weights_hash == ledger.final_weights_hash
    assert candidate.pre_protected_state.formation_count == len(ledger.formation_sessions)
    assert candidate.replayable
    assert candidate.frozen_at.tzinfo is not None
    # Everything the protected run needs to start from is bound before a permit
    # exists at all.
    assert candidate.frozen_at <= datetime.now(UTC) + timedelta(seconds=1)


# ===================================================================
# Remediation: the continuation actually continues
# ===================================================================


def test_the_protected_path_continues_the_frozen_state_rather_than_restarting(
    tmp_path: Path,
) -> None:
    """The sealed book decides the first protected fill, or nothing was continued.

    Two arms over the same fixture and the same policy. One opens on the frozen
    terminal state; the other opens flat. If the continuation were a fresh run
    with the sealed hash copied onto the result -- which is what it used to be --
    the two would produce identical ledgers and this test could not tell them
    apart.
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        finalized = wired.application.finalize(candidate=candidate, fixture=_fixture())
        continued = wired.pending.load_execution(finalized.package.protected_execution_ledger_hash)

        # The same protected axis, opened flat: a fresh walk, not a continuation.
        flat_store = PortfolioLedgerStore(
            tmp_path / "flat-arm", namespace=PENDING_FINALIZATION_NAMESPACE
        )
        flat_executor = PortfolioResearchExecutor(flat_store, packages=_PACKAGES)
        resolver = _Resolver(_resolved(sessions=PROTECTED_SESSIONS))
        authorities = resolver.resolve_authorities(workspace=workspace, spec=wired.spec)
        resolution = resolver.resolve(workspace=workspace, spec=wired.spec)
        authorities.require_resolution(resolution)
        flat_program = PortfolioResearchCompiler().compile(
            spec=wired.spec, authorities=authorities, resolved=resolution
        )
        flat = flat_executor.execute(
            workspace_id="qa-final",
            spec=wired.spec,
            program=flat_program,
            resolved=lambda: resolution,
            coverage=authorities.coverage,
            authorities_hash=authorities.authorities_hash,
        )
        flat_ledger = flat_store.load_execution(flat.execution_ledger_hash)

    sealed_boundary = candidate.pre_protected_state.terminal_boundary
    # The continuation opened on the frozen boundary, by identity.
    assert continued.initial_boundary == sealed_boundary
    assert continued.program_hash != flat_ledger.program_hash
    # And a flat start opened on a different state and produced a different path.
    assert flat_ledger.initial_boundary is not None
    assert flat_ledger.initial_boundary != sealed_boundary
    assert flat_ledger.ledger_hash != continued.ledger_hash
    # The difference shows up in the *first* protected formation, which is the
    # only formation the opening book can reach.
    assert continued.one_way_turnovers[0] != flat_ledger.one_way_turnovers[0]


def test_a_different_frozen_terminal_state_changes_the_first_protected_formation(
    tmp_path: Path,
) -> None:
    """Move the sealed book, and the opening trade moves with it.

    The sharper version of the test above: same axis, same policy, same
    everything except the weights in the frozen boundary. A continuation that
    reads its opening state cannot help but produce a different first turnover;
    one that restarts cannot help but produce the same.
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        finalized = wired.application.finalize(candidate=candidate, fixture=_fixture())
        original = wired.pending.load_execution(finalized.package.protected_execution_ledger_hash)
        boundary = candidate.pre_protected_state.terminal_boundary

        # A second boundary over the same axis holding nothing at all.
        empty = np.zeros(len(LISTINGS), dtype="<f8")
        empty_hash = wired.ledger.publish_lane(category="boundary-weights", values=empty)
        moved = SealedPortfolioBoundaryState.create(
            listing_count=boundary.listing_count,
            formation_session=boundary.formation_session,
            pretrade_weights_hash=empty_hash,
            pretrade_cash=1.0,
            optimizer_reference_hash=empty_hash,
            optimizer_reference_cash=1.0,
            sleeve_weights_hash=None,
            sleeve_count=0,
        )
        assert moved.boundary_hash != boundary.boundary_hash

        resolver = _Resolver(_resolved(sessions=PROTECTED_SESSIONS))
        authorities = resolver.resolve_authorities(workspace=workspace, spec=wired.spec)
        resolution = resolver.resolve(workspace=workspace, spec=wired.spec)
        authorities.require_resolution(resolution)
        carry = PortfolioContinuationCarry(
            boundary=moved,
            state=PortfolioWalkForwardState(
                pretrade_weights=empty.astype(float),
                pretrade_cash=1.0,
                optimizer_reference=empty.astype(float),
                optimizer_reference_cash=1.0,
            ),
            sleeve_weights=None,
        )
        moved_store = PortfolioLedgerStore(
            tmp_path / "moved-arm", namespace=PENDING_FINALIZATION_NAMESPACE
        )
        moved_store.publish_lane(category="boundary-weights", values=empty)
        moved_result = PortfolioResearchExecutor(moved_store, packages=_PACKAGES).execute(
            workspace_id="qa-final",
            spec=wired.spec,
            program=PortfolioResearchCompiler().compile(
                spec=wired.spec,
                authorities=authorities,
                resolved=resolution,
                continued_from_state_hash=moved.boundary_hash,
            ),
            resolved=lambda: resolution,
            coverage=authorities.coverage,
            authorities_hash=authorities.authorities_hash,
            continuation=carry,
        )
        moved_ledger = moved_store.load_execution(moved_result.execution_ledger_hash)

    assert moved_ledger.initial_boundary == moved
    assert moved_ledger.ledger_hash != original.ledger_hash
    assert moved_ledger.one_way_turnovers[0] != original.one_way_turnovers[0]


def test_a_continuation_refuses_a_boundary_it_cannot_restore(tmp_path: Path) -> None:
    """Missing state is a refusal, never a silent restart.

    The failure mode this replaces was the quiet one: a continuation that could
    not read its boundary would run flat and copy the sealed hash onto the
    result, and every downstream check would pass on a path that continued
    nothing.
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        boundary = candidate.pre_protected_state.terminal_boundary
        lane = (
            wired.ledger.root
            / "lanes"
            / "boundary-weights"
            / f"{boundary.pretrade_weights_hash}.parquet"
        )
        assert lane.is_file()
        lane.unlink()
        with pytest.raises(
            PortfolioFinalizationCompositionError, match="sealed_boundary_lane_unreadable"
        ):
            wired.application.finalize(candidate=candidate, fixture=_fixture())


def test_a_ledger_without_a_terminal_boundary_cannot_be_frozen(tmp_path: Path) -> None:
    """A predecessor ledger cannot become a candidate, because it cannot be continued."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        sealed = wired.ledger.load_execution(
            wired.ledger.load_result(wired.result_hash).execution_ledger_hash
        )
        identity = sealed.model_dump(
            mode="json",
            exclude={
                "ledger_hash",
                "initial_boundary",
                "final_boundary",
                "target_weights_hash",
                "realized_simple_returns_hash",
                "execution_available_hash",
            },
        )
        legacy = PortfolioExecutionLedger.model_validate(
            {**identity, "ledger_hash": canonical_hash(identity)}
        )

    assert legacy.final_boundary is None
    assert not legacy.replayable
    with pytest.raises(PortfolioFinalizationError, match="ledger_sealed_no_terminal_boundary"):
        SealedPreProtectedState.of(legacy)


# ===================================================================
# Remediation: nothing protected is visible before closure
# ===================================================================


def _public_inventory(store: PortfolioLedgerStore) -> set[str]:
    """Every file the ordinary public store holds, by relative path."""

    if not store.root.exists():
        return set()
    return {
        str(path.relative_to(store.root)).replace("\\", "/")
        for path in store.root.rglob("*")
        if path.is_file()
    }


def test_a_pending_protected_result_is_absent_from_every_public_index(
    tmp_path: Path,
) -> None:
    """Query the ordinary lookups, not just the absence of a handoff.

    Before closure the protected result must be unreachable through every route
    an ordinary reader has: by result, by report, by rendered page, by execution
    ledger, and through the two request-shaped indices a later development
    request would arrive on. Asserting "no handoff exists" would pass even if the
    whole protected path were sitting in the public store.
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        fixture = _fixture()
        before = _public_inventory(wired.ledger)

        # Drive the task to the point just before closure.
        adapter_seen: dict[str, object] = {}
        original = PortfolioFinalizationTaskAdapter.execute_stage

        def _stop_before_closure(self, *, task, execution, work_item):  # type: ignore[no-untyped-def]
            if work_item.stage_id == "verify_protected_closure":
                adapter_seen["package"] = self.package
                raise RuntimeError("stop before closure")
            return original(self, task=task, execution=execution, work_item=work_item)

        PortfolioFinalizationTaskAdapter.execute_stage = _stop_before_closure  # type: ignore[method-assign]
        try:
            with pytest.raises(BaseException, match=r"stop before closure|not_succeeded"):
                wired.application.finalize(candidate=candidate, fixture=fixture)
        finally:
            PortfolioFinalizationTaskAdapter.execute_stage = original  # type: ignore[method-assign]

        package = adapter_seen["package"]
        assert isinstance(package, FinalPortfolioEvaluationPackage)
        assert package.visibility == "PENDING_CLOSURE"

        # The protected artifacts exist -- in the pending namespace.
        assert wired.pending.load_result(package.protected_result_hash) is not None
        assert wired.pending.load_execution(package.protected_execution_ledger_hash)
        assert wired.pending.load_report(package.protected_report_hash)

        # And in the public store: nothing new at all.
        assert _public_inventory(wired.ledger) == before

        # Every ordinary public lookup refuses, one route at a time.
        for load, identity in (
            (wired.ledger.load_result, package.protected_result_hash),
            (wired.ledger.load_execution, package.protected_execution_ledger_hash),
            (wired.ledger.load_economics, package.protected_economic_ledger_hash),
            (wired.ledger.load_report, package.protected_report_hash),
        ):
            with pytest.raises(ValueError):
                load(identity)
        protected_program = wired.pending.load_execution(
            package.protected_execution_ledger_hash
        ).program_hash
        with pytest.raises(ValueError):
            wired.ledger.load_program(protected_program)
        assert wired.ledger.find_execution_for_program(protected_program) is None
        assert (
            wired.ledger.find_result_for(
                program_hash=protected_program, spec_hash=wired.spec.spec_hash
            )
            is None
        )
        # The request-shaped indices are the dangerous ones: they are keyed the
        # way a later development request arrives.
        assert (
            wired.ledger.find_program_for(
                holdings_spec_hash=candidate.holdings_spec_hash,
                authorities_hash=wired.ledger.load_program(candidate.program_hash).authorities_hash
                or "0" * 64,
            ).program_hash
            == candidate.program_hash
        )
        assert wired.store.find_handoff_for_package(package.package_hash) is None


def test_closure_is_what_moves_the_protected_artifacts_into_the_public_store(
    tmp_path: Path,
) -> None:
    """The release is the only door, and it opens exactly once."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        finalized = wired.application.finalize(candidate=candidate, fixture=_fixture())

        promoted = wired.ledger.load_result(finalized.package.protected_result_hash)
        assert promoted.result_hash == finalized.package.protected_result_hash
        assert wired.ledger.load_report(finalized.package.protected_report_hash)
        assert wired.ledger.load_html_by_uri(promoted.html_uri)
        # Byte-identical to what was validated, not a re-derivation.
        assert promoted == wired.pending.load_result(finalized.package.protected_result_hash)
        # But still not a cache hit for anybody's next development request: the
        # request-shaped indices were deliberately not promoted.
        pending_program = wired.pending.load_program(promoted.program_hash)
        assert pending_program.continued_from_state_hash is not None
        # No request-shaped index in public for the protected program, so a later
        # development request over the same holdings spec cannot land on it.
        assert (
            wired.ledger.find_program_for(
                holdings_spec_hash=pending_program.holdings_spec_hash,
                authorities_hash=pending_program.authorities_hash or "0" * 64,
            )
            is None
        )
        development_program = wired.ledger.load_program(candidate.program_hash)
        assert (
            wired.ledger.find_program_for(
                holdings_spec_hash=development_program.holdings_spec_hash,
                authorities_hash=development_program.authorities_hash or "0" * 64,
            ).program_hash
            == candidate.program_hash
        )
    assert wired.application.release.promotions == 1


# ===================================================================
# Remediation: a candidate must have been frozen beforehand
# ===================================================================


def test_the_gate_refuses_a_self_consistent_candidate_it_never_saw_frozen(
    tmp_path: Path,
) -> None:
    """A candidate that validates is not a candidate that was frozen.

    The forged candidate here is internally perfect: every hash checks out, the
    state binds its own ledger, the identity recomputes. It is refused for the
    only reason that matters -- nobody registered it before the permit was asked
    for, so "frozen beforehand" would have meant "constructed a moment ago".
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        result = wired.ledger.load_result(wired.result_hash)
        program = wired.ledger.load_program(result.program_hash)
        execution = wired.ledger.load_execution(result.execution_ledger_hash)
        forged = FrozenPortfolioCandidate.create(
            workspace_id="qa-final",
            spec_hash=result.spec_hash,
            holdings_spec_hash=program.holdings_spec_hash,
            program_hash=program.program_hash,
            numerical_input_assembly_hash=program.numerical_input_assembly_hash,
            development_task_id=wired.development_task_id,
            development_run_hash="7" * 64,
            control_receipt_hash="8" * 64,
            development_result_hash=result.result_hash,
            execution_ledger_hash=execution.ledger_hash,
            economic_ledger_hash=result.economic_ledger_hash,
            report_hash=result.report_hash,
            pre_protected_state=SealedPreProtectedState.of(execution),
            frozen_at=datetime.now(UTC),
        )
        # Self-consistent: it survives a full serialize/validate round trip, so
        # its own identity check passes and every binding it names is real.
        assert FrozenPortfolioCandidate.model_validate_json(forged.model_dump_json()) == forged

        gate = ProtectedValidationGate(
            store=ProtectedValidationStore(session.workspace / "runtime" / "artifacts"),
            registry=wired.store,
            inspector=PendingPackageInspector(wired.pending),
        )
        with pytest.raises(ProtectedValidationGateError, match="candidate_was_not_frozen"):
            gate.admit(candidate=forged, fixture=_fixture())

        # And the application refuses it too, rather than registering it on the
        # way past and manufacturing the prerequisite.
        with pytest.raises(
            PortfolioFinalizationCompositionError,
            match="candidate_was_not_frozen_before_finalization",
        ):
            wired.application.finalize(candidate=forged, fixture=_fixture())


def test_the_gate_refuses_a_candidate_edited_after_it_was_registered(
    tmp_path: Path,
) -> None:
    """Registration binds the exact bytes, not the name."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        registry = wired.store
        entry = registry.committed.path("by-candidate", candidate.candidate_hash)
        payload = json.loads(entry.read_text(encoding="utf-8"))
        payload["payload"]["workspace_id"] = "somewhere-else"
        entry.write_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())

        gate = ProtectedValidationGate(
            store=ProtectedValidationStore(session.workspace / "runtime" / "artifacts"),
            registry=registry,
            inspector=PendingPackageInspector(wired.pending),
        )
        with pytest.raises(ValueError, match=r"committed_index_tampered|candidate_"):
            gate.admit(candidate=candidate, fixture=_fixture())


# ===================================================================
# Remediation: every crash window
# ===================================================================


@pytest.mark.parametrize(
    "window",
    [
        "before_continuation_is_recorded",
        "before_package_publication",
        "before_permit_claim",
        "before_handoff_index",
    ],
)
def test_recovery_preserves_every_identity_across_each_crash_window(
    tmp_path: Path, window: str
) -> None:
    """Four injected interruptions, and not one repeated protected evaluation.

    Each window is a real gap between doing something and recording it. The
    guarantee under test is not "it finishes" but "it finishes as the *same*
    finalization": same task, candidate, permit, package, receipt and handoff
    identities, and exactly one walk of the protected book across both attempts.
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        fixture = _fixture()
        store = wired.store
        gate_store = wired.application.gate.store  # type: ignore[union-attr]
        tripped = {"done": False}

        def _trip_once(original, guard):  # type: ignore[no-untyped-def]
            def _wrapped(*args, **kwargs):  # type: ignore[no-untyped-def]
                if not tripped["done"] and guard(*args, **kwargs):
                    tripped["done"] = True
                    raise RuntimeError("crash: " + window)
                return original(*args, **kwargs)

            return _wrapped

        if window == "before_continuation_is_recorded":
            # The protected walk has run and published its ledgers; the record
            # that it ran has not been written yet.
            store.publish_continuation = _trip_once(  # type: ignore[method-assign]
                store.publish_continuation, lambda *a, **k: True
            )
        elif window == "before_package_publication":
            store.publish_package = _trip_once(  # type: ignore[method-assign]
                store.publish_package, lambda *a, **k: True
            )
        elif window == "before_permit_claim":
            # The receipt is sealed; the claim that spends the permit is not.
            gate_store.publish_claim = _trip_once(  # type: ignore[method-assign]
                gate_store.publish_claim, lambda *a, **k: True
            )
        else:
            store.publish_handoff = _trip_once(  # type: ignore[method-assign]
                store.publish_handoff, lambda *a, **k: True
            )

        with pytest.raises(BaseException, match=r"crash: |not_succeeded"):
            wired.application.finalize(candidate=candidate, fixture=fixture)
        assert tripped["done"], "the injected interruption never fired"

        task = next(
            value
            for value in session.task_control_registry.tasks()
            if value.task_kind == PortfolioFinalizationTaskAdapter.task_kind
        )
        first_permit = wired.application.gate.reopen_permit(candidate=candidate)
        finalized = wired.application.recover(
            task_id=task.task_id, candidate=candidate, fixture=fixture
        )

    # One finalization, whatever the machine did in the middle of it.
    assert finalized.task_id == task.task_id
    assert finalized.candidate.candidate_hash == candidate.candidate_hash
    assert finalized.receipt.closure == "CLOSED_EXACT"
    if first_permit is not None:
        assert finalized.permit.permit_hash == first_permit.permit_hash
    # The one that costs money if it is wrong.
    assert wired.continuation.evaluations == 1


def test_a_committed_artifact_survives_a_lost_content_file(tmp_path: Path) -> None:
    """The index entry carries the artifact, so a torn write heals forward.

    This is the window between recording that something happened and writing the
    file that says what it was. Because the entry carries the payload, recovery
    republishes the identical artifact instead of concluding it never existed and
    minting a second one with a new timestamp.
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        finalized = wired.application.finalize(candidate=candidate, fixture=_fixture())

        for category, identity in (
            ("packages", finalized.package.package_hash),
            ("handoffs", finalized.handoff.handoff_hash),
        ):
            content = wired.store.root / category / f"{identity}.json"
            assert content.is_file()
            content.unlink()

        healed_package = wired.store.find_package_for_permit(finalized.permit.permit_hash)
        healed_handoff = wired.store.find_handoff_for_package(finalized.package.package_hash)

    assert healed_package == finalized.package
    assert healed_handoff == finalized.handoff
    assert (wired.store.root / "packages" / f"{finalized.package.package_hash}.json").is_file()


def test_a_stage_reports_only_what_is_already_durable(tmp_path: Path) -> None:
    """Every stage's evidence hash must reopen from a store, in a fresh reader."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        finalized = wired.application.finalize(candidate=candidate, fixture=_fixture())
        artifact_root = session.workspace / "runtime" / "artifacts"
        reader = PortfolioFinalizationStore(artifact_root)
        validation = ProtectedValidationStore(artifact_root)
        # Read off the verified stage receipts -- what the runner actually
        # recorded -- and then reopen every one of those hashes through a store
        # built fresh right here. A stage that reported an in-memory artifact
        # would fail on the reopen rather than on the record.
        reached = {
            receipt.stage_id: tuple(value.content_hash for value in receipt.evidence)
            for receipt in session.task_control_registry.stage_receipts(finalized.task_id)
            if receipt.status == "VERIFIED"
        }

    assert set(reached) == set(FINALIZATION_STAGES)
    permit_hash, continuation_hash = (
        reached["claim_protected_permit"][0],
        reached["continue_protected_path"][0],
    )
    assert validation.load_permit(permit_hash).permit_hash == permit_hash
    durable_continuation = reader.find_continuation_for_permit(permit_hash)
    assert durable_continuation is not None
    assert durable_continuation.result_hash == continuation_hash
    assert reader.load_package(reached["seal_pending_package"][0])
    assert validation.load_receipt(reached["verify_protected_closure"][0])
    assert reader.load_handoff(reached["release_validated_handoff"][0])


# ===================================================================
# Remediation: a partial layer is never called exact
# ===================================================================


def test_a_replay_with_an_unproved_layer_cannot_be_called_exact() -> None:
    """The contract refuses the claim, so no code path can make it."""

    layers = tuple(
        ReadbackLayer(
            layer_id=layer_id,
            disposition="PARTIAL" if layer_id == "one_way_turnover" else "VERIFIED",
            identity_hash="c" * 64,
            detail="a layer that was not fully proved",
        )
        for layer_id in REPLAY_LAYERS
    )
    with pytest.raises(ValueError, match="replay_claimed_exact_with_an_unproved_layer"):
        StrongReplayReceipt.create(
            workspace_id="qa-final",
            result_hash="a" * 64,
            program_hash="b" * 64,
            numerical_input_assembly_hash="c" * 64,
            layers=layers,
            rederived_economic_ledger_hash="d" * 64,
            sealed_economic_ledger_hash="d" * 64,
            work=ReadbackWork(backtesting_transitions=1),
            disposition="REPLAYED_EXACT",
        )


def test_a_ledger_without_replay_children_is_refused_before_any_work(
    tmp_path: Path,
) -> None:
    """A path that sealed outputs but not inputs cannot be strongly replayed."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        result = wired.ledger.load_result(wired.result_hash)
        sealed = wired.ledger.load_execution(result.execution_ledger_hash)
        identity = sealed.model_dump(
            mode="json",
            exclude={
                "ledger_hash",
                "target_weights_hash",
                "realized_simple_returns_hash",
                "execution_available_hash",
                "initial_boundary",
                "final_boundary",
            },
        )
        stripped_hash = canonical_hash(identity)
        path = wired.ledger.root / "execution-ledgers" / f"{stripped_hash}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            json.dumps(
                {**identity, "ledger_hash": stripped_hash},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        stripped_result = result.model_dump(mode="json", exclude={"result_hash"}, exclude_none=True)
        stripped_result["execution_ledger_hash"] = stripped_hash
        stripped_result_hash = canonical_hash(
            {k: v for k, v in stripped_result.items() if k != "action"}
        )
        result_path = wired.ledger.root / "results" / f"{stripped_result_hash}.json"
        result_path.write_bytes(
            json.dumps(
                {**stripped_result, "result_hash": stripped_result_hash},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        receipt = strong_replay(
            wired.ledger,
            workspace_id="qa-final",
            result_hash=stripped_result_hash,
            spec=wired.spec,
            coverage=wired.coverage,
        )

    assert receipt.disposition == "REFUSED_UNSEALED_CHILDREN"
    assert receipt.work.total == 0
    assert all(v.disposition == "LEGACY" for v in receipt.layers)
    assert "not the inputs" in receipt.layer("one_way_turnover").detail


def test_readback_reports_a_finalization_artifact_that_will_not_open(
    tmp_path: Path,
) -> None:
    """An opener that cannot produce the artifact is a failure, not a PARTIAL."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        finalized = wired.application.finalize(candidate=candidate, fixture=_fixture())
        opener = HostFinalizationArtifactOpener(
            store=wired.store,
            validation=ProtectedValidationStore(session.workspace / "runtime" / "artifacts"),
        )
        # Remove the package's content *and* its index entry, so the healing
        # path cannot reconstruct it either.
        (wired.store.root / "packages" / f"{finalized.package.package_hash}.json").unlink()
        wired.store.committed.path("by-permit", finalized.permit.permit_hash).unlink()
        receipt = exact_readback(
            wired.ledger,
            workspace_id="qa-final",
            result_hash=finalized.package.protected_result_hash,
            package_hash=finalized.package.package_hash,
            validation_receipt_hash=finalized.receipt.receipt_hash,
            handoff_hash=finalized.handoff.handoff_hash,
            finalization=opener,
        )

    assert receipt.layer("final_evaluation_package").disposition == "FAILED"
    for layer_id in (
        "final_evaluation_package",
        "validation_receipt",
        "validated_handoff",
        "released_artifacts",
    ):
        assert receipt.layer(layer_id).disposition == "FAILED", layer_id
    assert receipt.work.total == 0


# ===================================================================
# Remediation 2: the tranche schedule survives the segment boundary
# ===================================================================

WHOLE_SESSIONS = (
    date(2024, 1, 2),
    date(2024, 1, 3),
    date(2024, 1, 4),
    date(2024, 1, 5),
    date(2024, 1, 8),
    date(2024, 1, 9),
    date(2024, 1, 10),
    date(2024, 1, 11),
)
"""One calendar, split two ways. Eight formations against three tranches, so a
prefix of 3 lands on the cycle and a prefix of 4 does not."""


def _run_axis(
    store: PortfolioLedgerStore,
    *,
    sessions: tuple[date, ...],
    spec: PortfolioResearchSpec,
    workspace: Path,
    carry: PortfolioContinuationCarry | None = None,
):  # type: ignore[no-untyped-def]
    """Execute one segment over one axis, returning result, ledger and provider."""

    prepared = _session_keyed_resolved(sessions)
    resolver = _Resolver(prepared, numerical=prepared)
    authorities = resolver.resolve_authorities(workspace=workspace, spec=spec)
    resolution = resolver.resolve(workspace=workspace, spec=spec)
    authorities.require_resolution(resolution)
    program = PortfolioResearchCompiler().compile(
        spec=spec,
        authorities=authorities,
        resolved=resolution,
        continued_from_state_hash=None if carry is None else carry.boundary.boundary_hash,
    )
    executor = PortfolioResearchExecutor(store, packages=_PACKAGES)
    result = executor.execute(
        workspace_id="qa-schedule",
        spec=spec,
        program=program,
        resolved=lambda: resolution,
        coverage=authorities.coverage,
        authorities_hash=authorities.authorities_hash,
        continuation=carry,
    )
    return result, store.load_execution(result.execution_ledger_hash), authorities.coverage


def _carry_from(store: PortfolioLedgerStore, ledger, listings: int):  # type: ignore[no-untyped-def]
    """Rehydrate a continuation carry from a sealed terminal boundary."""

    boundary = ledger.final_boundary
    assert boundary is not None

    def lane(category: str, content_hash: str, size: int) -> npt.NDArray[np.float64]:
        values: npt.NDArray[np.float64] = np.frombuffer(
            store.load_lane(category=category, content_hash=content_hash), dtype="<f8"
        )
        assert values.size == size
        return values.astype(np.float64, copy=True)

    sleeves = None
    if boundary.sleeve_weights_hash is not None:
        sleeves = lane(
            "boundary-sleeves", boundary.sleeve_weights_hash, boundary.sleeve_count * listings
        ).reshape((boundary.sleeve_count, listings))
    return PortfolioContinuationCarry(
        boundary=boundary,
        state=PortfolioWalkForwardState(
            pretrade_weights=lane("boundary-weights", boundary.pretrade_weights_hash, listings),
            pretrade_cash=boundary.pretrade_cash,
            optimizer_reference=lane(
                "boundary-weights", boundary.optimizer_reference_hash, listings
            ),
            optimizer_reference_cash=boundary.optimizer_reference_cash,
        ),
        sleeve_weights=sleeves,
    )


@dataclass(frozen=True, slots=True)
class _SplitPath:
    """One calendar run two ways: whole, and prefix + carried suffix."""

    whole_store: PortfolioLedgerStore
    whole_report: PortfolioDeclaredPathReport
    suffix_store: PortfolioLedgerStore
    suffix_result_hash: str
    suffix_ledger: PortfolioExecutionLedger
    suffix_report: PortfolioDeclaredPathReport
    suffix_spec: PortfolioResearchSpec
    suffix_coverage: PortfolioSupportCoverage
    boundary_session: date


def _split_at(tmp_path: Path, *, prefix: int, window_end: int) -> _SplitPath:
    """Run one axis whole and split, and report both at the same formation.

    The window is a descendant control, so it never enters the Program: both arms
    compile the same holdings identity and the split is genuinely the only
    difference between them.
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    listings = 80
    spec = PortfolioResearchSpec.create(
        study_start=WHOLE_SESSIONS[window_end], study_end=WHOLE_SESSIONS[window_end]
    )
    whole_store = PortfolioLedgerStore(tmp_path / "whole")
    whole_result, _whole_ledger, _ = _run_axis(
        whole_store, sessions=WHOLE_SESSIONS, spec=spec, workspace=workspace
    )

    prefix_store = PortfolioLedgerStore(tmp_path / "split")
    _prefix_result, prefix_ledger, _ = _run_axis(
        prefix_store,
        sessions=WHOLE_SESSIONS[:prefix],
        spec=PortfolioResearchSpec.default(),
        workspace=workspace,
    )
    carry = _carry_from(prefix_store, prefix_ledger, listings)
    suffix_store = PortfolioLedgerStore(tmp_path / "split-suffix")
    for category, content_hash in (
        ("boundary-weights", carry.boundary.pretrade_weights_hash),
        ("boundary-weights", carry.boundary.optimizer_reference_hash),
        ("boundary-sleeves", carry.boundary.sleeve_weights_hash),
    ):
        if content_hash is not None:
            suffix_store.publish_lane(
                category=category,
                values=np.frombuffer(
                    prefix_store.load_lane(category=category, content_hash=content_hash),
                    dtype="<f8",
                ),
            )
    suffix_result, suffix_ledger, coverage = _run_axis(
        suffix_store,
        sessions=WHOLE_SESSIONS[prefix:],
        spec=spec,
        workspace=workspace,
        carry=carry,
    )
    return _SplitPath(
        whole_store=whole_store,
        whole_report=whole_store.load_report(
            whole_store.load_result(whole_result.result_hash).report_hash
        ),
        suffix_store=suffix_store,
        suffix_result_hash=suffix_result.result_hash,
        suffix_ledger=suffix_ledger,
        suffix_report=suffix_store.load_report(
            suffix_store.load_result(suffix_result.result_hash).report_hash
        ),
        suffix_spec=spec,
        suffix_coverage=coverage,
        boundary_session=WHOLE_SESSIONS[prefix - 1],
    )


@pytest.mark.parametrize("prefix", [3, 4])
def test_strong_replay_reopens_the_boundary_and_verifies_the_continuation_report(
    tmp_path: Path, prefix: int
) -> None:
    """The sealed opening, uninterrupted book and replay agree at both split positions."""

    split = _split_at(tmp_path, prefix=prefix, window_end=prefix)
    book = split.suffix_report.window_end_book
    opening = split.suffix_store.load_opening_reference(split.suffix_ledger)
    listings = tuple(split.suffix_ledger.ordered_listing_ids)
    carried = {listings[index]: float(opening[index]) for index in range(len(listings))}

    assert bool(opening.any()), "the fixture must actually carry a book"
    assert book.change_boundary == "SEALED_CONTINUATION_BOUNDARY"
    assert book.preceding_formation_session == split.boundary_session
    assert book.formation_session == WHOLE_SESSIONS[prefix]

    # Every position is measured against the exact sealed boundary weight.
    for position in book.positions:
        assert position.preceding_weight == carried[position.listing_id]
        assert position.weight_change == position.weight - position.preceding_weight

    dispositions = {position.disposition for position in book.positions}
    # All four kinds of change are present, so none of them is being reported by
    # accident: a flat comparand can only ever produce OPENED.
    assert {"OPENED", "EXITED", "INCREASED", "REDUCED"} <= dispositions
    assert book.opened_count and book.exited_count
    assert book.opened_count < book.held_count, "most of this book was carried in"
    assert book.absolute_weight_change_total == sum(
        abs(position.weight_change) for position in book.positions
    )

    page = split.suffix_store.load_html_by_uri(
        split.suffix_store.load_result(split.suffix_result_hash).html_uri
    )
    assert "SEALED_CONTINUATION_BOUNDARY" in page
    assert "the sealed boundary this path continued from" in page
    assert "opened flat" not in page

    whole = split.whole_report.window_end_book
    suffix = split.suffix_report.window_end_book

    assert whole.change_boundary == "PRECEDING_FORMATION"
    assert suffix.change_boundary == "SEALED_CONTINUATION_BOUNDARY"
    assert whole.preceding_formation_session == suffix.preceding_formation_session
    assert whole.formation_session == suffix.formation_session
    assert [position.model_dump() for position in whole.positions] == [
        position.model_dump() for position in suffix.positions
    ]
    assert (whole.held_count, whole.opened_count, whole.exited_count) == (
        suffix.held_count,
        suffix.opened_count,
        suffix.exited_count,
    )
    assert whole.absolute_weight_change_total == suffix.absolute_weight_change_total
    assert whole.listing_axis_hash == suffix.listing_axis_hash

    receipt = strong_replay(
        split.suffix_store,
        workspace_id="qa-final",
        result_hash=split.suffix_result_hash,
        spec=split.suffix_spec,
        coverage=split.suffix_coverage,
    )

    assert receipt.disposition == "REPLAYED_EXACT"
    for layer in receipt.layers:
        assert layer.disposition == "VERIFIED", (layer.layer_id, layer.detail)
    # The rebuilt report matched by identity, and that identity carries a book
    # measured against the carried boundary rather than an assumed flat one.
    book = split.suffix_report.window_end_book
    assert book.change_boundary == "SEALED_CONTINUATION_BOUNDARY"
    assert book.exited_count > 0


def test_a_replay_without_its_sealed_opening_lane_fails_closed(tmp_path: Path) -> None:
    """Missing the opening book is a refusal, never an assumed flat one."""

    split = _split_at(tmp_path, prefix=3, window_end=3)
    boundary = split.suffix_ledger.initial_boundary
    assert boundary is not None
    lane = (
        split.suffix_store.root
        / "lanes"
        / "boundary-weights"
        / f"{boundary.optimizer_reference_hash}.parquet"
    )
    assert lane.exists()
    lane.unlink()

    with pytest.raises(ReadbackError, match="sealed_opening_boundary_unreadable"):
        strong_replay(
            split.suffix_store,
            workspace_id="qa-final",
            result_hash=split.suffix_result_hash,
            spec=split.suffix_spec,
            coverage=split.suffix_coverage,
        )


@pytest.mark.parametrize("prefix", [3, 4])
def test_a_continued_segment_matches_the_uninterrupted_path_exactly(
    tmp_path: Path, prefix: int
) -> None:
    """Splitting a path must not change it. Two prefixes, one aligned, one not.

    Three tranches and eight formations. At `prefix=3` the next scheduled sleeve
    is 0 and at `prefix=4` it is 1, so an implementation that resets the schedule
    is wrong in a different way in each arm -- and is wrong at all only because
    position zero means "this book has never traded".

    Everything the suffix produced is compared: the reviewed sleeves, the target
    books the policy asked for, the executed books, the turnovers, and the sealed
    terminal boundary. The boundary is the sharpest of these, because its hash
    covers both weight lanes, both cash balances and the schedule position at
    once -- if any of them drifted, the two paths end at different identities.
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    # The window is a descendant control, so it does not enter the Program and
    # the prefix run produces the same ledger under either spec. It is set to the
    # suffix on both compared arms so the two reports describe the same sessions,
    # and left alone on the prefix run, whose axis ends before that window opens.
    spec = PortfolioResearchSpec.create(study_start=WHOLE_SESSIONS[prefix])
    prefix_spec = PortfolioResearchSpec.default()
    listings = 80
    suffix = WHOLE_SESSIONS[prefix:]

    whole_store = PortfolioLedgerStore(tmp_path / "whole")
    _whole_result, whole_ledger, _ = _run_axis(
        whole_store, sessions=WHOLE_SESSIONS, spec=spec, workspace=workspace
    )

    prefix_store = PortfolioLedgerStore(tmp_path / "split")
    _prefix_result, prefix_ledger, _ = _run_axis(
        prefix_store, sessions=WHOLE_SESSIONS[:prefix], spec=prefix_spec, workspace=workspace
    )
    assert prefix_spec.holdings_spec_hash == spec.holdings_spec_hash
    carry = _carry_from(prefix_store, prefix_ledger, listings)
    suffix_store = PortfolioLedgerStore(tmp_path / "split-suffix")
    for category, content_hash in (
        ("boundary-weights", carry.boundary.pretrade_weights_hash),
        ("boundary-weights", carry.boundary.optimizer_reference_hash),
        ("boundary-sleeves", carry.boundary.sleeve_weights_hash),
    ):
        if content_hash is not None:
            suffix_store.publish_lane(
                category=category,
                values=np.frombuffer(
                    prefix_store.load_lane(category=category, content_hash=content_hash),
                    dtype="<f8",
                ),
            )
    _suffix_result, suffix_ledger, _ = _run_axis(
        suffix_store, sessions=suffix, spec=spec, workspace=workspace, carry=carry
    )

    # The schedule resumed rather than restarted.
    assert carry.boundary.decided_formation_count == prefix
    assert suffix_ledger.initial_boundary is not None
    assert suffix_ledger.initial_boundary.decided_formation_count == prefix

    def lanes(store: PortfolioLedgerStore, ledger, category: str, content_hash: str):  # type: ignore[no-untyped-def]
        return np.frombuffer(
            store.load_lane(category=category, content_hash=content_hash), dtype="<f8"
        ).reshape(ledger.executed_weights_shape)

    whole_executed = lanes(
        whole_store, whole_ledger, "executed-weights", whole_ledger.executed_weights_hash
    )[prefix:]
    suffix_executed = lanes(
        suffix_store, suffix_ledger, "executed-weights", suffix_ledger.executed_weights_hash
    )
    whole_targets = lanes(
        whole_store, whole_ledger, "target-weights", whole_ledger.target_weights_hash or ""
    )[prefix:]
    suffix_targets = lanes(
        suffix_store, suffix_ledger, "target-weights", suffix_ledger.target_weights_hash or ""
    )
    whole_pre_cap = lanes(
        whole_store, whole_ledger, "pre-cap-weights", whole_ledger.pre_cap_weights_hash
    )[prefix:]
    suffix_pre_cap = lanes(
        suffix_store, suffix_ledger, "pre-cap-weights", suffix_ledger.pre_cap_weights_hash
    )

    assert np.array_equal(suffix_targets, whole_targets), "the policy asked for a different book"
    assert np.array_equal(suffix_pre_cap, whole_pre_cap)
    assert np.array_equal(suffix_executed, whole_executed)
    assert suffix_ledger.one_way_turnovers == whole_ledger.one_way_turnovers[prefix:]
    assert suffix_ledger.gross_simple_returns == whole_ledger.gross_simple_returns[prefix:]
    assert suffix_ledger.decision_modes == whole_ledger.decision_modes[prefix:]
    assert (
        suffix_ledger.aggregate_cap_binding_counts
        == whole_ledger.aggregate_cap_binding_counts[prefix:]
    )
    assert suffix_ledger.risk_facts == whole_ledger.risk_facts[prefix:]
    # One identity covering both books, both cash balances and the schedule.
    assert suffix_ledger.final_boundary == whole_ledger.final_boundary

    # And the declared report over the same window agrees fact for fact.
    whole_report = whole_store.load_report(_whole_result.report_hash)
    suffix_report = suffix_store.load_report(_suffix_result.report_hash)
    assert suffix_report.window_guard.selected_start == whole_report.window_guard.selected_start
    assert suffix_report.window_guard.selected_end == whole_report.window_guard.selected_end
    assert suffix_report.window_end_distinct_names == whole_report.window_end_distinct_names
    assert suffix_report.window_end_effective_n == whole_report.window_end_effective_n
    assert suffix_report.window_cumulative_net_wealth == (whole_report.window_cumulative_net_wealth)
    assert suffix_report.mean_one_way_turnover == whole_report.mean_one_way_turnover
    assert suffix_report.risk_facts == whole_report.risk_facts
    # `window_unit_rows` is deliberately *not* compared. The report unit is
    # beta-stripped, and the Backtesting owner estimates beta once over the whole
    # materialized path rather than inside the selected window -- so an eight
    # formation path and a four formation path have different betas by contract,
    # and the rows would differ even if every fill were identical. The rows are a
    # fact about the path; everything asserted above is a fact about the window.
    assert len(suffix_report.window_unit_rows) == len(whole_report.window_unit_rows)


@pytest.mark.parametrize("prefix", [3, 4])
def test_the_first_continued_formation_does_not_restage_every_sleeve(
    tmp_path: Path, prefix: int
) -> None:
    """The failure this closes, stated directly and measured on the policy.

    A reset schedule reviews all three sleeves on the first continued formation
    and therefore rebuilds the whole book; a resumed one reviews exactly the
    sleeve that is next. Read off `due_sleeves` at both positions, and then off
    the turnover the continued segment actually paid.
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = PortfolioResearchSpec.default()
    listings = 80

    prefix_store = PortfolioLedgerStore(tmp_path / "prefix")
    _r, prefix_ledger, _c = _run_axis(
        prefix_store, sessions=WHOLE_SESSIONS[:prefix], spec=spec, workspace=workspace
    )
    carry = _carry_from(prefix_store, prefix_ledger, listings)
    suffix_store = PortfolioLedgerStore(tmp_path / "suffix")
    for category, content_hash in (
        ("boundary-weights", carry.boundary.pretrade_weights_hash),
        ("boundary-weights", carry.boundary.optimizer_reference_hash),
        ("boundary-sleeves", carry.boundary.sleeve_weights_hash),
    ):
        if content_hash is not None:
            suffix_store.publish_lane(
                category=category,
                values=np.frombuffer(
                    prefix_store.load_lane(category=category, content_hash=content_hash),
                    dtype="<f8",
                ),
            )
    _r2, suffix_ledger, _c2 = _run_axis(
        suffix_store,
        sessions=WHOLE_SESSIONS[prefix:],
        spec=spec,
        workspace=workspace,
        carry=carry,
    )

    # What the schedule says at the two positions.
    assert due_sleeves(formation_index=0, tranches=spec.tranches) == tuple(range(spec.tranches))
    assert due_sleeves(formation_index=prefix, tranches=spec.tranches) == (prefix % spec.tranches,)
    # And what the continued segment actually paid: a single-sleeve review
    # cannot cost what a full restage costs.
    first_continued = suffix_ledger.one_way_turnovers[0]
    opening = prefix_ledger.one_way_turnovers[0]
    assert first_continued < opening / 2.0, (first_continued, opening)


# ===================================================================
# Remediation 2: a legacy Program authorizes no reuse anywhere
# ===================================================================


def _serialize_legacy_path(
    wired: _Finalization, target: PortfolioLedgerStore | None = None
) -> tuple[str, str, str]:
    """Write a predecessor Program, ledger and result, at their own hashes.

    Serialized, not patched. The whole question is what happens when an artifact
    written before the assembly existed is *on disk* and something asks to reuse
    it, and a monkeypatched class cannot pose that question.
    """

    store = target or wired.ledger
    result = wired.ledger.load_result(wired.result_hash)
    program = wired.ledger.load_program(result.program_hash)
    ledger = wired.ledger.load_execution(result.execution_ledger_hash)

    def seal(
        category: str,
        identity_field: str,
        identity: dict[str, object],
        extra: dict[str, object] | None = None,
    ) -> str:
        """Write one artifact at the hash its own validator will recompute.

        `extra` carries fields the contract requires but excludes from identity
        -- a result's `action` is the only one -- so the file validates while the
        hash stays the one the predecessor writer would have produced.
        """

        content_hash = canonical_hash(identity)
        path = store.root / category / f"{content_hash}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            json.dumps(
                {**identity, **(extra or {}), identity_field: content_hash},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        return content_hash

    legacy_program = seal(
        "programs",
        "program_hash",
        program.model_dump(
            mode="json",
            exclude={
                "program_hash",
                "numerical_input_assembly_hash",
                "continued_from_state_hash",
            },
        ),
    )
    legacy_ledger = seal(
        "execution-ledgers",
        "ledger_hash",
        {
            **ledger.model_dump(mode="json", exclude={"ledger_hash"}),
            "program_hash": legacy_program,
        },
    )
    legacy_result = seal(
        "results",
        "result_hash",
        {
            k: v
            for k, v in result.model_dump(
                mode="json", exclude={"result_hash"}, exclude_none=True
            ).items()
            if k != "action"
        }
        | {"program_hash": legacy_program, "execution_ledger_hash": legacy_ledger},
        {"action": "PUBLISHED"},
    )
    if store is not wired.ledger:
        # The descendants the legacy result names are unchanged, so they are
        # copied rather than re-sealed: readback has to be able to walk the whole
        # predecessor path, which is exactly the thing that stays available.
        store.publish_economics(wired.ledger.load_economics(result.economic_ledger_hash))
        store.publish_report(wired.ledger.load_report(result.report_hash))
        store.publish_html(wired.ledger.load_html_by_uri(result.html_uri))
        for category, content_hash in (
            ("executed-weights", ledger.executed_weights_hash),
            ("pre-cap-weights", ledger.pre_cap_weights_hash),
            ("final-weights", ledger.final_weights_hash),
        ):
            store.publish_lane(
                category=category,
                values=np.frombuffer(
                    wired.ledger.load_lane(category=category, content_hash=content_hash),
                    dtype="<f8",
                ),
            )
    return legacy_program, legacy_ledger, legacy_result


def test_a_legacy_program_authorizes_no_reuse_on_any_route(tmp_path: Path) -> None:
    """Four reuse routes, four refusals, and readback still open.

    Reuse is not a performance detail here -- it is the claim that a stored
    answer is the answer to the question being asked. A path that cannot name
    the inputs its numbers came from cannot support that claim on any route, so
    each one refuses under the same name rather than three of them refusing and
    the fourth quietly returning a hit.
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        # Its own root: the live path already owns the by-holdings key, and one
        # program per key is a different test's subject. Here the question is
        # what the reuse routes do once a predecessor path is what they find.
        legacy = PortfolioLedgerStore(tmp_path / "legacy-root")
        legacy_program, _legacy_ledger, legacy_result = _serialize_legacy_path(wired, legacy)
        reopened = legacy.load_program(legacy_program)
        assert not reopened.replayable

        # 1. result reuse
        with pytest.raises(ValueError, match="program_assembly_absent_no_reuse"):
            legacy.find_result_for(program_hash=legacy_program, spec_hash=wired.spec.spec_hash)
        # 2. execution-ledger reuse
        with pytest.raises(ValueError, match="program_assembly_absent_no_reuse"):
            legacy.find_execution_for_program(legacy_program)
        # 3. PLAN / unit reuse -- both the by-holdings and the by-plan routes
        legacy.publish_program_index(
            holdings_spec_hash=reopened.holdings_spec_hash,
            authorities_hash=reopened.authorities_hash or "0" * 64,
            program_hash=legacy_program,
        )
        with pytest.raises(ValueError, match="program_assembly_absent_no_reuse"):
            legacy.find_program_for(
                holdings_spec_hash=reopened.holdings_spec_hash,
                authorities_hash=reopened.authorities_hash or "0" * 64,
            )
        legacy.publish_plan_index(
            spec_hash=wired.spec.spec_hash,
            authorities_hash=reopened.authorities_hash or "0" * 64,
            result_hash=legacy_result,
        )
        with pytest.raises(ValueError, match="program_assembly_absent_no_reuse"):
            legacy.find_planned_result(
                spec_hash=wired.spec.spec_hash,
                authorities_hash=reopened.authorities_hash or "0" * 64,
            )
        # 4. protected finalization
        with pytest.raises(
            PortfolioFinalizationCompositionError, match="candidate_program_assembly_absent"
        ):
            freeze_candidate(
                legacy,
                wired.store,
                workspace_id="qa-final",
                result_hash=legacy_result,
                development_task_id=wired.development_task_id,
                tasks=HostCompletedDevelopmentTasks(session.task_control_registry),
            )

        # Readback is untouched: the artifacts open by identity, and the receipt
        # says plainly which layer cannot be proved.
        receipt = exact_readback(legacy, workspace_id="qa-final", result_hash=legacy_result)
        assert receipt.work.total == 0
        assert receipt.layer("numerical_input_assembly").disposition == "LEGACY"
        assert receipt.layer("portfolio_program").disposition == "VERIFIED"


def test_the_executor_refuses_a_legacy_program_before_touching_an_index(
    tmp_path: Path,
) -> None:
    """The refusal lands before either reuse lookup and before the thunk runs."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        legacy_program, _ledger, _result = _serialize_legacy_path(wired)
        reopened = wired.ledger.load_program(legacy_program)

        def _must_not_resolve():  # type: ignore[no-untyped-def]
            raise AssertionError("the resolution thunk was evaluated")

        with pytest.raises(ValueError, match="program_assembly_absent_no_reuse"):
            PortfolioResearchExecutor(wired.ledger, packages=_PACKAGES).execute(
                workspace_id="qa-final",
                spec=wired.spec,
                program=reopened,
                resolved=_must_not_resolve,
                coverage=_Resolver(_resolved(sessions=DEVELOPMENT_SESSIONS))
                .resolve_authorities(workspace=workspace, spec=wired.spec)
                .coverage,
                authorities_hash=reopened.authorities_hash or "0" * 64,
            )


# ===================================================================
# Remediation 2: the development task is verified, not quoted
# ===================================================================


def test_freezing_verifies_the_completed_development_task(tmp_path: Path) -> None:
    """The happy path, and what the projection actually binds."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        tasks = HostCompletedDevelopmentTasks(session.task_control_registry)
        run = tasks.open_completed_run(wired.development_task_id)
        assert run is not None
        assert run.task_kind == PORTFOLIO_PUBLIC_TASK_KIND
        assert run.completed
        assert wired.result_hash in run.published_result_hashes

        candidate = _candidate(wired)
        assert candidate.development_run_hash == run.projection_hash
        # And the registered freeze receipt carries the same verified projection.
        receipt = wired.store.open_freeze_receipt(candidate.candidate_hash)
        assert receipt is not None
        assert receipt.development_run_hash == run.projection_hash


def test_freezing_refuses_a_task_that_does_not_exist(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        with pytest.raises(
            PortfolioFinalizationCompositionError, match="development_task_not_found"
        ):
            freeze_candidate(
                wired.ledger,
                wired.store,
                workspace_id="qa-final",
                result_hash=wired.result_hash,
                development_task_id=str(uuid4()),
                tasks=HostCompletedDevelopmentTasks(session.task_control_registry),
            )


def test_freezing_refuses_a_task_of_the_wrong_kind(tmp_path: Path) -> None:
    """A finished task of some other kind is still not a development run."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        finalized = wired.application.finalize(candidate=candidate, fixture=_fixture())
        # The *finalization* task completed too, and is emphatically not the
        # development task the frozen artifacts came from.
        with pytest.raises(
            PortfolioFinalizationCompositionError, match="development_task_wrong_kind"
        ):
            freeze_candidate(
                wired.ledger,
                wired.store,
                workspace_id="qa-final",
                result_hash=wired.result_hash,
                development_task_id=str(finalized.task_id),
                tasks=HostCompletedDevelopmentTasks(session.task_control_registry),
            )


def test_freezing_refuses_an_unfinished_task(tmp_path: Path) -> None:
    """Admitted is not completed, and a queued task has published nothing."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        result = wired.ledger.load_result(wired.result_hash)
        program = wired.ledger.load_program(result.program_hash)
        envelope, goal, plan = portfolio_research_task_contract(
            workspace_id="qa-final",
            spec=wired.spec,
            program=program,
            admission_hash="9" * 64,
            authorities_hash=program.authorities_hash or "0" * 64,
            workspace_manifest_hash="a" * 64,
            strategy_catalog_hash=_Resolver(_resolved()).strategy_catalog_hash,
            selected_strategy_package_id=TEST_PACKAGE.strategy_id,
        )
        admitted = session.task_control_registry.admit(
            input_envelope=envelope, goal=goal, plan=plan, observed_at=datetime.now(UTC)
        )
        tasks = HostCompletedDevelopmentTasks(session.task_control_registry)
        # It is the right kind and it exists; it simply has not published
        # anything, so the projection refuses to describe it as a run at all.
        assert tasks.open_completed_run(str(admitted.record.task_id)) is None
        with pytest.raises(
            PortfolioFinalizationCompositionError, match="development_task_not_found"
        ):
            freeze_candidate(
                wired.ledger,
                wired.store,
                workspace_id="qa-final",
                result_hash=wired.result_hash,
                development_task_id=str(admitted.record.task_id),
                tasks=HostCompletedDevelopmentTasks(session.task_control_registry),
            )


def test_freezing_refuses_a_result_the_named_task_did_not_publish(tmp_path: Path) -> None:
    """Two development runs, and the wrong one is named for the artifacts."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        # A second development path over a different axis: a real, completed
        # development task that published a different result.
        other = PortfolioResearchApplication(
            workspace_id="qa-final",
            workspace=workspace,
            manifest_binding=lambda: "a" * 64,
            session=session,
            resolver=_Resolver(_resolved(sessions=PROTECTED_SESSIONS)),
        )
        completed = other.run(spec=wired.spec)
        other_task_id = str(completed.pipeline_manifest.task_id)
        assert completed.result.result_hash != wired.result_hash

        tasks = HostCompletedDevelopmentTasks(session.task_control_registry)
        with pytest.raises(
            PortfolioFinalizationCompositionError,
            match="development_task_did_not_publish_this_result",
        ):
            freeze_candidate(
                wired.ledger,
                wired.store,
                workspace_id="qa-final",
                result_hash=wired.result_hash,
                development_task_id=other_task_id,
                tasks=tasks,
            )
        # And the mirror: the other task's own result freezes cleanly under it.
        frozen = freeze_candidate(
            wired.ledger,
            wired.store,
            workspace_id="qa-final",
            result_hash=completed.result.result_hash,
            development_task_id=other_task_id,
            tasks=tasks,
        )
        assert frozen.development_result_hash == completed.result.result_hash


# ===================================================================
# Remediation 2: closure opens the package's children
# ===================================================================


def _open_gate(session, wired: _Finalization) -> ProtectedValidationGate:  # type: ignore[no-untyped-def]
    return ProtectedValidationGate(
        store=ProtectedValidationStore(session.workspace / "runtime" / "artifacts"),
        registry=wired.store,
        inspector=PendingPackageInspector(wired.pending),
    )


def _forged_package(permit, candidate, fixture, **overrides):  # type: ignore[no-untyped-def]
    """A package that is perfect about itself and wrong about its children."""

    values: dict[str, object] = {
        "permit_hash": permit.permit_hash,
        "candidate_hash": candidate.candidate_hash,
        "continued_from_state_hash": candidate.pre_protected_state.state_hash,
        "fixture_hash": fixture.fixture_hash,
        "protected_result_hash": "a" * 64,
        "protected_execution_ledger_hash": "b" * 64,
        "protected_economic_ledger_hash": "c" * 64,
        "protected_report_hash": "d" * 64,
        "protected_formation_count": len(fixture.sessions),
        "protected_listing_count": len(LISTINGS),
    }
    values.update(overrides)
    return FinalPortfolioEvaluationPackage.create(**values)


def test_a_package_naming_children_that_do_not_exist_is_refused(tmp_path: Path) -> None:
    """The forgery this closes: every hash well-formed, nothing behind any of them.

    The package validates its own identity, names the right permit, the right
    candidate, the right frozen state and the right fixture. Before closure
    opened the children, that was enough -- so a caller could mint a package out
    of arbitrary hex and have the Gate seal a receipt over it.

    And the permit is spent either way. A refused closure that left the
    authorization intact would turn a forged package into a free retry.
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        fixture = _fixture()
        gate = _open_gate(session, wired)
        permit = gate.admit(candidate=candidate, fixture=fixture)
        forged = _forged_package(permit, candidate, fixture)

        receipt = gate.close(permit=permit, candidate=candidate, package=forged)
        assert receipt.closure == "REFUSED_PACKAGE_CHILDREN_ABSENT"
        assert not receipt.closed
        assert CLAIM_LIMIT_NO_RESELECTION not in receipt.claim_limits
        # Spent.
        with pytest.raises(ProtectedValidationGateError, match="permit_already_claimed"):
            gate.close(permit=permit, candidate=candidate, package=forged)


def test_a_package_borrowing_another_finalizations_children_is_refused(
    tmp_path: Path,
) -> None:
    """Real artifacts, from the wrong run. Existence is not enough.

    Every child here opens. They belong to a *different* finalization, so the
    package's own claims and the result's descendants disagree -- which is
    exactly the check that needs all four opened at once.
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        first = wired.application.finalize(candidate=_candidate(wired), fixture=_fixture())

        # A second, genuine development path and candidate in the same workspace.
        second_development = PortfolioResearchApplication(
            workspace_id="qa-final",
            workspace=workspace,
            manifest_binding=lambda: "a" * 64,
            session=session,
            resolver=_Resolver(_resolved(sessions=DEVELOPMENT_SESSIONS[:2])),
        )
        completed = second_development.run(spec=wired.spec)
        second_candidate = freeze_candidate(
            wired.ledger,
            wired.store,
            workspace_id="qa-final",
            result_hash=completed.result.result_hash,
            development_task_id=str(completed.pipeline_manifest.task_id),
            tasks=HostCompletedDevelopmentTasks(session.task_control_registry),
        )
        gate = _open_gate(session, wired)
        permit = gate.admit(candidate=second_candidate, fixture=_fixture())
        # The first finalization's real, existing children, under the second
        # candidate's permit.
        borrowed = _forged_package(
            permit,
            second_candidate,
            _fixture(),
            protected_result_hash=first.package.protected_result_hash,
            protected_execution_ledger_hash="b" * 64,
            protected_economic_ledger_hash=first.package.protected_economic_ledger_hash,
            protected_report_hash=first.package.protected_report_hash,
        )
        receipt = gate.close(permit=permit, candidate=second_candidate, package=borrowed)

    assert receipt.closure in {
        "REFUSED_PACKAGE_CHILDREN_ABSENT",
        "REFUSED_PACKAGE_BINDINGS_INCOHERENT",
    }
    assert not receipt.closed


def test_a_package_whose_children_do_not_refer_to_each_other_is_refused(
    tmp_path: Path,
) -> None:
    """All four children exist and open; the result names a different report."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        fixture = _fixture()
        gate = _open_gate(session, wired)
        permit = gate.admit(candidate=candidate, fixture=fixture)
        # Run the protected path so real children exist, then claim a report
        # that exists but is not this result's.
        real = wired.continuation.evaluate(permit=permit, candidate=candidate)
        development_report = wired.ledger.load_result(wired.result_hash).report_hash
        wired.pending.publish_report(wired.ledger.load_report(development_report))
        incoherent = _forged_package(
            permit,
            candidate,
            fixture,
            protected_result_hash=real.protected_result_hash,
            protected_execution_ledger_hash=real.execution_ledger_hash,
            protected_economic_ledger_hash=real.economic_ledger_hash,
            protected_report_hash=development_report,
        )
        receipt = gate.close(permit=permit, candidate=candidate, package=incoherent)

    assert receipt.closure == "REFUSED_PACKAGE_BINDINGS_INCOHERENT"
    assert not receipt.closed


def test_a_package_over_the_wrong_axis_is_refused(tmp_path: Path) -> None:
    """The children are coherent and real; they describe another window.

    The permit names an exact fixture -- these sessions, this listing axis, this
    count. A protected run over a different axis is a different evaluation, and
    the only place that can be caught is against the ledger the package points
    at.
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        # The continuation runs over the full protected axis, but the permit is
        # issued for a shorter one.
        narrow = _fixture(sessions=PROTECTED_SESSIONS[:2])
        gate = _open_gate(session, wired)
        permit = gate.admit(candidate=candidate, fixture=narrow)
        wide = _fixture()
        wide_permit_run = wired.continuation.evaluate(
            permit=ProtectedEvaluationPermit.create(
                candidate_hash=candidate.candidate_hash,
                configuration_hash=candidate.spec_hash,
                pre_protected_state_hash=candidate.pre_protected_state.state_hash,
                fixture=wide,
                issued_at=datetime.now(UTC),
            ),
            candidate=candidate,
        )
        mismatched = _forged_package(
            permit,
            candidate,
            narrow,
            protected_result_hash=wide_permit_run.protected_result_hash,
            protected_execution_ledger_hash=wide_permit_run.execution_ledger_hash,
            protected_economic_ledger_hash=wide_permit_run.economic_ledger_hash,
            protected_report_hash=wide_permit_run.report_hash,
            protected_formation_count=len(narrow.sessions),
        )
        receipt = gate.close(permit=permit, candidate=candidate, package=mismatched)

    assert receipt.closure == "REFUSED_PACKAGE_AXIS_MISMATCH"
    assert not receipt.closed


def test_the_inspection_carries_no_metric_and_no_verdict() -> None:
    """The port is narrow by construction, not by convention."""

    fields = set(ProtectedPackageInspection.model_fields)
    assert "closure" not in fields and "disposition" not in fields
    with pytest.raises(ValueError, match="inspection_carries_a_metric"):

        class _Leaky(ProtectedPackageInspection):
            cumulative_return: float = 0.0

        _Leaky.create(package_hash="a" * 64)


# ===================================================================
# Remediation 2: release is atomic at the observable boundary
# ===================================================================


def _public_discovery_routes(wired: _Finalization, package) -> list[object]:  # type: ignore[no-untyped-def]
    """Every ordinary way a public reader could *arrive at* a protected result.

    Discovery and safe projection, not by-identity reads. Copying artifacts into
    the public store is admitted preparation, so a reader who already holds the
    exact hash can open the bytes; what must not exist before the release marker
    is any route that *hands* somebody that hash -- the reuse indices, and the
    released-artifact projection every public surface goes through.
    """

    ledger = wired.ledger
    protected_program = wired.pending.load_execution(
        package.protected_execution_ledger_hash
    ).program_hash
    pending_program = wired.pending.load_program(protected_program)
    routes: list[object] = []
    for call in (
        lambda: ledger.find_result_for(
            program_hash=protected_program, spec_hash=wired.spec.spec_hash
        ),
        lambda: ledger.find_execution_for_program(protected_program),
        lambda: ledger.find_program_for(
            holdings_spec_hash=pending_program.holdings_spec_hash,
            authorities_hash=pending_program.authorities_hash or "0" * 64,
        ),
        lambda: HostReleasedArtifacts(store=wired.store, public=ledger).open_released(
            package.package_hash
        ),
    ):
        try:
            routes.append(call())
        except Exception:
            routes.append(None)
    return routes


@pytest.mark.parametrize(
    "window",
    [
        "during_artifact_adoption",
        "after_copying_before_marker",
        "after_handoff_before_marker",
        "after_marker_before_task_evidence",
    ],
)
def test_release_is_atomic_across_every_window(tmp_path: Path, window: str) -> None:
    """Four crashes through the release, and one released result at the end.

    The guarantee has two halves and both are read off the machine rather than
    argued. Before the marker is committed, no ordinary public route reaches the
    protected result -- not by identity, not through either reuse index. And
    after recovery, the package, receipt, handoff and released result carry the
    identities the first attempt produced, with the protected book walked once.
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        fixture = _fixture()
        store = wired.store
        release = wired.application.release
        tripped = {"done": False}

        def _trip(original):  # type: ignore[no-untyped-def]
            def _wrapped(*args, **kwargs):  # type: ignore[no-untyped-def]
                if not tripped["done"]:
                    tripped["done"] = True
                    raise RuntimeError("crash: " + window)
                return original(*args, **kwargs)

            return _wrapped

        def _trip_after(original):  # type: ignore[no-untyped-def]
            def _wrapped(*args, **kwargs):  # type: ignore[no-untyped-def]
                out = original(*args, **kwargs)
                if not tripped["done"]:
                    tripped["done"] = True
                    raise RuntimeError("crash: " + window)
                return out

            return _wrapped

        if window == "during_artifact_adoption":
            # The store the release actually writes through, not a second
            # handle onto the same root.
            public = wired.application.ledger
            public.adopt_result_from = _trip(public.adopt_result_from)  # type: ignore[method-assign]
        elif window == "after_copying_before_marker":
            release.promote = _trip_after(release.promote)  # type: ignore[method-assign]
        elif window == "after_handoff_before_marker":
            store.publish_release = _trip(store.publish_release)  # type: ignore[method-assign]
        else:
            store.publish_release = _trip_after(store.publish_release)  # type: ignore[method-assign]

        with pytest.raises(BaseException, match=r"crash: |not_succeeded"):
            wired.application.finalize(candidate=candidate, fixture=fixture)
        assert tripped["done"], "the injected interruption never fired"

        first_adapter = wired.application.adapter
        assert first_adapter is not None
        package = first_adapter.package
        assert package is not None
        released_before = store.find_release_for_package(package.package_hash)
        if window != "after_marker_before_task_evidence":
            # No marker, therefore no release -- whatever files happen to exist.
            assert released_before is None
            assert _public_discovery_routes(wired, package) == [None] * 4

        task = next(
            value
            for value in session.task_control_registry.tasks()
            if value.task_kind == PortfolioFinalizationTaskAdapter.task_kind
        )
        finalized = wired.application.recover(
            task_id=task.task_id, candidate=candidate, fixture=fixture
        )
        marker = store.find_release_for_package(finalized.package.package_hash)

    # One finalization, one release, one walk of the protected book.
    assert finalized.task_id == task.task_id
    assert finalized.package.package_hash == package.package_hash
    assert finalized.receipt.closure == "CLOSED_EXACT"
    assert wired.continuation.evaluations == 1
    assert marker is not None
    assert marker.package_hash == finalized.package.package_hash
    assert marker.validation_receipt_hash == finalized.receipt.receipt_hash
    assert marker.handoff_hash == finalized.handoff.handoff_hash
    assert marker.released_result_hash == finalized.package.protected_result_hash
    assert marker.released_report_hash == finalized.package.protected_report_hash
    # And now the artifacts really are in the public store, byte for byte.
    assert wired.ledger.load_result(marker.released_result_hash) == wired.pending.load_result(
        marker.released_result_hash
    )


def test_nothing_public_is_observable_before_the_release_marker(tmp_path: Path) -> None:
    """The ordering itself, stated at the two instants that matter."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        seen: dict[str, object] = {}
        original = wired.application.release.promote

        def _observe(package):  # type: ignore[no-untyped-def]
            # Immediately before the copy: nothing public, no marker.
            seen["before_copy"] = _public_discovery_routes(wired, package)
            seen["marker_before"] = wired.store.find_release_for_package(package.package_hash)
            out = original(package)
            # Immediately after the copy, and still before the marker: the bytes
            # are there and the result is still not released.
            seen["marker_after_copy"] = wired.store.find_release_for_package(package.package_hash)
            seen["result_after_copy"] = wired.ledger.load_result(package.protected_result_hash)
            seen["discovery_after_copy"] = wired.ledger.find_result_for(
                program_hash=wired.pending.load_execution(
                    package.protected_execution_ledger_hash
                ).program_hash,
                spec_hash=wired.spec.spec_hash,
            )
            return out

        wired.application.release.promote = _observe  # type: ignore[method-assign]
        finalized = wired.application.finalize(candidate=candidate, fixture=_fixture())
        marker = wired.store.find_release_for_package(finalized.package.package_hash)

    assert seen["before_copy"] == [None] * 4
    assert seen["marker_before"] is None
    # Copying alone releases nothing: the marker is still absent, and no reuse
    # index discovers the result even though its bytes are now in place.
    assert seen["marker_after_copy"] is None
    assert seen["discovery_after_copy"] is None
    assert seen["result_after_copy"] is not None
    assert marker is not None
    assert marker.handoff_hash == finalized.handoff.handoff_hash


# ===================================================================
# Remediation 2: a real upstream opener, and identity-checked layers
# ===================================================================


def test_finalization_layers_fail_on_another_finalizations_artifacts(tmp_path: Path) -> None:
    """Valid artifacts, wrong run: every mismatched layer must fail.

    Two complete finalizations in one workspace. Each artifact opens; each is
    genuine. Cross them, and the package no longer describes the result being
    read back, the receipt no longer closes the package in hand, and the handoff
    no longer cites either -- which is the whole reason the three are checked
    against each other rather than one at a time.
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        first = wired.application.finalize(candidate=_candidate(wired), fixture=_fixture())

        second_development = PortfolioResearchApplication(
            workspace_id="qa-final",
            workspace=workspace,
            manifest_binding=lambda: "a" * 64,
            session=session,
            resolver=_Resolver(_resolved(sessions=DEVELOPMENT_SESSIONS[:2])),
        )
        completed = second_development.run(spec=wired.spec)
        second_candidate = freeze_candidate(
            wired.ledger,
            wired.store,
            workspace_id="qa-final",
            result_hash=completed.result.result_hash,
            development_task_id=str(completed.pipeline_manifest.task_id),
            tasks=HostCompletedDevelopmentTasks(session.task_control_registry),
        )
        # The same protected fixture: the two finalizations differ by the book
        # they continue, not by the window, which is what makes their artifacts
        # confusable enough to be worth crossing.
        second = wired.application.finalize(candidate=second_candidate, fixture=_fixture())
        opener = HostFinalizationArtifactOpener(
            store=wired.store,
            validation=ProtectedValidationStore(session.workspace / "runtime" / "artifacts"),
        )

        def read(result_hash: str, package, receipt, handoff):  # type: ignore[no-untyped-def]
            return exact_readback(
                wired.ledger,
                workspace_id="qa-final",
                result_hash=result_hash,
                package_hash=package.package_hash,
                validation_receipt_hash=receipt.receipt_hash,
                handoff_hash=handoff.handoff_hash,
                finalization=opener,
            )

        released = first.package.protected_result_hash
        coherent = read(released, first.package, first.receipt, first.handoff)
        wrong_package = read(released, second.package, first.receipt, first.handoff)
        wrong_receipt = read(released, first.package, second.receipt, first.handoff)
        wrong_handoff = read(released, first.package, first.receipt, second.handoff)

    for layer_id in ("final_evaluation_package", "validation_receipt", "validated_handoff"):
        assert coherent.layer(layer_id).disposition == "VERIFIED", layer_id
    assert wrong_package.layer("final_evaluation_package").disposition == "FAILED"
    assert "another result" in wrong_package.layer("final_evaluation_package").detail
    assert wrong_receipt.layer("validation_receipt").disposition == "FAILED"
    assert "different package" in wrong_receipt.layer("validation_receipt").detail
    assert wrong_handoff.layer("validated_handoff").disposition == "FAILED"
    assert "different package" in wrong_handoff.layer("validated_handoff").detail


# ===================================================================
# Closure batch: the frozen configuration binds the protected run
# ===================================================================


@pytest.mark.parametrize(
    ("label", "overrides"),
    [
        ("cost", {"cost_bps_per_side": "10"}),
        ("study_window", {"study_start": PROTECTED_SESSIONS[1]}),
        ("report_unit", {"report_unit": "SIMPLE_CUMULATIVE"}),
        ("holdings_control", {"top_k": 30}),
    ],
)
def test_a_continuation_under_another_configuration_does_no_protected_work(
    tmp_path: Path, label: str, overrides: dict[str, object]
) -> None:
    """A protected run is permitted for one configuration, and only that one.

    Cost, the study window, the report unit and every holdings control decide
    what the protected numbers *are*. Running under a different one produces a
    coherent package about a question nobody permitted -- so it is refused
    before the axis is even resolved, and the counter proves no book was walked.
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        # The continuation is re-pointed at a different spec; everything else,
        # including the permit and the candidate, is untouched.
        wired.continuation.spec = PortfolioResearchSpec.create(**overrides)
        # A holdings control moves the holdings spec *and* the request, so the
        # spec check fires first. Either refusal is the right one; what matters
        # is that neither of them let a protected book be walked.
        del label
        with pytest.raises(
            BaseException,
            match=r"protected_spec_is_not_the_frozen_configuration"
            r"|protected_holdings_spec_is_not_the_frozen_one|not_succeeded",
        ):
            wired.application.finalize(candidate=candidate, fixture=_fixture())

    assert wired.continuation.evaluations == 0


def test_a_continuation_in_another_workspace_does_no_protected_work(tmp_path: Path) -> None:
    """The candidate names the workspace it was frozen in; so must the run."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        wired.continuation.workspace_id = "somewhere-else"
        with pytest.raises(
            BaseException, match=r"protected_workspace_is_not_the_frozen_one|not_succeeded"
        ):
            wired.application.finalize(candidate=candidate, fixture=_fixture())

    assert wired.continuation.evaluations == 0


def test_the_gate_refuses_a_coherent_package_from_another_configuration(
    tmp_path: Path,
) -> None:
    """The Gate's own half: a package the runner never saw, and never would.

    Everything here is real. A second development path is run under a *different*
    cost, finalized on the permitted fixture through its own candidate, and its
    protected package -- coherent, children present, axis correct -- is offered
    for closure under the first candidate's permit.

    The runner's refusal cannot catch this, because no runner was involved. The
    Gate compares the configuration identities the inspection reports against the
    candidate and the permit, and that is the only thing standing here.
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        fixture = _fixture()
        gate = _open_gate(session, wired)
        permit = gate.admit(candidate=candidate, fixture=fixture)

        # A genuine finalization under a different cost, over the same fixture.
        other_spec = PortfolioResearchSpec.create(cost_bps_per_side="10")
        other_development = PortfolioResearchApplication(
            workspace_id="qa-final",
            workspace=workspace,
            manifest_binding=lambda: "a" * 64,
            session=session,
            resolver=_Resolver(_resolved(sessions=DEVELOPMENT_SESSIONS)),
        )
        completed = other_development.run(spec=other_spec)
        other_candidate = freeze_candidate(
            wired.ledger,
            wired.store,
            workspace_id="qa-final",
            result_hash=completed.result.result_hash,
            development_task_id=str(completed.pipeline_manifest.task_id),
            tasks=HostCompletedDevelopmentTasks(session.task_control_registry),
        )
        assert other_candidate.control_receipt_hash != candidate.control_receipt_hash
        other_continuation = ExecutorProtectedContinuation(
            workspace_id="qa-final",
            workspace=workspace,
            spec=other_spec,
            resolver=_Resolver(_resolved(sessions=PROTECTED_SESSIONS)),
            executor=PortfolioResearchExecutor(wired.pending, packages=_PACKAGES),
            compiler=PortfolioResearchCompiler(),
            development_ledger=wired.ledger,
        )
        other_permit = ProtectedEvaluationPermit.create(
            candidate_hash=other_candidate.candidate_hash,
            configuration_hash=other_candidate.spec_hash,
            pre_protected_state_hash=other_candidate.pre_protected_state.state_hash,
            fixture=fixture,
            issued_at=datetime.now(UTC),
        )
        produced = other_continuation.evaluate(permit=other_permit, candidate=other_candidate)
        # Re-labelled for the *first* permit and candidate. Every child is real.
        crossed = FinalPortfolioEvaluationPackage.create(
            permit_hash=permit.permit_hash,
            candidate_hash=candidate.candidate_hash,
            continued_from_state_hash=candidate.pre_protected_state.state_hash,
            fixture_hash=fixture.fixture_hash,
            protected_result_hash=produced.protected_result_hash,
            protected_execution_ledger_hash=produced.execution_ledger_hash,
            protected_economic_ledger_hash=produced.economic_ledger_hash,
            protected_report_hash=produced.report_hash,
            protected_formation_count=produced.formation_count,
            protected_listing_count=produced.listing_count,
        )
        receipt = gate.close(permit=permit, candidate=candidate, package=crossed)

    assert receipt.closure == "REFUSED_CONFIGURATION_MISMATCH"
    assert not receipt.closed


def test_the_gate_refuses_a_program_whose_axis_disagrees_with_its_ledger(
    tmp_path: Path,
) -> None:
    """The ledger matches the fixture; the Program says another window.

    Two independent axis claims, and until both were read the second one could
    say anything. Only the Program moves here: the real protected ledger is left
    exactly as the run sealed it, so it still matches the permitted fixture and
    every child still refers to every other. The forged Program recomputes to its
    own hash and declares a window one session short.
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        fixture = _fixture()
        gate = _open_gate(session, wired)
        permit = gate.admit(candidate=candidate, fixture=fixture)
        produced = wired.continuation.evaluate(permit=permit, candidate=candidate)

        ledger = wired.pending.load_execution(produced.execution_ledger_hash)
        program = wired.pending.load_program(ledger.program_hash)
        moved = {
            **program.model_dump(mode="json", exclude={"program_hash"}),
            "formation_end": PROTECTED_SESSIONS[-2].isoformat(),
        }
        forged_program = PortfolioExecutionProgram.model_validate(
            {**moved, "program_hash": canonical_hash(moved)}
        )
        wired.pending.content.publish_model(
            category="programs", value=forged_program, identity_field="program_hash"
        )

        # The report names the Program and binds its own coverage to it, so both
        # move with it. The *ledger* does not: it stays exactly as sealed.
        report = wired.pending.load_report(produced.report_hash)
        forged_coverage = PortfolioLedgerCoverage.of(
            program_hash=forged_program.program_hash,
            ledger_hash=ledger.ledger_hash,
            formation_sessions=ledger.formation_sessions,
            ordered_listing_ids=ledger.ordered_listing_ids,
            source_coverage_hash=report.ledger_coverage.source_coverage_hash,
        )
        forged_report_identity = {
            **report.model_dump(mode="json", exclude={"report_hash"}),
            "program_hash": forged_program.program_hash,
            "ledger_coverage": forged_coverage.model_dump(mode="json"),
        }
        forged_report_hash = canonical_hash(forged_report_identity)
        (wired.pending.root / "reports" / f"{forged_report_hash}.json").write_bytes(
            json.dumps(
                {**forged_report_identity, "report_hash": forged_report_hash},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        result = wired.pending.load_result(produced.protected_result_hash)
        forged_result_identity = {
            k: v
            for k, v in result.model_dump(
                mode="json", exclude={"result_hash"}, exclude_none=True
            ).items()
            if k != "action"
        } | {
            "program_hash": forged_program.program_hash,
            "report_hash": forged_report_hash,
        }
        forged_result_hash = canonical_hash(forged_result_identity)
        (wired.pending.root / "results" / f"{forged_result_hash}.json").write_bytes(
            json.dumps(
                {
                    **forged_result_identity,
                    "action": "PUBLISHED",
                    "result_hash": forged_result_hash,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        package = FinalPortfolioEvaluationPackage.create(
            permit_hash=permit.permit_hash,
            candidate_hash=candidate.candidate_hash,
            continued_from_state_hash=candidate.pre_protected_state.state_hash,
            fixture_hash=fixture.fixture_hash,
            protected_result_hash=forged_result_hash,
            protected_execution_ledger_hash=produced.execution_ledger_hash,
            protected_economic_ledger_hash=produced.economic_ledger_hash,
            protected_report_hash=forged_report_hash,
            protected_formation_count=produced.formation_count,
            protected_listing_count=produced.listing_count,
        )
        inspection = PendingPackageInspector(wired.pending).inspect(package)
        receipt = gate.close(permit=permit, candidate=candidate, package=package)

    # Every child opened, and the ledger still describes the permitted fixture.
    assert not inspection.absent_children
    assert inspection.formation_sessions == tuple(fixture.sessions)
    # Only the Program disagrees, and the ledger-to-Program child binding catches it.
    assert inspection.program_formation_end != fixture.sessions[-1]
    assert receipt.closure == "REFUSED_PACKAGE_BINDINGS_INCOHERENT"
    assert not receipt.closed


def test_the_gate_compares_the_program_listing_count_to_the_opened_ledger(
    tmp_path: Path,
) -> None:
    """A package cannot repeat a false Program width and make it true."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        fixture = _fixture()
        gate = _open_gate(session, wired)
        permit = gate.admit(candidate=candidate, fixture=fixture)
        produced = wired.continuation.evaluate(permit=permit, candidate=candidate)
        package = FinalPortfolioEvaluationPackage.create(
            permit_hash=permit.permit_hash,
            candidate_hash=candidate.candidate_hash,
            continued_from_state_hash=candidate.pre_protected_state.state_hash,
            fixture_hash=fixture.fixture_hash,
            protected_result_hash=produced.protected_result_hash,
            protected_execution_ledger_hash=produced.execution_ledger_hash,
            protected_economic_ledger_hash=produced.economic_ledger_hash,
            protected_report_hash=produced.report_hash,
            protected_formation_count=produced.formation_count,
            protected_listing_count=produced.listing_count,
        )
        inspection = PendingPackageInspector(wired.pending).inspect(package)
        false_values = inspection.model_dump(mode="python", exclude={"inspection_hash"})
        false_values["program_listing_count"] = produced.listing_count + 1
        false_inspection = ProtectedPackageInspection.create(**false_values)

    assert (
        ProtectedValidationGate._closure(
            permit=permit,
            candidate=candidate,
            package=package,
            inspection=inspection,
        )
        == "CLOSED_EXACT"
    )
    assert (
        ProtectedValidationGate._closure(
            permit=permit,
            candidate=candidate,
            package=package,
            inspection=false_inspection,
        )
        == "REFUSED_PACKAGE_AXIS_MISMATCH"
    )


def test_the_gate_refuses_a_report_whose_comparison_child_is_absent(tmp_path: Path) -> None:
    """The report's benchmark child belongs to the protected package closure."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        fixture = _fixture()
        gate = _open_gate(session, wired)
        permit = gate.admit(candidate=candidate, fixture=fixture)
        produced = wired.continuation.evaluate(permit=permit, candidate=candidate)
        report = wired.pending.load_report(produced.report_hash)
        (
            wired.pending.root
            / "benchmark-comparisons"
            / f"{report.benchmark_comparison_hash}.json"
        ).unlink()
        package = FinalPortfolioEvaluationPackage.create(
            permit_hash=permit.permit_hash,
            candidate_hash=candidate.candidate_hash,
            continued_from_state_hash=candidate.pre_protected_state.state_hash,
            fixture_hash=fixture.fixture_hash,
            protected_result_hash=produced.protected_result_hash,
            protected_execution_ledger_hash=produced.execution_ledger_hash,
            protected_economic_ledger_hash=produced.economic_ledger_hash,
            protected_report_hash=produced.report_hash,
            protected_formation_count=produced.formation_count,
            protected_listing_count=produced.listing_count,
        )
        receipt = gate.close(permit=permit, candidate=candidate, package=package)

    assert receipt.closure == "REFUSED_PACKAGE_CHILDREN_ABSENT"
    assert not receipt.closed


# ===================================================================
# Closure batch: the release marker is what makes a closure readable
# ===================================================================


def _readback_with(wired: _Finalization, session, result_hash: str, package, receipt, handoff):  # type: ignore[no-untyped-def]
    return exact_readback(
        wired.ledger,
        workspace_id="qa-final",
        result_hash=result_hash,
        package_hash=package.package_hash,
        validation_receipt_hash=receipt.receipt_hash,
        handoff_hash=handoff.handoff_hash,
        finalization=HostFinalizationArtifactOpener(
            store=wired.store,
            validation=ProtectedValidationStore(session.workspace / "runtime" / "artifacts"),
        ),
    )


def test_adopted_but_unreleased_artifacts_do_not_read_back_as_verified(
    tmp_path: Path,
) -> None:
    """The window this closes: bytes in place, marker absent, hashes known.

    A caller that already holds the package, receipt and handoff identities can
    open all three the moment adoption finishes -- content addressing does not
    care whether anything released them. Before the marker, none of the four
    finalization layers may be VERIFIED, and the reason has to be the missing
    release rather than a missing artifact.
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        seen: dict[str, object] = {}
        original = wired.store.publish_release

        def _stop_before_marker(value):  # type: ignore[no-untyped-def]
            seen["package"] = wired.application.adapter.package  # type: ignore[union-attr]
            seen["receipt"] = wired.application.adapter.receipt  # type: ignore[union-attr]
            seen["handoff"] = wired.store.find_handoff_for_package(value.package_hash)
            raise RuntimeError("crash after adoption, before the marker")

        wired.store.publish_release = _stop_before_marker  # type: ignore[method-assign]
        with pytest.raises(BaseException, match=r"before the marker|not_succeeded"):
            wired.application.finalize(candidate=candidate, fixture=_fixture())
        wired.store.publish_release = original  # type: ignore[method-assign]

        package = seen["package"]
        receipt = seen["receipt"]
        handoff = seen["handoff"]
        assert isinstance(package, FinalPortfolioEvaluationPackage)
        assert isinstance(receipt, PortfolioValidationReceipt)
        assert handoff is not None
        # The bytes really are adopted: a by-identity read succeeds.
        assert wired.ledger.load_result(package.protected_result_hash) is not None
        unreleased = _readback_with(
            wired, session, package.protected_result_hash, package, receipt, handoff
        )
        assert wired.store.find_release_for_package(package.package_hash) is None

        # Recovery commits the marker, and only then does the closure read back.
        task = next(
            value
            for value in session.task_control_registry.tasks()
            if value.task_kind == PortfolioFinalizationTaskAdapter.task_kind
        )
        finalized = wired.application.recover(
            task_id=task.task_id, candidate=candidate, fixture=_fixture()
        )
        released = _readback_with(
            wired,
            session,
            finalized.package.protected_result_hash,
            finalized.package,
            finalized.receipt,
            finalized.handoff,
        )

    for layer_id in (
        "final_evaluation_package",
        "validation_receipt",
        "validated_handoff",
        "released_artifacts",
    ):
        assert unreleased.layer(layer_id).disposition == "FAILED", layer_id
        assert "not been released" in unreleased.layer(layer_id).detail, layer_id
        assert released.layer(layer_id).disposition == "VERIFIED", layer_id
    assert unreleased.work.total == 0
    assert released.work.total == 0
    # Same finalization, one protected evaluation, and the marker binds all six.
    assert finalized.package.package_hash == package.package_hash
    assert finalized.receipt.receipt_hash == receipt.receipt_hash
    assert finalized.handoff.handoff_hash == handoff.handoff_hash
    assert wired.continuation.evaluations == 1
    assert finalized.release.package_hash == finalized.package.package_hash
    assert finalized.release.validation_receipt_hash == finalized.receipt.receipt_hash
    assert finalized.release.handoff_hash == finalized.handoff.handoff_hash
    assert finalized.release.released_result_hash == finalized.package.protected_result_hash


def test_a_deleted_release_marker_unproves_the_closure(tmp_path: Path) -> None:
    """A released closure stops reading back the moment its marker is gone."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        finalized = wired.application.finalize(candidate=candidate, fixture=_fixture())
        before = _readback_with(
            wired,
            session,
            finalized.package.protected_result_hash,
            finalized.package,
            finalized.receipt,
            finalized.handoff,
        )
        wired.store.committed.path("released", finalized.package.package_hash).unlink()
        (wired.store.root / "releases" / f"{finalized.release.marker_hash}.json").unlink()
        after = _readback_with(
            wired,
            session,
            finalized.package.protected_result_hash,
            finalized.package,
            finalized.receipt,
            finalized.handoff,
        )

    assert before.layer("released_artifacts").disposition == "VERIFIED"
    assert after.layer("released_artifacts").disposition == "FAILED"
    assert after.layer("validated_handoff").disposition == "FAILED"


def test_a_crossed_release_marker_is_refused(tmp_path: Path) -> None:
    """A real marker from another finalization does not release this one."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        first = wired.application.finalize(candidate=_candidate(wired), fixture=_fixture())

        second_development = PortfolioResearchApplication(
            workspace_id="qa-final",
            workspace=workspace,
            manifest_binding=lambda: "a" * 64,
            session=session,
            resolver=_Resolver(_resolved(sessions=DEVELOPMENT_SESSIONS[:2])),
        )
        completed = second_development.run(spec=wired.spec)
        second_candidate = freeze_candidate(
            wired.ledger,
            wired.store,
            workspace_id="qa-final",
            result_hash=completed.result.result_hash,
            development_task_id=str(completed.pipeline_manifest.task_id),
            tasks=HostCompletedDevelopmentTasks(session.task_control_registry),
        )
        second = wired.application.finalize(candidate=second_candidate, fixture=_fixture())

        # The first package's identities, read back against the second release.
        crossed = _readback_with(
            wired,
            session,
            second.package.protected_result_hash,
            first.package,
            first.receipt,
            first.handoff,
        )

    assert crossed.layer("released_artifacts").disposition == "FAILED"
    assert crossed.layer("final_evaluation_package").disposition == "FAILED"


def test_release_stage_verification_reopens_the_marker(tmp_path: Path) -> None:
    """The runner's own verifier, not just the caller's readback."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        finalized = wired.application.finalize(candidate=candidate, fixture=_fixture())
        adapter = wired.application.adapter
        assert adapter is not None
        work_item = next(
            value
            for value in session.task_control_registry.task(finalized.task_id).plan.work_items
            if value.stage_id == "release_validated_handoff"
        )
        evidence = (
            TaskEvidence(
                evidence_kind=FINALIZATION_EVIDENCE_KIND,
                reference=(
                    "playpen://portfolio-strategy-lab/finalization/release_validated_handoff/"
                    + finalized.handoff.handoff_hash
                ),
                content_hash=finalized.handoff.handoff_hash,
            ),
        )
        record = session.task_control_registry.task(finalized.task_id)
        # Recovery invokes verification on a fresh adapter before any execute
        # method can populate its remembered permit/package/receipt.
        adapter = type(adapter)(
            **{
                name: getattr(adapter, name)
                for name in (
                    "workspace_id",
                    "candidate",
                    "fixture",
                    "gate",
                    "continuation",
                    "release",
                    "store",
                )
            }
        )
        for definition, item in zip(
            record.plan.work_items,
            session.task_control_registry.work_items(record.task_id),
            strict=True,
        ):
            adapter.verify_stage(
                task=record, execution=None, work_item=definition, evidence=item.evidence
            )
            if definition.stage_id in {"claim_protected_permit", "verify_protected_closure"}:
                forged = item.evidence[0].model_copy(update={"content_hash": "f" * 64})
                with pytest.raises(PortfolioFinalizationError, match="evidence_not_durable"):
                    adapter.verify_stage(
                        task=record, execution=None, work_item=definition, evidence=(forged,)
                    )
        assert (
            adapter.permits_issued,
            adapter.protected_continuations,
            adapter.package_publications,
            adapter.closures,
            adapter.handoff_publications,
            adapter.release_commits,
        ) == (0,) * 6
        # With the marker, it verifies.
        adapter.verify_stage(task=record, execution=None, work_item=work_item, evidence=evidence)
        wired.store.committed.path("released", finalized.package.package_hash).unlink()
        (wired.store.root / "releases" / f"{finalized.release.marker_hash}.json").unlink()
        with pytest.raises(PortfolioFinalizationError, match="release_marker_absent"):
            adapter.verify_stage(
                task=record, execution=None, work_item=work_item, evidence=evidence
            )


def test_recovery_refuses_to_complete_without_the_release_marker(tmp_path: Path) -> None:
    """A CompletedFinalization is a released one, or it is not returned."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(session)
        candidate = _candidate(wired)
        finalized = wired.application.finalize(candidate=candidate, fixture=_fixture())
        wired.store.committed.path("released", finalized.package.package_hash).unlink()
        (wired.store.root / "releases" / f"{finalized.release.marker_hash}.json").unlink()
        # Deleting the marker and asking again re-commits it, because the stage
        # is idempotent -- so the refusal is measured on a store that cannot
        # commit one.
        wired.store.publish_release = lambda value: (_ for _ in ()).throw(  # type: ignore[method-assign]
            RuntimeError("no marker")
        )
        with pytest.raises(BaseException, match=r"no marker|release_absent|not_succeeded"):
            wired.application.recover(
                task_id=finalized.task_id, candidate=candidate, fixture=_fixture()
            )


# ===================================================================
# Closure batch: anchor_plus_spy replays like anything else
# ===================================================================


def test_a_secondary_comparator_path_replays_every_layer(tmp_path: Path) -> None:
    """`anchor_plus_spy` is an admitted control, so it is inside exact replay.

    The secondary series is an input the run resolved from outside the path --
    it cannot be rederived from fills any more than the eligible-universe anchor
    can. So it is sealed as its own artifact and reopened, and what the replay
    proves is that the comparison identity and the whole report projection follow
    from it and from the rerun.
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = PortfolioResearchSpec.create(secondary_benchmark_view="anchor_plus_spy")
    with WorkspaceApplicationSession.acquire(workspace) as session:
        development = PortfolioResearchApplication(
            workspace_id="qa-spy",
            workspace=workspace,
            manifest_binding=lambda: "a" * 64,
            session=session,
            resolver=_Resolver(
                _resolved(benchmark_view="anchor_plus_spy", sessions=DEVELOPMENT_SESSIONS)
            ),
        )
        completed = development.run(spec=spec)
        coverage = (
            _Resolver(_resolved(benchmark_view="anchor_plus_spy", sessions=DEVELOPMENT_SESSIONS))
            .resolve_authorities(workspace=workspace, spec=spec)
            .coverage
        )
        report = development.ledger.load_report(completed.result.report_hash)
        sealed_comparison = development.ledger.load_comparison(report.benchmark_comparison_hash)
        receipt = strong_replay(
            development.ledger,
            workspace_id="qa-spy",
            result_hash=completed.result.result_hash,
            spec=spec,
            coverage=coverage,
        )

    assert sealed_comparison.secondary_benchmark_id == "SPY_TOTAL_RETURN"
    assert sealed_comparison.secondary_simple_returns is not None
    assert receipt.disposition == "REPLAYED_EXACT"
    for layer in receipt.layers:
        assert layer.disposition == "VERIFIED", (layer.layer_id, layer.detail)
    assert receipt.work.forbidden_in_replay == 0


def test_a_secondary_comparator_path_without_the_sealed_series_is_refused(
    tmp_path: Path,
) -> None:
    """A predecessor artifact says so by name, not as a numerical mismatch."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = PortfolioResearchSpec.create(secondary_benchmark_view="anchor_plus_spy")
    with WorkspaceApplicationSession.acquire(workspace) as session:
        development = PortfolioResearchApplication(
            workspace_id="qa-spy",
            workspace=workspace,
            manifest_binding=lambda: "a" * 64,
            session=session,
            resolver=_Resolver(
                _resolved(benchmark_view="anchor_plus_spy", sessions=DEVELOPMENT_SESSIONS)
            ),
        )
        completed = development.run(spec=spec)
        coverage = (
            _Resolver(_resolved(benchmark_view="anchor_plus_spy", sessions=DEVELOPMENT_SESSIONS))
            .resolve_authorities(workspace=workspace, spec=spec)
            .coverage
        )
        report = development.ledger.load_report(completed.result.report_hash)
        # The pre-sealing shape: the comparison was never published.
        (
            development.ledger.root
            / "benchmark-comparisons"
            / f"{report.benchmark_comparison_hash}.json"
        ).unlink()
        receipt = strong_replay(
            development.ledger,
            workspace_id="qa-spy",
            result_hash=completed.result.result_hash,
            spec=spec,
            coverage=coverage,
        )

    assert receipt.disposition == "REFUSED_UNSEALED_CHILDREN"
    assert receipt.work.total == 0
    assert "secondary comparator" in receipt.layer("report_metrics").detail


# ===================================================================
# Closure batch: upstream layers are proved by opening real artifacts
# ===================================================================


def _publish_risk_surface(artifact_root: Path, *, listings: tuple[str, ...]) -> tuple[str, Path]:
    """Seal one real Risk surface and its Parquet child in the owner store."""

    epoch_values: dict[str, object] = {
        "kind": "RiskUniverseEpoch",
        "market_profile_id": "qa-synthetic",
        "universe_manifest_revision": "1" * 64,
        "panel_snapshot_hash": "2" * 64,
        "ordered_listing_ids": listings,
        "data_validity_class": "CURRENT_UNIVERSE_RESEARCH_ONLY",
        "minimum_valid_return_sessions": 315,
    }
    epoch_identity = RiskUniverseEpoch.model_construct(
        **epoch_values, epoch_hash="0" * 64
    ).model_dump(mode="json", exclude={"epoch_hash"})
    epoch = RiskUniverseEpoch(**epoch_identity, epoch_hash=canonical_hash(epoch_identity))
    sessions = tuple(date(2024, 1, 2) + timedelta(days=index) for index in range(315))
    rows: list[dict[str, object]] = []
    for formation_index, formation in enumerate(sessions):
        for listing in listings:
            values: dict[str, object] = {
                "formation_session": formation,
                "return_start_session": formation - timedelta(days=1),
                "listing_id": listing,
                "symbol": listing,
                "open_total_return_log": 0.0001 * (formation_index + 1),
                "entry_source_row_hash": canonical_hash(
                    {"listing": listing, "formation": formation.isoformat(), "side": "entry"}
                ),
                "exit_source_row_hash": canonical_hash(
                    {"listing": listing, "formation": formation.isoformat(), "side": "exit"}
                ),
                "action_set_hash": "3" * 64,
            }
            rows.append({**values, "row_hash": canonical_hash(values)})
    store = RiskReturnArtifactStore(artifact_root)
    chunk = store.publish_chunk(pa.Table.from_pylist(rows))
    surface_values: dict[str, object] = {
        "kind": "CausalRiskReturnSurface",
        "epoch": epoch,
        "first_formation_session": date(2024, 1, 2),
        "last_formation_session": sessions[-1],
        "formation_count": 315,
        "return_unit": "one-session-open-to-open-log-return",
        "formula_identity": "split-adjusted-open-plus-dividend-log-gross-return",
        "source_watermark_hash": "5" * 64,
        "chunks": (chunk,),
        "limitations": ("Synthetic QA surface; carries no scientific claim.",),
    }
    identity = CausalRiskReturnSurface.model_construct(
        **surface_values, surface_hash="0" * 64
    ).model_dump(mode="json", exclude={"surface_hash"})
    surface = CausalRiskReturnSurface(**identity, surface_hash=canonical_hash(identity))
    store.publish_manifest(surface)
    chunk_path = store.root / "chunks" / f"{chunk.content_hash}.parquet"
    return str(surface.surface_hash), chunk_path


def _publish_alpha_evidence(root: Path) -> tuple[str, AlphaProductRecipe, Path]:
    """Write a complete synthetic Alpha package under a synthetic recipe."""

    root.mkdir(parents=True, exist_ok=True)
    daily = root / "iw184-daily-evidence.parquet"
    daily.write_bytes(b"non-reserved synthetic Alpha daily evidence")
    daily_sha = hashlib.sha256(daily.read_bytes()).hexdigest()
    recipe_values = INSTALLED_ALPHA_PRODUCT_RECIPE.model_dump(mode="json", exclude={"recipe_hash"})
    recipe_values["evidence_daily_parquet_sha256"] = daily_sha
    draft = AlphaProductRecipe.model_construct(**recipe_values, recipe_hash="0" * 64)
    recipe = AlphaProductRecipe(
        **recipe_values,
        recipe_hash=recipe_identity(draft, exclude=frozenset({"recipe_hash"})),
    )
    vintages = ("2024-01", "2023-10", "2023-07", "2023-04")
    formation = DEVELOPMENT_SESSIONS[-1]
    listings = LISTINGS
    rows = [(formation, listing) for listing in listings]
    months = (formation.strftime("%Y-%m"),)
    session_axis_hash = canonical_hash([formation.isoformat()])
    model_lineage: dict[str, object] = {}
    (root / "prediction-axes").mkdir()
    (root / "model-predictions").mkdir()
    first_model_payload: Path | None = None
    for vintage in vintages:
        axis_payload = root / "prediction-axes" / f"{vintage}.npz"
        np.savez(
            axis_payload,
            row_sessions=np.asarray([value for value, _listing in rows], dtype="datetime64[D]"),
            row_listing_ids=np.asarray([listing for _value, listing in rows], dtype="<U36"),
        )
        axis_sha = hashlib.sha256(axis_payload.read_bytes()).hexdigest()
        (root / "prediction-axes" / f"{vintage}.json").write_text(
            json.dumps(
                {
                    "kind": "IW184QuarterlyPredictionAxis",
                    "quarter_start": vintage,
                    "artifact_sha256": axis_sha,
                    "row_count": len(rows),
                    "transform_session_axis_hash": session_axis_hash,
                    "operational_months": list(months),
                    "prediction_months": list(months),
                }
            ),
            encoding="utf-8",
        )
        vintage_root = root / "model-predictions" / vintage
        vintage_root.mkdir()
        seed_lineage: dict[str, object] = {}
        for seed in recipe.seeds:
            model_recipe_hash = canonical_hash({"vintage": vintage, "seed": seed})
            identity = {
                "kind": "IW184Fixed1260QuarterlySeedModel",
                "candidate": recipe.feature_axis_id,
                "candidate_axis_hash": recipe.feature_axis_hash,
                "feature_view_id": "SYNTHETIC_NON_RESERVED",
                "annual_root_hash": "4" * 64,
                "source_resolution_hash": "5" * 64,
                "recipe_hash": model_recipe_hash,
                "seed": seed,
                "quarter_start": vintage,
                "training_window_sessions": recipe.training_window_sessions,
                "training_first": "2019-01-02",
                "training_last": "2023-12-29",
                "training_session_axis_hash": "6" * 64,
                "purge_session": "2024-01-01",
                "purge_session_count": recipe.purge_sessions,
                "prediction_months": list(months),
                "operational_months": list(months),
                "transform_session_axis_hash": session_axis_hash,
                "score_aggregation": recipe.score_aggregation,
            }
            identity_hash = canonical_hash(identity)
            payload = vintage_root / f"seed-{seed}.npz"
            np.savez(
                payload,
                scores=np.linspace(0.0, 1.0, len(rows), dtype=np.float64),
                identity_hash=np.asarray(identity_hash, dtype="<U64"),
            )
            if first_model_payload is None:
                first_model_payload = payload
            artifact_sha = hashlib.sha256(payload.read_bytes()).hexdigest()
            estimator_hash = canonical_hash({"estimator": seed, "vintage": vintage})
            (vintage_root / f"seed-{seed}.json").write_text(
                json.dumps(
                    {
                        **identity,
                        "ordered_feature_ids_hash": recipe.feature_axis_hash,
                        "identity_hash": identity_hash,
                        "estimator_content_hash": estimator_hash,
                        "artifact_sha256": artifact_sha,
                        "prediction_axis_sha256": axis_sha,
                        "prediction_row_count": len(rows),
                        "training_row_count": recipe.training_window_sessions,
                    }
                ),
                encoding="utf-8",
            )
            seed_lineage[str(seed)] = {
                "artifact_sha256": artifact_sha,
                "estimator_content_hash": estimator_hash,
                "identity_hash": identity_hash,
                "training_first": identity["training_first"],
                "training_last": identity["training_last"],
                "training_row_count": recipe.training_window_sessions,
            }
        model_lineage[vintage] = {
            "prediction_axis_sha256": axis_sha,
            "row_count": len(rows),
            "seeds": seed_lineage,
        }
    manifest = {
        "kind": "IW184Fixed1260QuarterlySeed3Vintage4WeightSensitivityDiagnostic",
        "disposition": recipe.evidence_disposition,
        "definition": {
            "feature_axis": recipe.feature_axis_id,
            "feature_count": recipe.feature_count,
            "live_model_count": recipe.live_model_count,
            "training_window_sessions": recipe.training_window_sessions,
            "purge_session_count": recipe.purge_sessions,
            "seeds": list(recipe.seeds),
            "vintage_count": recipe.vintage_count,
            "vintage_weights": list(recipe.vintage_weights),
            "seed_aggregation": "EQUAL_MEAN_WITHIN_VINTAGE",
            "score_aggregation": recipe.score_aggregation,
        },
        "daily_evidence": {
            "name": daily.name,
            "row_count": 1,
            "sha256": daily_sha,
        },
        "model_lineage": model_lineage,
    }
    path = root / f"{EVIDENCE_PACKAGE_ID}.json"
    path.write_bytes(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
    assert first_model_payload is not None
    return hashlib.sha256(path.read_bytes()).hexdigest(), recipe, first_model_payload


def test_upstream_layers_are_proved_by_opening_the_real_artifacts(tmp_path: Path) -> None:
    """Four owners, four opens, and an arbitrary hash proves nothing.

    The recipes are installed objects, so opening one means resolving the recipe
    this build actually runs. The evidence manifest is a file whose sha256 the
    reader recomputes before it will look at it. The Risk surface comes back out
    of its own store and re-validates its own identity on load. In every case the
    identity readback compares against the Program is the one the *artifact*
    reported, never the one the caller asked for.
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    artifact_root = workspace / "runtime" / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    alpha_root = tmp_path / "alpha-evidence"
    listings = tuple(f"listing-{index:04d}" for index in range(80))
    surface_hash, risk_chunk_path = _publish_risk_surface(artifact_root, listings=listings)
    risk_chunk_bytes = risk_chunk_path.read_bytes()
    manifest_sha, alpha_recipe, alpha_child_path = _publish_alpha_evidence(alpha_root)

    with WorkspaceApplicationSession.acquire(workspace) as session:
        resolution = _resolved(sessions=DEVELOPMENT_SESSIONS)
        bound = dataclasses.replace(
            resolution,
            alpha_recipe_hash=alpha_recipe.recipe_hash,
            alpha_evidence_manifest_hash=manifest_sha,
            risk_recipe_hash=INSTALLED_RISK_DECOMPOSITION_RECIPE.recipe_hash,
            risk_return_surface_hash=surface_hash,
        )
        development = PortfolioResearchApplication(
            workspace_id="qa-upstream",
            workspace=workspace,
            manifest_binding=lambda: "a" * 64,
            session=session,
            resolver=_Resolver(bound, numerical=bound),
        )
        completed = development.run(spec=PortfolioResearchSpec.default())
        opener = HostUpstreamEvidence(
            artifact_root=artifact_root,
            alpha_evidence_root=alpha_root,
            alpha_evidence_closure=product_evidence_closure_reader(recipe=alpha_recipe),
            alpha_recipe=alpha_recipe,
        )
        proved = exact_readback(
            development.ledger,
            workspace_id="qa-upstream",
            result_hash=completed.result.result_hash,
            upstream=opener,
        )
        # Keep the manifest intact but remove one Parquet child.
        risk_chunk_path.unlink()
        deleted = exact_readback(
            development.ledger,
            workspace_id="qa-upstream",
            result_hash=completed.result.result_hash,
            upstream=opener,
        )
        risk_chunk_path.write_bytes(risk_chunk_bytes)
        # Keep Alpha's manifest intact but tamper one model payload below it.
        alpha_child_path.write_bytes(b"tampered model payload")
        tampered = exact_readback(
            development.ledger,
            workspace_id="qa-upstream",
            result_hash=completed.result.result_hash,
            upstream=opener,
        )

    for layer_id in (
        "alpha_recipe",
        "alpha_evidence_package",
        "risk_recipe",
        "risk_return_surface",
    ):
        assert proved.layer(layer_id).disposition == "VERIFIED", layer_id
        assert "opened" in proved.layer(layer_id).detail
    assert proved.work.total == 0
    assert deleted.layer("risk_return_surface").disposition == "FAILED"
    assert deleted.layer("risk_recipe").disposition == "VERIFIED"
    assert tampered.layer("alpha_evidence_package").disposition == "FAILED"
    assert tampered.layer("alpha_recipe").disposition == "VERIFIED"


def test_an_arbitrary_identity_cannot_be_opened(tmp_path: Path) -> None:
    """No record, no receipt, no way in: the opener has to find an artifact.

    `record()` is gone -- the opener has no write surface at all -- so this is
    both a behavioural and a structural statement.
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    artifact_root = workspace / "runtime" / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    opener = HostUpstreamEvidence(artifact_root=artifact_root)

    assert not hasattr(opener, "record")
    for role in (
        "ALPHA_EVIDENCE_MANIFEST",
        "RISK_RETURN_SURFACE",
    ):
        assert opener.open_upstream(role=role, identity_hash="a" * 64) is None
    # The installed recipes open, and report *their own* identity -- so a
    # request for a different one is answered, and then fails the comparison.
    opened = opener.open_upstream(role="ALPHA_RECIPE", identity_hash="a" * 64)
    assert opened is not None
    assert opened.identity_hash == INSTALLED_ALPHA_PRODUCT_RECIPE.recipe_hash
    layer = _upstream_layer(
        "alpha_recipe",
        role="ALPHA_RECIPE",
        identity="a" * 64,
        opener=opener,
        proved_detail="opened",
        unreachable_detail="not opened",
    )
    assert layer.disposition == "FAILED"
    assert "different identity" in layer.detail


def test_the_application_read_back_seam_needs_no_prerequisite(tmp_path: Path) -> None:
    """The production caller reopens the complete released closure at zero work."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    artifact_root = workspace / "runtime" / "artifacts"
    alpha_root = tmp_path / "complete-alpha-evidence"
    surface_hash, _chunk_path = _publish_risk_surface(artifact_root, listings=LISTINGS)
    manifest_sha, alpha_recipe, _alpha_child = _publish_alpha_evidence(alpha_root)

    def bound(sessions: tuple[date, ...]):  # type: ignore[no-untyped-def]
        resolution = _resolved(sessions=sessions)
        return dataclasses.replace(
            resolution,
            alpha_recipe_hash=alpha_recipe.recipe_hash,
            alpha_evidence_manifest_hash=manifest_sha,
            risk_recipe_hash=INSTALLED_RISK_DECOMPOSITION_RECIPE.recipe_hash,
            risk_return_surface_hash=surface_hash,
        )

    with WorkspaceApplicationSession.acquire(workspace) as session:
        wired = _wire(
            session,
            development_resolution=bound(DEVELOPMENT_SESSIONS),
            protected_resolution=bound(PROTECTED_SESSIONS),
            alpha_evidence_root=alpha_root,
            alpha_recipe=alpha_recipe,
        )
        candidate = _candidate(wired)
        finalized = wired.application.finalize(candidate=candidate, fixture=_fixture())
        receipt = wired.application.read_back(
            finalized.package.protected_result_hash, package=finalized.package
        )
        downstream = HostValidatedPortfolioHandoffReader(artifact_root).open(
            finalized.handoff.handoff_hash
        )

    assert receipt.work.total == 0
    assert tuple(layer.layer_id for layer in receipt.layers) == READBACK_LAYERS
    for layer in receipt.layers:
        assert layer.disposition == "VERIFIED", (layer.layer_id, layer.detail)
    assert downstream.handoff == finalized.handoff
    assert downstream.package == finalized.package
    assert downstream.receipt == finalized.receipt
    assert downstream.report.report_hash == finalized.package.protected_report_hash
