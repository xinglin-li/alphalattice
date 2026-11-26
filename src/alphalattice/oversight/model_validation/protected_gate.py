"""The Protected Validation Gate: admission, one permit, closure, one receipt.

The successor. It replaces the Holdout runtime beside it -- which is retired
through consumer-specific sub-Gates rather than ported -- and it is deliberately
small, because almost everything the predecessor did belonged to Portfolio.

What the Gate owns: whether a candidate was frozen before anyone asked about
protected evidence, whether a permit has already been spent, whether the package
handed back continues the exact state the permit named, and what may be claimed
about the result. What it does not own, and cannot reach from here: any score,
array, Backtesting call, metric formula, report, task runner or pointer. It
imports the frozen Portfolio contracts *to verify them* and nothing else from
the numerical path.

Two refusals carry the protocol.

**One permit per candidate.** `admit` refuses a second issuance for a candidate
that already has one. Without that, "one-time" would mean "one at a time", and a
second evaluation on the same fixture is exactly the reselection the firewall
exists to prevent.

**One closure per permit.** `close` spends the permit. A second closure -- of the
same permit, with any package -- is refused, so a package cannot be swapped
after a receipt has been sealed.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.control.workspace_runtime.content_store import (
    CommittedIndex,
    CommittedKind,
    ContentAddressedStore,
    ContentAddressedStoreError,
)
from alphalattice.investment.portfolio_strategy_lab.application.finalization import (
    CLAIM_LIMIT_NO_RESELECTION,
    CLAIM_LIMIT_SINGLE_EVALUATION,
    CLAIM_LIMIT_SYNTHETIC_ONLY,
    PROTECTED_FIXTURE_DISPOSITION,
    ClosureDisposition,
    FinalPortfolioEvaluationPackage,
    FrozenCandidateRegistry,
    FrozenPortfolioCandidate,
    PortfolioValidationReceipt,
    ProtectedEvaluationPermit,
    ProtectedFixture,
    ProtectedPackageInspection,
    ProtectedPackageInspector,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash


class ProtectedValidationGateError(ValueError):
    """Stable refusal for an admission, permit, claim or closure failure."""


class PermitClaimRecord(BaseModel):  # type: ignore[misc]
    """Whether a permit has been spent, held by the Gate and nobody else."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["PermitClaimRecord"] = "PermitClaimRecord"
    permit_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    package_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    claimed_at: datetime
    claim_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Create a claim record with its content hash derived from the fields."""
        identity = cls.model_construct(**values, claim_hash="0" * 64).model_dump(
            mode="json", exclude={"claim_hash"}
        )
        return cls(**identity, claim_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require a timezone aware claim time and matching content hash."""
        if self.claimed_at.tzinfo is None or self.claimed_at.utcoffset() is None:
            raise ProtectedValidationGateError("protected_validation.claim_clock_invalid")
        if self.claim_hash != canonical_hash(self.model_dump(mode="json", exclude={"claim_hash"})):
            raise ProtectedValidationGateError("protected_validation.claim_identity_invalid")
        return self


_PERMITS = CommittedKind("by-candidate", "permits", ProtectedEvaluationPermit, "permit_hash")
_CLAIMS = CommittedKind("by-permit", "claims", PermitClaimRecord, "claim_hash")
_RECEIPTS = CommittedKind(
    "by-permit-receipt", "receipts", PortfolioValidationReceipt, "receipt_hash"
)


class ProtectedValidationStore:
    """The Gate's own content-addressed artifacts: permits, claims, receipts.

    Separate from the Portfolio ledger because the authority is separate. The
    Gate must be able to say "this permit was already spent" without asking the
    owner whose package it is about to judge.
    """

    def __init__(self, artifact_root: Path) -> None:
        """Open the gate's committed content store below ``artifact_root``."""
        self.root = artifact_root.resolve() / "protected-validation"
        self.content = ContentAddressedStore(self.root, uri_prefix="playpen://protected-validation")
        # The same commit rule Portfolio's finalization store uses, because the
        # two authorities have to agree about what "already happened" means
        # after a crash. A permit carries an issue time, so it can never be
        # recomputed: the index entry has to carry it or it is lost.
        self.committed = CommittedIndex(self.root, self.content)

    def publish_permit(self, value: ProtectedEvaluationPermit) -> str:
        """Commit a permit under its candidate identity."""
        return self.committed.commit(_PERMITS, value.candidate_hash, value)

    def load_permit(self, permit_hash: str) -> ProtectedEvaluationPermit:
        """Load a permit and verify its content identity."""
        return self.committed.load(_PERMITS, permit_hash)

    def find_permit_for_candidate(self, candidate_hash: str) -> ProtectedEvaluationPermit | None:
        """Find the candidate's existing permit and verify its index binding."""
        permit = self.committed.open(_PERMITS, candidate_hash)
        if permit is not None and permit.candidate_hash != candidate_hash:
            raise ProtectedValidationGateError("protected_validation.permit_index_tampered")
        return permit

    def publish_claim(self, value: PermitClaimRecord) -> str:
        """Commit a permit-spending claim under the permit identity."""
        return self.committed.commit(_CLAIMS, value.permit_hash, value)

    def find_claim(self, permit_hash: str) -> PermitClaimRecord | None:
        """Find a permit's claim and verify its index binding."""
        claim = self.committed.open(_CLAIMS, permit_hash)
        if claim is not None and claim.permit_hash != permit_hash:
            raise ProtectedValidationGateError("protected_validation.claim_index_tampered")
        return claim

    def publish_receipt(self, value: PortfolioValidationReceipt) -> str:
        """Seal the receipt under the permit it answers, before any claim.

        Committed first so the claim can never point at a receipt that is not
        there. A crash between the two leaves a receipt nobody has claimed --
        recoverable, and the closure simply completes.
        """
        return self.committed.commit(_RECEIPTS, value.permit_hash, value)

    def find_receipt_for_permit(self, permit_hash: str) -> PortfolioValidationReceipt | None:
        """Find the receipt sealed for a permit before its claim was recorded."""
        return self.committed.open(_RECEIPTS, permit_hash)

    def load_receipt(self, receipt_hash: str) -> PortfolioValidationReceipt:
        """Load a sealed receipt and verify its content identity."""
        return self.committed.load(_RECEIPTS, receipt_hash)


class ProtectedValidationGate:
    """One capability, injected into the Portfolio finalization adapter.

    Not a Desk, not a task, not a runner. Product Host constructs it and hands it
    across the port; Portfolio never learns its type.
    """

    def __init__(
        self,
        *,
        store: ProtectedValidationStore,
        registry: FrozenCandidateRegistry,
        inspector: ProtectedPackageInspector,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        """Bind the store, frozen registry, package inspector and clock."""
        self.store = store
        # Read-only, and required. Closure without it can only check a package
        # against itself, and a package is a set of hashes -- so a forged one
        # naming children that do not exist would close exactly.
        self.inspector = inspector
        # Read-only, and required. Without it the Gate could only check the
        # candidate against itself, which is what let a caller construct a
        # candidate and have it admitted in the same breath.
        self.registry = registry
        self.clock = clock

    # ------------------------------------------------------------- admission

    def admit(
        self, *, candidate: FrozenPortfolioCandidate, fixture: ProtectedFixture
    ) -> ProtectedEvaluationPermit:
        """Open the pre-existing freeze receipt, verify it, issue the single permit.

        The order matters and is the whole finding this method exists to answer.
        A candidate is self-describing: it validates its own hash over its own
        fields, so a caller who wants a permit can build one that says whatever
        admission would like to hear. What it cannot do is have published a
        freeze receipt *before* asking. So the first thing this does is open the
        registry and require the receipt to already be there.
        """
        if fixture.disposition != PROTECTED_FIXTURE_DISPOSITION:
            raise ProtectedValidationGateError("protected_validation.fixture_not_admitted")
        self._require_previously_frozen(candidate)
        if candidate.pre_protected_state.execution_ledger_hash != candidate.execution_ledger_hash:
            raise ProtectedValidationGateError("protected_validation.candidate_state_mismatch")
        if candidate.frozen_at > self.clock():
            raise ProtectedValidationGateError("protected_validation.candidate_not_yet_frozen")
        existing = self.store.find_permit_for_candidate(candidate.candidate_hash)
        if existing is not None:
            # One evaluation of one frozen candidate. A second permit is how a
            # protected fixture becomes a selection surface.
            raise ProtectedValidationGateError("protected_validation.permit_already_issued")
        permit = ProtectedEvaluationPermit.create(
            candidate_hash=candidate.candidate_hash,
            configuration_hash=candidate.spec_hash,
            pre_protected_state_hash=candidate.pre_protected_state.state_hash,
            fixture=fixture,
            issued_at=self.clock(),
        )
        self.store.publish_permit(permit)
        return permit

    # --------------------------------------------------------------- closure

    def close(
        self,
        *,
        permit: ProtectedEvaluationPermit,
        candidate: FrozenPortfolioCandidate,
        package: FinalPortfolioEvaluationPackage,
    ) -> PortfolioValidationReceipt:
        """Verify continuity and the exact package, then spend the permit."""
        try:
            sealed = self.store.load_permit(permit.permit_hash)
        except ContentAddressedStoreError as error:
            raise ProtectedValidationGateError(
                "protected_validation.permit_not_issued_here"
            ) from error
        if sealed != permit:
            raise ProtectedValidationGateError("protected_validation.permit_tampered")
        claimed = self.store.find_claim(permit.permit_hash)
        if claimed is not None:
            # The permit is spent. Closing again -- even with the same package --
            # would let a second receipt exist for one authorization.
            raise ProtectedValidationGateError("protected_validation.permit_already_claimed")
        sealed_receipt = self.store.find_receipt_for_permit(permit.permit_hash)
        if sealed_receipt is not None:
            # A crash between sealing the receipt and recording the claim. The
            # receipt is the expensive, opinionated half and it is already
            # durable; finishing the claim is the honest completion, and
            # re-deciding closure here would let one permit produce two answers
            # depending on when the machine died.
            if sealed_receipt.package_hash != package.package_hash:
                raise ProtectedValidationGateError(
                    "protected_validation.receipt_is_for_another_package"
                )
            self._record_claim(
                permit=permit, candidate=candidate, package=package, receipt=sealed_receipt
            )
            return sealed_receipt

        # Opened before judging. The inspection performs no arithmetic and
        # carries no metric -- the contract refuses a result-shaped field name --
        # so the Gate learns what the package's children *bind* without learning
        # a single number out of them.
        inspection = self.inspector.inspect(package)
        closure = self._closure(
            permit=permit, candidate=candidate, package=package, inspection=inspection
        )
        receipt = PortfolioValidationReceipt.create(
            permit_hash=permit.permit_hash,
            candidate_hash=candidate.candidate_hash,
            package_hash=package.package_hash,
            fixture_hash=permit.fixture.fixture_hash,
            pre_protected_state_hash=candidate.pre_protected_state.state_hash,
            closure=closure,
            claim_limits=(
                (
                    CLAIM_LIMIT_NO_RESELECTION,
                    CLAIM_LIMIT_SINGLE_EVALUATION,
                    CLAIM_LIMIT_SYNTHETIC_ONLY,
                )
                if closure == "CLOSED_EXACT"
                else (CLAIM_LIMIT_SINGLE_EVALUATION,)
            ),
            sealed_at=self.clock(),
        )
        # Receipt first, claim second. The claim is what spends the permit, so
        # it must never exist without the receipt it spends it for.
        self.store.publish_receipt(receipt)
        self._record_claim(permit=permit, candidate=candidate, package=package, receipt=receipt)
        return receipt

    def _record_claim(
        self,
        *,
        permit: ProtectedEvaluationPermit,
        candidate: FrozenPortfolioCandidate,
        package: FinalPortfolioEvaluationPackage,
        receipt: PortfolioValidationReceipt,
    ) -> None:
        self.store.publish_claim(
            PermitClaimRecord.create(
                permit_hash=permit.permit_hash,
                candidate_hash=candidate.candidate_hash,
                package_hash=package.package_hash,
                receipt_hash=receipt.receipt_hash,
                claimed_at=self.clock(),
            )
        )

    def _require_previously_frozen(self, candidate: FrozenPortfolioCandidate) -> None:
        """Two independently sealed records have to agree, or there is no freeze.

        Both halves are needed. The receipt alone proves something was frozen;
        the registered candidate proves it was *this* one, byte for byte. A Gate
        that checked only the receipt would admit a candidate whose fields had
        been edited after registration, because the receipt names the candidate
        by hash and a hash the caller supplies is not a hash the caller earned.
        """
        receipt = self.registry.open_freeze_receipt(candidate.candidate_hash)
        if receipt is None:
            raise ProtectedValidationGateError("protected_validation.candidate_was_not_frozen")
        registered = self.registry.open_candidate(candidate.candidate_hash)
        if registered is None:
            raise ProtectedValidationGateError("protected_validation.candidate_not_registered")
        if registered != candidate:
            raise ProtectedValidationGateError("protected_validation.candidate_differs_from_frozen")
        if not receipt.describes(candidate):
            raise ProtectedValidationGateError("protected_validation.freeze_receipt_mismatch")
        if receipt.frozen_at > self.clock():
            raise ProtectedValidationGateError("protected_validation.candidate_not_yet_frozen")

    # ---------------------------------------------------------- resumption

    def reopen_permit(
        self, *, candidate: FrozenPortfolioCandidate
    ) -> ProtectedEvaluationPermit | None:
        """Read a candidate's existing permit without issuing another."""
        return self.store.find_permit_for_candidate(candidate.candidate_hash)

    def reopen_receipt(
        self, *, permit: ProtectedEvaluationPermit
    ) -> PortfolioValidationReceipt | None:
        """Read the receipt produced by a spent permit, when present."""
        claim = self.store.find_claim(permit.permit_hash)
        return None if claim is None else self.store.load_receipt(claim.receipt_hash)

    @staticmethod
    def _closure(
        *,
        permit: ProtectedEvaluationPermit,
        candidate: FrozenPortfolioCandidate,
        package: FinalPortfolioEvaluationPackage,
        inspection: ProtectedPackageInspection,
    ) -> ClosureDisposition:
        """Which of the eight ways this could fail actually happened.

        Distinguished rather than collapsed to a boolean: a caller that gets
        `REFUSED_STATE_DISCONTINUOUS` knows the book moved, and a caller that
        gets `REFUSED_CANDIDATE_MISMATCH` knows it sent the wrong candidate.

        The order runs outward: what the permit says, then what the package
        claims, then what the package's children actually are. The last group is
        the one that needs the inspection, and it is the one a self-consistent
        forgery survives without it.
        """
        if permit.candidate_hash != candidate.candidate_hash:
            return "REFUSED_CANDIDATE_MISMATCH"
        if permit.configuration_hash != candidate.spec_hash:
            return "REFUSED_CANDIDATE_MISMATCH"
        if permit.pre_protected_state_hash != candidate.pre_protected_state.state_hash:
            return "REFUSED_STATE_DISCONTINUOUS"
        if package.continued_from_state_hash != candidate.pre_protected_state.state_hash:
            return "REFUSED_STATE_DISCONTINUOUS"
        if package.permit_hash != permit.permit_hash:
            return "REFUSED_PERMIT_INVALID"
        if (
            package.candidate_hash != candidate.candidate_hash
            or package.fixture_hash != permit.fixture.fixture_hash
        ):
            return "REFUSED_PACKAGE_MISMATCH"
        if inspection.package_hash != package.package_hash:
            return "REFUSED_PACKAGE_MISMATCH"
        if inspection.absent_children:
            # The whole point of opening them: a package can name anything.
            return "REFUSED_PACKAGE_CHILDREN_ABSENT"
        if (
            inspection.result_spec_hash != candidate.spec_hash
            or inspection.result_spec_hash != permit.configuration_hash
            or inspection.program_holdings_spec_hash != candidate.holdings_spec_hash
            or inspection.report_control_receipt_hash != candidate.control_receipt_hash
        ):
            # The configuration the protected numbers were produced under. Cost,
            # study window, report unit, benchmark view and every holdings
            # control decide what those numbers are, and a package produced under
            # a changed one is a different evaluation wearing this permit's
            # shape. Checked here and not only at the runner: the Gate has to
            # refuse a forged package nobody's executor ever saw.
            return "REFUSED_CONFIGURATION_MISMATCH"
        if (
            inspection.result_execution_ledger_hash != package.protected_execution_ledger_hash
            or inspection.result_economic_ledger_hash != package.protected_economic_ledger_hash
            or inspection.result_report_hash != package.protected_report_hash
            or inspection.ledger_program_hash != inspection.result_program_hash
            or inspection.economic_execution_ledger_hash != package.protected_execution_ledger_hash
            or inspection.report_program_hash != inspection.result_program_hash
            or inspection.report_execution_ledger_hash != package.protected_execution_ledger_hash
            or inspection.report_economic_ledger_hash != package.protected_economic_ledger_hash
            or inspection.report_comparison_hash is None
            or inspection.comparison_execution_ledger_hash
            != package.protected_execution_ledger_hash
            or inspection.comparison_economic_ledger_hash != package.protected_economic_ledger_hash
        ):
            # Children that exist but do not refer to each other: a package
            # assembled from two different finalizations reads as valid until
            # somebody opens all four and checks that they agree.
            return "REFUSED_PACKAGE_BINDINGS_INCOHERENT"
        frozen_boundary = candidate.pre_protected_state.terminal_boundary.boundary_hash
        if (
            inspection.initial_boundary_hash != frozen_boundary
            or inspection.program_continued_from_state_hash != frozen_boundary
        ):
            # The path has to *open* on the permitted book, and its Program has
            # to say so. Either alone would admit a run that started somewhere
            # else and relabelled itself afterwards.
            return "REFUSED_STATE_DISCONTINUOUS"
        fixture = permit.fixture
        if (
            inspection.formation_sessions != tuple(fixture.sessions)
            or inspection.listing_axis_hash != fixture.listing_axis_hash
            or package.protected_formation_count != len(fixture.sessions)
        ):
            return "REFUSED_PACKAGE_AXIS_MISMATCH"
        if (
            inspection.program_formation_start != fixture.sessions[0]
            or inspection.program_formation_end != fixture.sessions[-1]
            or inspection.program_formation_count != len(fixture.sessions)
            or inspection.program_listing_count != package.protected_listing_count
            or inspection.program_listing_count != inspection.ledger_listing_count
            or inspection.program_formation_sessions_hash
            != inspection.ledger_formation_sessions_hash
            or inspection.program_ordered_listing_ids_hash
            != inspection.ledger_ordered_listing_ids_hash
        ):
            # A Program and the path it produced can disagree. The ledger already
            # matched the fixture above; this is the other half -- the Program's
            # own axis claims have to match both, or a valid ledger could close
            # under a Program compiled for another window.
            return "REFUSED_PACKAGE_AXIS_MISMATCH"
        return "CLOSED_EXACT"


__all__ = [
    "PermitClaimRecord",
    "ProtectedValidationGate",
    "ProtectedValidationGateError",
    "ProtectedValidationStore",
]
