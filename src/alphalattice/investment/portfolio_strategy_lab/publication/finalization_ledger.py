"""Durable finalization artifacts: freeze receipt, candidate, package, handoff.

Portfolio's own store, beside the Portfolio ledger rather than inside it. A
candidate and a protected package are not descendants of a declared path -- they
are what a *finalization* produced about one -- and filing them under the
research ledger's indices would let a reuse lookup for a development request
return a protected artifact.

Every write here goes through one commit rule, because every one of them sits in
a window a crash can land in.

**The index carries the artifact.** A commit writes a single index entry that
contains the artifact's whole canonical payload, with one atomic replace; the
content file is a materialization of that entry and is written afterwards. So
there is no half-committed state a reader can misread: either the index entry is
absent and nothing happened, or it is present and the artifact is fully
recoverable from it. A reader that finds the entry and no content file heals
forward by republishing from the payload it already holds.

The alternative -- content first, index last -- cannot close these windows,
because three of the four artifacts carry a timestamp. Recomputing them after a
crash produces a *different* identity, so an orphaned content file could never
be found again and the protocol would issue a second permit, or evaluate a
second time, for want of a file that was already on disk.
"""

from __future__ import annotations

from pathlib import Path

from alphalattice.control.workspace_runtime.content_store import (
    CommittedIndex,
    CommittedKind,
    ContentAddressedStore,
)
from alphalattice.investment.portfolio_strategy_lab.application.finalization import (
    FinalPortfolioEvaluationPackage,
    FrozenPortfolioCandidate,
    PortfolioCandidateFreezeReceipt,
    ProtectedContinuationResult,
    ReleasedPortfolioArtifacts,
    ValidatedPortfolioHandoff,
)


class FinalizationLedgerStoreError(ValueError):
    """Stable finalization storage refusal."""


_CANDIDATES = CommittedKind(
    "by-candidate", "candidates", FrozenPortfolioCandidate, "candidate_hash"
)
_FREEZE_RECEIPTS = CommittedKind(
    "freeze-receipts", "freeze-receipts", PortfolioCandidateFreezeReceipt, "receipt_hash"
)
_CONTINUATIONS = CommittedKind(
    "by-permit-continuation", "continuations", ProtectedContinuationResult, "result_hash"
)
_PACKAGES = CommittedKind("by-permit", "packages", FinalPortfolioEvaluationPackage, "package_hash")
_HANDOFFS = CommittedKind("by-package", "handoffs", ValidatedPortfolioHandoff, "handoff_hash")
_RELEASES = CommittedKind("released", "releases", ReleasedPortfolioArtifacts, "marker_hash")


class PortfolioFinalizationStore:
    """Content-addressed finalization artifacts behind four one-time commits."""

    def __init__(self, artifact_root: Path) -> None:
        """Bind finalization artifacts and deterministic committed-index ownership.

        Args:
            artifact_root: Caller-owned workspace artifact directory.
        """
        self.root = artifact_root.resolve() / "portfolio-strategy-lab" / "finalization"
        self.content = ContentAddressedStore(
            self.root, uri_prefix="playpen://portfolio-strategy-lab/finalization"
        )
        self.committed = CommittedIndex(self.root, self.content)

    # ------------------------------------------------------------ freeze

    def publish_freeze(
        self, *, candidate: FrozenPortfolioCandidate, receipt: PortfolioCandidateFreezeReceipt
    ) -> str:
        """Register one frozen candidate: the candidate itself and its receipt.

        Both under the candidate hash, and the receipt last. A Gate that finds
        the receipt is guaranteed to find the candidate beside it, so admission
        never has to reason about a half-registered freeze.
        """
        if not receipt.describes(candidate):
            raise FinalizationLedgerStoreError("portfolio_finalization.freeze_receipt_mismatch")
        self.committed.commit(_CANDIDATES, candidate.candidate_hash, candidate)
        return self.committed.commit(_FREEZE_RECEIPTS, candidate.candidate_hash, receipt)

    def open_freeze_receipt(self, candidate_hash: str) -> PortfolioCandidateFreezeReceipt | None:
        """Reopen one registered candidate freeze receipt by its exact committed key.

        Args:
            candidate_hash: Exact registered lookup identity.

        Returns:
            Committed validated artifact, or None without an entry.
        """
        return self.committed.open(_FREEZE_RECEIPTS, candidate_hash)

    def open_candidate(self, candidate_hash: str) -> FrozenPortfolioCandidate | None:
        """Reopen one registered frozen candidate by its exact committed key.

        Args:
            candidate_hash: Exact registered lookup identity.

        Returns:
            Committed validated artifact, or None without an entry.
        """
        return self.committed.open(_CANDIDATES, candidate_hash)

    def find_candidate_for_result(self, result_hash: str) -> str | None:
        """The candidate frozen from one development result, if any.

        A read over the by-candidate index rather than a second index: freezes
        are few, and a result that was frozen twice is the ambiguity this
        refuses rather than resolves by order.
        """
        root = self.root / "index" / "by-candidate"
        if not root.is_dir():
            return None
        matches = [
            candidate.candidate_hash
            for path in sorted(root.glob("*.json"))
            if (candidate := self.open_candidate(path.stem)) is not None
            and candidate.development_result_hash == result_hash
        ]
        if len(matches) > 1:
            raise FinalizationLedgerStoreError("portfolio_finalization.result_frozen_twice")
        return matches[0] if matches else None

    def load_candidate(self, candidate_hash: str) -> FrozenPortfolioCandidate:
        """Require one registered frozen candidate to reopen.

        Args:
            candidate_hash: Exact frozen candidate identity.

        Returns:
            Registered validated frozen candidate.

        Raises:
            FinalizationLedgerStoreError: Candidate is not registered.
        """
        candidate = self.open_candidate(candidate_hash)
        if candidate is None:
            raise FinalizationLedgerStoreError("portfolio_finalization.candidate_not_registered")
        return candidate

    # ------------------------------------------------------- continuation

    def publish_continuation(self, value: ProtectedContinuationResult) -> str:
        """Seal what the protected run produced, keyed by the permit that allowed it.

        Committed *before* the package is built, so the stage that computes can
        report a durable identity. Recovery reopens this instead of running the
        continuation a second time.
        """
        return self.committed.commit(_CONTINUATIONS, value.permit_hash, value)

    def find_continuation_for_permit(self, permit_hash: str) -> ProtectedContinuationResult | None:
        """Reopen one committed protected continuation by its exact committed key.

        Args:
            permit_hash: Exact registered lookup identity.

        Returns:
            Committed validated artifact, or None without an entry.
        """
        return self.committed.open(_CONTINUATIONS, permit_hash)

    # ------------------------------------------------------------- package

    def publish_package(self, value: FinalPortfolioEvaluationPackage) -> str:
        """Commit the exact finalization package under its declared predecessor key.

        Args:
            value: Validated sealed package contract.

        Returns:
            Committed content URI; index ownership refuses a conflicting value for the same key.
        """
        return self.committed.commit(_PACKAGES, value.permit_hash, value)

    def load_package(self, package_hash: str) -> FinalPortfolioEvaluationPackage:
        """Reopen the exact committed finalization package.

        Args:
            package_hash: Exact artifact self identity.

        Returns:
            Validated committed artifact.
        """
        return self.committed.load(_PACKAGES, package_hash)

    def find_package_for_permit(self, permit_hash: str) -> FinalPortfolioEvaluationPackage | None:
        """The package this permit produced, if the continuation already sealed one."""
        return self.committed.open(_PACKAGES, permit_hash)

    # ------------------------------------------------------------- handoff

    def publish_handoff(self, value: ValidatedPortfolioHandoff) -> str:
        """Commit the exact finalization handoff under its declared predecessor key.

        Args:
            value: Validated sealed handoff contract.

        Returns:
            Committed content URI; index ownership refuses a conflicting value for the same key.
        """
        return self.committed.commit(_HANDOFFS, value.package_hash, value)

    def load_handoff(self, handoff_hash: str) -> ValidatedPortfolioHandoff:
        """Reopen the exact committed finalization handoff.

        Args:
            handoff_hash: Exact artifact self identity.

        Returns:
            Validated committed artifact.
        """
        return self.committed.load(_HANDOFFS, handoff_hash)

    def find_handoff_for_package(self, package_hash: str) -> ValidatedPortfolioHandoff | None:
        """Reopen one committed validated handoff by its exact committed key.

        Args:
            package_hash: Exact registered lookup identity.

        Returns:
            Committed validated artifact, or None without an entry.
        """
        return self.committed.open(_HANDOFFS, package_hash)

    # ------------------------------------------------------------- release

    def publish_release(self, value: ReleasedPortfolioArtifacts) -> str:
        """Commit the final release that makes promoted artifacts observable.

        Commit the release. The last write of a finalization, and the only
        one that makes the promoted artifacts observable.
        """
        return self.committed.commit(_RELEASES, value.package_hash, value)

    def find_release_for_package(self, package_hash: str) -> ReleasedPortfolioArtifacts | None:
        """Whether this package has actually been released.

        The question every public reader of a protected result has to ask first.
        Copied files answer "is it here"; only this answers "may it be read".
        """
        return self.committed.open(_RELEASES, package_hash)


__all__ = ["FinalizationLedgerStoreError", "PortfolioFinalizationStore"]
