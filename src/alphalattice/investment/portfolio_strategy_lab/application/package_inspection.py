"""Open a pending protected package's children and report what they bind.

The Validation Gate has to answer a question it cannot answer from the package
alone: are the artifacts this package names actually there, do they refer to each
other, and do they describe the window the permit authorized? A package is a set
of hashes, so a forged one whose children do not exist -- or whose children are
some *other* finalization's -- is internally perfect.

Portfolio owns this because only Portfolio knows where a pending protected result
lives and what its descendants are. The Gate consumes it across a port and never
learns which store answered.

The reader is deliberately incurious. It opens artifacts, copies out identities,
and reports absences by name. It computes nothing, compares nothing, and reaches
no verdict -- comparing what it found against what was permitted is the Gate's
job, and a reader that also judged would be the second authority this whole
protocol exists to avoid.
"""

from __future__ import annotations

from collections.abc import Callable

from alphalattice.investment.portfolio_strategy_lab.application.advancement import (
    ordered_listing_axis_hash,
)
from alphalattice.investment.portfolio_strategy_lab.application.finalization import (
    FinalPortfolioEvaluationPackage,
    ProtectedPackageInspection,
)
from alphalattice.investment.portfolio_strategy_lab.publication.portfolio_ledger import (
    PortfolioLedgerStore,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash


class PendingPackageInspector:
    """Reads one pending namespace. Constructed by composition, injected as a port."""

    def __init__(self, pending: PortfolioLedgerStore) -> None:
        """Bind pending package inspection to its ledger readback owner.

        Args:
            pending: Current pending-package ledger store.
        """
        self.pending = pending

    def inspect(self, package: FinalPortfolioEvaluationPackage) -> ProtectedPackageInspection:
        """Open the result and its four descendants, or name what is missing.

        Every open is guarded separately so the report distinguishes "the result
        is not there" from "the result is there and names a report that is not".
        Both are refusals, and they are different mistakes.
        """
        absent: list[str] = []
        result = self._open(
            "result", package.protected_result_hash, absent, self.pending.load_result
        )
        if result is None:
            return ProtectedPackageInspection.create(
                package_hash=package.package_hash, absent_children=tuple(absent)
            )
        program = self._open("program", result.program_hash, absent, self.pending.load_program)
        ledger = self._open(
            "execution_ledger", result.execution_ledger_hash, absent, self.pending.load_execution
        )
        economics = self._open(
            "economic_ledger", result.economic_ledger_hash, absent, self.pending.load_economics
        )
        report = self._open("report", result.report_hash, absent, self.pending.load_report)
        comparison = (
            None
            if report is None
            else self._open(
                "benchmark_comparison",
                report.benchmark_comparison_hash,
                absent,
                self.pending.load_comparison,
            )
        )
        # The package's own claims about its children are checked for existence
        # here too: it names them independently of the result, and the two can
        # disagree.
        for label, identity, load in (
            (
                "package_execution_ledger",
                package.protected_execution_ledger_hash,
                self.pending.load_execution,
            ),
            (
                "package_economic_ledger",
                package.protected_economic_ledger_hash,
                self.pending.load_economics,
            ),
            ("package_report", package.protected_report_hash, self.pending.load_report),
        ):
            self._open(label, identity, absent, load)
        return ProtectedPackageInspection.create(
            package_hash=package.package_hash,
            absent_children=tuple(absent),
            result_spec_hash=result.spec_hash,
            program_holdings_spec_hash=None if program is None else program.holdings_spec_hash,
            report_control_receipt_hash=(
                None if report is None else report.control_receipt.receipt_hash
            ),
            program_formation_start=None if program is None else program.formation_start,
            program_formation_end=None if program is None else program.formation_end,
            program_formation_count=None if program is None else program.formation_count,
            program_formation_sessions_hash=(
                None if program is None else program.formation_sessions_hash
            ),
            program_listing_count=None if program is None else program.listing_count,
            program_ordered_listing_ids_hash=(
                None if program is None else program.ordered_listing_ids_hash
            ),
            ledger_formation_sessions_hash=(
                None
                if ledger is None
                else str(
                    canonical_hash(tuple(value.isoformat() for value in ledger.formation_sessions))
                )
            ),
            ledger_ordered_listing_ids_hash=(
                None if ledger is None else str(canonical_hash(tuple(ledger.ordered_listing_ids)))
            ),
            ledger_listing_count=None if ledger is None else len(ledger.ordered_listing_ids),
            result_program_hash=result.program_hash,
            result_execution_ledger_hash=result.execution_ledger_hash,
            result_economic_ledger_hash=result.economic_ledger_hash,
            result_report_hash=result.report_hash,
            ledger_program_hash=None if ledger is None else ledger.program_hash,
            economic_execution_ledger_hash=(
                None if economics is None else economics.execution_ledger_hash
            ),
            report_program_hash=None if report is None else report.program_hash,
            report_execution_ledger_hash=(None if report is None else report.execution_ledger_hash),
            report_economic_ledger_hash=(None if report is None else report.economic_ledger_hash),
            report_comparison_hash=(None if report is None else report.benchmark_comparison_hash),
            comparison_execution_ledger_hash=(
                None if comparison is None else comparison.execution_ledger_hash
            ),
            comparison_economic_ledger_hash=(
                None if comparison is None else comparison.economic_ledger_hash
            ),
            program_continued_from_state_hash=(
                None if program is None else program.continued_from_state_hash
            ),
            initial_boundary_hash=(
                None
                if ledger is None or ledger.initial_boundary is None
                else ledger.initial_boundary.boundary_hash
            ),
            formation_sessions=() if ledger is None else tuple(ledger.formation_sessions),
            listing_axis_hash=(
                None
                if ledger is None
                else ordered_listing_axis_hash(tuple(ledger.ordered_listing_ids))
            ),
        )

    @staticmethod
    def _open[ArtifactT](
        label: str,
        identity: str,
        absent: list[str],
        load: Callable[[str], ArtifactT],
    ) -> ArtifactT | None:
        """Open one artifact, or record its label among the absences.

        Every failure mode collapses to "absent" on purpose. A hash that does not
        resolve, one whose file will not parse, and one whose stored identity
        disagrees with the name it was asked for are three ways of saying the
        package points at something that is not there, and the Gate refuses on
        all three identically.
        """

        try:
            return load(identity)
        except Exception:
            absent.append(label)
            return None


__all__ = ["PendingPackageInspector"]
