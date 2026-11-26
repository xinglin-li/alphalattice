"""Protected finalization: what Portfolio owns, and the one port it consumes.

A finalization is not a second research run. It takes a candidate and a
configuration that were frozen *before* anyone asked about protected evidence,
continues the exact sealed book state through the same Backtesting owner the
development path used, and seals a package that stays invisible until a gate
verifies it. Every type here exists to make one of those clauses checkable.

Three ownership lines run through this module and none of them may be crossed.

**Portfolio owns the numbers.** The frozen candidate, the sealed pre-protected
state, the protected continuation and the package are Portfolio's; the Gate never
computes one of them. So they live here, beside the ledgers they are built from.

**The Gate owns admission and closure.** It decides whether a candidate was
really frozen, issues at most one permit, verifies closure and seals a receipt
that carries no metric. `ProtectedEvaluationPort` is the entire surface Portfolio
sees of it -- two calls, typed both ways, no score, no array, no calculator, no
callback and no task runner. The contracts cross that port, so they are declared
here rather than in the Gate: Portfolio must be able to consume a permit without
importing the Validation runtime.

**Nothing returns to selection.** The receipt states its claim limits, the
handoff copies no metric, and neither carries a mutable pointer. A protected
result that could be read back into a comparison would make the whole protocol
decorative.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.investment.portfolio_strategy_lab.application.advancement import (
    ordered_listing_axis_hash,
    ordered_session_axis_hash,
)
from alphalattice.investment.portfolio_strategy_lab.application.contracts import (
    PortfolioExecutionLedger,
    SealedPortfolioBoundaryState,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.sealing import seal_model_from_dump

PROTECTED_FIXTURE_DISPOSITION: Literal["ISOLATED_SYNTHETIC_NON_RESERVED_FIXTURE"] = (
    "ISOLATED_SYNTHETIC_NON_RESERVED_FIXTURE"
)
"""Every permit in this Gate names a fixture, never the reserved region.

Stated as a value on the contract rather than a comment, so a run against real
protected evidence could not be produced by this code path at all -- there is no
other member to pass.
"""

CLAIM_LIMIT_NO_RESELECTION: Literal["PROTECTED_EVIDENCE_MAY_NOT_RETURN_TO_SELECTION_OR_TUNING"] = (
    "PROTECTED_EVIDENCE_MAY_NOT_RETURN_TO_SELECTION_OR_TUNING"
)
CLAIM_LIMIT_SINGLE_EVALUATION = "ONE_EVALUATION_OF_ONE_FROZEN_CANDIDATE_ON_ONE_FIXTURE"
CLAIM_LIMIT_SYNTHETIC_ONLY = "SYNTHETIC_FIXTURE_CARRIES_NO_SCIENTIFIC_CLAIM"

_RESULT_SHAPED_FIELD_WORDS = (
    "return",
    "sharpe",
    "alpha",
    "beta",
    "wealth",
    "turnover",
    "drawdown",
    "volatility",
    "metric",
    "score",
    "pnl",
)
"""Field-name words that would make the validation receipt a results surface.

Module-level rather than a class attribute: a tuple on a pydantic model becomes
a private attribute descriptor, and the check would silently stop checking.
"""

ClosureDisposition = Literal[
    "CLOSED_EXACT",
    "REFUSED_CANDIDATE_MISMATCH",
    "REFUSED_STATE_DISCONTINUOUS",
    "REFUSED_PACKAGE_MISMATCH",
    "REFUSED_PERMIT_INVALID",
    "REFUSED_PACKAGE_CHILDREN_ABSENT",
    "REFUSED_PACKAGE_BINDINGS_INCOHERENT",
    "REFUSED_PACKAGE_AXIS_MISMATCH",
    "REFUSED_CONFIGURATION_MISMATCH",
]
"""Eight outcomes, because eight different things went wrong.

The last four are what a package *contains* rather than what it claims: a
package is a set of hashes, and a self-consistent one whose children do not
exist, do not refer to each other, describe some other window, or were produced
under some other configuration would pass every check that reads only the
package itself.
"""


class PortfolioFinalizationError(ValueError):
    """Stable refusal for a finalization identity, authority or closure failure."""


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class SealedPreProtectedState(_Contract):
    """The exact book at the last development formation, sealed before any permit.

    Sealed *before* the Gate is asked anything. That order is the point: a state
    captured after a permit was issued could have been chosen to suit the
    protected window, and no later check could tell the difference.
    """

    kind: Literal["SealedPreProtectedState"] = "SealedPreProtectedState"
    execution_ledger_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    last_formation_session: date
    formation_count: int = Field(gt=0)
    formation_axis_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    listing_axis_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    final_weights_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    """One component of the boundary below, kept because existing readers cite
    it by this name. `terminal_boundary` is the whole state."""

    terminal_boundary: SealedPortfolioBoundaryState
    """Every component the continuation has to restore: both books, both cash
    balances and the policy's sleeve state.

    Required, not optional. A pre-protected state that named only the reference
    book could be sealed for a ledger that cannot actually be continued, and the
    failure would surface as a plausible protected path rather than a refusal.
    """

    cumulative_gross_wealth: float = Field(gt=0.0)
    cumulative_one_way_turnover: float = Field(ge=0.0)
    missed_execution_count: int = Field(ge=0)
    state_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def of(cls, ledger: PortfolioExecutionLedger) -> Self:
        """Read the boundary straight off the admitted ledger, computing nothing new.

        Refuses a ledger that did not seal its terminal boundary. That is the
        predecessor shape, and it cannot be continued: the reference book alone
        is one drift away from the book the next trade is priced against, and no
        later check could tell a wrong start from a right one.
        """
        if ledger.final_boundary is None:
            raise PortfolioFinalizationError(
                "portfolio_finalization.ledger_sealed_no_terminal_boundary"
            )
        values: dict[str, object] = {
            "kind": "SealedPreProtectedState",
            "execution_ledger_hash": ledger.ledger_hash,
            "last_formation_session": ledger.formation_sessions[-1],
            "formation_count": len(ledger.formation_sessions),
            "formation_axis_hash": ordered_session_axis_hash(ledger.formation_sessions),
            "listing_axis_hash": ordered_listing_axis_hash(ledger.ordered_listing_ids),
            "final_weights_hash": ledger.final_weights_hash,
            "terminal_boundary": ledger.final_boundary,
            "cumulative_gross_wealth": _compound(ledger.gross_simple_returns),
            "cumulative_one_way_turnover": float(sum(ledger.one_way_turnovers)),
            "missed_execution_count": ledger.missed_execution_count,
        }
        provisional = cls.model_construct(**values, state_hash="0" * 64).model_dump(
            mode="json", exclude={"state_hash"}
        )
        return cls(**provisional, state_hash=canonical_hash(provisional))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require terminal optimizer/session authority and exact pre-finalization state identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PortfolioFinalizationError: Terminal reference differs from final weights, terminal date
                differs from the last formation or state_hash differs.
        """
        if self.terminal_boundary.optimizer_reference_hash != self.final_weights_hash:
            raise PortfolioFinalizationError("portfolio_finalization.state_boundary_disagrees")
        if self.terminal_boundary.formation_session != self.last_formation_session:
            raise PortfolioFinalizationError(
                "portfolio_finalization.state_boundary_session_invalid"
            )
        if self.state_hash != canonical_hash(self.model_dump(mode="json", exclude={"state_hash"})):
            raise PortfolioFinalizationError("portfolio_finalization.state_identity_invalid")
        return self


def _compound(returns: tuple[float, ...]) -> float:
    wealth = 1.0
    for value in returns:
        wealth *= 1.0 + value
    return wealth


class FrozenPortfolioCandidate(_Contract):
    """One candidate and configuration, frozen with the development run that made it.

    Everything here is an identity the development path already published. The
    contract adds no number of its own, which is what lets the Gate verify
    "this was frozen beforehand" without re-running anything.
    """

    kind: Literal["FrozenPortfolioCandidate"] = "FrozenPortfolioCandidate"
    workspace_id: str = Field(min_length=1, max_length=120)
    spec_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    holdings_spec_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    control_receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    """The exact installed control values the development report was sealed with.

    Sealed on the candidate so closure can check the *configuration* a protected
    package was produced under without holding a spec. The spec hash alone is not
    enough at the Gate: it is one field on a result, and a package assembled
    around a differently configured run carries a different control receipt while
    every other identity still looks plausible.
    """

    program_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    numerical_input_assembly_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """One digest over every input the Program bound that decides a number.

    `None` for a predecessor Program compiled before those inputs were bound. A
    candidate without it can be admitted for readback but can authorize neither
    unit reuse nor strong replay: there is nothing to prove the numbers came from
    the inputs the Program claims.
    """

    development_task_id: str = Field(min_length=1, max_length=64)
    development_run_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    """The verified completed-run projection this candidate was frozen against.

    A free-form task id is a claim; this is the identity of what Task Control
    actually holds -- the kind, the terminal lifecycle, and the result identities
    the task's own verified evidence names. Freezing checks the artifacts against
    it, so a candidate can no longer be frozen over a task that never ran, never
    finished, or published something else.
    """

    development_result_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_ledger_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    economic_ledger_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    pre_protected_state: SealedPreProtectedState
    frozen_at: datetime
    candidate_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal one explicitly frozen development candidate.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical candidate_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        identity = cls.model_construct(**values, candidate_hash="0" * 64).model_dump(
            mode="json", exclude={"candidate_hash"}
        )
        return cls(**identity, candidate_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require candidate/state ledger agreement and an aware freeze clock.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PortfolioFinalizationError: Pre-finalization state refers to another execution ledger,
                freeze time lacks timezone authority or candidate_hash differs.
        """
        if self.pre_protected_state.execution_ledger_hash != self.execution_ledger_hash:
            raise PortfolioFinalizationError(
                "portfolio_finalization.candidate_state_ledger_mismatch"
            )
        if self.frozen_at.tzinfo is None or self.frozen_at.utcoffset() is None:
            raise PortfolioFinalizationError("portfolio_finalization.candidate_clock_invalid")
        if self.candidate_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"candidate_hash"})
        ):
            raise PortfolioFinalizationError("portfolio_finalization.candidate_identity_invalid")
        return self

    @property
    def replayable(self) -> bool:
        """Whether this candidate may authorize unit reuse or strong replay."""
        return self.numerical_input_assembly_hash is not None


class CompletedDevelopmentRun(_Contract):
    """What a completed development Task durably says about the run it produced.

    A projection, built by whoever can read Task Control, so freezing does not
    have to trust a task id somebody typed. The id alone is a string: it proves
    nothing about whether that task finished, what kind of task it was, or
    whether the artifacts being frozen are the ones it actually published.
    """

    kind: Literal["CompletedDevelopmentRun"] = "CompletedDevelopmentRun"
    task_id: str = Field(min_length=1, max_length=64)
    task_kind: str = Field(min_length=1, max_length=120)
    lifecycle: str = Field(min_length=1, max_length=40)
    spec_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    program_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    authorities_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    published_result_hashes: tuple[str, ...] = Field(min_length=1)
    """Every result identity the task's verified stage receipts name.

    Read off durable evidence rather than from the caller, which is what makes
    "this task published that result" checkable at all.
    """

    projection_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal one completed development-run projection.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical projection_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        identity = cls.model_construct(**values, projection_hash="0" * 64).model_dump(
            mode="json", exclude={"projection_hash"}
        )
        return cls(**identity, projection_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require exact declared development-run projection identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PortfolioFinalizationError: The canonical projection differs from projection_hash.
        """
        if self.projection_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"projection_hash"})
        ):
            raise PortfolioFinalizationError("portfolio_finalization.run_identity_invalid")
        return self

    @property
    def completed(self) -> bool:
        """Check whether the retained development-run lifecycle succeeded.

        Returns:
            True only for SUCCEEDED; this property performs no execution.
        """
        return self.lifecycle == "SUCCEEDED"


class CompletedDevelopmentTaskRegistry(Protocol):
    """Read-only access to what Task Control durably recorded about one task."""

    def open_completed_run(self, task_id: str) -> CompletedDevelopmentRun | None:
        """The projection for this task, or `None` if there is no such task."""
        ...


class PortfolioCandidateFreezeReceipt(_Contract):
    """Durable proof that a candidate was frozen before anyone asked for a permit.

    The prerequisite the Gate checks, and the reason it is a separate artifact
    rather than a field on the candidate: a candidate is self-describing, so a
    caller can build one that says anything and it will validate. What a caller
    cannot do is produce a freeze receipt that was already in the registry when
    admission opened it.

    It carries every binding the candidate claims, so verifying the candidate
    against it is a comparison of two independently sealed records rather than a
    record checked against itself.
    """

    kind: Literal["PortfolioCandidateFreezeReceipt"] = "PortfolioCandidateFreezeReceipt"
    workspace_id: str = Field(min_length=1, max_length=120)
    candidate_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    development_task_id: str = Field(min_length=1, max_length=64)
    development_run_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    development_result_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    program_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    spec_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    control_receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    numerical_input_assembly_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    execution_ledger_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    economic_ledger_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    pre_protected_state_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    frozen_at: datetime
    receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def of(cls, candidate: FrozenPortfolioCandidate) -> Self:
        """Seal a metric-free receipt selecting the exact frozen candidate and run lineage.

        Args:
            candidate: Explicitly frozen candidate with aware clock and retained state/result/ledger
                identities.

        Returns:
            Validated receipt with the candidate, development, execution/economic and
            pre-finalization state identities.
        """
        values: dict[str, object] = {
            "kind": "PortfolioCandidateFreezeReceipt",
            "workspace_id": candidate.workspace_id,
            "candidate_hash": candidate.candidate_hash,
            "development_task_id": candidate.development_task_id,
            "development_run_hash": candidate.development_run_hash,
            "development_result_hash": candidate.development_result_hash,
            "program_hash": candidate.program_hash,
            "spec_hash": candidate.spec_hash,
            "control_receipt_hash": candidate.control_receipt_hash,
            "numerical_input_assembly_hash": candidate.numerical_input_assembly_hash,
            "execution_ledger_hash": candidate.execution_ledger_hash,
            "economic_ledger_hash": candidate.economic_ledger_hash,
            "pre_protected_state_hash": candidate.pre_protected_state.state_hash,
            "frozen_at": candidate.frozen_at,
        }
        identity = cls.model_construct(**values, receipt_hash="0" * 64).model_dump(
            mode="json", exclude={"receipt_hash"}
        )
        return cls(**identity, receipt_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require a timezone-aware freeze receipt and exact identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PortfolioFinalizationError: Freeze time lacks timezone authority or receipt_hash
                differs.
        """
        if self.frozen_at.tzinfo is None or self.frozen_at.utcoffset() is None:
            raise PortfolioFinalizationError("portfolio_finalization.freeze_clock_invalid")
        if self.receipt_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"receipt_hash"})
        ):
            raise PortfolioFinalizationError("portfolio_finalization.freeze_identity_invalid")
        return self

    def describes(self, candidate: FrozenPortfolioCandidate) -> bool:
        """Whether this receipt is the one that froze exactly this candidate."""
        return self == type(self).of(candidate)


class ProtectedFixture(_Contract):
    """An isolated, non-reserved synthetic window a permit may name.

    The fixture is a first-class contract rather than a pair of dates because the
    firewall is about *which* evidence, not about an interval: a development
    study window that happened to overlap these dates is still not this fixture,
    and the identity is what says so.
    """

    kind: Literal["ProtectedFixture"] = "ProtectedFixture"
    fixture_id: str = Field(min_length=1, max_length=120)
    disposition: Literal["ISOLATED_SYNTHETIC_NON_RESERVED_FIXTURE"] = PROTECTED_FIXTURE_DISPOSITION
    sessions: tuple[date, ...] = Field(min_length=1)
    listing_axis_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    session_axis_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def of(cls, *, fixture_id: str, sessions: tuple[date, ...], listings: tuple[str, ...]) -> Self:
        """Seal controlled fixture authority without exposing a physical workspace path.

        Args:
            fixture_id: Controlled owner-issued fixture authority key.
            sessions: Ordered session axis supplied within the controlled owner boundary.
            listings: Ordered listing axis used only to derive its identity.

        Returns:
            Validated fixture authority with ordered axis hashes and installed disposition.
        """
        values: dict[str, object] = {
            "kind": "ProtectedFixture",
            "fixture_id": fixture_id,
            "disposition": PROTECTED_FIXTURE_DISPOSITION,
            "sessions": sessions,
            "listing_axis_hash": ordered_listing_axis_hash(listings),
            "session_axis_hash": ordered_session_axis_hash(sessions),
        }
        identity = cls.model_construct(**values, fixture_hash="0" * 64).model_dump(
            mode="json", exclude={"fixture_hash"}
        )
        return cls(**identity, fixture_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require exact controlled fixture session axis and declared identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PortfolioFinalizationError: Session-axis hash differs from the retained ordered dates or
                fixture_hash differs.
        """
        if ordered_session_axis_hash(self.sessions) != self.session_axis_hash:
            raise PortfolioFinalizationError("portfolio_finalization.fixture_axis_invalid")
        if self.fixture_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"fixture_hash"})
        ):
            raise PortfolioFinalizationError("portfolio_finalization.fixture_identity_invalid")
        return self

    def overlaps(self, *, start: date, end: date) -> bool:
        """Whether a requested window touches this fixture at all."""
        return any(start <= value <= end for value in self.sessions)


class ProtectedEvaluationPermit(_Contract):
    """One content-addressed authorization to evaluate one candidate on one fixture.

    Issued by the Gate, consumed by Portfolio. It carries no evidence and no
    metric: it names what may be evaluated and binds the exact state the
    continuation has to start from, so a permit cannot be reused against a book
    that has moved since.
    """

    kind: Literal["ProtectedEvaluationPermit"] = "ProtectedEvaluationPermit"
    candidate_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    pre_protected_state_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fixture: ProtectedFixture
    issued_at: datetime
    permit_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal one declared controlled evaluation permit record.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical permit_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        identity = cls.model_construct(**values, permit_hash="0" * 64).model_dump(
            mode="json", exclude={"permit_hash"}
        )
        return cls(**identity, permit_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require a timezone-aware permit record and exact declared identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PortfolioFinalizationError: Issue time lacks timezone authority or permit_hash differs.
        """
        if self.issued_at.tzinfo is None or self.issued_at.utcoffset() is None:
            raise PortfolioFinalizationError("portfolio_finalization.permit_clock_invalid")
        if self.permit_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"permit_hash"})
        ):
            raise PortfolioFinalizationError("portfolio_finalization.permit_identity_invalid")
        return self


class FinalPortfolioEvaluationPackage(_Contract):
    """The protected continuation, sealed and pending until a gate closes it.

    `PENDING_CLOSURE` is a real state, not a label: the Host refuses to release a
    report for a package in it, so the metrics exist and are unreachable until a
    receipt says otherwise. Releasing on a field the package sets itself would
    make the gate advisory.
    """

    kind: Literal["FinalPortfolioEvaluationPackage"] = "FinalPortfolioEvaluationPackage"
    permit_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    continued_from_state_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    protected_result_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    """The sealed result the release promotes, named here rather than looked up.

    A lookup would need an index from report to result, and an index is one more
    thing that can point somewhere else. The package already had to be exact
    about what it contains; this makes it exact about what gets published.
    """

    protected_execution_ledger_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    protected_economic_ledger_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    protected_report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    protected_formation_count: int = Field(gt=0)
    protected_listing_count: int = Field(gt=0)
    """The decision axis width the protected path ran on.

    On the package because the Program declares one too, and the two have to be
    checked against each other -- an axis is a pair, and a Gate that verified only
    its session half would admit a run over another universe.
    """

    visibility: Literal["PENDING_CLOSURE"] = "PENDING_CLOSURE"
    package_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal one declared final evaluation package.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical package_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        identity = cls.model_construct(**values, package_hash="0" * 64).model_dump(
            mode="json", exclude={"package_hash"}
        )
        return cls(**identity, package_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require exact declared final evaluation package identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PortfolioFinalizationError: The canonical package payload differs from package_hash.
        """
        if self.package_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"package_hash"})
        ):
            raise PortfolioFinalizationError("portfolio_finalization.package_identity_invalid")
        return self


class ProtectedPackageInspection(_Contract):
    """What opening a pending package's children found. Identities only.

    Deliberately not a summary and deliberately not a verdict: it reports the
    bindings it read so the Gate can compare them against the permit it issued.
    The Gate must be able to check a package without opening a ledger itself and
    without learning a single number out of one.
    """

    kind: Literal["ProtectedPackageInspection"] = "ProtectedPackageInspection"
    package_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    absent_children: tuple[str, ...] = ()
    """Child identities the package names that could not be opened at all."""

    result_spec_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The request the protected result was sealed under.

    Configuration is not a detail of a protected package: cost, study window,
    report unit, benchmark view and every holdings control decide what the
    numbers *are*. A package produced under a changed configuration is a
    different evaluation wearing the permitted one's shape.
    """

    program_holdings_spec_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    report_control_receipt_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    program_formation_start: date | None = None
    program_formation_end: date | None = None
    program_formation_count: int | None = Field(default=None, ge=0)
    program_formation_sessions_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    program_listing_count: int | None = Field(default=None, ge=0)
    program_ordered_listing_ids_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """What the Program itself claims about the axis it was compiled for.

    Reported separately from the ledger's axis, and compared against it, because
    a Program and the path it produced can disagree: the ledger can match the
    permitted fixture exactly while the Program that compiled it declares another
    window, and nothing that reads only one of the two would notice.
    """

    ledger_formation_sessions_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    ledger_ordered_listing_ids_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    ledger_listing_count: int | None = Field(default=None, ge=0)
    """The ledger's axis, hashed the way the *Program* hashes its own.

    So the Gate compares like with like. The fixture's axis hashes use the
    ordered-axis identity; the Program uses a plain canonical hash of the same
    ordering, and comparing across the two functions would compare nothing.
    """

    result_program_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    result_execution_ledger_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    result_economic_ledger_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    result_report_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    ledger_program_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    economic_execution_ledger_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    report_program_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    report_execution_ledger_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    report_economic_ledger_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    report_comparison_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    comparison_execution_ledger_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    comparison_economic_ledger_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    program_continued_from_state_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    initial_boundary_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    formation_sessions: tuple[date, ...] = ()
    listing_axis_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    inspection_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal one metric-free controlled package inspection.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical inspection_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        identity = cls.model_construct(**values, inspection_hash="0" * 64).model_dump(
            mode="json", exclude={"inspection_hash"}
        )
        return cls(**identity, inspection_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require a metric-free inspection schema and exact declared identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PortfolioFinalizationError: A field name is result-shaped or inspection_hash differs.
        """
        for name in type(self).model_fields:
            if any(word in name for word in _RESULT_SHAPED_FIELD_WORDS):
                # The inspection crosses into the Gate, so it is held to the
                # same rule as the receipt: no number may ride along.
                raise PortfolioFinalizationError(
                    "portfolio_finalization.inspection_carries_a_metric:" + name
                )
        if self.inspection_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"inspection_hash"})
        ):
            raise PortfolioFinalizationError("portfolio_finalization.inspection_identity_invalid")
        return self


class ProtectedPackageInspector(Protocol):
    """Opens a pending package's children so the Gate can verify their bindings.

    Portfolio-owned, because only Portfolio knows where a pending protected
    result lives and what its descendants are. Narrow and read-only: it opens,
    reports identities, and returns. It computes nothing, and the Gate could not
    ask it to.
    """

    def inspect(self, package: FinalPortfolioEvaluationPackage) -> ProtectedPackageInspection:
        """Open every child the package names and report what they bind."""
        ...


class PortfolioValidationReceipt(_Contract):
    """The Gate's answer: closure, claim limits, and deliberately no number.

    Metric-free is enforced, not asserted. The validator refuses any field whose
    name reads like a result, so a later edit that adds `sharpe` or
    `cumulative_return` here fails at construction rather than quietly turning
    the receipt into a second results surface.
    """

    kind: Literal["PortfolioValidationReceipt"] = "PortfolioValidationReceipt"
    permit_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    package_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    pre_protected_state_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    closure: ClosureDisposition
    claim_limits: tuple[str, ...] = Field(min_length=1)
    reselection_disposition: Literal["PROTECTED_EVIDENCE_MAY_NOT_RETURN_TO_SELECTION_OR_TUNING"] = (
        CLAIM_LIMIT_NO_RESELECTION
    )
    sealed_at: datetime
    receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal one metric-free closure validation receipt.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical receipt_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        return seal_model_from_dump(cls, values, field="receipt_hash")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require metric-free closure evidence, aware time and the no-reselection claim.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PortfolioFinalizationError: A field name is result-shaped, seal time lacks timezone
                authority, CLOSED_EXACT contradicts the required no-reselection claim or
                receipt_hash differs.
        """
        for name in type(self).model_fields:
            if any(word in name for word in _RESULT_SHAPED_FIELD_WORDS):
                raise PortfolioFinalizationError(
                    "portfolio_finalization.validation_receipt_carries_a_metric:" + name
                )
        if self.sealed_at.tzinfo is None or self.sealed_at.utcoffset() is None:
            raise PortfolioFinalizationError("portfolio_finalization.receipt_clock_invalid")
        if (self.closure == "CLOSED_EXACT") != (CLAIM_LIMIT_NO_RESELECTION in self.claim_limits):
            raise PortfolioFinalizationError("portfolio_finalization.receipt_claim_limits_invalid")
        if self.receipt_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"receipt_hash"})
        ):
            raise PortfolioFinalizationError("portfolio_finalization.receipt_identity_invalid")
        return self

    @property
    def closed(self) -> bool:
        """Check whether the retained finalization closure is exact.

        Returns:
            True only for CLOSED_EXACT; no release permission is issued here.
        """
        return self.closure == "CLOSED_EXACT"


class ValidatedPortfolioHandoff(_Contract):
    """What downstream evidence work may open, and nothing more.

    Identities only. A handoff that copied a metric would let Alternative
    Evidence read a protected number without reopening the package that owns it,
    and the first thing anyone would do with that number is compare it.
    """

    kind: Literal["ValidatedPortfolioHandoff"] = "ValidatedPortfolioHandoff"
    workspace_id: str = Field(min_length=1, max_length=120)
    finalization_task_id: str = Field(min_length=1, max_length=64)
    candidate_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    package_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    released_report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    strategy_discovery_disposition: Literal["HANDOFF_MAY_NOT_RETURN_TO_STRATEGY_DISCOVERY"] = (
        "HANDOFF_MAY_NOT_RETURN_TO_STRATEGY_DISCOVERY"
    )
    released_at: datetime
    handoff_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal one declared validated portfolio handoff.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical handoff_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        identity = cls.model_construct(**values, handoff_hash="0" * 64).model_dump(
            mode="json", exclude={"handoff_hash"}
        )
        return cls(**identity, handoff_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require a timezone-aware validated handoff record and exact identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PortfolioFinalizationError: Release time lacks timezone authority or handoff_hash
                differs.
        """
        if self.released_at.tzinfo is None or self.released_at.utcoffset() is None:
            raise PortfolioFinalizationError("portfolio_finalization.handoff_clock_invalid")
        if self.handoff_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"handoff_hash"})
        ):
            raise PortfolioFinalizationError("portfolio_finalization.handoff_identity_invalid")
        return self


class ReleasedPortfolioArtifacts(_Contract):
    """The one commit that makes a protected result observable.

    Copying artifacts into the public store is *preparation*: it is idempotent,
    it is recoverable, and on its own it releases nothing. This marker is the
    release. It binds all four identities together -- the package that was
    evaluated, the receipt that closed it, the handoff that was minted for it,
    and the exact result those refer to -- so "released" is a single atomic fact
    rather than an inference from which files happen to exist.

    Without it, a crash midway through copying would leave a public store that
    looks like a released result and is not one, and nothing downstream could
    tell the difference.
    """

    kind: Literal["ReleasedPortfolioArtifacts"] = "ReleasedPortfolioArtifacts"
    workspace_id: str = Field(min_length=1, max_length=120)
    package_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    handoff_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    released_result_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    released_report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    released_at: datetime
    marker_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal one declared final artifact release marker.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical marker_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        identity = cls.model_construct(**values, marker_hash="0" * 64).model_dump(
            mode="json", exclude={"marker_hash"}
        )
        return cls(**identity, marker_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require a timezone-aware release marker and exact identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PortfolioFinalizationError: Release time lacks timezone authority or marker_hash
                differs.
        """
        if self.released_at.tzinfo is None or self.released_at.utcoffset() is None:
            raise PortfolioFinalizationError("portfolio_finalization.release_clock_invalid")
        if self.marker_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"marker_hash"})
        ):
            raise PortfolioFinalizationError("portfolio_finalization.release_identity_invalid")
        return self


class ProtectedContinuationResult(_Contract):
    """What the protected continuation produced, by identity only.

    The continuation runs inside Portfolio through the same executor and the same
    Backtesting owner the development path used. What crosses back into the
    finalization adapter is a set of identities, so the adapter -- which is about
    lifecycle, not numbers -- never handles an array.
    """

    kind: Literal["ProtectedContinuationResult"] = "ProtectedContinuationResult"
    permit_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    continued_from_state_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    protected_result_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_ledger_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    economic_ledger_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    formation_count: int = Field(gt=0)
    listing_count: int = Field(gt=0)
    backtesting_owner: Literal["SHARED_PORTFOLIO_BACKTESTING"] = "SHARED_PORTFOLIO_BACKTESTING"
    """Named on the result so the protocol proof does not rest on a comment.

    The protected run has to use the development owner; a continuation that
    quietly grew its own mechanics would produce numbers nobody could compare.
    """

    result_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal one declared controlled continuation result.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical result_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        identity = cls.model_construct(**values, result_hash="0" * 64).model_dump(
            mode="json", exclude={"result_hash"}
        )
        return cls(**identity, result_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require exact declared controlled continuation identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PortfolioFinalizationError: The canonical continuation payload differs from result_hash.
        """
        if self.result_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"result_hash"})
        ):
            raise PortfolioFinalizationError("portfolio_finalization.continuation_identity_invalid")
        return self


class ProtectedPathContinuation(Protocol):
    """Portfolio's own protected run, wired by composition to the shared executor."""

    def evaluate(
        self, *, permit: ProtectedEvaluationPermit, candidate: FrozenPortfolioCandidate
    ) -> ProtectedContinuationResult:
        """Continue the sealed book across the permitted fixture. Portfolio-owned."""
        ...


class FinalizationReleaseAuthority(Protocol):
    """Product Host's atomic bind of package and receipt, and nothing before it.

    A second injected port rather than a Portfolio method, because deciding when
    a protected report becomes visible is Host authority: Portfolio computed the
    numbers and must not also be the thing that decides they may be read.
    """

    def release(
        self,
        *,
        package: FinalPortfolioEvaluationPackage,
        receipt: PortfolioValidationReceipt,
        candidate: FrozenPortfolioCandidate,
        finalization_task_id: str,
    ) -> ValidatedPortfolioHandoff:
        """Bind the exact package to the exact receipt, or refuse."""
        ...

    def promote(self, package: FinalPortfolioEvaluationPackage) -> str | None:
        """Copy the validated artifacts where a released reader will need them.

        Preparation. Separate from `release` because it is idempotent and
        index-free, and because it must be able to run *after* the handoff is
        durable without being part of what mints it.
        """
        ...


class FrozenCandidateRegistry(Protocol):
    """Read-only access to what Portfolio actually froze, for the Gate to check.

    The Gate cannot verify "this was frozen beforehand" from a candidate alone --
    a candidate is a self-describing record and a caller can build one that says
    anything. It needs to open the registry Portfolio publishes into and find
    the receipt already there.

    Read-only by shape: there is no publish here, so admission cannot create the
    prerequisite it is about to check.
    """

    def open_freeze_receipt(self, candidate_hash: str) -> PortfolioCandidateFreezeReceipt | None:
        """The freeze receipt for this candidate, if one was ever published."""
        ...

    def open_candidate(self, candidate_hash: str) -> FrozenPortfolioCandidate | None:
        """The candidate as it was published, for comparison against the caller's."""
        ...


class ProtectedEvaluationPort(Protocol):
    """The whole surface Portfolio sees of the Validation Gate.

    Two calls. Nothing crossing this boundary is a score, an array, a metric
    calculator, a callback or a task runner, and Portfolio never learns which
    implementation answered -- which is what lets the finalization adapter be
    written once and injected by composition.
    """

    def admit(
        self, *, candidate: FrozenPortfolioCandidate, fixture: ProtectedFixture
    ) -> ProtectedEvaluationPermit:
        """Open the pre-existing freeze receipt, verify it, issue one permit.

        The verification is against the registry, not against the argument: a
        candidate that is internally consistent but was never registered is
        refused, because otherwise "frozen beforehand" would mean nothing more
        than "constructed before this call".
        """
        ...

    def close(
        self,
        *,
        permit: ProtectedEvaluationPermit,
        candidate: FrozenPortfolioCandidate,
        package: FinalPortfolioEvaluationPackage,
    ) -> PortfolioValidationReceipt:
        """Verify state continuity and the exact package, then seal a receipt."""
        ...

    def reopen_permit(
        self, *, candidate: FrozenPortfolioCandidate
    ) -> ProtectedEvaluationPermit | None:
        """The permit already issued for this candidate, if there is one.

        A read, not an authority: a resumed finalization must recover the permit
        it was issued rather than ask for another, and asking again would --
        correctly -- be refused.
        """
        ...

    def reopen_receipt(
        self, *, permit: ProtectedEvaluationPermit
    ) -> PortfolioValidationReceipt | None:
        """The receipt already sealed against this permit, if closure happened.

        Same reason: a run interrupted between closure and release has to reopen
        the receipt, and closing again would be refused because the permit is
        spent.
        """
        ...


__all__ = [
    "CLAIM_LIMIT_NO_RESELECTION",
    "CLAIM_LIMIT_SINGLE_EVALUATION",
    "CLAIM_LIMIT_SYNTHETIC_ONLY",
    "PROTECTED_FIXTURE_DISPOSITION",
    "ClosureDisposition",
    "CompletedDevelopmentRun",
    "CompletedDevelopmentTaskRegistry",
    "FinalPortfolioEvaluationPackage",
    "FinalizationReleaseAuthority",
    "FrozenCandidateRegistry",
    "FrozenPortfolioCandidate",
    "PortfolioCandidateFreezeReceipt",
    "PortfolioFinalizationError",
    "PortfolioValidationReceipt",
    "ProtectedContinuationResult",
    "ProtectedEvaluationPermit",
    "ProtectedEvaluationPort",
    "ProtectedFixture",
    "ProtectedPackageInspection",
    "ProtectedPackageInspector",
    "ProtectedPathContinuation",
    "ReleasedPortfolioArtifacts",
    "SealedPreProtectedState",
    "ValidatedPortfolioHandoff",
]
