"""Installed rebalance clocks: when a formation is a decision and when it is a hold.

The engine rebalanced at every formation index and there was no knob, so a
weekly strategy could only be expressed *inside a policy*, by having the policy
remember which formation it was on and return its previous target on the other
four. That works and it is the wrong owner: cadence is a property of the
execution clock, not of the objective, and a cadence hidden inside one policy
cannot be compared across policies -- which is exactly the comparison a study of
turnover needs.

It also could not be done from outside without being wrong. A wrapper around the
decision provider sees ``reference_weights``, which is the *executed* book before
drift; returning that on a hold session would trade the drifted book back to its
pre-drift weights and charge turnover for the privilege. A hold has to be a hold
of the **pretrade** book, and only the segment loop holds that.

The loop asks the clock and passes its explicit REBALANCE or HOLD mode to the
provider. A hold carries the pretrade book without an adapter target decision or
forecast. The shared execution mechanics derive turnover from the filled book.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from alphalattice.kernel.shared_kernel.identity import canonical_hash

from .contracts import PortfolioWalkForwardError, RebalanceClockBinding

EVERY_FORMATION_CLOCK_ID = "EVERY_FORMATION"
EVERY_N_FORMATIONS_CLOCK_ID = "EVERY_N_FORMATIONS"
WHOLE_BOOK_EVERY_N_FORMATIONS_CLOCK_ID = "WHOLE_BOOK_EVERY_N_FORMATIONS"


@dataclass(frozen=True, slots=True)
class ScoredFormationClock:
    """A Host-verified score axis over a complete economic calendar."""

    formation_indices: tuple[int, ...]
    clock_id: str = "SCORED_FORMATIONS"

    def rebalances(self, *, formation_index: int, segment_start_index: int) -> bool:
        """Test membership in the explicit host-verified scored formation axis.

        Args:
            formation_index: Position on the complete formation axis.
            segment_start_index: Segment-relative anchor; ignored by globally anchored clocks.

        Returns:
            Whether this formation is a rebalance rather than a hold.
        """
        del segment_start_index
        return formation_index in self.formation_indices

    @property
    def binding(self) -> RebalanceClockBinding:
        """Seal this clock's declared identity and replay parameters.

        Returns:
            The clock binding, including scored indices when this clock uses them.
        """
        values = {
            "clock_id": self.clock_id,
            "parameters": {},
            "formation_indices": self.formation_indices,
        }
        return RebalanceClockBinding(
            clock_id=self.clock_id,
            formation_indices=self.formation_indices,
            binding_hash=canonical_hash(values),
        )


@dataclass(frozen=True, slots=True)
class EveryFormationClock:
    """Decide on every formation. What the engine did before there was a clock."""

    clock_id: str = EVERY_FORMATION_CLOCK_ID

    def rebalances(self, *, formation_index: int, segment_start_index: int) -> bool:
        """Declare a rebalance on every formation.

        Args:
            formation_index: Position on the complete formation axis.
            segment_start_index: Segment-relative anchor; ignored by globally anchored clocks.

        Returns:
            Whether this formation is a rebalance rather than a hold.
        """
        del formation_index, segment_start_index
        return True

    @property
    def binding(self) -> RebalanceClockBinding:
        """Seal this clock's declared identity and replay parameters.

        Returns:
            The clock binding, including scored indices when this clock uses them.
        """
        return _binding(self.clock_id, {})


@dataclass(frozen=True, slots=True)
class EveryNFormationsClock:
    """Decide every ``interval`` formations, counting from the segment's own start.

    Counting from the segment start rather than from the absolute formation index
    is the load-bearing choice. A fold's validation region begins wherever the
    split policy put it, and phase measured against the global axis would give
    each fold a different first decision session -- so a cadence comparison would
    be partly a comparison of which folds happened to start on a decision.

    An ``interval`` of one is refused rather than silently accepted: that is
    ``EVERY_FORMATION``, and two installed ids that mean the same thing are two
    identities for one experiment.
    """

    interval: int
    clock_id: str = EVERY_N_FORMATIONS_CLOCK_ID

    def __post_init__(self) -> None:
        """Reject an interval that duplicates the every-formation clock.

        Raises:
            PortfolioWalkForwardError: The interval is below two formations.
        """
        if self.interval < 2:
            raise PortfolioWalkForwardError("portfolio_backtesting.rebalance_interval_invalid")

    def rebalances(self, *, formation_index: int, segment_start_index: int) -> bool:
        """Test the cadence anchored to this segment's first formation.

        Args:
            formation_index: Position on the complete formation axis.
            segment_start_index: Segment-relative anchor; ignored by globally anchored clocks.

        Returns:
            Whether this formation is a rebalance rather than a hold.
        """
        return (formation_index - segment_start_index) % self.interval == 0

    @property
    def binding(self) -> RebalanceClockBinding:
        """Seal this clock's declared identity and replay parameters.

        Returns:
            The clock binding, including scored indices when this clock uses them.
        """
        return _binding(self.clock_id, {"interval": int(self.interval)})


@dataclass(frozen=True, slots=True)
class WholeBookEveryNFormationsClock:
    """Rebalance one complete book every ``interval`` formations at one phase.

    Unlike :class:`EveryNFormationsClock`, this clock is deliberately anchored
    to the complete formation axis.  Each declared offset is therefore an
    independent whole-book path; it is neither a fold-relative cadence nor a
    partitioned sleeve schedule.  A later report may aggregate independent
    paths, but this owner never composites their trades.
    """

    interval: int
    offset: int
    clock_id: str = WHOLE_BOOK_EVERY_N_FORMATIONS_CLOCK_ID

    def __post_init__(self) -> None:
        """Require an interval of at least two and an offset within that interval.

        Raises:
            PortfolioWalkForwardError: The interval or whole-book phase offset is invalid.
        """
        if self.interval < 2:
            raise PortfolioWalkForwardError("portfolio_backtesting.rebalance_interval_invalid")
        if self.offset < 0 or self.offset >= self.interval:
            raise PortfolioWalkForwardError("portfolio_backtesting.rebalance_offset_invalid")

    def rebalances(self, *, formation_index: int, segment_start_index: int) -> bool:
        """Test the whole-book cadence anchored to the complete formation axis.

        Args:
            formation_index: Position on the complete formation axis.
            segment_start_index: Segment-relative anchor; ignored by globally anchored clocks.

        Returns:
            Whether this formation is a rebalance rather than a hold.
        """
        del segment_start_index
        return formation_index % self.interval == self.offset

    @property
    def binding(self) -> RebalanceClockBinding:
        """Seal this clock's declared identity and replay parameters.

        Returns:
            The clock binding, including scored indices when this clock uses them.
        """
        return _binding(
            self.clock_id,
            {"interval": int(self.interval), "offset": int(self.offset)},
        )


def _binding(clock_id: str, parameters: dict[str, int]) -> RebalanceClockBinding:
    return RebalanceClockBinding(
        clock_id=clock_id,
        parameters=tuple(sorted(parameters.items())),
        binding_hash=canonical_hash({"clock_id": clock_id, "parameters": parameters}),
    )


_INSTALLED = MappingProxyType(
    {
        EVERY_FORMATION_CLOCK_ID: EveryFormationClock,
        EVERY_N_FORMATIONS_CLOCK_ID: EveryNFormationsClock,
        WHOLE_BOOK_EVERY_N_FORMATIONS_CLOCK_ID: WholeBookEveryNFormationsClock,
        "SCORED_FORMATIONS": ScoredFormationClock,
    }
)


def build_installed_rebalance_clock_catalog() -> MappingProxyType[str, type]:
    """The explicit installed clock map; no discovery and no runtime mutation."""
    return _INSTALLED


def resolve_rebalance_clock(clock_id: str, **parameters: int) -> object:
    """Build one installed clock, or refuse an id the catalog does not carry."""
    factory = _INSTALLED.get(clock_id)
    if factory is None:
        raise PortfolioWalkForwardError(
            "portfolio_backtesting.rebalance_clock_not_installed:" + clock_id
        )
    return factory(**parameters)


__all__ = [
    "EVERY_FORMATION_CLOCK_ID",
    "EVERY_N_FORMATIONS_CLOCK_ID",
    "WHOLE_BOOK_EVERY_N_FORMATIONS_CLOCK_ID",
    "EveryFormationClock",
    "EveryNFormationsClock",
    "ScoredFormationClock",
    "WholeBookEveryNFormationsClock",
    "build_installed_rebalance_clock_catalog",
    "resolve_rebalance_clock",
]
