"""How the book an optimizer prices turnover against is valued, and when.

The optimizer reference is **not** the pre-trade book. The pre-trade book sits at
the current formation's entry open, which is in the future at the moment the
decision is taken; the reference is the book as it stood at the last instant the
decision could actually see. Confusing the two is a whole session of hindsight,
and it is why the engine carries both.

One rule states every case::

    reference(decision at close(S)) = the book at open(S), valued through the
                                      lane's own method

Under ``EXECUTED_BOOK_AT_OPEN_T_PROXY`` the book is left at that open and the
reference is a full trading day stale against the close it is decided at. Under
``MARKED_TO_MARKET_AT_CLOSE_T`` it is re-valued by session ``S``'s own intraday
move -- a published fact, complete at ``close(S)``, and therefore knowable by a
decision taken there.

The mark is applied **at the point of use**, keyed on the decision session's
label. Nothing here indexes a marks array beside a formation axis: that is the
one-day-early error the whole remediation exists to remove, and no check inside a
loop that knows only positions could ever see it. What the execution snapshot's
own ``entry_session`` column is for is the *other* half of the claim -- that the
book being marked was really filled at that open -- and ``require_carry_segment``
is where it is checked.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from typing import Final

import numpy as np

from .contracts import (
    EXECUTED_BOOK_AT_OPEN_T_PROXY,
    MARKED_TO_MARKET_AT_CLOSE_T,
    FloatArray,
    PortfolioWalkForwardError,
    ReferenceMarkMethod,
)
from .execution import mark_book_to_session_close


@dataclass(frozen=True, slots=True)
class ReferenceMarkLane:
    """One installed way of valuing the reference, with everything it needs.

    A single object rather than a method name beside some arrays, because the
    two must not be able to disagree. A Campaign that sealed
    ``MARKED_TO_MARKET_AT_CLOSE_T`` into its identity while the loop ran the open
    proxy would publish a claim nothing executed, and no amount of checking at
    the seal could catch it -- the numbers are produced somewhere else. Here the
    declaration and the arithmetic are the same value, so the two cannot diverge.
    """

    method: ReferenceMarkMethod
    marks_by_session: Mapping[date, Sequence[float]] | None = None
    entry_session_by_formation: Mapping[date, date] | None = None
    state_transition_binding_hash: str | None = None
    """The binding this lane was resolved beside, so a consumer can tie the two.

    ``None`` on the proxy lane the research paths use, which publishes no
    Campaign evidence and therefore seals no transition identity.
    """

    def __post_init__(self) -> None:
        """Require close-mark inputs and reject marks attached to an open proxy.

        Raises:
            PortfolioWalkForwardError: Close marking lacks its surface, entry projection or binding,
                or a proxy carries marks.
        """
        resolved = (
            self.marks_by_session,
            self.entry_session_by_formation,
            self.state_transition_binding_hash,
        )
        if self.marks_to_close and any(value is None for value in resolved):
            raise PortfolioWalkForwardError("portfolio_strategy_lab.reference_mark_lane_incomplete")
        if not self.marks_to_close and self.marks_by_session is not None:
            # A proxy holding marks is a lane that could mark and chose not to,
            # which is indistinguishable from one that meant to and did not.
            raise PortfolioWalkForwardError(
                "portfolio_strategy_lab.reference_mark_proxy_carries_marks"
            )

    @property
    def marks_to_close(self) -> bool:
        """Identify whether this lane values the reference at the decision session close.

        Returns:
            True for the installed close-mark method.
        """
        return self.method == MARKED_TO_MARKET_AT_CLOSE_T

    def require_entry_projection(self, formation_sessions: Sequence[date]) -> None:
        """Every formation must have an owner-derived fill session. Nothing more.

        The whole-axis check, and deliberately *only* existence. A decision axis
        is not a carry chain: an out-of-fold axis skips the sessions its Alpha
        purge removed, so two scored formations can be adjacent by index and days
        apart on the exchange calendar. Asserting ``entry(previous) == next`` over
        the whole axis refuses those pairs -- and the four purge gaps in the
        installed 1,260-session axis are exactly that shape, so the resolver
        refused every real Campaign while every fixture with a contiguous axis
        passed.

        Adjacency belongs to ``require_carry_segment``, which is asked only about
        edges the engine actually carries state across.
        """
        projection = self.entry_session_by_formation
        if projection is None:
            return
        for session in formation_sessions:
            if session not in projection:
                raise PortfolioWalkForwardError(
                    "portfolio_strategy_lab.reference_mark_entry_session_absent"
                )

    def require_carry_segment(self, formation_sessions: Sequence[date]) -> None:
        """Check that this segment is an edge the engine can actually carry across.

        The reference handed to formation ``i`` is the book formation ``i-1``
        filled, and the lane values it at ``open(t_i)``. That is the same book
        only when ``entry_session(t_{i-1})`` really is ``t_i``.

        Two different things break it, and they are named apart because a reader
        can do nothing useful with one message for both:

        * ``entry`` *after* the next decision -- a method that fills later than
          the next close, so the previous decision's orders are still unfilled
          when the next one is taken and there is no book to value at all;
        * ``entry`` *before* it -- a lawful **gap** in the decision axis. An
          out-of-fold axis skips the sessions its Alpha purge removed, and the
          execution owner is explicit that an intended execution session inside
          such a gap is correct (``_admit_execution_clock``). The refusal here is
          not about the method: it is that this engine cannot carry state across
          the gap. Formation ``i-1``'s outcome window ends at
          ``open(holding_end)`` and formation ``i`` fills at
          ``open(entry(t_i))``, and on a gapped axis those are different
          instants with no return between them in the workspace -- so the
          pre-trade book would be a session stale and nothing here could
          reconstruct it.

        Refused rather than approximated, and refused in the engine rather than
        at the resolver: admission is about whether inputs were knowable, and a
        gapped axis is perfectly admissible. What it is not is runnable.
        """
        if self.entry_session_by_formation is None:
            # Nothing to check against. The state-carry limitation is a property
            # of the *return axis*, not of the mark method -- the pre-trade chain
            # is as stale across a gap under the proxy as under the close mark --
            # but only a caller holding the execution owner's projection can see
            # it. The proxy lane the research paths run on carries none, so those
            # paths are unchecked here and remain so; closing that means giving
            # them a projection, which is a change to five call sites and not to
            # this rule.
            return
        self.require_entry_projection(formation_sessions)
        for current, following in pairwise(formation_sessions):
            self._require_carry_edge(source=current, expected=following)

    def _require_carry_edge(self, *, source: date, expected: date) -> None:
        """One state-carry edge, with the two failures told apart."""
        projection = self.entry_session_by_formation
        if projection is None:
            return
        entry = projection.get(source)
        if entry is None:
            raise PortfolioWalkForwardError(
                "portfolio_strategy_lab.reference_mark_entry_session_absent"
            )
        if entry > expected:
            raise PortfolioWalkForwardError(
                "portfolio_strategy_lab.reference_mark_entry_after_next_decision"
            )
        if entry < expected:
            raise PortfolioWalkForwardError(
                "portfolio_strategy_lab.reference_mark_state_carry_gap_unsupported"
            )

    def require_carry_transition(
        self, *, last_decided: date, embargo_session: date, next_decision: date
    ) -> None:
        """Check the one edge that crosses an embargo, stated rather than assumed.

        Two fills, both from the execution owner's own column. The last decided
        formation must fill on the embargo session, and the embargo formation --
        which nobody traded -- must fill on the next decision's session. That is
        what makes the book entering the embargo the book at the next decision's
        open, which is the whole reason the reference is *taken* across an
        embargo rather than drifted.

        Separate from ``require_carry_segment`` because the sessions involved are
        not adjacent on the decision axis: the embargo formation is not decided
        on and therefore is not in it.
        """
        self._require_carry_edge(source=last_decided, expected=embargo_session)
        self._require_carry_edge(source=embargo_session, expected=next_decision)

    def value_at_decision(
        self, *, weights: FloatArray, cash: float, decision_session: date
    ) -> tuple[FloatArray, float]:
        """Value the book held at ``open(decision_session)`` for that session's close.

        The proxy returns the pair unchanged and is honest about it. The close
        mark applies that session's own published intraday move, which is the
        only thing that has to be looked up -- by label, never by position.
        """
        if not self.marks_to_close:
            return np.asarray(weights, dtype=np.float64), float(cash)
        assert self.marks_by_session is not None
        return mark_book_to_session_close(
            weights=np.asarray(weights, dtype=np.float64),
            cash=float(cash),
            marks_by_session=self.marks_by_session,
            session=decision_session,
        )


OPEN_PROXY_REFERENCE_MARK_LANE: Final = ReferenceMarkLane(method=EXECUTED_BOOK_AT_OPEN_T_PROXY)
"""What a path with no published mark surface runs on, named rather than implied.

Every research path in this repository -- regularization, the research loop, the
policy validator -- prices turnover against the entry open, and did so as an
unstated property of the loop. It is the same arithmetic under a name, so a
reader can see which of the two installed methods a result was measured on
instead of inferring it from the absence of an argument.
"""


__all__ = ["OPEN_PROXY_REFERENCE_MARK_LANE", "ReferenceMarkLane"]
