"""Conditional Portfolio intent and observed settlement, without a second engine."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import Literal, Self
from uuid import UUID

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.capabilities.portfolio_backtesting.contracts import (
    FloatArray,
    PortfolioCostPolicy,
    PortfolioPerSideCostAssumption,
    PortfolioWalkForwardState,
)
from alphalattice.capabilities.portfolio_backtesting.execution import (
    drift_holdings,
    execute_portfolio_entry,
    mark_book_to_session_close,
)
from alphalattice.capabilities.portfolio_backtesting.state import validate_portfolio_state
from alphalattice.foundation.causal_outcomes.execution.compile import _ELIGIBLE
from alphalattice.foundation.causal_outcomes.execution.contracts import (
    CausalExecutionSchedulePoint,
    LocalQAMarketSnapshot,
)
from alphalattice.foundation.causal_outcomes.execution.readers import local_qa_point_rows
from alphalattice.investment.portfolio_strategy_lab.application.calibration import (
    PreparedPortfolioBookInput,
    PreparedPortfolioComponentInput,
)
from alphalattice.investment.portfolio_strategy_lab.application.strategy_package import (
    FrozenStrategyPackage,
)
from alphalattice.investment.portfolio_strategy_lab.application.tranche_book_execution import (
    CappedSleeveBookProvider,
    ComponentBookPlan,
    ComponentBookProvider,
    build_component_book,
    first_short_formation,
)
from alphalattice.investment.portfolio_strategy_lab.policies.post_observed_authority import (
    FrozenHistoricalBookRecipe,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash


class _Sealed(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        payload = cls.model_construct(**values).model_dump(mode="json", exclude={"content_hash"})
        return cls(**payload, content_hash=canonical_hash(payload))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def identity(self) -> Self:
        if self.content_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"content_hash"})
        ):
            raise ValueError("portfolio_update.identity_invalid")
        return self


class PortfolioEntryBook(_Sealed):
    """Book immediately after one entry. It contains no later holding return.

    Sleeves remain the policy's target attribution, as in the historical engine;
    the next projection reconciles failed fills with the executed weights.
    """

    schedule: CausalExecutionSchedulePoint
    next_position: int = Field(ge=1)
    weights: tuple[float, ...]
    cash: float
    sleeves: tuple[tuple[float, ...], ...]
    component_sleeve_counts: tuple[int, ...] = Field(default=(), exclude_if=lambda v: not v)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def state(self) -> Self:
        """Require normalized portfolio carry and finite nonnegative component sleeve states.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Weights/cash/reference state, sleeve shape/finiteness or per-component
                sleeve count/mass violates the declared carry contract.
        """
        weights: FloatArray = np.asarray(self.weights, dtype=np.float64)
        validate_portfolio_state(
            PortfolioWalkForwardState(
                pretrade_weights=weights,
                pretrade_cash=self.cash,
                optimizer_reference=weights,
                optimizer_reference_cash=self.cash,
            ),
            len(self.weights),
        )
        sleeves: FloatArray = np.asarray(self.sleeves, dtype=np.float64)
        if (
            sleeves.ndim != 2
            or sleeves.shape[1] != weights.size
            or not np.isfinite(sleeves).all()
            or np.any(sleeves < 0)
            or (not self.component_sleeve_counts and abs(float(sleeves.sum()) - 1.0) > 1e-8)
        ):
            raise ValueError("portfolio_update.sleeve_state_invalid")
        if self.component_sleeve_counts:
            if any(n <= 0 for n in self.component_sleeve_counts) or sum(
                self.component_sleeve_counts
            ) != len(sleeves):
                raise ValueError("portfolio_update.component_sleeve_axis_invalid")
            start = 0
            for count in self.component_sleeve_counts:
                if abs(float(sleeves[start : start + count].sum()) - 1.0) > 1e-8:
                    raise ValueError("portfolio_update.component_sleeve_mass_invalid")
                start += count
        return self


class PortfolioDecisionCheckpoint(_Sealed):
    """A full starting state and one fixed model/axis epoch, admitted explicitly.

    An operator's QA admission, or a person's activation of a reviewed book that runs its
    strategy forward (LS1, OW12): then the person's time and the book are named, and the
    starting state is that book's sealed last state.
    """

    kind: Literal["PortfolioDecisionCheckpoint"] = "PortfolioDecisionCheckpoint"
    purpose: Literal[
        "POST_OBSERVED_QA_NOT_FORWARD_ADMISSION", "PERSON_ACTIVATED_FORWARD_RESEARCH"
    ] = "POST_OBSERVED_QA_NOT_FORWARD_ADMISSION"
    activated_at: datetime | None = Field(default=None, exclude_if=lambda v: v is None)
    book_task_id: UUID | None = Field(default=None, exclude_if=lambda v: v is None)
    package: FrozenStrategyPackage
    recipe: FrozenHistoricalBookRecipe
    ordered_listing_ids: tuple[str, ...]
    listing_labels: tuple[str, ...]
    model_authority_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    inference_recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    epoch_start: date
    epoch_end: date
    formation_sessions: tuple[date, ...]
    initial_book: PortfolioEntryBook
    source_hashes: tuple[str, ...] = Field(min_length=1)
    component_authority_hashes: tuple[str, ...] = Field(default=(), exclude_if=lambda v: not v)
    component_recipe_hashes: tuple[str, ...] = Field(default=(), exclude_if=lambda v: not v)
    lineage_root_hash: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$", exclude_if=lambda v: v is None
    )
    previous_checkpoint_hash: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$", exclude_if=lambda v: v is None
    )

    @property
    def history_hash(self) -> str:
        """Read the retained continuation lineage root or the checkpoint content fallback.

        Returns:
            lineage_root_hash when present, otherwise content_hash.
        """
        return self.lineage_root_hash or self.content_hash

    @property
    def model_recipe_hashes(self) -> tuple[str, ...]:
        """Read per-component model recipes with compatible single-component fallback.

        Returns:
            Declared component_recipe_hashes, otherwise the one inference_recipe_hash.
        """
        return self.component_recipe_hashes or (self.inference_recipe_hash,)

    @property
    def model_authority_hashes(self) -> tuple[str, ...]:
        """Read per-component model authority with compatible single-component fallback.

        Returns:
            Declared component_authority_hashes, otherwise the one model_authority_hash.
        """
        return self.component_authority_hashes or (self.model_authority_hash,)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def binding(self) -> Self:
        """Require coherent frozen policy, component authority, axes and entry carry position.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Package/policy/allocation/component plan disagrees, component model
                authority is partial, listing/formation/book/sleeve axes differ, entry
                position/clock is invalid or ordered component authority/recipe hashes contradict
                the declared aggregate bindings.
        """
        policy, component = self.package.frozen_policy, self.recipe.components
        axis, book = self.ordered_listing_ids, self.initial_book
        if (
            policy is None
            or policy.policy_recipe_hash != self.recipe.recipe_hash
            or self.package.strategy_id != self.recipe.strategy_id
            or not component
            or sum(c.allocation_basis_points for c in component) != 10_000
            or self.package.component_ids != tuple(c.component_id for c in component)
            or len(self.package.component_plan) != len(component)
            or len(self.model_recipe_hashes) != len(component)
            or (bool(self.component_authority_hashes) != bool(self.component_recipe_hashes))
            or (len(component) > 1 and not self.component_authority_hashes)
            or not axis
            or axis != tuple(sorted(set(axis)))
            or len(self.listing_labels) != len(axis)
            or len(book.weights) != len(axis)
            or len(book.sleeves) != self.recipe.tranches * len(component)
            or (
                len(component) > 1
                and book.component_sleeve_counts != (self.recipe.tranches,) * len(component)
            )
            or len(self.formation_sessions) <= self.recipe.sizing_activation_formation
            or self.formation_sessions != tuple(sorted(set(self.formation_sessions)))
            or book.next_position >= len(self.formation_sessions)
            or self.formation_sessions[book.next_position] != book.schedule.entry_session
            or self.formation_sessions[book.next_position - 1] != book.schedule.formation_session
            or not self.epoch_start <= book.schedule.entry_session <= self.epoch_end
        ):
            raise ValueError("portfolio_update.checkpoint_invalid")
        if self.component_authority_hashes and (
            len(self.component_authority_hashes) != len(component)
            or self.model_authority_hash
            != canonical_hash(
                tuple(zip(self.package.component_ids, self.component_authority_hashes, strict=True))
            )
            or self.inference_recipe_hash
            != canonical_hash(
                tuple(zip(self.package.component_ids, self.component_recipe_hashes, strict=True))
            )
        ):
            raise ValueError("portfolio_update.component_authority_binding_invalid")
        # A person's activation names its time and its book; a QA admission names neither.
        forward = self.purpose == "PERSON_ACTIVATED_FORWARD_RESEARCH"
        if (
            forward != (self.activated_at is not None)
            or forward != (self.book_task_id is not None)
            or (self.activated_at is not None and self.activated_at.utcoffset() is None)
        ):
            raise ValueError("portfolio_update.checkpoint_activation_invalid")
        return self


class PortfolioConditionalProposal(_Sealed):
    """Rule-bound intent: estimates and conditional execution weights are distinct."""

    kind: Literal["PortfolioConditionalProposal"] = "PortfolioConditionalProposal"
    checkpoint_hash: str
    input: PreparedPortfolioComponentInput | PreparedPortfolioBookInput
    book: PortfolioEntryBook
    schedule: CausalExecutionSchedulePoint
    source_snapshot_hash: str
    reference_weights: tuple[float, ...]
    close_weights: tuple[float, ...]
    close_cash: float
    estimated_weights: tuple[float, ...]
    estimated_weight_changes: tuple[float, ...]
    estimate_disposition: Literal["CLOSE_MARKED_ESTIMATE_NOT_EXECUTION_TARGET"] = (
        "CLOSE_MARKED_ESTIMATE_NOT_EXECUTION_TARGET"
    )
    execution_semantics: Literal["SEALED_INPUTS_AT_OBSERVED_ENTRY_NO_NEW_RESEARCH"] = (
        "SEALED_INPUTS_AT_OBSERVED_ENTRY_NO_NEW_RESEARCH"
    )


class PortfolioObservedSettlement(_Sealed):
    """Retain observed QA entry/outcome state and declared per-side cost lanes.

    Entry settlement carries no realized return; outcome settlement carries all realized gross/net
    lanes. Daily-bar QA observations do not establish verified venue execution. Optional pretrade
    weights retain the explicit settled carry axis.
    """

    kind: Literal["PortfolioObservedSettlement"] = "PortfolioObservedSettlement"
    phase: Literal["ENTRY_SETTLED", "OUTCOME_SETTLED"]
    execution_basis: Literal["DAILY_BAR_QA_NOT_VERIFIED_VENUE_EXECUTION"] = (
        "DAILY_BAR_QA_NOT_VERIFIED_VENUE_EXECUTION"
    )
    proposal_hash: str
    source_snapshot_hash: str
    formation_session: date
    entry: PortfolioEntryBook
    target_weights: tuple[float, ...]
    turnover: float
    missed_executions: int = Field(ge=0)
    pretrade_weights: tuple[float, ...] | None = Field(default=None, exclude_if=lambda v: v is None)
    cost_fraction_5bps: float
    cost_fraction_10bps: float
    gross_return: float | None = None
    net_return_5bps: float | None = None
    net_return_10bps: float | None = None

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def outcome(self) -> Self:
        """Require phase-appropriate returns, finite economics and valid pretrade mass.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Pretrade axes/weights/mass are invalid, entry/outcome return presence
                contradicts phase, or turnover/cost/return values violate finiteness/nonnegative
                turnover.
        """
        if self.pretrade_weights is not None and (
            len(self.pretrade_weights) != len(self.entry.weights)
            or not np.isfinite(self.pretrade_weights).all()
            or any(v < 0 or v > 1 for v in self.pretrade_weights)
            or sum(self.pretrade_weights) > 1 + 1e-12
        ):
            raise ValueError("portfolio_update.pretrade_weights_invalid")
        returns = (self.gross_return, self.net_return_5bps, self.net_return_10bps)
        if (
            (self.phase == "ENTRY_SETTLED" and any(v is not None for v in returns))
            or (self.phase == "OUTCOME_SETTLED" and any(v is None for v in returns))
            or any(v is not None and not np.isfinite(v) for v in returns)
            or not np.isfinite(
                (self.turnover, self.cost_fraction_5bps, self.cost_fraction_10bps)
            ).all()
            or self.turnover < 0
        ):
            raise ValueError("portfolio_update.settlement_invalid")
        return self


class PortfolioSourceRevision(_Sealed):
    """Input-impact evidence, not adoption of a re-optimized historical book."""

    previous_market_hash: str
    revised_market_hash: str
    changed_bar_count: int = Field(ge=0)
    actions_changed: bool
    continuation_basis: Literal["AS_ISSUED"] = "AS_ISSUED"
    replay_status: Literal["NOT_ADOPTED_NOT_PERFORMED"] = "NOT_ADOPTED_NOT_PERFORMED"


class PortfolioUpdatePublication(_Sealed):
    """Bind post-observed QA book/update publication to source and continuation authority.

    Pending proposal, observed entry/events, optional source correction and retained HTML identity
    describe the issued update. The claim is post-observed QA, with Risk and CRO explicitly
    unevaluated for this proposal; publication is not timely advice.
    """

    kind: Literal["PortfolioUpdatePublication"] = "PortfolioUpdatePublication"
    plan_hash: str
    checkpoint_hash: str
    parent_hash: str | None
    observed_through: date
    published_at: datetime
    book: PortfolioEntryBook
    pending_proposal: PortfolioConditionalProposal | None
    active_entry: PortfolioObservedSettlement | None
    events: tuple[PortfolioObservedSettlement, ...]
    source_snapshot_hash: str
    html_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    claim: Literal["POST_OBSERVED_QA_NOT_TIMELY_ADVICE"] = "POST_OBSERVED_QA_NOT_TIMELY_ADVICE"
    risk_status: Literal["NOT_EVALUATED_FOR_THIS_PROPOSAL"] = "NOT_EVALUATED_FOR_THIS_PROPOSAL"
    cro_status: Literal["NOT_REVIEWED_FOR_THIS_PROPOSAL"] = "NOT_REVIEWED_FOR_THIS_PROPOSAL"
    input_checkpoint_hash: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$", exclude_if=lambda v: v is None
    )
    revised_source_hash: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$", exclude_if=lambda v: v is None
    )
    source_revision: PortfolioSourceRevision | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    continuation_basis: Literal["AS_ISSUED"] = Field(
        default="AS_ISSUED", exclude_if=lambda v: v == "AS_ISSUED"
    )


@dataclass(frozen=True, slots=True)
class PortfolioUpdatePositions:
    """One source-exact position basis for presentation and downstream review."""

    basis: Literal["CONDITIONAL_ESTIMATE", "OBSERVED_RESEARCH_ENTRY"]
    position_hash: str
    schedule: CausalExecutionSchedulePoint
    weights: tuple[float, ...]
    preceding: tuple[float, ...] | None

    @property
    def changes(self) -> tuple[float, ...] | None:
        """Subtract preceding weights on the exact declared listing axis when available.

        Returns:
            Ordered current-minus-preceding changes, or None for absent preceding positions.

        Raises:
            ValueError: Current and preceding weight axes have different lengths.
        """
        if self.preceding is None:
            return None
        return tuple(a - b for a, b in zip(self.weights, self.preceding, strict=True))

    @property
    def effective_n(self) -> float:
        """Compute reciprocal weight concentration for the declared position view.

        Returns:
            One over summed squared weights, or zero when squared concentration is nonpositive.
        """
        hhi = sum(v * v for v in self.weights)
        return 0.0 if hhi <= 0 else 1.0 / hhi


def portfolio_update_positions(
    value: PortfolioUpdatePublication, history: tuple[PortfolioUpdatePublication, ...] = ()
) -> PortfolioUpdatePositions:
    """Read sealed facts, never re-run the book to reconstruct missing facts."""
    proposal = value.pending_proposal
    if proposal is not None:
        if (
            tuple(
                a - b
                for a, b in zip(proposal.estimated_weights, proposal.close_weights, strict=True)
            )
            != proposal.estimated_weight_changes
        ):
            raise ValueError("portfolio_update.estimate_changes_invalid")
        return PortfolioUpdatePositions(
            "CONDITIONAL_ESTIMATE",
            proposal.content_hash,
            proposal.schedule,
            proposal.estimated_weights,
            proposal.close_weights,
        )
    # Later publications can carry the same book after its entry event completed.
    entries = [
        entry
        for publication in (*history, value)
        for entry in (*publication.events, publication.active_entry)
        if entry is not None and entry.entry.content_hash == value.book.content_hash
    ]
    preceding = entries[-1].pretrade_weights if entries else None
    return PortfolioUpdatePositions(
        "OBSERVED_RESEARCH_ENTRY",
        value.book.content_hash,
        value.book.schedule,
        value.book.weights,
        preceding,
    )


def reindex_entry_book(
    book: PortfolioEntryBook, before: tuple[str, ...], after: tuple[str, ...]
) -> PortfolioEntryBook:
    """Change coordinates only. A carried sleeve is never sold or normalized here."""
    if before == after:
        return book
    if len(before) != len(book.weights) or not set(before) <= set(after):
        raise ValueError("portfolio_update.carried_listing_cannot_disappear")
    positions = {listing: i for i, listing in enumerate(before)}

    def remap(row: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(row[positions[v]] if v in positions else 0.0 for v in after)

    return PortfolioEntryBook.create(
        **{
            **{k: getattr(book, k) for k in type(book).model_fields if k != "content_hash"},
            "weights": remap(book.weights),
            "sleeves": tuple(remap(row) for row in book.sleeves),
        }
    )


def transition_decision_checkpoint(
    previous: PortfolioDecisionCheckpoint,
    *,
    candidate_labels: dict[str, str],
    source_hash: str,
) -> PortfolioDecisionCheckpoint:
    """Admitted candidates plus retained valuation names share one history root."""
    axis = tuple(sorted(set(previous.ordered_listing_ids) | set(candidate_labels)))
    labels = dict(zip(previous.ordered_listing_ids, previous.listing_labels, strict=True))
    labels.update(candidate_labels)
    if axis == previous.ordered_listing_ids:
        return previous
    return PortfolioDecisionCheckpoint.create(
        **{
            **{k: getattr(previous, k) for k in type(previous).model_fields if k != "content_hash"},
            "ordered_listing_ids": axis,
            "listing_labels": tuple(labels[v] for v in axis),
            "initial_book": reindex_entry_book(
                previous.initial_book, previous.ordered_listing_ids, axis
            ),
            "lineage_root_hash": previous.history_hash,
            "previous_checkpoint_hash": previous.content_hash,
            "source_hashes": (previous.content_hash, source_hash),
        }
    )


def market_on_axis(snapshot: LocalQAMarketSnapshot, axis: tuple[str, ...]) -> LocalQAMarketSnapshot:
    """Project observed market metadata and rows onto an explicitly admitted listing axis.

    Args:
        snapshot: Sealed local QA market snapshot.
        axis: Requested ordered listing axis contained in the snapshot population.

    Returns:
        Original snapshot for an identical axis, otherwise a re-sealed axis/subset-row projection.

    Raises:
        ValueError: A requested listing is absent from source authority.
    """
    if snapshot.ordered_listing_ids == axis:
        return snapshot
    if not set(axis) <= set(snapshot.ordered_listing_ids):
        raise ValueError("portfolio_update.valuation_axis_missing")
    return LocalQAMarketSnapshot.create(
        **{
            **{k: getattr(snapshot, k) for k in type(snapshot).model_fields if k != "content_hash"},
            "ordered_listing_ids": axis,
            "bars": tuple(v for v in snapshot.bars if v.listing_id in axis),
            "actions": tuple(v for v in snapshot.actions if v.listing_id in axis),
        }
    )


def continue_issued_observations(
    previous: LocalQAMarketSnapshot,
    current: LocalQAMarketSnapshot,
    *,
    raw_previous: LocalQAMarketSnapshot | None = None,
) -> LocalQAMarketSnapshot:
    """Keep the issued prefix; only future observations extend its valuation.

    Provider bars are split-adjusted. A newly effective split translates the
    old prefix by its declared factor before appending new-basis bars. Corrections
    to an already observed split cannot establish a unit conversion and refuse.
    Other late revisions remain in the separately sealed current snapshot.
    """
    if current.through < previous.through or not set(previous.ordered_listing_ids) <= set(
        current.ordered_listing_ids
    ):
        raise ValueError("portfolio_update.correction_axis_or_clock_invalid")
    factual = raw_previous or previous
    if (
        factual.through != previous.through
        or factual.ordered_listing_ids != previous.ordered_listing_ids
    ):
        raise ValueError("portfolio_update.raw_basis_axis_or_clock_invalid")
    old_splits = tuple(v for v in factual.actions if v.action_kind == "SPLIT")
    if old_splits != tuple(
        v
        for v in current.actions
        if v.action_kind == "SPLIT"
        and v.effective_date <= previous.through
        and v.listing_id in previous.ordered_listing_ids
    ):
        raise ValueError("portfolio_update.revised_split_basis_requires_review")
    factors = dict.fromkeys(previous.ordered_listing_ids, 1.0)
    for action in current.actions:
        if action.action_kind == "SPLIT" and action.effective_date > previous.through:
            ratio = action.new_shares_per_old_share
            if ratio is None or not np.isfinite(ratio) or ratio <= 0:
                raise ValueError("portfolio_update.split_conversion_unavailable")
            if action.listing_id in factors:
                factors[action.listing_id] /= ratio
    known = {(v.session_date, v.listing_id): v for v in current.bars}
    # Establish the conversion from successive factual versions. Issued prices
    # may deliberately differ after a correction and cannot prove this ratio.
    for listing, factor in factors.items():
        if factor == 1.0:
            continue
        pairs = [
            (v, known.get((v.session_date, listing)))
            for v in factual.bars
            if v.listing_id == listing
        ]
        if not pairs or any(b is None for _, b in pairs):
            raise ValueError("portfolio_update.split_conversion_unverified")
        if not all(
            b is not None
            and np.allclose(
                np.asarray((a.open, a.high, a.low, a.close)) * factor,
                (b.open, b.high, b.low, b.close),
                rtol=1e-12,
                atol=0,
            )
            for a, b in pairs
        ):
            raise ValueError("portfolio_update.split_conversion_unverified")
    for bar in previous.bars:
        factor = factors[bar.listing_id]
        known[bar.session_date, bar.listing_id] = replace(
            bar,
            open=bar.open * factor,
            high=bar.high * factor,
            low=bar.low * factor,
            close=bar.close * factor,
        )
    old_actions = tuple(
        replace(v, cash_amount=v.cash_amount * factors[v.listing_id])
        if v.cash_amount is not None
        else v
        for v in previous.actions
    )
    actions = old_actions + tuple(
        v
        for v in current.actions
        if v.effective_date > previous.through or v.listing_id not in previous.ordered_listing_ids
    )
    return LocalQAMarketSnapshot.create(
        **{
            **{k: getattr(current, k) for k in type(current).model_fields if k != "content_hash"},
            "source_hash": canonical_hash(
                [previous.content_hash, current.content_hash, "AS_ISSUED"]
                + ([] if raw_previous is None else [raw_previous.content_hash])
            ),
            "bars": tuple(known[k] for k in sorted(known)),
            "actions": tuple(
                sorted(actions, key=lambda v: (v.listing_id, v.effective_date, v.action_kind))
            ),
        }
    )


def source_revision_impact(
    previous: LocalQAMarketSnapshot | None, current: LocalQAMarketSnapshot
) -> PortfolioSourceRevision | None:
    """Detect revised historical bars/actions relative to a preceding market snapshot.

    Args:
        previous: Prior source snapshot, or None when no comparison exists.
        current: Current revised/extended market snapshot.

    Returns:
        Sealed source revision with changed prior-bar count and action-change flag, or None when no
        prior snapshot/change is observed.
    """
    if previous is None:
        return None
    known = {(v.session_date, v.listing_id): v for v in current.bars}
    changed = sum(known.get((v.session_date, v.listing_id)) != v for v in previous.bars)
    actions_changed = previous.actions != tuple(
        v
        for v in current.actions
        if v.effective_date <= previous.through and v.listing_id in previous.ordered_listing_ids
    )
    if not changed and not actions_changed:
        return None
    return PortfolioSourceRevision.create(
        previous_market_hash=previous.content_hash,
        revised_market_hash=current.content_hash,
        changed_bar_count=changed,
        actions_changed=actions_changed,
    )


def _values(values: FloatArray) -> tuple[float, ...]:
    return tuple(float(v) for v in values)


def _provider(
    checkpoint: PortfolioDecisionCheckpoint,
    book: PortfolioEntryBook,
    prepared: PreparedPortfolioComponentInput | PreparedPortfolioBookInput,
) -> ComponentBookProvider:
    recipe = checkpoint.recipe
    inputs = (
        prepared.components if isinstance(prepared, PreparedPortfolioBookInput) else (prepared,)
    )
    if len(inputs) != len(recipe.components):
        raise ValueError("portfolio_update.component_input_count_invalid")
    components = []
    for index, (component, item) in enumerate(zip(recipe.components, inputs, strict=True)):
        formation = item.formation_input()
        if item.ordered_listing_ids != checkpoint.ordered_listing_ids:
            source_columns = {v: i for i, v in enumerate(item.ordered_listing_ids)}
            formation = replace(
                formation,
                scores=np.asarray(
                    [
                        formation.scores[source_columns[v]] if v in source_columns else np.nan
                        for v in checkpoint.ordered_listing_ids
                    ]
                ),
                decision_eligible=np.asarray(
                    [
                        formation.decision_eligible[source_columns[v]]
                        if v in source_columns
                        else False
                        for v in checkpoint.ordered_listing_ids
                    ],
                    dtype=np.bool_,
                ),
            )
        # A session whose tradable, scored names are fewer than a rebalance selects is refused
        # before the update decides anything, by that session, never inside the decision (V519,
        # V500's class); the proposal and its settlement both decide through here.
        if first_short_formation((formation,), selected=recipe.top_k, walked=1) is not None:
            raise ValueError(f"portfolio_update.eligible_pool_short:{formation.formation_session}")
        provider = CappedSleeveBookProvider(
            top_k=recipe.top_k,
            exit_rank=recipe.exit_rank,
            tranches=recipe.tranches,
            aggregate_name_cap=recipe.aggregate_name_cap,
            aggregate_cap_start_formation=recipe.aggregate_cap_start_formation,
            formations=(formation,),
            ordered_listing_ids=checkpoint.ordered_listing_ids,
            initial_sleeve_weights=np.asarray(
                book.sleeves[index * recipe.tranches : (index + 1) * recipe.tranches]
            ),
            schedule_offset=book.next_position,
            weight_rule=component.weight_rule,
            review_phase=recipe.review_phase,
            sizing_activation_formation=recipe.sizing_activation_formation,
            sleeve_cap_equal_weight_multiple=recipe.sleeve_cap_equal_weight_multiple,
        )
        components.append(
            ComponentBookPlan(
                component_id=component.component_id,
                allocation_basis_points=component.allocation_basis_points,
                sleeve_rows=recipe.tranches,
                provider=provider,
            )
        )
    return build_component_book(tuple(components))


def require_proposal_position(
    *,
    checkpoint: PortfolioDecisionCheckpoint,
    prepared: PreparedPortfolioComponentInput | PreparedPortfolioBookInput,
    position: int,
    expected_session: date,
) -> None:
    """PLAN and execution share the same input/epoch/continuation refusal."""
    day = prepared.formation_session
    if (
        prepared.strategy_package_hash != checkpoint.package.package_hash
        or not set(prepared.ordered_listing_ids) <= set(checkpoint.ordered_listing_ids)
        or expected_session != day
        or not 0 <= position < len(checkpoint.formation_sessions)
        or checkpoint.formation_sessions[position] != day
        or not checkpoint.epoch_start <= day <= checkpoint.epoch_end
    ):
        raise ValueError("portfolio_update.proposal_input_or_position_mismatch")
    items = prepared.components if isinstance(prepared, PreparedPortfolioBookInput) else (prepared,)
    if (
        isinstance(prepared, PreparedPortfolioBookInput)
        and prepared.component_ids != checkpoint.package.component_ids
    ):
        raise ValueError("portfolio_update.component_order_invalid")
    if len(items) != len(checkpoint.recipe.components):
        raise ValueError("portfolio_update.component_input_count_invalid")
    for item, component, recipe_hash in zip(
        items, checkpoint.recipe.components, checkpoint.model_recipe_hashes, strict=True
    ):
        if (
            item.component_recipe_hash != recipe_hash
            or (
                component.weight_rule == "mu.iv0"
                and (
                    item.rule is None
                    or checkpoint.formation_sessions[item.rule.activation_position]
                    != item.rule.activation_session
                )
            )
            or (component.weight_rule == "ew" and item.rule is not None)
            or any(
                live and score == component.outsider_sentinel
                for live, score in zip(item.decision_eligible, item.scores, strict=True)
            )
        ):
            raise ValueError("portfolio_update.component_input_policy_invalid")


def _decision_eligible_listings(
    prepared: PreparedPortfolioComponentInput | PreparedPortfolioBookInput,
) -> set[str]:
    components = (
        prepared.components if isinstance(prepared, PreparedPortfolioBookInput) else (prepared,)
    )
    return {
        listing
        for component in components
        for listing, eligible in zip(
            component.ordered_listing_ids, component.decision_eligible, strict=True
        )
        if eligible
    }


def make_proposal(
    *,
    checkpoint: PortfolioDecisionCheckpoint,
    book: PortfolioEntryBook,
    prepared: PreparedPortfolioComponentInput | PreparedPortfolioBookInput,
    observed: LocalQAMarketSnapshot,
) -> PortfolioConditionalProposal:
    """Estimate a conditional formation proposal using explicit observed close marks.

    The numerical owner retains the executed-open reference carry; the close estimate is separate.
    Held or eligible listings require finite positive observed opens/closes, while unused
    unsupported listings can contribute zero marks.

    Args:
        checkpoint: Frozen policy/component/axis and schedule authority.
        book: Current issued entry-book carry.
        prepared: Exact formation/component calibration input.
        observed: QA market observations through this formation on the checkpoint axis.

    Returns:
        Sealed conditional proposal with open reference, close estimate and estimated target
        changes.

    Raises:
        ValueError: Proposal position, source axis/clock/schedule or required observed close marks
            are invalid.
    """
    day = prepared.formation_session
    require_proposal_position(
        checkpoint=checkpoint,
        prepared=prepared,
        position=book.next_position,
        expected_session=book.schedule.entry_session,
    )
    if observed.ordered_listing_ids != checkpoint.ordered_listing_ids or day != observed.through:
        raise ValueError("portfolio_update.proposal_input_or_position_mismatch")
    point = next((v for v in observed.schedule if v.formation_session == day), None)
    if point is None:
        raise ValueError("portfolio_update.schedule_missing")
    bars = {(v.session_date, v.listing_id): v for v in observed.bars}
    eligible = _decision_eligible_listings(prepared)
    marks = []
    for index, listing in enumerate(checkpoint.ordered_listing_ids):
        bar = bars.get((day, listing))
        if (
            bar is None
            or not np.isfinite((bar.open, bar.close)).all()
            or min(bar.open, bar.close) <= 0
        ):
            if (
                not any(row[index] > 0 for row in (book.weights, *book.sleeves))
                and listing not in eligible
            ):
                marks.append(0.0)
                continue
            raise ValueError("portfolio_update.close_mark_unavailable")
        marks.append(bar.close / bar.open - 1.0)
    # The frozen component engine's reference lane is its executed-open proxy.
    # The explicit close estimate is a different value, never written into carry.
    close, cash = mark_book_to_session_close(
        weights=np.asarray(book.weights), cash=book.cash, marks_by_session={day: marks}, session=day
    )
    estimate = _provider(checkpoint, book, prepared)(
        formation_index=0,
        reference_weights=np.asarray(book.weights),
        pretrade_weights=close,
        decision_mode="REBALANCE",
    )
    return PortfolioConditionalProposal.create(
        checkpoint_hash=checkpoint.content_hash,
        input=prepared,
        book=book,
        schedule=point,
        source_snapshot_hash=observed.content_hash,
        reference_weights=book.weights,
        close_weights=_values(close),
        close_cash=cash,
        estimated_weights=_values(estimate.target_weights),
        estimated_weight_changes=_values(estimate.target_weights - close),
    )


def _drift_to_exit(
    book: PortfolioEntryBook, observed: LocalQAMarketSnapshot
) -> tuple[FloatArray, float, float]:
    rows = local_qa_point_rows(observed, book.schedule).to_pylist()
    returns = np.asarray(
        [np.nan if r["simple_return"] is None else r["simple_return"] for r in rows]
    )
    if book.schedule.holding_end_session > observed.through:
        raise ValueError("portfolio_update.holding_observation_pending")
    return drift_holdings(weights=np.asarray(book.weights), cash=book.cash, returns=returns)


def settle_entry(
    *,
    checkpoint: PortfolioDecisionCheckpoint,
    proposal: PortfolioConditionalProposal,
    observed: LocalQAMarketSnapshot,
) -> PortfolioObservedSettlement:
    """Settle a conditional proposal using observed QA entry prices and eligibility.

    Args:
        checkpoint: Exact policy/component and listing authority.
        proposal: Issued conditional proposal and carry state.
        observed: Market observations covering the declared entry session.

    Returns:
        Entry-settled book/sleeves, actual turnover/missed count and registered 5/10 bps-per-side
        cost fractions.

    Raises:
        ValueError: Checkpoint/source/entry-carry binding differs or required entry price/status is
            absent/unknown.
    """
    if (
        proposal.checkpoint_hash != checkpoint.content_hash
        or observed.ordered_listing_ids != checkpoint.ordered_listing_ids
        or proposal.schedule.entry_session > observed.through
        or proposal.book.schedule.holding_end_session != proposal.schedule.entry_session
    ):
        raise ValueError("portfolio_update.entry_observation_or_state_invalid")
    rows = local_qa_point_rows(observed, proposal.schedule).to_pylist()
    eligible_statuses = {v.value for v in _ELIGIBLE}
    pretrade, cash, _ = _drift_to_exit(proposal.book, observed)
    provider = _provider(checkpoint, proposal.book, proposal.input)
    decision = provider(
        formation_index=0,
        reference_weights=np.asarray(proposal.reference_weights),
        pretrade_weights=pretrade,
        decision_mode="REBALANCE",
    )
    eligible = _decision_eligible_listings(proposal.input)
    for index, row in enumerate(rows):
        if (
            checkpoint.ordered_listing_ids[index] in eligible
            or pretrade[index] > 0
            or decision.target_weights[index] > 0
        ):
            if row["entry_open_split_adjusted"] is None:
                raise ValueError("portfolio_update.entry_price_missing")
            if row["entry_status"] == "ELIGIBILITY_UNKNOWN":
                raise ValueError("portfolio_update.entry_status_unknown")
    available: np.ndarray[tuple[int], np.dtype[np.bool_]] = np.asarray(
        [row["entry_status"] in eligible_statuses for row in rows], dtype=np.bool_
    )
    executed, executed_cash, turnover, missed = execute_portfolio_entry(
        decision=decision,
        pretrade_weights=pretrade,
        pretrade_cash=cash,
        execution_available=available,
    )
    sleeves = provider.sleeve_state
    assert sleeves is not None
    book = PortfolioEntryBook.create(
        schedule=proposal.schedule,
        next_position=proposal.book.next_position + 1,
        weights=_values(executed),
        cash=executed_cash,
        sleeves=tuple(_values(row) for row in sleeves),
        component_sleeve_counts=proposal.book.component_sleeve_counts,
    )
    policy = PortfolioCostPolicy()
    costs = tuple(
        float(
            policy.cost_fraction(
                one_way_turnover=np.asarray([turnover]),
                cost_bps=PortfolioPerSideCostAssumption.from_bps_per_side(
                    str(bps)
                ).platform_cost_bps,
            )[0]
        )
        for bps in (5, 10)
    )
    return PortfolioObservedSettlement.create(
        phase="ENTRY_SETTLED",
        proposal_hash=proposal.content_hash,
        source_snapshot_hash=observed.content_hash,
        formation_session=proposal.schedule.formation_session,
        entry=book,
        target_weights=_values(decision.target_weights),
        turnover=turnover,
        missed_executions=missed,
        pretrade_weights=_values(pretrade),
        cost_fraction_5bps=costs[0],
        cost_fraction_10bps=costs[1],
    )


def settle_outcome(
    entry: PortfolioObservedSettlement, observed: LocalQAMarketSnapshot
) -> PortfolioObservedSettlement:
    """Settle observed holding outcomes and registered per-side net-return lanes.

    Args:
        entry: Exact observed entry settlement with retained turnover.
        observed: QA market observations covering the declared holding exit.

    Returns:
        Re-sealed outcome settlement with observed source identity, gross return and net 5/10
        bps-per-side returns.
    """
    _, _, gross = _drift_to_exit(entry.entry, observed)
    policy = PortfolioCostPolicy()
    net = tuple(
        float(
            policy.net_simple_returns(
                gross_simple_returns=np.asarray([gross]),
                one_way_turnovers=np.asarray([entry.turnover]),
                cost_bps=PortfolioPerSideCostAssumption.from_bps_per_side(
                    str(bps)
                ).platform_cost_bps,
            )[0]
        )
        for bps in (5, 10)
    )
    values = {
        name: getattr(entry, name) for name in type(entry).model_fields if name != "content_hash"
    }
    values.update(
        phase="OUTCOME_SETTLED",
        source_snapshot_hash=observed.content_hash,
        gross_return=gross,
        net_return_5bps=net[0],
        net_return_10bps=net[1],
    )
    return PortfolioObservedSettlement.create(**values)


def advance_decision_state(
    *,
    checkpoint: PortfolioDecisionCheckpoint,
    previous: PortfolioUpdatePublication | None,
    prepared: PreparedPortfolioComponentInput | PreparedPortfolioBookInput | None,
    observed: LocalQAMarketSnapshot,
    plan_hash: str,
    published_at: datetime,
    previous_checkpoint: PortfolioDecisionCheckpoint | None = None,
    revised_source_hash: str | None = None,
    source_revision: PortfolioSourceRevision | None = None,
) -> PortfolioUpdatePublication:
    """Evaluate one observed prefix, then optionally issue the next close proposal."""
    if observed.ordered_listing_ids != checkpoint.ordered_listing_ids:
        raise ValueError("portfolio_update.listing_epoch_mismatch")
    if previous is not None and (
        previous.checkpoint_hash != checkpoint.history_hash
        or observed.through < previous.observed_through
    ):
        raise ValueError("portfolio_update.parent_or_observation_invalid")
    book = checkpoint.initial_book if previous is None else previous.book
    pending = None if previous is None else previous.pending_proposal
    active = None if previous is None else previous.active_entry
    prior_checkpoint = previous_checkpoint or checkpoint
    prior_market = market_on_axis(observed, prior_checkpoint.ordered_listing_ids)
    events = []
    if active is not None and active.entry.schedule.holding_end_session <= observed.through:
        events.append(settle_outcome(active, prior_market))
        active = None
    if pending is not None and pending.schedule.entry_session <= observed.through:
        active = settle_entry(checkpoint=prior_checkpoint, proposal=pending, observed=prior_market)
        events.append(active)
        book, pending = active.entry, None
        if active.entry.schedule.holding_end_session <= observed.through:
            events.append(settle_outcome(active, prior_market))
            active = None
    book = reindex_entry_book(
        book, prior_checkpoint.ordered_listing_ids, checkpoint.ordered_listing_ids
    )
    if pending is not None and prior_checkpoint != checkpoint:
        raise ValueError("portfolio_update.axis_transition_waits_for_sealed_entry")
    if active is not None and prior_checkpoint != checkpoint:
        active = PortfolioObservedSettlement.create(
            **{
                **{k: getattr(active, k) for k in type(active).model_fields if k != "content_hash"},
                "entry": book,
                "pretrade_weights": None
                if active.pretrade_weights is None
                else tuple(
                    dict(
                        zip(
                            prior_checkpoint.ordered_listing_ids,
                            active.pretrade_weights,
                            strict=True,
                        )
                    ).get(v, 0.0)
                    for v in checkpoint.ordered_listing_ids
                ),
                "target_weights": tuple(
                    dict(
                        zip(
                            prior_checkpoint.ordered_listing_ids, active.target_weights, strict=True
                        )
                    ).get(v, 0.0)
                    for v in checkpoint.ordered_listing_ids
                ),
            }
        )
    if prepared is not None:
        if pending is not None:
            raise ValueError("portfolio_update.previous_entry_pending")
        pending = make_proposal(
            checkpoint=checkpoint, book=book, prepared=prepared, observed=observed
        )
    if not events and prepared is None and source_revision is None:
        raise ValueError("portfolio_update.no_new_observations")
    return PortfolioUpdatePublication.create(
        plan_hash=plan_hash,
        checkpoint_hash=checkpoint.history_hash,
        input_checkpoint_hash=checkpoint.content_hash
        if checkpoint.lineage_root_hash is not None
        else None,
        parent_hash=None if previous is None else previous.content_hash,
        observed_through=observed.through,
        published_at=published_at,
        book=book,
        pending_proposal=pending,
        active_entry=active,
        events=tuple(events),
        source_snapshot_hash=observed.content_hash,
        revised_source_hash=revised_source_hash,
        source_revision=source_revision,
    )
