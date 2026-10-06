"""Causal temporal aggregation of an Alpha formation score.

An Alpha method emits one score per listing per formation session. This owner
declares a second, separate method that consumes those scores and emits another
score, using only values the listing already had. It is a **return-signal
method**, not an evaluation step: the evaluation compares declared methods and
never transforms a score on its own authority, so an aggregate that is going to
be compared has to be a method somebody named first.

Three things this owner deliberately does not conflate:

* **Alpha score temporal aggregation** -- what is here. A causal function of one
  listing's own formation scores at ``t`` and earlier, producing another score
  at ``t``.
* **Portfolio rebalance and holding policy** -- when the book is allowed to
  trade and how positions carry. A different desk. Nothing here proposes,
  changes or implies one.
* **Realized holding period** -- an outcome statistic of a book that was
  actually traded. It configures nothing.

Aggregating a score changes the score. It does not change how often the book
rebalances, and the turnover a downstream evaluation reports is the consequence
of a different score entering the same unchanged daily book.

FORMULA SPECIFICATION -- ``TRAILING_MEAN_FORMATION_SCORE``
--------------------------------------------------------

For listing ``i`` at formation session ``t``, let ``P(t)`` be the position of
``t`` in the ordered formation session axis of the surface being aggregated, and
let ``span`` be the declared trailing window in sessions. Then

    aggregated[i, t] = mean over { score[i, s] : P(t) - span < P(s) <= P(t),
                                   score[i, s] is finite }

and the accumulation runs from the most recent session backwards, ``k = 0`` to
``span - 1``, so the summation order is fixed rather than incidental.

* **Window unit** is *formation sessions on the surface's own axis*, never
  observation count. A listing absent for ten sessions has a window that has
  advanced ten sessions, so absence costs depth instead of silently reaching
  further into the past.
* **min_periods is 1**, declared rather than tuned. A row's own score is always
  finite on an Alpha methodology surface, so every input row produces an output
  row and **the row axis cannot move with the span**. That is the point of the
  choice: a span comparison must not be confounded with a coverage change, and
  an audit after the fact is weaker than a construction that cannot fail. The
  cost is that a listing near its first scored session is aggregated shallowly;
  the realized depth is measured and published rather than assumed.
* **A listing that is absent contributes nothing.** IPO, delisting and universe
  re-entry are all the same case: sessions where the listing has no finite score
  are not in the mean. Re-entry does not carry anything across the gap beyond
  what the window itself still spans, and nothing is filled.
* **No backward fill, no future value, no target contact.** The aggregation axis
  is the formation session axis of the input surface; target availability plays
  no part in deciding it, and the computation is structurally incapable of
  reading a later session because the window is bounded above by ``P(t)``.
* **Ties are ties.** The mean can produce a constant cross-section or equal
  values for two listings. No tie-break is manufactured here; the evaluation's
  own constant-score policy decides what a constant session means.
* **``span == 1`` is bitwise degenerate.** The output is the input, exactly,
  including signed zero: the ``k = 0`` term is selected rather than accumulated
  and the divisor is exactly ``1.0``.

The mixture of models inside one window is intended and is what a live system
would have had. A window at a formation in outer fold ``k`` reaches back into
formations owned by fold ``k-1``, whose scores came from an earlier fit. Those
values were genuinely available at the time, and refusing them would model a
system that discards its own history at every refit boundary.

This module owns only the declared trailing-mean arithmetic. Historical schema
types below retain read-only compatibility with superseded cost-based span
evidence; the executable selector and numerical replay runtime were retired.
"""

from __future__ import annotations

from datetime import date
from typing import Final, Literal, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.capabilities.causal_inputs.contracts import (
    AnchoredInstantPolicy,
    SessionAnchor,
)
from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog
from alphalattice.foundation.feature_engine.catalog.observation_clock import (
    installed_feature_availability_policy,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.research_authoring.contracts import MarketPhase

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]

_HASH = r"^[0-9a-f]{64}$"

ALPHA_SCORE_AGGREGATION_METHOD_ID: Final = "TRAILING_MEAN_FORMATION_SCORE"
"""The one installed aggregation method. A domain name, not a version."""

APPROVED_AGGREGATION_SPAN_DOMAIN: Final[tuple[int, ...]] = (1, 21, 42, 63)
"""The whole candidate domain. A span outside it is not selectable."""

AGGREGATION_CONTROL_SPAN: Final = 1
"""Declared before the experiment: what fold zero uses, having no prior fold."""

AGGREGATION_SELECTION_RULE_ID: Final = (
    "MAXIMUM_MATURED_PRIOR_FOLD_NET_DECILE_SPREAD_5BPS_THEN_SMALLEST_SPAN"
)
AGGREGATION_SELECTION_COST_BPS: Final = 5.0
AGGREGATION_SELECTION_LANE_ID: Final = "RAW_ECONOMIC_SIMPLE_RETURN"
AGGREGATION_MATURITY_POLICY: Final = "OUTCOME_MATURITY_STRICTLY_BEFORE_CURRENT_FOLD_FIRST_FORMATION"

AGGREGATION_WINDOW_UNIT: Final = "FORMATION_SESSIONS_ON_SURFACE_AXIS"
AGGREGATION_MINIMUM_PERIODS_POLICY: Final = "ONE_ROW_AXIS_INVARIANT"
AGGREGATION_ABSENCE_POLICY: Final = "ABSENT_SESSION_CONTRIBUTES_NOTHING_NO_FILL"
AGGREGATION_TIE_POLICY: Final = "TIES_PRESERVED_NO_TIE_BREAK_MANUFACTURED"


class AlphaScoreAggregationError(ValueError):
    """Fail-closed boundary for the aggregation owner."""


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class AlphaScoreAggregationProgram(_Contract):
    """Historical schema for frozen superseded score-aggregation Programs."""

    kind: Literal["AlphaScoreAggregationProgram"] = "AlphaScoreAggregationProgram"
    method_id: Literal["TRAILING_MEAN_FORMATION_SCORE"]
    candidate_span_domain: tuple[int, ...] = Field(min_length=1)
    control_span_sessions: int = Field(ge=1)
    selection_rule_id: str = Field(min_length=1, max_length=96)
    window_unit: str = Field(min_length=1, max_length=64)
    minimum_periods_policy: str = Field(min_length=1, max_length=64)
    absence_policy: str = Field(min_length=1, max_length=64)
    tie_policy: str = Field(min_length=1, max_length=64)
    selection_lane_id: Literal["RAW_ECONOMIC_SIMPLE_RETURN"]
    selection_cost_bps: float = Field(gt=0.0)
    maturity_policy: Literal["OUTCOME_MATURITY_STRICTLY_BEFORE_CURRENT_FOLD_FIRST_FORMATION"]
    capability_hash: str = Field(pattern=_HASH)
    implementation_binding_hash: str = Field(pattern=_HASH)
    installed_catalog_hash: str = Field(pattern=_HASH)
    program_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require the installed aggregation policies and an admitted control span.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            AlphaScoreAggregationError: Span domain/control, selection rule, cost, lane, maturity
                policy or program identity differs.
        """
        if self.control_span_sessions not in self.candidate_span_domain:
            raise AlphaScoreAggregationError(
                "alpha_research.score_aggregation_control_span_not_in_domain"
            )
        if self.candidate_span_domain != tuple(sorted(set(self.candidate_span_domain))):
            raise AlphaScoreAggregationError(
                "alpha_research.score_aggregation_span_domain_unordered"
            )
        if (
            self.selection_rule_id != AGGREGATION_SELECTION_RULE_ID
            or self.selection_cost_bps != AGGREGATION_SELECTION_COST_BPS
            or self.selection_lane_id != AGGREGATION_SELECTION_LANE_ID
            or self.maturity_policy != AGGREGATION_MATURITY_POLICY
        ):
            raise AlphaScoreAggregationError(
                "alpha_research.score_aggregation_program_policy_invalid"
            )
        if self.program_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"program_hash"})
        ):
            raise AlphaScoreAggregationError("alpha_research.score_aggregation_program_invalid")
        return self


class AlphaScoreAggregationFoldSelection(_Contract):
    """What one outer fold chose, and the evidence it was allowed to see."""

    kind: Literal["AlphaScoreAggregationFoldSelection"] = "AlphaScoreAggregationFoldSelection"
    fold_index: int = Field(ge=0)
    selected_span_sessions: int = Field(ge=1)
    selection_basis: Literal["PRIOR_FOLD_EVIDENCE", "DECLARED_CONTROL_NO_PRIOR_EVIDENCE"]
    prior_fold_row_count: int = Field(ge=0)
    prior_fold_session_count: int = Field(ge=0)
    matured_prior_fold_row_count: int = Field(ge=0)
    matured_prior_fold_session_count: int = Field(ge=0)
    current_fold_first_formation_session: date
    scored_row_count: int = Field(ge=1)
    candidate_criteria: tuple[tuple[int, float | None], ...] = Field(min_length=1)
    """``(span, prior-fold criterion)`` for every candidate, in span order.

    ``None`` where the candidate produced no measurable book on the prior folds.
    Kept rather than dropped so a selection can be read back and disputed.
    """


class AlphaScoreAggregationEvidence(_Contract):
    """One arm's aggregated surface, bound to the rows it was computed over."""

    kind: Literal["AlphaScoreAggregationEvidence"] = "AlphaScoreAggregationEvidence"
    identity_class: Literal["DEVELOPMENT_EVIDENCE_ONLY_NEVER_PUBLISHED"] = (
        "DEVELOPMENT_EVIDENCE_ONLY_NEVER_PUBLISHED"
    )
    program_hash: str = Field(pattern=_HASH)
    methodology_id: str = Field(min_length=1, max_length=96)
    horizon_sessions: int = Field(ge=1)
    upstream_score_projection_hash: str = Field(pattern=_HASH)
    upstream_dossier_hash: str = Field(pattern=_HASH)
    upstream_decision_receipt_hash: str = Field(pattern=_HASH)
    target_recipe_binding_hash: str = Field(pattern=_HASH)
    outcome_snapshot_hash: str = Field(pattern=_HASH)
    outcome_method_binding_hash: str = Field(pattern=_HASH)
    execution_clock_id: str = Field(min_length=1, max_length=128)
    row_axis_hash: str = Field(pattern=_HASH)
    fold_axis_hash: str = Field(pattern=_HASH)
    maturity_axis_hash: str = Field(pattern=_HASH)
    input_score_value_hash: str = Field(pattern=_HASH)
    raw_economic_return_value_hash: str = Field(pattern=_HASH)
    aggregated_score_value_hash: str = Field(pattern=_HASH)
    ordered_fold_selections: tuple[AlphaScoreAggregationFoldSelection, ...] = Field(min_length=1)
    mean_realized_depth: float = Field(gt=0.0)
    minimum_realized_depth: int = Field(ge=1)
    evidence_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal ordered fold selection evidence.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical evidence_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        draft = dict(values)
        draft.pop("evidence_hash", None)
        provisional = cls.model_construct(**draft, evidence_hash="0" * 64)
        identity = provisional.model_dump(mode="json", exclude={"evidence_hash"})
        return cls(**draft, evidence_hash=str(canonical_hash(identity)))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require sorted unique fold selections and exact evidence identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            AlphaScoreAggregationError: Fold indices repeat/are unordered or the evidence identity
                differs.
        """
        folds = tuple(value.fold_index for value in self.ordered_fold_selections)
        if folds != tuple(sorted(set(folds))):
            raise AlphaScoreAggregationError("alpha_research.score_aggregation_fold_axis_invalid")
        if self.evidence_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"evidence_hash"})
        ):
            raise AlphaScoreAggregationError("alpha_research.score_aggregation_evidence_invalid")
        return self


class AlphaScoreAggregationSurface(_Contract):
    """Durable selected aggregate values bound to their exact evidence."""

    kind: Literal["AlphaScoreAggregationSurface"] = "AlphaScoreAggregationSurface"
    identity_class: Literal["DEVELOPMENT_EVIDENCE_ONLY_NEVER_PUBLISHED"] = (
        "DEVELOPMENT_EVIDENCE_ONLY_NEVER_PUBLISHED"
    )
    evidence_hash: str = Field(pattern=_HASH)
    program_hash: str = Field(pattern=_HASH)
    row_axis_hash: str = Field(pattern=_HASH)
    aggregated_score_value_hash: str = Field(pattern=_HASH)
    row_count: int = Field(ge=1)
    surface_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal one temporal aggregation surface.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical surface_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        draft = dict(values)
        draft.pop("surface_hash", None)
        provisional = cls.model_construct(**draft, surface_hash="0" * 64)
        identity = provisional.model_dump(mode="json", exclude={"surface_hash"})
        return cls(**draft, surface_hash=str(canonical_hash(identity)))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require exact temporal surface content identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            AlphaScoreAggregationError: The canonical surface payload differs from its declared
                hash.
        """
        if self.surface_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"surface_hash"})
        ):
            raise AlphaScoreAggregationError("alpha_research.score_aggregation_surface_invalid")
        return self


def score_value_hash(values: FloatArray) -> str:
    """Byte identity of a score vector, non-finite values included."""
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()


def row_axis_hash(row_sessions: tuple[date, ...], row_listing_ids: tuple[str, ...]) -> str:
    """Hash ordered formation/listing pairs without reordering the row axis.

    Args:
        row_sessions: Formation date for each row.
        row_listing_ids: Listing identity for each corresponding row.

    Returns:
        Canonical hash of ISO-date/listing row keys.

    Raises:
        ValueError: Session and listing axes have different lengths.
    """
    return str(
        canonical_hash(
            [
                f"{session.isoformat()}|{listing}"
                for session, listing in zip(row_sessions, row_listing_ids, strict=True)
            ]
        )
    )


def aggregate_trailing_mean(
    *,
    row_sessions: tuple[date, ...],
    row_listing_ids: tuple[str, ...],
    scores: FloatArray,
    trailing_span_sessions: int,
) -> tuple[FloatArray, IntArray]:
    """Apply the frozen formula. Returns the aggregate and its realized depth.

    The dense session-by-listing intermediate is deliberate. A cumulative-sum
    window would be faster and would **not** reproduce the input bitwise at
    ``span == 1``, because ``cumsum[p] - cumsum[p-1]`` is not ``x[p]`` in
    floating point. Degeneracy at the control span is a property this method
    promises, so the accumulation is direct and its order is fixed.
    """
    if trailing_span_sessions not in APPROVED_AGGREGATION_SPAN_DOMAIN:
        raise AlphaScoreAggregationError("alpha_research.score_aggregation_span_not_in_domain")
    count = len(row_sessions)
    if count != len(row_listing_ids) or scores.shape != (count,) or count == 0:
        raise AlphaScoreAggregationError("alpha_research.score_aggregation_row_axis_invalid")
    if not bool(np.isfinite(scores).all()):
        raise AlphaScoreAggregationError("alpha_research.score_aggregation_input_nonfinite")

    sessions = tuple(sorted(set(row_sessions)))
    listings = tuple(sorted(set(row_listing_ids)))
    session_position = {value: index for index, value in enumerate(sessions)}
    listing_position = {value: index for index, value in enumerate(listings)}
    rows: IntArray = np.fromiter(
        (session_position[value] for value in row_sessions), dtype=np.int64, count=count
    )
    columns: IntArray = np.fromiter(
        (listing_position[value] for value in row_listing_ids), dtype=np.int64, count=count
    )

    panel: FloatArray = np.full((len(sessions), len(listings)), np.nan, dtype=np.float64)
    if len(np.unique(rows * len(listings) + columns)) != count:
        raise AlphaScoreAggregationError("alpha_research.score_aggregation_row_duplicated")
    panel[rows, columns] = scores

    finite = np.isfinite(panel)
    total: FloatArray = np.where(finite, panel, 0.0)
    depth: IntArray = finite.astype(np.int64)
    for lag in range(1, trailing_span_sessions):
        shifted_finite = finite[:-lag]
        shifted_value = panel[:-lag]
        total[lag:] += np.where(shifted_finite, shifted_value, 0.0)
        depth[lag:] += shifted_finite.astype(np.int64)

    # ``depth`` is at least one wherever an input row exists, because the row's
    # own score is finite; the row axis therefore cannot move with the span.
    aggregated_panel = np.divide(total, depth, out=np.full_like(total, np.nan), where=depth > 0)
    aggregated = np.ascontiguousarray(aggregated_panel[rows, columns], dtype=np.float64)
    realized = np.ascontiguousarray(depth[rows, columns], dtype=np.int64)
    if int(realized.min()) < 1 or not bool(np.isfinite(aggregated).all()):
        raise AlphaScoreAggregationError("alpha_research.score_aggregation_output_incomplete")
    aggregated.setflags(write=False)
    realized.setflags(write=False)
    return aggregated, realized


# Fixed Panel-model score handoff. This lives beside Alpha score temporal
# methods because Alpha owns the score clock; strategy schedules stay outside.
SEALED_PANEL_ALPHA_PROGRAM_HASH = "f3efd784dad27f84e2882822d399d3ec08760bf2c748c856c8e90ee99c45c883"
SEALED_PANEL_ALPHA_ROOT_HASH = "3ec1b4175eec87901f69752f3a2a1e53998550a4b9ead277ca52eae116329cbb"
SEALED_PANEL_ALPHA_MODEL_RECIPE_ID = "SPARSE_SESSION_AMPLITUDE_LIGHTGBM"
SEALED_PANEL_ALPHA_VIEW_ID = "SPARSE_SESSION_AMPLITUDE"
SEALED_PANEL_ALPHA_FEATURE_COLUMN_COUNT = 195
SEALED_PANEL_ALPHA_DECISION_SESSION_COUNT = 1_260
FIXED_ALPHA_ROW_AXIS_RECEIPT_CATEGORY = "development/fixed-alpha-row-axis-receipts"
FIXED_ALPHA_ROW_AXIS_LANE_CATEGORY = "development/fixed-alpha-row-axis-lanes"
PANEL_SCORE_READINESS_RECEIPT_CATEGORY = "development/panel-score-readiness-receipts"
PANEL_SCORE_READINESS_POLICY = "DEVELOPMENT_PANEL_SCORE_FORMATION_READINESS_BUDGET"
PANEL_SCORE_READINESS_MINUTES_AFTER_CLOSE = 60

_PANEL_SCORE_READINESS_RATIONALE = (
    "Installed development operational policy, not a campaign-wall-time measurement. "
    "It budgets one hour after close(T) for the already-trained score producer to "
    "resolve installed close(T) facts and form that session's score. The historical "
    "five-fold batch duration is intentionally not divided or reused as nightly latency."
)


class PanelScoreTemporalAuthorityError(ValueError):
    """The durable Alpha graph cannot support its claimed score clock."""


def _seal[ContractT: _Contract](
    model: type[ContractT], field: str, /, **values: object
) -> ContractT:
    draft = model.model_construct(**values, **{field: "0" * 64})
    return model(
        **values,
        **{field: str(canonical_hash(draft.model_dump(mode="json", exclude={field})))},
    )


class FixedAlphaFoldRowAxis(_Contract):
    """One historical OOF row lane bound to both exact score children."""

    fold_index: int = Field(ge=0)
    validation_sessions: tuple[date, ...] = Field(min_length=1)
    row_count_by_session: tuple[int, ...] = Field(min_length=1)
    listing_position_lane_hash: str = Field(pattern=_HASH)
    listing_position_dtype: Literal["<u2"] = "<u2"
    row_count: int = Field(ge=1)
    row_axis_hash: str = Field(pattern=_HASH)
    fixed_score_artifact_hash: str = Field(pattern=_HASH)
    fixed_score_value_hash: str = Field(pattern=_HASH)
    control_score_artifact_hash: str = Field(pattern=_HASH)
    control_score_value_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_axis(self) -> Self:
        """Require sorted unique sessions and positive row counts on the fixed fold axis.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PanelScoreTemporalAuthorityError: Session order, per-session count axis or total row
                count is inconsistent.
        """
        if (
            self.validation_sessions != tuple(sorted(set(self.validation_sessions)))
            or len(self.row_count_by_session) != len(self.validation_sessions)
            or any(value < 1 for value in self.row_count_by_session)
            or sum(self.row_count_by_session) != self.row_count
        ):
            raise PanelScoreTemporalAuthorityError(
                "alpha_research.fixed_alpha_row_axis_receipt_invalid"
            )
        return self


class FixedAlphaRowAxisReceipt(_Contract):
    """Alpha-owned exact row labels bound to one fixed-study score graph."""

    kind: Literal["FixedAlphaRowAxisReceipt"] = "FixedAlphaRowAxisReceipt"
    identity_class: Literal["DEVELOPMENT_ONLY"] = "DEVELOPMENT_ONLY"
    program_hash: str = Field(pattern=_HASH)
    # A historical compatibility receipt may bind an already-existing root.
    # Successor roots bind this receipt handle instead, avoiding an identity cycle.
    root_hash: str | None = Field(default=None, pattern=_HASH)
    source_resolution_hash: str = Field(pattern=_HASH)
    panel_snapshot_hash: str = Field(pattern=_HASH)
    fixed_model_recipe_id: str
    fixed_feature_view_id: str
    control_model_recipe_id: Literal["RELATIVE_CONTROL_RIDGE"] = "RELATIVE_CONTROL_RIDGE"
    ordered_listing_ids: tuple[str, ...] = Field(min_length=2)
    listing_axis_hash: str = Field(pattern=_HASH)
    folds: tuple[FixedAlphaFoldRowAxis, ...] = Field(min_length=1)
    receipt_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal the fixed model/view row-axis receipt.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical receipt_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        return _seal(cls, "receipt_hash", **values)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require installed model/view authority and exact listing/fold axis receipts.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PanelScoreTemporalAuthorityError: Fixed recipe/view, listing order/hash, contiguous fold
                indices or receipt identity differs.
        """
        if (
            self.fixed_model_recipe_id != SEALED_PANEL_ALPHA_MODEL_RECIPE_ID
            or self.fixed_feature_view_id != SEALED_PANEL_ALPHA_VIEW_ID
            or self.ordered_listing_ids != tuple(sorted(set(self.ordered_listing_ids)))
            or self.listing_axis_hash != canonical_hash(list(self.ordered_listing_ids))
            or tuple(value.fold_index for value in self.folds) != tuple(range(len(self.folds)))
            or self.receipt_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"receipt_hash"}))
        ):
            raise PanelScoreTemporalAuthorityError(
                "alpha_research.fixed_alpha_row_axis_receipt_invalid"
            )
        return self


def resolve_panel_score_readiness_policy(handle: str) -> AnchoredInstantPolicy:
    """Resolve the Alpha producer's strategy-independent formation budget."""
    if handle != PANEL_SCORE_READINESS_POLICY:
        raise PanelScoreTemporalAuthorityError(
            "alpha_research.panel_score_readiness_policy_not_installed"
        )
    return AnchoredInstantPolicy.create(
        policy_id=handle,
        basis="INSTALLED_OPERATIONAL_POLICY",
        anchor=SessionAnchor(offset_sessions=0, event="OFFICIAL_CLOSE"),
        minutes_after_anchor=PANEL_SCORE_READINESS_MINUTES_AFTER_CLOSE,
        rationale=_PANEL_SCORE_READINESS_RATIONALE,
    )


class PanelScoreFormationReadinessReceipt(_Contract):
    """One producer policy projected across every formation in a score surface."""

    kind: Literal["PanelScoreFormationReadinessReceipt"] = "PanelScoreFormationReadinessReceipt"
    identity_class: Literal["DEVELOPMENT_ONLY"] = "DEVELOPMENT_ONLY"
    program_hash: str = Field(pattern=_HASH)
    source_identity_hash: str = Field(pattern=_HASH)
    score_surface_hash: str = Field(pattern=_HASH)
    formation_count: int = Field(ge=1)
    formation_axis_hash: str = Field(pattern=_HASH)
    observed_through: AnchoredInstantPolicy
    source_available: AnchoredInstantPolicy
    derived_ready: AnchoredInstantPolicy
    receipt_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal the panel-score source and derived readiness receipt.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical receipt_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        return _seal(cls, "receipt_hash", **values)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_readiness(self) -> Self:
        """Require the official-close source clock and installed derived readiness policy.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PanelScoreTemporalAuthorityError: Source anchor/event/delay, derived policy or readiness
                receipt identity differs.
        """
        expected_ready = resolve_panel_score_readiness_policy(self.derived_ready.policy_id)
        if (
            self.observed_through.anchor.offset_sessions != 0
            or self.observed_through.anchor.event != "OFFICIAL_CLOSE"
            or self.observed_through.minutes_after_anchor != 0
            or self.source_available.anchor.offset_sessions != 0
            or self.source_available.anchor.event != "OFFICIAL_CLOSE"
            or self.source_available.minutes_after_anchor != 0
            or self.derived_ready != expected_ready
            or self.receipt_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"receipt_hash"}))
        ):
            raise PanelScoreTemporalAuthorityError(
                "alpha_research.panel_score_formation_readiness_invalid"
            )
        return self


class AlphaPanelSourceIdentity(_Contract):
    """Bind exact panel axes and Feature authority to the official-close source policy.

    The source-resolution and panel-snapshot identities accompany session, listing and factor axes.
    Feature catalog, observation, availability and source-authority bindings select the governing
    policy; this record itself publishes no score values.
    """

    kind: Literal["AlphaPanelSourceIdentity"] = "AlphaPanelSourceIdentity"
    source_resolution_hash: str = Field(pattern=_HASH)
    panel_snapshot_hash: str = Field(pattern=_HASH)
    ordered_session_axis_hash: str = Field(pattern=_HASH)
    ordered_listing_axis_hash: str = Field(pattern=_HASH)
    ordered_factor_axis_hash: str = Field(pattern=_HASH)
    feature_catalog_hash: str = Field(pattern=_HASH)
    formula_observation_policy_hash: str = Field(pattern=_HASH)
    source_availability_catalog_hash: str = Field(pattern=_HASH)
    source_authority_binding_hash: str = Field(pattern=_HASH)
    governing_availability_policy_id: str
    governing_availability_policy_hash: str = Field(pattern=_HASH)
    governing_availability_phase: Literal["OFFICIAL_CLOSE"] = "OFFICIAL_CLOSE"
    governing_availability_delay_sessions: Literal[0] = 0
    identity_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal the declared Alpha panel source identity.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical identity_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        return _seal(cls, "identity_hash", **values)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require exact declared Alpha panel source identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PanelScoreTemporalAuthorityError: The canonical source payload differs from its identity
                hash.
        """
        if self.identity_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"identity_hash"})
        ):
            raise PanelScoreTemporalAuthorityError(
                "alpha_research.panel_score_source_identity_invalid"
            )
        return self


def resolve_alpha_panel_source_identity(
    *,
    source_resolution_hash: str,
    panel_snapshot_hash: str,
    ordered_session_axis_hash: str,
    ordered_listing_axis_hash: str,
    ordered_factor_axis_hash: str,
    formula_observation_policy_hash: str | None,
    source_availability_policy_hash: str | None,
    source_authority_binding_hash: str | None,
    catalog: FeatureCatalog | None = None,
) -> AlphaPanelSourceIdentity:
    """Resolve panel source identity against current installed Feature authority.

    Args:
        source_resolution_hash: Exact source-resolution receipt.
        panel_snapshot_hash: Exact source panel snapshot.
        ordered_session_axis_hash: Ordered session-axis identity.
        ordered_listing_axis_hash: Ordered listing-axis identity.
        ordered_factor_axis_hash: Ordered factor-axis identity.
        formula_observation_policy_hash: Declared formula clock binding.
        source_availability_policy_hash: Declared source-availability binding.
        source_authority_binding_hash: Declared source-authority binding.
        catalog: The catalog the Panel was built under, a workspace's activations
            included; the shipped one when none is named.

    Returns:
        Sealed source identity bound to the installed Feature catalog and official-close
        availability policy.

    Raises:
        PanelScoreTemporalAuthorityError: The declared observation/availability/authority tuple
            differs from the catalog or its clock is not admitted.
    """
    # The catalog the Panel was built under, a workspace's activations included (EX); the
    # shipped one when the caller names none.
    catalog = FeatureCatalog.load() if catalog is None else catalog
    binding = catalog.binding
    availability = installed_feature_availability_policy()
    if (
        formula_observation_policy_hash,
        source_availability_policy_hash,
        source_authority_binding_hash,
    ) != (
        binding.formula_observation_policy_hash,
        binding.source_availability_policy_hash,
        binding.source_authority_binding_hash,
    ):
        raise PanelScoreTemporalAuthorityError(
            "alpha_research.panel_score_feature_clock_owner_mismatch"
        )
    if (
        availability.available_after_phase is not MarketPhase.OFFICIAL_CLOSE
        or availability.publication_delay_sessions != 0
    ):
        raise PanelScoreTemporalAuthorityError(
            "alpha_research.panel_score_source_availability_not_close_t"
        )
    return AlphaPanelSourceIdentity.create(
        source_resolution_hash=source_resolution_hash,
        panel_snapshot_hash=panel_snapshot_hash,
        ordered_session_axis_hash=ordered_session_axis_hash,
        ordered_listing_axis_hash=ordered_listing_axis_hash,
        ordered_factor_axis_hash=ordered_factor_axis_hash,
        feature_catalog_hash=binding.catalog_hash,
        formula_observation_policy_hash=binding.formula_observation_policy_hash,
        source_availability_catalog_hash=binding.source_availability_policy_hash,
        source_authority_binding_hash=binding.source_authority_binding_hash,
        governing_availability_policy_id=availability.policy_id,
        governing_availability_policy_hash=availability.policy_hash,
    )


class PanelScoreFoldIdentity(_Contract):
    """Bind one ordered validation fold to its artifact, source and score-value axes."""

    kind: Literal["PanelScoreFoldIdentity"] = "PanelScoreFoldIdentity"
    fold_index: int = Field(ge=0)
    artifact_hash: str = Field(pattern=_HASH)
    source_surface_hash: str = Field(pattern=_HASH)
    validation_session_count: int = Field(ge=1)
    validation_session_axis_hash: str = Field(pattern=_HASH)
    row_count: int = Field(ge=1)
    row_axis_hash: str = Field(pattern=_HASH)
    score_value_hash: str = Field(pattern=_HASH)
    identity_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal one declared panel-score fold identity.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical identity_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        return _seal(cls, "identity_hash", **values)


class PanelScoreProducerIdentity(_Contract):
    """Record development-only score production with fixed view and resource authority.

    Formation/fold axes, source lineage, score surface and numerical call counts describe the
    producer. Solver calls are forbidden and resource measurement remains in the declared
    parent-process scope.
    """

    kind: Literal["PanelScoreProducerIdentity"] = "PanelScoreProducerIdentity"
    identity_class: Literal["DEVELOPMENT_ONLY"] = "DEVELOPMENT_ONLY"
    program_hash: str = Field(pattern=_HASH)
    root_hash: str = Field(pattern=_HASH)
    model_recipe_id: str
    feature_view_id: str
    source: AlphaPanelSourceIdentity
    feature_preflight_hash: str = Field(pattern=_HASH)
    feature_column_count: int = Field(ge=1)
    feature_axis_hash: str = Field(pattern=_HASH)
    ordered_formation_sessions: tuple[date, ...] = Field(min_length=1)
    decision_session_axis_hash: str = Field(pattern=_HASH)
    folds: tuple[PanelScoreFoldIdentity, ...] = Field(min_length=1)
    score_surface_hash: str = Field(pattern=_HASH)
    fit_call_count: int = Field(ge=0)
    predict_call_count: int = Field(ge=0)
    metric_call_count: int = Field(ge=0)
    solver_call_count: Literal[0] = 0
    resource_measurement_scope: Literal["PARENT_PROCESS_ONLY"] = "PARENT_PROCESS_ONLY"
    worker_resource_receipt_hashes: tuple[str, ...] = ()
    producer_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal the declared panel-score producer identity.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical producer_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        return _seal(cls, "producer_hash", **values)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_fixed_handoff(self) -> Self:
        """Require the fixed model/view shape, formation population and fold score identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            PanelScoreTemporalAuthorityError: Installed fixed recipe/view/counts, sorted unique
                formation axis/hash, contiguous fold indices, total validation sessions, score
                surface or producer hash differs.
        """
        sessions = self.ordered_formation_sessions
        expected_surface_hash = panel_score_surface_hash(self.folds)
        if (
            self.model_recipe_id != SEALED_PANEL_ALPHA_MODEL_RECIPE_ID
            or self.feature_view_id != SEALED_PANEL_ALPHA_VIEW_ID
            or self.feature_column_count != SEALED_PANEL_ALPHA_FEATURE_COLUMN_COUNT
            or len(sessions) != SEALED_PANEL_ALPHA_DECISION_SESSION_COUNT
            or sessions != tuple(sorted(set(sessions)))
            or self.decision_session_axis_hash
            != canonical_hash([value.isoformat() for value in sessions])
            or tuple(value.fold_index for value in self.folds) != tuple(range(len(self.folds)))
            or sum(value.validation_session_count for value in self.folds)
            != SEALED_PANEL_ALPHA_DECISION_SESSION_COUNT
            or self.score_surface_hash != expected_surface_hash
            or self.producer_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"producer_hash"}))
        ):
            raise PanelScoreTemporalAuthorityError(
                "alpha_research.panel_score_fixed_handoff_invalid"
            )
        return self


def panel_score_surface_hash(folds: tuple[PanelScoreFoldIdentity, ...]) -> str:
    """Hash ordered fold artifact/source/axis/value bindings for one score surface.

    Args:
        folds: Fold identities in declared validation order.

    Returns:
        Canonical hash of the ordered fold bindings without sorting or recomputing scores.
    """
    return str(
        canonical_hash(
            [
                {
                    "artifact_hash": value.artifact_hash,
                    "source_surface_hash": value.source_surface_hash,
                    "validation_session_axis_hash": value.validation_session_axis_hash,
                    "row_axis_hash": value.row_axis_hash,
                    "score_value_hash": value.score_value_hash,
                }
                for value in folds
            ]
        )
    )


def resolve_panel_score_producer_identity(
    *,
    program_hash: str,
    root_hash: str,
    model_recipe_id: str,
    source: AlphaPanelSourceIdentity,
    feature_preflight_hash: str,
    feature_column_count: int,
    feature_axis_hash: str,
    ordered_formation_sessions: tuple[date, ...],
    folds: tuple[PanelScoreFoldIdentity, ...],
    fit_call_count: int,
    predict_call_count: int,
    metric_call_count: int,
    solver_call_count: int,
    resource_measurement_scope: str,
    worker_resource_receipt_hashes: tuple[str, ...],
) -> PanelScoreProducerIdentity:
    """Seal fixed-view score production while refusing solver or resource-scope drift.

    Args:
        program_hash: Exact score-production program.
        root_hash: Declared production root.
        model_recipe_id: Installed fixed-model recipe identity.
        source: Sealed panel source identity.
        feature_preflight_hash: Exact feature preflight receipt.
        feature_column_count: Declared feature-axis size.
        feature_axis_hash: Ordered feature-axis identity.
        ordered_formation_sessions: Declared formation-date axis.
        folds: Ordered validation fold identities.
        fit_call_count: Observed numerical fit calls.
        predict_call_count: Observed numerical prediction calls.
        metric_call_count: Observed metric calls.
        solver_call_count: Observed solver calls, required to be zero.
        resource_measurement_scope: Required parent-process measurement scope.
        worker_resource_receipt_hashes: Retained worker resource evidence identities.

    Returns:
        Validated producer identity with installed feature-view ID and derived formation/surface
        hashes.

    Raises:
        PanelScoreTemporalAuthorityError: Solver use, resource scope or the declared fixed handoff
            violates admission.
    """
    if solver_call_count != 0 or resource_measurement_scope != "PARENT_PROCESS_ONLY":
        raise PanelScoreTemporalAuthorityError(
            "alpha_research.panel_score_resource_authority_invalid"
        )
    return PanelScoreProducerIdentity.create(
        program_hash=program_hash,
        root_hash=root_hash,
        model_recipe_id=model_recipe_id,
        feature_view_id=SEALED_PANEL_ALPHA_VIEW_ID,
        source=source,
        feature_preflight_hash=feature_preflight_hash,
        feature_column_count=feature_column_count,
        feature_axis_hash=feature_axis_hash,
        ordered_formation_sessions=ordered_formation_sessions,
        decision_session_axis_hash=canonical_hash(
            [value.isoformat() for value in ordered_formation_sessions]
        ),
        folds=folds,
        score_surface_hash=panel_score_surface_hash(folds),
        fit_call_count=fit_call_count,
        predict_call_count=predict_call_count,
        metric_call_count=metric_call_count,
        solver_call_count=solver_call_count,
        resource_measurement_scope=resource_measurement_scope,
        worker_resource_receipt_hashes=worker_resource_receipt_hashes,
    )


__all__ = [
    "AGGREGATION_ABSENCE_POLICY",
    "AGGREGATION_CONTROL_SPAN",
    "AGGREGATION_MATURITY_POLICY",
    "AGGREGATION_MINIMUM_PERIODS_POLICY",
    "AGGREGATION_SELECTION_COST_BPS",
    "AGGREGATION_SELECTION_LANE_ID",
    "AGGREGATION_SELECTION_RULE_ID",
    "AGGREGATION_TIE_POLICY",
    "AGGREGATION_WINDOW_UNIT",
    "ALPHA_SCORE_AGGREGATION_METHOD_ID",
    "APPROVED_AGGREGATION_SPAN_DOMAIN",
    "FIXED_ALPHA_ROW_AXIS_LANE_CATEGORY",
    "FIXED_ALPHA_ROW_AXIS_RECEIPT_CATEGORY",
    "PANEL_SCORE_READINESS_MINUTES_AFTER_CLOSE",
    "PANEL_SCORE_READINESS_POLICY",
    "PANEL_SCORE_READINESS_RECEIPT_CATEGORY",
    "SEALED_PANEL_ALPHA_DECISION_SESSION_COUNT",
    "SEALED_PANEL_ALPHA_FEATURE_COLUMN_COUNT",
    "SEALED_PANEL_ALPHA_MODEL_RECIPE_ID",
    "SEALED_PANEL_ALPHA_PROGRAM_HASH",
    "SEALED_PANEL_ALPHA_ROOT_HASH",
    "AlphaPanelSourceIdentity",
    "AlphaScoreAggregationError",
    "AlphaScoreAggregationEvidence",
    "AlphaScoreAggregationFoldSelection",
    "AlphaScoreAggregationProgram",
    "AlphaScoreAggregationSurface",
    "FixedAlphaFoldRowAxis",
    "FixedAlphaRowAxisReceipt",
    "PanelScoreFoldIdentity",
    "PanelScoreFormationReadinessReceipt",
    "PanelScoreProducerIdentity",
    "PanelScoreTemporalAuthorityError",
    "aggregate_trailing_mean",
    "panel_score_surface_hash",
    "resolve_alpha_panel_source_identity",
    "resolve_panel_score_producer_identity",
    "resolve_panel_score_readiness_policy",
    "row_axis_hash",
    "score_value_hash",
]
