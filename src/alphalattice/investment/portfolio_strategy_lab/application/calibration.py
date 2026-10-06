"""Frozen causal sizing evidence, owned and consumed by Portfolio."""

from __future__ import annotations

from datetime import date
from hashlib import sha256
from io import BytesIO
from typing import Literal, Self, cast

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.control.workspace_runtime.content_store import columns_digest
from alphalattice.investment.alpha_research.publication.contracts import (
    FrozenComponentScoreSnapshot,
)
from alphalattice.investment.portfolio_strategy_lab.application.tranche_book_execution import (
    TrancheFormationInputs,
)
from alphalattice.investment.portfolio_strategy_lab.policies.buffered_rank_return import (
    complete_matured_observation_indices,
    complete_matured_rank_curve,
    rank_bucket_observations,
)
from alphalattice.investment.portfolio_strategy_lab.policies.contracts import (
    CausalRankReturnCurveSlice,
)
from alphalattice.investment.portfolio_strategy_lab.publication.artifacts import (
    PortfolioResearchArtifactStore,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

type FloatArray = npt.NDArray[np.float64]
type BoolArray = npt.NDArray[np.bool_]


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)

    @classmethod
    def create(cls, **values: object) -> Self:
        identity = cls.model_construct(**values, content_hash="0" * 64).model_dump(
            mode="json", exclude={"content_hash"}
        )
        return cls(**identity, content_hash=canonical_hash(identity))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        if self.content_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"content_hash"})
        ):
            raise ValueError("portfolio_calibration.identity_invalid")
        return self


class FrozenRankCalibrationRule(_Contract):
    """The installed complete-matured rule, not user-editable tuning controls.

    Its activation is part of the frozen research contract just like its buckets
    and lookback: the book's 521st formation, as the historical book's sizing
    activation is. The session is that formation on the axis the rule serves: the
    G6 research's calendar put it on 2021-09-07, a user's research book has its own
    (LS1, V459). It is held where the axis is known: a decision checkpoint admits a
    prepared input only when its formations name this session at this position.
    Another policy requires its own explicit admission, not a changed seed.
    """

    bucket_count: Literal[20] = 20
    lookback: Literal[252] = 252
    minimum_members: Literal[80] = 80
    tie_order: Literal["DECLARED_LISTING_AXIS"] = "DECLARED_LISTING_AXIS"
    window: Literal["LATEST_COMPLETE_MATURED_OBSERVATIONS"] = "LATEST_COMPLETE_MATURED_OBSERVATIONS"
    return_recipe: Literal["NEXT_OPEN_TO_OPEN_ONE_SESSION"] = "NEXT_OPEN_TO_OPEN_ONE_SESSION"
    activation_session: date
    activation_position: Literal[521] = 521
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class CalibrationObservations(_Contract):
    """Missing outcomes stay missing in packed arrays; no zero-fill or future tail."""

    kind: Literal["CalibrationObservations"] = "CalibrationObservations"
    purpose: Literal["POST_OBSERVED_CALIBRATION_QA"] = "POST_OBSERVED_CALIBRATION_QA"
    origin: Literal["FROZEN_RESEARCH_INPUTS", "SYNTHETIC_QA_INPUTS", "LOCAL_QA_OBSERVATIONS"]
    strategy_package_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    component_recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    rule: FrozenRankCalibrationRule
    formation_sessions: tuple[date, ...]
    holding_end_sessions: tuple[date | None, ...]
    ordered_listing_ids: tuple[str, ...]
    source_hashes: tuple[str, ...]
    score_snapshot_hashes: tuple[str, ...] = ()
    array_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_axes(self) -> Self:
        """Require nonempty causal formation/listing axes and explicit valid source identities.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Formation dates repeat/are unordered, listing/source axes are
                absent/repeated, holding-end coverage or causality disagrees, or source/score hash
                syntax is invalid.
        """
        if (
            not self.formation_sessions
            or self.formation_sessions != tuple(sorted(set(self.formation_sessions)))
            or len(self.holding_end_sessions) != len(self.formation_sessions)
            or len(set(self.ordered_listing_ids)) != len(self.ordered_listing_ids)
            or not self.ordered_listing_ids
            or not self.source_hashes
            or any(
                end is not None and end <= day
                for day, end in zip(self.formation_sessions, self.holding_end_sessions, strict=True)
            )
            or any(
                len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
                for value in (*self.source_hashes, *self.score_snapshot_hashes)
            )
        ):
            raise ValueError("portfolio_calibration.observation_axis_invalid")
        return self


_ARRAY_NAMES = ("scores", "returns", "eligible")


def _legacy_npz_digest(scores: FloatArray, returns: FloatArray, eligible: BoolArray) -> str:
    # The digest observations sealed before V210 carry: their arrays packed as one npz.
    data = BytesIO()
    np.savez(data, scores=scores, returns=returns, eligible=eligible)
    return sha256(data.getvalue()).hexdigest()


def publish_observations(
    store: PortfolioResearchArtifactStore,
    *,
    scores: FloatArray,
    returns: FloatArray,
    eligible: BoolArray,
    **metadata: object,
) -> CalibrationObservations:
    """Verify and publish exact calibration score, return and eligibility arrays.

    Args:
        store: Portfolio artifact publication owner.
        scores: Formation-by-listing calibration scores.
        returns: Realized simple returns on the same axis.
        eligible: Boolean calibration eligibility on the same axis.
        metadata: Explicit observation, rule and source lineage fields.

    Returns:
        Sealed observations selecting the published arrays and metadata.

    Raises:
        ValueError: Arrays violate the declared observation axes or numerical contract.
    """
    columns = {"scores": scores, "returns": returns, "eligible": eligible}
    value = CalibrationObservations.create(**metadata, array_hash=columns_digest(columns))
    _check_arrays(value, scores, returns, eligible)
    store.publish_columns(category="calibration-arrays", columns=columns)
    store.publish(category="calibration-observations", value=value, identity_field="content_hash")
    return value


def _check_arrays(
    value: CalibrationObservations, scores: FloatArray, returns: FloatArray, eligible: BoolArray
) -> None:
    shape = (len(value.formation_sessions), len(value.ordered_listing_ids))
    if (
        scores.shape != shape
        or returns.shape != shape
        or eligible.shape != shape
        or scores.dtype != np.float64
        or returns.dtype != np.float64
        or eligible.dtype != np.bool_
        or np.isinf(scores).any()
        or np.isinf(returns).any()
    ):
        raise ValueError("portfolio_calibration.observation_values_invalid")


def publish_consumed_observations(
    store: PortfolioResearchArtifactStore,
    *,
    decision_session: date,
    scores: FloatArray,
    returns: FloatArray,
    eligible: BoolArray,
    formation_sessions: tuple[date, ...],
    holding_end_sessions: tuple[date | None, ...],
    rule: FrozenRankCalibrationRule,
    **metadata: object,
) -> CalibrationObservations:
    """Seal the consumed window and current row, not another full history copy.

    The immutable seed and source revision remain parents. Selecting from the
    complete history precedes slicing, so missing observations or calendar gaps
    cannot move the maturity rule. The current row binds today's eligibility.
    """
    observed = rank_bucket_observations(
        scores=scores,
        realized_simple_returns=returns,
        decision_eligible=eligible,
        rank_keys=np.arange(scores.shape[1]),
        bucket_count=rule.bucket_count,
    )
    selected = complete_matured_observation_indices(
        complete_indices=np.flatnonzero(np.isfinite(observed).all(axis=1)),
        observation_sessions=formation_sessions,
        holding_end_sessions=holding_end_sessions,
        decision_session=decision_session,
        lookback=rule.lookback,
    )
    kept = sorted({*selected, formation_sessions.index(decision_session)})
    return publish_observations(
        store,
        scores=scores[kept],
        returns=returns[kept],
        eligible=eligible[kept],
        rule=rule,
        formation_sessions=tuple(formation_sessions[i] for i in kept),
        holding_end_sessions=tuple(holding_end_sessions[i] for i in kept),
        **metadata,
    )


def load_observations(
    store: PortfolioResearchArtifactStore, content_hash: str
) -> tuple[CalibrationObservations, FloatArray, FloatArray, BoolArray]:
    """Read exact calibration arrays with current columns or compatible historical NPZ.

    Args:
        store: Portfolio artifact readback owner.
        content_hash: Exact observation metadata identity.

    Returns:
        Validated observations, scores, simple returns and eligibility in the declared shape.

    Raises:
        ValueError: Stored fields, element counts or reconstructed arrays violate the observation
            contract.
    """
    value = store.load(
        category="calibration-observations",
        content_hash=content_hash,
        model=CalibrationObservations,
        identity_field="content_hash",
    )
    shape = (len(value.formation_sessions), len(value.ordered_listing_ids))
    if store.holds_array(category="calibration-arrays", content_hash=value.array_hash):
        columns = store.load_columns(category="calibration-arrays", content_hash=value.array_hash)
        if tuple(columns) != _ARRAY_NAMES or any(
            v.size != shape[0] * shape[1] for v in columns.values()
        ):
            raise ValueError("portfolio_calibration.observation_fields_invalid")
        scores, returns, eligible = (columns[name].reshape(shape) for name in _ARRAY_NAMES)
    else:
        # Observations sealed before V210 packed their arrays as one npz `.bin`.
        data = store.load_packed_bytes(category="calibration-arrays", content_hash=value.array_hash)
        with np.load(BytesIO(data), allow_pickle=False) as arrays:
            if set(arrays.files) != set(_ARRAY_NAMES):
                raise ValueError("portfolio_calibration.observation_fields_invalid")
            scores, returns, eligible = (arrays[name] for name in _ARRAY_NAMES)
    _check_arrays(value, scores, returns, eligible)
    return value, scores, returns, eligible


class PreparedPortfolioComponentInput(_Contract):
    """Executable input, explicitly not a recommendation or realized result."""

    kind: Literal["PreparedPortfolioComponentInput"] = "PreparedPortfolioComponentInput"
    disposition: Literal["PORTFOLIO_INPUT_QA_NOT_RECOMMENDATION"] = (
        "PORTFOLIO_INPUT_QA_NOT_RECOMMENDATION"
    )
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    score_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    observation_hash: str | None = Field(pattern=r"^[0-9a-f]{64}$")
    strategy_package_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    component_recipe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    formation_session: date
    rule: FrozenRankCalibrationRule | None
    ordered_listing_ids: tuple[str, ...]
    scores: tuple[float | None, ...]
    decision_eligible: tuple[bool, ...]
    bucket_means: tuple[float | None, ...]
    selected_observation_sessions: tuple[date, ...]
    latest_holding_end_session: date | None
    sizing_rule: Literal["ew", "mu.iv0"]
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_input(self) -> Self:
        """Require component axes and causal sizing/calibration evidence for the selected rule.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Listing/score/eligibility axes disagree, active scores are absent/nonfinite,
                equal-weight inputs carry calibration evidence, matured observations violate clocks,
                or activated rank sizing lacks complete calibration.
        """
        width = len(self.ordered_listing_ids)
        if self.rule is None:
            if (
                not width
                or len(set(self.ordered_listing_ids)) != width
                or len(self.scores) != width
                or len(self.decision_eligible) != width
                or self.sizing_rule != "ew"
                or self.observation_hash is not None
                or self.bucket_means
                or self.selected_observation_sessions
                or self.latest_holding_end_session is not None
                or any(value is not None and not np.isfinite(value) for value in self.scores)
                or any(
                    live and score is None
                    for live, score in zip(self.decision_eligible, self.scores, strict=True)
                )
            ):
                raise ValueError("portfolio_calibration.ew_input_invalid")
            return self
        finite = all(value is not None and np.isfinite(value) for value in self.bucket_means)
        if (
            not width
            or len(set(self.ordered_listing_ids)) != width
            or len(self.scores) != width
            or len(self.decision_eligible) != width
            or len(self.bucket_means) != self.rule.bucket_count
            or any(value is not None and not np.isfinite(value) for value in self.scores)
            or any(
                active and value is None
                for active, value in zip(self.decision_eligible, self.scores, strict=True)
            )
            or self.selected_observation_sessions
            != tuple(sorted(set(self.selected_observation_sessions)))
            or any(day >= self.formation_session for day in self.selected_observation_sessions)
            or (
                self.latest_holding_end_session is not None
                and self.latest_holding_end_session > self.formation_session
            )
            or (self.sizing_rule == "mu.iv0")
            != (self.formation_session >= self.rule.activation_session)
            or (
                self.sizing_rule == "mu.iv0"
                and (
                    not finite
                    or len(self.selected_observation_sessions) != self.rule.lookback
                    or self.latest_holding_end_session is None
                )
            )
        ):
            raise ValueError("portfolio_calibration.prepared_input_invalid")
        return self

    def formation_input(self, *, local_index: int = 0) -> TrancheFormationInputs:
        """No I/O, second sizing algorithm or implicit position-state initialization."""
        if self.rule is None:
            return TrancheFormationInputs(
                formation_session=self.formation_session,
                scores=np.asarray([np.nan if v is None else v for v in self.scores]),
                decision_eligible=np.asarray(self.decision_eligible, dtype=np.bool_),
                risk_allocation=None,
                causal_rank_return_curve=None,
            )
        means = np.asarray([np.nan if value is None else value for value in self.bucket_means])
        available = (
            bool(np.isfinite(means).all())
            and len(self.selected_observation_sessions) == self.rule.lookback
        )
        return TrancheFormationInputs(
            formation_session=self.formation_session,
            scores=np.asarray([np.nan if value is None else value for value in self.scores]),
            decision_eligible=np.asarray(self.decision_eligible, dtype=np.bool_),
            risk_allocation=None,
            causal_rank_return_curve=CausalRankReturnCurveSlice(
                formation_index=local_index,
                formation_session=self.formation_session,
                bucket_means=means,
                bucket_support_counts=tuple(len(self.selected_observation_sessions) for _ in means),
                admitted_formation_count=len(self.selected_observation_sessions),
                disposition="AVAILABLE" if available else "INSUFFICIENT_MATURED_FORMATION_HISTORY",
                curve_hash=self.content_hash,
            ),
        )

    def readout(self) -> dict[str, object]:
        """Project component sizing, support and matured calibration evidence without fitting.

        Returns:
            Read model with formation, sizing/support/authority facts and zero fit/prediction calls.
        """
        if self.rule is None:
            return {
                "formation_session": self.formation_session.isoformat(),
                "sizing_rule": "ew",
                "calibration": "NOT_CONSUMED",
                "eligible_listing_count": sum(self.decision_eligible),
                "purpose": self.disposition,
                "prediction_calls": 0,
                "fit_calls": 0,
            }
        return {
            "formation_session": self.formation_session.isoformat(),
            "matured_observation_count": len(self.selected_observation_sessions),
            "latest_observation_session": self.selected_observation_sessions[-1].isoformat()
            if self.selected_observation_sessions
            else None,
            "latest_holding_end_session": str(self.latest_holding_end_session)
            if self.latest_holding_end_session
            else None,
            "sizing_rule": self.sizing_rule,
            "activation_session": self.rule.activation_session.isoformat(),
            "eligible_listing_count": sum(self.decision_eligible),
            "purpose": self.disposition,
            "prediction_calls": 0,
            "fit_calls": 0,
        }


class PreparedPortfolioBookInput(_Contract):
    """One sealed component tuple, not a second calibration or execution owner."""

    kind: Literal["PreparedPortfolioBookInput"] = "PreparedPortfolioBookInput"
    component_ids: tuple[str, ...]
    components: tuple[PreparedPortfolioComponentInput, ...] = Field(min_length=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_components(self) -> Self:
        """Require unique component identifiers and common package/formation/listing authority.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Component IDs/counts repeat or package, formation or listing axes differ
                across components.
        """
        if (
            len(self.component_ids) != len(self.components)
            or len(set(self.component_ids)) != len(self.components)
            or any(
                item.strategy_package_hash != self.strategy_package_hash
                or item.formation_session != self.formation_session
                or item.ordered_listing_ids != self.ordered_listing_ids
                for item in self.components
            )
        ):
            raise ValueError("portfolio_calibration.book_input_axis_invalid")
        return self

    @property
    def strategy_package_hash(self) -> str:
        """Read the common strategy package identity from the first component.

        Returns:
            First component strategy_package_hash.

        Raises:
            IndexError: The component tuple is empty.
        """
        return self.components[0].strategy_package_hash

    @property
    def formation_session(self) -> date:
        """Read the common formation session from the first component.

        Returns:
            First component formation_session.

        Raises:
            IndexError: The component tuple is empty.
        """
        return self.components[0].formation_session

    @property
    def ordered_listing_ids(self) -> tuple[str, ...]:
        """Read the common ordered listing axis from the first component.

        Returns:
            First component ordered_listing_ids.

        Raises:
            IndexError: The component tuple is empty.
        """
        return self.components[0].ordered_listing_ids

    def readout(self) -> dict[str, object]:
        """Project formation and per-component calibration read models in declared order.

        Returns:
            Read model keyed by the declared component identities.
        """
        return {
            "formation_session": self.formation_session.isoformat(),
            "components": {
                name: item.readout()
                for name, item in zip(self.component_ids, self.components, strict=True)
            },
        }


def compile_ew_component_input(
    *,
    request_hash: str,
    score: FrozenComponentScoreSnapshot,
    current_eligible: BoolArray,
    outsider_sentinel: float | None,
) -> PreparedPortfolioComponentInput:
    """Prepare equal-weight score eligibility without consuming rank calibration.

    Args:
        request_hash: Exact component input request.
        score: Frozen component score snapshot.
        current_eligible: Current listing-axis eligibility mask.
        outsider_sentinel: Optional explicit score marking a listing outside candidate support.

    Returns:
        Equal-weight input using finite declared scores and excluding the outsider sentinel.

    Raises:
        ValueError: Current eligibility shape differs from the score listing axis.
    """
    if current_eligible.shape != (len(score.ordered_listing_ids),):
        raise ValueError("portfolio_calibration.ew_eligibility_axis_invalid")
    return PreparedPortfolioComponentInput.create(
        request_hash=request_hash,
        score_snapshot_hash=score.snapshot_hash,
        observation_hash=None,
        strategy_package_hash=score.strategy_package_hash,
        component_recipe_hash=score.component_recipe_hash,
        formation_session=score.formation_session,
        rule=None,
        ordered_listing_ids=score.ordered_listing_ids,
        scores=score.scores,
        decision_eligible=tuple(
            bool(live and value is not None and value != outsider_sentinel)
            for live, value in zip(current_eligible, score.scores, strict=True)
        ),
        bucket_means=(),
        selected_observation_sessions=(),
        latest_holding_end_session=None,
        sizing_rule="ew",
    )


def select_calibration_scores(
    values: tuple[FrozenComponentScoreSnapshot, ...],
    *,
    package_hash: str,
    component_recipe_hash: str,
    issued: dict[date, str],
    current: FrozenComponentScoreSnapshot | None = None,
) -> tuple[FrozenComponentScoreSnapshot, ...]:
    """One history selection rule; axes may evolve, scientific identities may not."""
    selected: dict[date, FrozenComponentScoreSnapshot] = {}
    for value in values:
        if (
            value.strategy_package_hash != package_hash
            or value.component_recipe_hash != component_recipe_hash
        ):
            continue
        day = value.formation_session
        if current is not None and day >= current.formation_session:
            continue
        if day in issued and issued[day] != value.snapshot_hash:
            continue
        if day in selected and selected[day] != value:
            raise ValueError("portfolio_calibration.score_history_selection_ambiguous")
        selected[day] = value
    if current is not None:
        if (
            current.strategy_package_hash != package_hash
            or current.component_recipe_hash != component_recipe_hash
        ):
            raise ValueError("portfolio_calibration.score_binding_mismatch")
        selected[current.formation_session] = current
    return tuple(selected[d] for d in sorted(selected))


def compile_portfolio_calibration(
    *,
    request_hash: str,
    score: FrozenComponentScoreSnapshot,
    observations: CalibrationObservations,
    scores: FloatArray,
    returns: FloatArray,
    eligible: BoolArray,
    current_eligible: BoolArray,
) -> PreparedPortfolioComponentInput:
    """Compile mature rank-bucket sizing evidence bound to exact current score values.

    Current score/eligibility must match retained observation values on the component listing
    subsequence. Only complete matured observation windows are consumed; activated sizing requires
    complete finite calibration and sufficient live eligible support.

    Args:
        request_hash: Exact component input request.
        score: Frozen current component score snapshot.
        observations: Sealed calibration observation/rule/source metadata.
        scores: Exact retained calibration scores.
        returns: Exact retained realized simple returns.
        eligible: Exact retained calibration eligibility.
        current_eligible: Current eligibility on the component listing axis.

    Returns:
        Validated prepared input with selected mature sessions, rank means and activation-dependent
        sizing.

    Raises:
        ValueError: Observation content/source axes/current values differ, current formation is
            absent, matured calibration is insufficient or live selection support is below the
            installed minimum.
    """
    _check_arrays(observations, scores, returns, eligible)
    if observations.array_hash not in {
        columns_digest({"scores": scores, "returns": returns, "eligible": eligible}),
        _legacy_npz_digest(scores, returns, eligible),
    }:
        raise ValueError("portfolio_calibration.observation_content_mismatch")
    if (
        not set(score.ordered_listing_ids) <= set(observations.ordered_listing_ids)
        or score.strategy_package_hash != observations.strategy_package_hash
        or score.component_recipe_hash != observations.component_recipe_hash
        or current_eligible.shape != (len(score.scores),)
    ):
        raise ValueError("portfolio_calibration.score_binding_mismatch")
    if score.formation_session not in observations.formation_sessions:
        raise ValueError("portfolio_calibration.current_formation_absent")
    row = observations.formation_sessions.index(score.formation_session)
    columns = [observations.ordered_listing_ids.index(v) for v in score.ordered_listing_ids]
    if not np.array_equal(current_eligible, eligible[row, columns]) or not np.array_equal(
        scores[row, columns],
        np.asarray([np.nan if v is None else v for v in score.scores]),
        equal_nan=True,
    ):
        raise ValueError("portfolio_calibration.current_values_unbound")
    observed = rank_bucket_observations(
        scores=scores,
        realized_simple_returns=returns,
        decision_eligible=eligible,
        rank_keys=np.arange(scores.shape[1]),
        bucket_count=observations.rule.bucket_count,
    )
    curves, _ = complete_matured_rank_curve(
        observations=observed,
        observation_sessions=observations.formation_sessions,
        holding_end_sessions=observations.holding_end_sessions,
        decision_sessions=(score.formation_session,),
        lookback=observations.rule.lookback,
    )
    selected = complete_matured_observation_indices(
        complete_indices=np.flatnonzero(np.isfinite(observed).all(axis=1)),
        observation_sessions=observations.formation_sessions,
        holding_end_sessions=observations.holding_end_sessions,
        decision_session=score.formation_session,
        lookback=observations.rule.lookback,
    )
    active = score.formation_session >= observations.rule.activation_session
    if active and not np.isfinite(curves[0]).all():
        raise ValueError("portfolio_calibration.insufficient_matured_observations")
    if int(np.count_nonzero(current_eligible & np.asarray(score.live))) < 35:
        raise ValueError("portfolio_calibration.current_selection_support_insufficient")
    return PreparedPortfolioComponentInput.create(
        request_hash=request_hash,
        score_snapshot_hash=score.snapshot_hash,
        observation_hash=observations.content_hash,
        strategy_package_hash=score.strategy_package_hash,
        component_recipe_hash=score.component_recipe_hash,
        formation_session=score.formation_session,
        rule=observations.rule,
        ordered_listing_ids=score.ordered_listing_ids,
        scores=score.scores,
        decision_eligible=tuple(
            bool(a and b) for a, b in zip(score.live, current_eligible, strict=True)
        ),
        bucket_means=tuple(float(v) if np.isfinite(v) else None for v in curves[0]),
        selected_observation_sessions=tuple(observations.formation_sessions[i] for i in selected),
        latest_holding_end_session=max(
            cast(date, observations.holding_end_sessions[i]) for i in selected
        )
        if selected
        else None,
        sizing_rule="mu.iv0" if active else "ew",
    )
