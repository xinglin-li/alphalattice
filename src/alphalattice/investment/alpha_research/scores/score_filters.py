"""Installed causal score filters over durable formation-score surfaces."""

from __future__ import annotations

from datetime import date
from types import MappingProxyType
from typing import Final, Literal, Self, cast

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.investment.alpha_research.scores.temporal_aggregation import (
    ALPHA_SCORE_AGGREGATION_METHOD_ID,
    AlphaScoreAggregationError,
    aggregate_trailing_mean,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]
type AlphaScoreFilterMethodId = Literal[
    "RAW_SCORE_TEMPORAL_FILTER",
    "TRAILING_MEAN_FORMATION_SCORE",
    "EWMA_FORMATION_SCORE",
]
type AlphaScoreFilterInputMode = Literal[
    "RAW_SCORE",
    "CROSS_SECTION_STANDARDIZED_SCORE",
]

_HASH = r"^[0-9a-f]{64}$"
RAW_SCORE_TEMPORAL_FILTER_METHOD_ID: Final = "RAW_SCORE_TEMPORAL_FILTER"
EWMA_FORMATION_SCORE_METHOD_ID: Final = "EWMA_FORMATION_SCORE"
RAW_SCORE_INPUT_MODE: Final = "RAW_SCORE"
CROSS_SECTION_STANDARDIZED_SCORE_INPUT_MODE: Final = "CROSS_SECTION_STANDARDIZED_SCORE"
SCORE_FILTER_ROW_AXIS_POLICY: Final = "INPUT_ROW_AXIS_INVARIANT"
SCORE_FILTER_MINIMUM_PERIODS: Final = 1
SCORE_FILTER_SELECTION_POLICY_ID: Final = "MATURED_PRIOR_FOLD_SESSION_RANK_IC_PAIRED"
SCORE_FILTER_CANDIDATE_SPAN_DOMAIN: Final[tuple[int, ...]] = (1, 21, 42, 63)


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class AlphaScoreFilterDescriptor(_Contract):
    """One numerical score-filter method, independent of scientific selection."""

    kind: Literal["AlphaScoreFilterDescriptor"] = "AlphaScoreFilterDescriptor"
    method_id: AlphaScoreFilterMethodId
    owner: Literal["alpha_research.scores.score_filters"] = "alpha_research.scores.score_filters"
    formula_id: str
    implementation_semantics_hash: str = Field(pattern=_HASH)
    admitted_span_domain: tuple[int, ...] = Field(min_length=1)
    admitted_input_modes: tuple[AlphaScoreFilterInputMode, ...] = (
        "RAW_SCORE",
        "CROSS_SECTION_STANDARDIZED_SCORE",
    )
    window_unit: Literal["FORMATION_SESSIONS_ON_SURFACE_AXIS"] = (
        "FORMATION_SESSIONS_ON_SURFACE_AXIS"
    )
    minimum_periods: Literal[1] = 1
    absence_policy: Literal["ABSENT_SESSION_CONTRIBUTES_NOTHING_NO_FILL"] = (
        "ABSENT_SESSION_CONTRIBUTES_NOTHING_NO_FILL"
    )
    row_axis_policy: Literal["INPUT_ROW_AXIS_INVARIANT"] = "INPUT_ROW_AXIS_INVARIANT"
    target_contact: Literal["NONE"] = "NONE"
    descriptor_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(cls, **values: object) -> Self:
        """Seal one declared score-filter descriptor.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical descriptor_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        provisional = cls.model_construct(**values, descriptor_hash="0" * 64)
        return cls(
            **values,
            descriptor_hash=str(
                canonical_hash(provisional.model_dump(mode="json", exclude={"descriptor_hash"}))
            ),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require a sorted unique span domain and exact descriptor identity.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            AlphaScoreAggregationError: The admitted span domain repeats/is unordered or the
                descriptor hash differs.
        """
        if self.admitted_span_domain != tuple(
            sorted(set(self.admitted_span_domain))
        ) or self.descriptor_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"descriptor_hash"})
        ):
            raise AlphaScoreAggregationError("alpha_research.score_filter_descriptor_invalid")
        return self


class AlphaScoreFilterCatalog:
    """Installed numerical methods; evaluation and Portfolio policy live elsewhere."""

    def __init__(self, descriptors: tuple[AlphaScoreFilterDescriptor, ...]) -> None:
        """Index declared score-filter descriptors by unique method identity.

        Args:
            descriptors: Ordered descriptors to index; an empty catalog is permitted.

        Raises:
            AlphaScoreAggregationError: A method identity is declared more than once.
        """
        if len({value.method_id for value in descriptors}) != len(descriptors):
            raise AlphaScoreAggregationError("alpha_research.score_filter_method_duplicated")
        self._descriptors = MappingProxyType({value.method_id: value for value in descriptors})

    @property
    def descriptors(self) -> tuple[AlphaScoreFilterDescriptor, ...]:
        """Read all retained descriptors in sorted method order.

        Returns:
            Immutable tuple ordered by method_id.
        """
        return tuple(self._descriptors[key] for key in sorted(self._descriptors))

    def resolve(self, method_id: str) -> AlphaScoreFilterDescriptor:
        """Resolve one explicitly installed score-filter descriptor.

        Args:
            method_id: Exact installed method identity.

        Returns:
            Typed descriptor for that method.

        Raises:
            AlphaScoreAggregationError: The requested method is not installed.
        """
        descriptor = self._descriptors.get(cast(AlphaScoreFilterMethodId, method_id))
        if descriptor is None:
            raise AlphaScoreAggregationError("alpha_research.score_filter_method_not_installed")
        return descriptor


def _method_semantics_hash(method_id: AlphaScoreFilterMethodId) -> str:
    return str(
        canonical_hash(
            {
                "owner": "alpha_research.scores.score_filters",
                "method_id": method_id,
                "formation_window": True,
                "minimum_periods": 1,
                "absence": "NO_CONTRIBUTION_NO_FILL",
                "row_axis": "INVARIANT",
                "raw_mode": "PRESERVE_CROSS_SESSION_CONFIDENCE_MAGNITUDE",
                "standardized_mode": "PER_SESSION_MEAN_POPULATION_STD_Z",
                "constant_session": "DETERMINISTIC_ZERO",
                "span_one": "EXACT_WITHIN_INPUT_MODE",
                "ewma": "ADJUST_FALSE_WITH_ABSENT_SESSION_DECAY",
                "trailing_owner": (
                    "alpha_research.scores.temporal_aggregation.aggregate_trailing_mean"
                    if method_id == ALPHA_SCORE_AGGREGATION_METHOD_ID
                    else None
                ),
            }
        )
    )


def build_installed_alpha_score_filter_catalog() -> AlphaScoreFilterCatalog:
    """Install only methods consumed by the declared filter frontier."""
    return AlphaScoreFilterCatalog(
        tuple(
            AlphaScoreFilterDescriptor.create(
                method_id=method_id,
                formula_id=formula_id,
                implementation_semantics_hash=_method_semantics_hash(
                    cast(AlphaScoreFilterMethodId, method_id)
                ),
                admitted_span_domain=domain,
            )
            for method_id, formula_id, domain in (
                (
                    RAW_SCORE_TEMPORAL_FILTER_METHOD_ID,
                    "IDENTITY_CURRENT_FORMATION_SCORE",
                    (1,),
                ),
                (
                    ALPHA_SCORE_AGGREGATION_METHOD_ID,
                    "CAUSAL_TRAILING_MEAN_MIN_PERIODS_ONE",
                    SCORE_FILTER_CANDIDATE_SPAN_DOMAIN,
                ),
                (
                    EWMA_FORMATION_SCORE_METHOD_ID,
                    "CAUSAL_ADJUST_FALSE_EWMA_WITH_SESSION_GAP_DECAY",
                    SCORE_FILTER_CANDIDATE_SPAN_DOMAIN,
                ),
            )
        )
    )


class AlphaScoreFilterSpec(_Contract):
    """Identity-bound method selection; callers never supply numerical callbacks."""

    kind: Literal["AlphaScoreFilterSpec"] = "AlphaScoreFilterSpec"
    method_id: AlphaScoreFilterMethodId
    span_sessions: int = Field(ge=1)
    input_mode: AlphaScoreFilterInputMode
    method_descriptor_hash: str = Field(pattern=_HASH)
    spec_id: str
    spec_hash: str = Field(pattern=_HASH)

    @classmethod
    def create(
        cls,
        *,
        method_id: str,
        span_sessions: int,
        input_mode: str = RAW_SCORE_INPUT_MODE,
    ) -> Self:
        """Seal an installed score-filter method, span and input mode.

        Args:
            method_id: Installed score-filter method.
            span_sessions: Span admitted by its descriptor.
            input_mode: Admitted raw or declared standardized score input mode.

        Returns:
            Spec bound to the installed descriptor and method/input/span identity.

        Raises:
            AlphaScoreAggregationError: Method, span or input mode is not admitted.
        """
        descriptor = build_installed_alpha_score_filter_catalog().resolve(method_id)
        if (
            span_sessions not in descriptor.admitted_span_domain
            or input_mode not in descriptor.admitted_input_modes
        ):
            raise AlphaScoreAggregationError("alpha_research.score_filter_spec_not_admitted")
        spec_id = f"{descriptor.method_id}::{input_mode}::span-{span_sessions}"
        draft = {
            "method_id": descriptor.method_id,
            "span_sessions": span_sessions,
            "input_mode": input_mode,
            "method_descriptor_hash": descriptor.descriptor_hash,
            "spec_id": spec_id,
        }
        provisional = cls.model_construct(**draft, spec_hash="0" * 64)
        return cls(
            **draft,
            spec_hash=str(
                canonical_hash(provisional.model_dump(mode="json", exclude={"spec_hash"}))
            ),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Reconcile filter admission and identity with the installed descriptor.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            AlphaScoreAggregationError: Method/span/input admission, descriptor binding, generated
                spec ID or canonical spec hash differs.
        """
        descriptor = build_installed_alpha_score_filter_catalog().resolve(self.method_id)
        expected_id = f"{self.method_id}::{self.input_mode}::span-{self.span_sessions}"
        if (
            self.span_sessions not in descriptor.admitted_span_domain
            or self.input_mode not in descriptor.admitted_input_modes
            or self.method_descriptor_hash != descriptor.descriptor_hash
            or self.spec_id != expected_id
            or self.spec_hash != canonical_hash(self.model_dump(mode="json", exclude={"spec_hash"}))
        ):
            raise AlphaScoreAggregationError("alpha_research.score_filter_spec_invalid")
        return self


def _cross_section_standardize(
    *,
    row_sessions: tuple[date, ...],
    row_listing_ids: tuple[str, ...],
    scores: FloatArray,
) -> FloatArray:
    count = len(row_sessions)
    if count != len(row_listing_ids) or scores.shape != (count,) or count == 0:
        raise AlphaScoreAggregationError("alpha_research.score_filter_row_axis_invalid")
    output: FloatArray = np.empty(count, dtype=np.float64)
    grouped: dict[date, list[int]] = {}
    for position, session in enumerate(row_sessions):
        grouped.setdefault(session, []).append(position)
    for positions in grouped.values():
        selected: IntArray = np.asarray(positions, dtype=np.int64)
        values = scores[selected]
        if not bool(np.isfinite(values).all()):
            raise AlphaScoreAggregationError("alpha_research.score_filter_input_nonfinite")
        centered = values - float(np.mean(values))
        dispersion = float(np.std(centered, ddof=0))
        output[selected] = 0.0 if dispersion == 0.0 else centered / dispersion
    output.setflags(write=False)
    return output


def _aggregate_ewma(
    *,
    row_sessions: tuple[date, ...],
    row_listing_ids: tuple[str, ...],
    scores: FloatArray,
    span_sessions: int,
) -> tuple[FloatArray, IntArray]:
    """Causal adjust-false EWMA; absent sessions decay state but emit no row."""

    if span_sessions not in SCORE_FILTER_CANDIDATE_SPAN_DOMAIN:
        raise AlphaScoreAggregationError("alpha_research.score_filter_span_not_in_domain")
    count = len(row_sessions)
    if count != len(row_listing_ids) or scores.shape != (count,) or count == 0:
        raise AlphaScoreAggregationError("alpha_research.score_filter_row_axis_invalid")
    if not bool(np.isfinite(scores).all()):
        raise AlphaScoreAggregationError("alpha_research.score_filter_input_nonfinite")
    if span_sessions == 1:
        exact = np.ascontiguousarray(scores, dtype=np.float64)
        depth: IntArray = np.ones(count, dtype=np.int64)
        exact.setflags(write=False)
        depth.setflags(write=False)
        return exact, depth

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
    if len(np.unique(rows * len(listings) + columns)) != count:
        raise AlphaScoreAggregationError("alpha_research.score_filter_row_duplicated")
    panel: FloatArray = np.full((len(sessions), len(listings)), np.nan, dtype=np.float64)
    panel[rows, columns] = scores
    alpha = 2.0 / (float(span_sessions) + 1.0)
    decay = 1.0 - alpha
    state: FloatArray = np.full(len(listings), np.nan, dtype=np.float64)
    observed_depth: IntArray = np.zeros(len(listings), dtype=np.int64)
    output: FloatArray = np.full_like(panel, np.nan)
    depth_panel: IntArray = np.zeros(panel.shape, dtype=np.int64)
    for position in range(len(sessions)):
        present = np.isfinite(panel[position])
        initialized = np.isfinite(state)
        state[initialized] *= decay
        continuing = present & initialized
        state[continuing] += alpha * panel[position, continuing]
        starting = present & ~initialized
        state[starting] = panel[position, starting]
        observed_depth[present] += 1
        output[position, present] = state[present]
        depth_panel[position, present] = observed_depth[present]
    filtered = np.ascontiguousarray(output[rows, columns], dtype=np.float64)
    realized = np.ascontiguousarray(depth_panel[rows, columns], dtype=np.int64)
    if not bool(np.isfinite(filtered).all()) or int(realized.min()) < 1:
        raise AlphaScoreAggregationError("alpha_research.score_filter_output_incomplete")
    filtered.setflags(write=False)
    realized.setflags(write=False)
    return filtered, realized


def apply_alpha_score_filter(
    *,
    spec: AlphaScoreFilterSpec,
    row_sessions: tuple[date, ...],
    row_listing_ids: tuple[str, ...],
    scores: FloatArray,
) -> tuple[FloatArray, IntArray]:
    """Apply explicit input preparation, then one installed temporal method."""
    admitted_scores = (
        scores
        if spec.input_mode == RAW_SCORE_INPUT_MODE
        else _cross_section_standardize(
            row_sessions=row_sessions,
            row_listing_ids=row_listing_ids,
            scores=scores,
        )
    )
    if spec.method_id == RAW_SCORE_TEMPORAL_FILTER_METHOD_ID:
        if spec.span_sessions != 1:
            raise AlphaScoreAggregationError("alpha_research.raw_score_filter_span_invalid")
        return _aggregate_ewma(
            row_sessions=row_sessions,
            row_listing_ids=row_listing_ids,
            scores=admitted_scores,
            span_sessions=1,
        )
    if spec.method_id == ALPHA_SCORE_AGGREGATION_METHOD_ID:
        return cast(
            tuple[FloatArray, IntArray],
            aggregate_trailing_mean(
                row_sessions=row_sessions,
                row_listing_ids=row_listing_ids,
                scores=admitted_scores,
                trailing_span_sessions=spec.span_sessions,
            ),
        )
    if spec.method_id == EWMA_FORMATION_SCORE_METHOD_ID:
        return _aggregate_ewma(
            row_sessions=row_sessions,
            row_listing_ids=row_listing_ids,
            scores=admitted_scores,
            span_sessions=spec.span_sessions,
        )
    raise AlphaScoreAggregationError("alpha_research.score_filter_method_not_installed")


__all__ = [
    "CROSS_SECTION_STANDARDIZED_SCORE_INPUT_MODE",
    "EWMA_FORMATION_SCORE_METHOD_ID",
    "RAW_SCORE_INPUT_MODE",
    "RAW_SCORE_TEMPORAL_FILTER_METHOD_ID",
    "SCORE_FILTER_CANDIDATE_SPAN_DOMAIN",
    "SCORE_FILTER_MINIMUM_PERIODS",
    "SCORE_FILTER_ROW_AXIS_POLICY",
    "SCORE_FILTER_SELECTION_POLICY_ID",
    "AlphaScoreFilterCatalog",
    "AlphaScoreFilterDescriptor",
    "AlphaScoreFilterInputMode",
    "AlphaScoreFilterMethodId",
    "AlphaScoreFilterSpec",
    "apply_alpha_score_filter",
    "build_installed_alpha_score_filter_catalog",
]
