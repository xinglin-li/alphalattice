"""Frozen-panel and causal-outcome array construction for Alpha Research."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol

import numpy as np
import numpy.typing as npt

from alphalattice.foundation.causal_outcomes.execution.methods import (
    ExecutionOutcomeMethodSeal,
)
from alphalattice.foundation.causal_outcomes.execution.readers import (
    CausalOutcomeDevelopmentRows,
)
from alphalattice.foundation.feature_engine.panels.reader import FeaturePanelReader
from alphalattice.investment.alpha_research.inputs.development_foundation import (
    AlphaFoundationAuthority,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.validation.contracts import (
    RebalanceEvidencePoint,
    ResearchSplitPlan,
    ResearchSplitSpec,
    SplitWindow,
    ValidationTimeline,
)
from alphalattice.kernel.validation.enums import RebalanceFrequency, SplitMode
from alphalattice.kernel.validation.splitting import build_research_split

from ..experiments.contracts import AlphaFoldCommitment, seal_contract
from ..targets.authority import AlphaTargetMethod, InstalledLaneTargetMethod
from ..targets.execution_outcome import AlphaTargetPolicy
from .training import AlphaTrainingInputBinding

type FloatArray = npt.NDArray[np.float64]
type BoolArray = npt.NDArray[np.bool_]


class AlphaArrayBoundaryError(ValueError):
    """Raised when immutable sources cannot reconstruct the frozen Alpha surface."""


class AlphaSplitGeometry(Protocol):
    """The split shape a plan is built from, without naming which policy carries it.

    The frozen ``AlphaSplitPolicy`` declares ``embargo_sessions`` as ``Literal[0]``
    and its ``policy_hash`` is sealed into Goal and current Programs, so a
    development run that needs a real embargo cannot be expressed by widening it.
    ``AlphaDevelopmentSplitPolicy`` already carries the same shape with free
    integers, and this Protocol is what lets one plan builder accept either --
    the frozen lanes keep their exact geometry and identity, and a development
    run states an embargo derived from the method it actually reads.
    """

    @property
    def train_sessions(self) -> int: ...

    @property
    def validation_sessions(self) -> int: ...

    @property
    def step_sessions(self) -> int: ...

    @property
    def purge_sessions(self) -> int: ...

    @property
    def embargo_sessions(self) -> int: ...

    @property
    def sealed_holdout_sessions(self) -> int: ...

    @property
    def minimum_folds(self) -> int: ...

    @property
    def expected_complete_folds(self) -> int: ...


def _readonly[DType: np.generic](value: npt.NDArray[DType]) -> npt.NDArray[DType]:
    result = np.asarray(value)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class AlphaFoldArrays:
    """Retain one committed fold with read-only features, targets and provenance.

    Training and validation matrices share the declared factor order. Completeness masks separate
    model training from the common comparison surface; optional economic returns preserve a distinct
    evaluation lane.
    """

    commitment: AlphaFoldCommitment
    ordered_factor_ids: tuple[str, ...]
    training_sessions: tuple[date, ...]
    training_listing_ids: tuple[str, ...]
    training_features: FloatArray
    training_targets: FloatArray
    training_feature_complete: BoolArray
    training_outcome_complete: BoolArray
    validation_sessions: tuple[date, ...]
    validation_row_sessions: tuple[date, ...]
    validation_listing_ids: tuple[str, ...]
    validation_features: FloatArray
    validation_targets: FloatArray
    validation_feature_complete: BoolArray
    validation_outcome_complete: BoolArray
    validation_feature_row_hashes: tuple[str | None, ...]
    validation_outcome_row_hashes: tuple[str | None, ...]
    training_economic_returns: FloatArray | None = None
    validation_economic_returns: FloatArray | None = None
    training_row_sessions: tuple[date, ...] = ()
    training_feature_row_hashes: tuple[str | None, ...] = ()
    training_outcome_row_hashes: tuple[str | None, ...] = ()
    training_input_binding: AlphaTrainingInputBinding | None = None

    def __post_init__(self) -> None:
        """Require committed feature/row axes, complete provenance and non-writeable fold arrays.

        Raises:
            AlphaArrayBoundaryError: Feature authority, shapes, session/row ordering, training
                binding, provenance lengths or array writeability violates the fold boundary.
        """
        factor_count = len(self.ordered_factor_ids)
        train_count = len(self.training_listing_ids)
        validation_count = len(self.validation_listing_ids)
        if (
            not self.ordered_factor_ids
            or len(self.ordered_factor_ids) > 128
            or self.ordered_factor_ids != tuple(dict.fromkeys(self.ordered_factor_ids))
        ):
            raise AlphaArrayBoundaryError("ALPHA_FACTOR_AUTHORITY_INVALID")
        if self.training_features.shape != (train_count, factor_count):
            raise AlphaArrayBoundaryError("ALPHA_TRAINING_FEATURE_SHAPE_MISMATCH")
        if self.validation_features.shape != (validation_count, factor_count):
            raise AlphaArrayBoundaryError("ALPHA_VALIDATION_FEATURE_SHAPE_MISMATCH")
        for label, value, count in (
            ("training targets", self.training_targets, train_count),
            ("training feature mask", self.training_feature_complete, train_count),
            ("training outcome mask", self.training_outcome_complete, train_count),
            ("validation targets", self.validation_targets, validation_count),
            ("validation feature mask", self.validation_feature_complete, validation_count),
            ("validation outcome mask", self.validation_outcome_complete, validation_count),
        ):
            if value.shape != (count,):
                raise AlphaArrayBoundaryError(f"ALPHA_{label.upper().replace(' ', '_')}_SHAPE")
        for label, value, count in (
            ("training economic returns", self.training_economic_returns, train_count),
            ("validation economic returns", self.validation_economic_returns, validation_count),
        ):
            if value is not None and value.shape != (count,):
                raise AlphaArrayBoundaryError(f"ALPHA_{label.upper().replace(' ', '_')}_SHAPE")
        if len(self.validation_row_sessions) != validation_count:
            raise AlphaArrayBoundaryError("ALPHA_VALIDATION_SESSION_SHAPE_MISMATCH")
        if self.training_input_binding is not None and (
            len(self.training_row_sessions) != train_count
            or len(self.training_feature_row_hashes) != train_count
            or len(self.training_outcome_row_hashes) != train_count
            or self.training_input_binding.scope != "DEVELOPMENT_FOLD"
            or self.training_input_binding.ordered_feature_ids != self.ordered_factor_ids
            or self.training_input_binding.fold_commitment_hash != self.commitment.commitment_hash
        ):
            raise AlphaArrayBoundaryError("ALPHA_TRAINING_INPUT_BINDING_MISMATCH")
        if (
            len(self.validation_feature_row_hashes) != validation_count
            or len(self.validation_outcome_row_hashes) != validation_count
        ):
            raise AlphaArrayBoundaryError("ALPHA_VALIDATION_PROVENANCE_SHAPE_MISMATCH")
        if tuple(sorted(set(self.training_sessions))) != self.training_sessions:
            raise AlphaArrayBoundaryError("ALPHA_TRAINING_SESSION_ORDER_MISMATCH")
        if tuple(sorted(set(self.validation_sessions))) != self.validation_sessions:
            raise AlphaArrayBoundaryError("ALPHA_VALIDATION_SESSION_ORDER_MISMATCH")
        row_keys = tuple(
            zip(self.validation_row_sessions, self.validation_listing_ids, strict=True)
        )
        if row_keys != tuple(sorted(row_keys)) or len(row_keys) != len(set(row_keys)):
            raise AlphaArrayBoundaryError("ALPHA_VALIDATION_ROW_ORDER_MISMATCH")
        for value in (
            self.training_features,
            self.training_targets,
            self.training_feature_complete,
            self.training_outcome_complete,
            self.validation_features,
            self.validation_targets,
            self.validation_feature_complete,
            self.validation_outcome_complete,
            *(
                value
                for value in (
                    self.training_economic_returns,
                    self.validation_economic_returns,
                )
                if value is not None
            ),
        ):
            if value.flags.writeable:
                raise AlphaArrayBoundaryError("ALPHA_ARRAY_MUST_BE_READ_ONLY")

    @property
    def training_model_mask(self) -> BoolArray:
        """Select training rows with both complete features and complete outcomes.

        Returns:
            Non-writeable conjunction of the two training completeness masks.
        """
        return _readonly(self.training_feature_complete & self.training_outcome_complete)

    @property
    def common_comparison_mask(self) -> BoolArray:
        """Select validation rows complete on both the feature and causal-outcome surface.

        Returns:
            Non-writeable common comparison mask shared across candidate evaluation.
        """
        return _readonly(self.validation_feature_complete & self.validation_outcome_complete)

    @property
    def economic_validation_targets(self) -> FloatArray:
        """Read the explicit economic-return lane or its historical target fallback.

        Returns:
            Validation economic returns when supplied, otherwise validation_targets.
        """
        return (
            self.validation_targets
            if self.validation_economic_returns is None
            else self.validation_economic_returns
        )


@dataclass(frozen=True, slots=True)
class PreparedAlphaArrays:
    """Retain a split plan and its ordered committed development folds."""

    split_plan: ResearchSplitPlan
    folds: tuple[AlphaFoldArrays, ...]

    def __post_init__(self) -> None:
        """Require contiguous zero-based fold indices in the prepared fold sequence.

        Raises:
            AlphaArrayBoundaryError: Fold commitment indices differ from the sequence positions.
        """
        if tuple(value.commitment.fold_index for value in self.folds) != tuple(
            range(len(self.folds))
        ):
            raise AlphaArrayBoundaryError("ALPHA_FOLD_ORDER_MISMATCH")

    @property
    def common_surface_row_count(self) -> int:
        """Count common feature/outcome-complete comparison rows across all retained folds.

        Returns:
            Sum of each fold common comparison mask count.
        """
        return sum(int(value.common_comparison_mask.sum()) for value in self.folds)


@dataclass(frozen=True, slots=True)
class AlphaFoldArrayPlan:
    """Durable split identity plus readers used to load one fold at a time."""

    foundation: AlphaFoundationAuthority
    """The frozen constitution, or a development-only binding of the same shape.

    Structural on purpose. A development run is authorized by a published Panel,
    published causal outcomes and one Factor development checkpoint -- not by the
    four distinct pieces of admission evidence ``ResearchFoundationBinding``
    names. Widening here lets a development binding supply what it genuinely has,
    instead of borrowing four field names for evidence that does not exist and
    producing a foundation that validates while being false.

    ``ResearchFoundationBinding`` satisfies this Protocol unmodified, so the
    current and Goal paths keep their exact types and this plan keeps accepting
    them.
    """

    ordered_listing_ids: tuple[str, ...]
    feature_reader: FeaturePanelReader
    outcome_reader: CausalOutcomeDevelopmentRows
    feature_panel_manifest_ref: str
    causal_outcome_manifest_ref: str
    split_plan: ResearchSplitPlan
    panel_sessions: tuple[date, ...] = ()
    """The Panel's whole verified calendar, as ``prepare_alpha_fold_plan`` read it.

    The sessions the split plan was cut from, and the calendar the array
    workspace positions the program on (its formation session is the last of
    them). Carried so the workspace works from the calendar its split was
    built on rather than reading the Panel a second time for the same value.
    A plan assembled without one cannot enter a workspace (its formation
    session is unavailable), which is the refusal an empty calendar always had.
    """

    target_policy: AlphaTargetPolicy | None = None
    target_method: AlphaTargetMethod | None = None
    """The whole resolved target method, for a development run.

    Carried entire rather than split into ``target_policy`` plus a
    standardization id. Splitting a resolved method into loose fields means the
    surface reassembles something *shaped* like the method, and nothing checks
    that the reassembly is what the Host resolved. Present, the surface asks the
    method to compile itself; absent, the frozen lane route runs exactly as
    before.

    Typed on the Protocol rather than on the development recipe, because a
    method whose composition has no lane -- the canonical target and its
    unbounded control -- has to reach the same executor. What the plan needs is
    something that can compile a surface and name its own identity, not a
    particular class.
    """

    outcome_method: ExecutionOutcomeMethodSeal | None = None
    """The resolved causal-outcome method authority for a development run.

    A development writer may not build a target from a snapshot whose method
    nobody resolved. A legacy snapshot stays readable and is not a development
    input.
    """

    target_standardization_id: str | None = None
    """The standardization a frozen-lane caller named, when one did.

    ``None`` means "use the lane's derived key", which is what every frozen lane
    does and is why current identity is unaffected. Development runs express the
    same thing through ``target_recipe`` instead.
    """

    sector_by_listing_id: Mapping[str, str] | None = None
    additional_factor_ids: tuple[str, ...] = ()
    feature_context_hash: str | None = None
    ordered_base_feature_ids: tuple[str, ...] | None = None
    """The base factor axis this run actually reads, when it is a subset.

    ``None`` means "the whole foundation axis", which is what every current and
    Goal run does, so their arrays are byte-identical to before.

    It exists because a development document can *declare* a subset of the Factor
    evidence axis, and until this field there was nowhere for that declaration to
    reach the numbers. The axis was recorded in the input-binding hash and then
    ignored: matrices, budgets, the training binding and the estimator were all
    built from ``foundation.ordered_factor_ids``. A run could declare eight
    features, seal an identity naming eight, and fit sixty-one.

    Deliberately *not* achieved by narrowing the foundation's own axis. That axis
    is upstream authority -- the parent Factor evidence answered for all of it --
    and overwriting it to fake a selection would destroy the record of what the
    selection was made from.
    """

    @property
    def base_feature_ids(self) -> tuple[str, ...]:
        """The one axis every array in this plan is built over.

        A property rather than a field each caller resolves, because the defect
        being closed was precisely that different consumers disagreed about which
        axis was in force.
        """
        if self.ordered_base_feature_ids is None:
            return tuple(self.foundation.ordered_factor_ids)
        return self.ordered_base_feature_ids

    @property
    def fold_count(self) -> int:
        """Read the number of development windows in this admitted split plan.

        Returns:
            Number of split-plan windows.
        """
        return len(self.split_plan.windows)


@dataclass(frozen=True, slots=True)
class AlphaCurrentRefitArrays:
    """Retain current-refit training and formation matrices with exact optional input authority.

    The record separates training targets/masks from formation feature completeness and retains row
    provenance when an input binding is supplied. Its training cutoff cannot follow formation.
    """

    ordered_factor_ids: tuple[str, ...]
    ordered_listing_ids: tuple[str, ...]
    training_sessions: tuple[date, ...]
    training_cutoff: date
    formation_session: date
    training_features: FloatArray
    training_targets: FloatArray
    training_mask: BoolArray
    current_features: FloatArray
    current_feature_complete: BoolArray
    training_row_sessions: tuple[date, ...] = ()
    training_feature_row_hashes: tuple[str | None, ...] = ()
    training_outcome_row_hashes: tuple[str | None, ...] = ()
    current_feature_row_hashes: tuple[str | None, ...] = ()
    training_economic_returns: FloatArray | None = None
    training_input_binding: AlphaTrainingInputBinding | None = None

    def __post_init__(self) -> None:
        """Require current-refit shapes, ordered training clocks and complete bound provenance.

        Raises:
            AlphaArrayBoundaryError: Training/current axes, cutoff/formation clocks, masks or
                current-refit binding/provenance/economic-return fields disagree.
        """
        if (
            self.training_sessions != tuple(sorted(set(self.training_sessions)))
            or self.training_sessions[-1] != self.training_cutoff
            or self.training_cutoff > self.formation_session
            or self.training_features.shape[1] != len(self.ordered_factor_ids)
            or self.current_features.shape
            != (len(self.ordered_listing_ids), len(self.ordered_factor_ids))
            or self.training_targets.shape != (self.training_features.shape[0],)
            or self.training_mask.shape != self.training_targets.shape
            or self.current_feature_complete.shape != (len(self.ordered_listing_ids),)
        ):
            raise AlphaArrayBoundaryError("ALPHA_CURRENT_REFIT_ARRAY_SHAPE_MISMATCH")
        if self.training_input_binding is not None and (
            len(self.training_row_sessions) != self.training_features.shape[0]
            or len(self.training_feature_row_hashes) != self.training_features.shape[0]
            or len(self.training_outcome_row_hashes) != self.training_features.shape[0]
            or len(self.current_feature_row_hashes) != len(self.ordered_listing_ids)
            or self.training_economic_returns is None
            or self.training_economic_returns.shape != self.training_targets.shape
            or self.training_input_binding.scope != "CURRENT_REFIT"
            or self.training_input_binding.ordered_feature_ids != self.ordered_factor_ids
            or self.training_input_binding.fold_commitment_hash is not None
        ):
            raise AlphaArrayBoundaryError("ALPHA_CURRENT_TRAINING_INPUT_BINDING_MISMATCH")


class AlphaArrayWorkspaceLike(Protocol):
    """Minimal numerical workspace boundary shared by base and decorated arrays."""

    def fold_lease(self, fold_index: int) -> AbstractContextManager[AlphaFoldArrays]: ...

    def load_fold(self, fold_index: int) -> AlphaFoldArrays: ...

    def prepare_current_refit(self) -> AlphaCurrentRefitArrays: ...


def build_alpha_split_plan(
    sessions: tuple[date, ...],
    *,
    frozen_at: datetime,
    policy: AlphaSplitGeometry,
) -> ResearchSplitPlan:
    """Build the admitted rolling split geometry on a canonical calendar and aware freeze clock.

    Args:
        sessions: Sorted unique Panel sessions used to build the validation timeline.
        frozen_at: Timezone-aware evidence/benchmark availability and as-of timestamp.
        policy: Admitted training/validation/step/purge/embargo/holdout geometry and expected
            complete folds.

    Returns:
        Rolling research split with exactly the admitted complete fold count.

    Raises:
        AlphaArrayBoundaryError: The freeze clock is naive, sessions are not canonical or complete
            fold count differs.
    """
    if frozen_at.tzinfo is None or frozen_at.utcoffset() is None:
        raise AlphaArrayBoundaryError("ALPHA_FROZEN_CLOCK_INVALID")
    if sessions != tuple(sorted(set(sessions))):
        raise AlphaArrayBoundaryError("ALPHA_PANEL_CALENDAR_NOT_ORDERED")
    timeline = ValidationTimeline(
        points=tuple(
            RebalanceEvidencePoint(
                session_date=value,
                evidence_available_at=frozen_at,
                benchmark_available_at=frozen_at,
            )
            for value in sessions
        )
    )
    spec = ResearchSplitSpec(
        mode=SplitMode.ROLLING,
        as_of_timestamp=frozen_at,
        train_sessions=policy.train_sessions,
        validation_sessions=policy.validation_sessions,
        step_sessions=policy.step_sessions,
        purge_sessions=policy.purge_sessions,
        embargo_sessions=policy.embargo_sessions,
        holdout_sessions=policy.sealed_holdout_sessions,
        minimum_folds=policy.minimum_folds,
    )
    plan = build_research_split(timeline, spec)
    if len(plan.windows) != policy.expected_complete_folds:
        raise AlphaArrayBoundaryError(f"ALPHA_COMPLETE_FOLD_COUNT_MISMATCH:{len(plan.windows)}")
    return plan


def build_alpha_fold_commitment(window: SplitWindow) -> AlphaFoldCommitment:
    """Seal one split window training/validation bounds, counts and exact session identities.

    Args:
        window: Split window supplying ordered training and validation sessions.

    Returns:
        Fold commitment binding fold index, endpoint/count summaries and both canonical session
        hashes.
    """
    train_sessions = tuple(window.train_sessions)
    validation_sessions = tuple(window.validation_sessions)
    values = {
        "fold_index": int(window.fold_index),
        "train_first": train_sessions[0],
        "train_last": train_sessions[-1],
        "validation_first": validation_sessions[0],
        "validation_last": validation_sessions[-1],
        "train_session_count": len(train_sessions),
        "validation_session_count": len(validation_sessions),
        "train_sessions_hash": canonical_hash(train_sessions),
        "validation_sessions_hash": canonical_hash(validation_sessions),
    }
    return seal_contract(AlphaFoldCommitment, values, "commitment_hash")


def prepare_alpha_fold_plan(
    *,
    foundation: AlphaFoundationAuthority,
    ordered_listing_ids: tuple[str, ...],
    feature_reader: FeaturePanelReader,
    outcome_reader: CausalOutcomeDevelopmentRows,
    feature_panel_manifest_ref: str,
    causal_outcome_manifest_ref: str,
    frozen_at: datetime,
    split_policy: AlphaSplitGeometry,
    target_policy: AlphaTargetPolicy | None = None,
    target_method: AlphaTargetMethod | None = None,
    outcome_method: ExecutionOutcomeMethodSeal | None = None,
    target_standardization_id: str | None = None,
    sector_by_listing_id: Mapping[str, str] | None = None,
    ordered_base_feature_ids: tuple[str, ...] | None = None,
    panel_sessions: tuple[date, ...] | None = None,
) -> AlphaFoldArrayPlan:
    """Cut the split from the Panel's calendar and bind the readers that load it.

    ``panel_sessions`` is the calendar ``feature_reader.available_sessions``
    answered for ``feature_panel_manifest_ref`` when the caller has already
    read it in this same preparation (the executor sizes the split policy from
    it first); left unset, the plan reads it here. Either way it is the one
    value the Panel's immutable chunks verify to, and the split plan binds it
    through its timeline hash.
    """
    if foundation.research_cadence is not RebalanceFrequency.DAILY:
        raise AlphaArrayBoundaryError("ALPHA_RESEARCH_CADENCE_NOT_DAILY")
    if ordered_listing_ids != tuple(sorted(set(ordered_listing_ids))):
        raise AlphaArrayBoundaryError("ALPHA_LISTING_AUTHORITY_INVALID")
    if target_method is not None:
        # The two target routes are mutually exclusive by construction. A plan
        # carrying both a resolved method and a loose lane policy has two
        # answers to "which method compiled this", and whichever the compiler
        # happened to read would silently become the truth.
        if target_policy is not None or target_standardization_id is not None:
            raise AlphaArrayBoundaryError("ALPHA_TARGET_ROUTE_AMBIGUOUS")
        if outcome_method is None or outcome_method.disposition != "METHOD_BOUND":
            # A development target is built from method-bound evidence or not at
            # all; a legacy snapshot stays readable and stays out of here.
            raise AlphaArrayBoundaryError("ALPHA_DEVELOPMENT_TARGET_METHOD_UNBOUND")
        # The lane a frozen-lane method admits is *derived* from the resolved
        # method, never accepted alongside it, and a method with no lane derives
        # none. Exclusivity is a property of what a caller may supply.
        if isinstance(target_method, InstalledLaneTargetMethod):
            target_policy = target_method.recipe.policy
            target_standardization_id = target_method.standardization_id
        # A lane-free method leaves both unset. Its standardization is not a
        # loose routing key the surface has to apply; the method resolves and
        # applies its own, which is the whole reason it needed a boundary of its
        # own rather than a wider lane enum.
    if target_standardization_id is not None and target_policy is None:
        # A named standardization with no lane to apply it to is a request the
        # surface cannot honour, and the failure would otherwise be silent: the
        # plan would carry a key nothing reads.
        raise AlphaArrayBoundaryError("ALPHA_TARGET_STANDARDIZATION_WITHOUT_POLICY")
    if ordered_base_feature_ids is not None:
        _assert_admissible_base_axis(
            ordered_base_feature_ids,
            (
                *foundation.ordered_factor_ids,
                *(getattr(foundation, "ordered_context_ids", None) or ()),
            ),
        )
    sessions = (
        feature_reader.available_sessions(feature_panel_manifest_ref)
        if panel_sessions is None
        else panel_sessions
    )
    return AlphaFoldArrayPlan(
        foundation=foundation,
        ordered_listing_ids=ordered_listing_ids,
        feature_reader=feature_reader,
        outcome_reader=outcome_reader,
        feature_panel_manifest_ref=feature_panel_manifest_ref,
        causal_outcome_manifest_ref=causal_outcome_manifest_ref,
        split_plan=build_alpha_split_plan(sessions, frozen_at=frozen_at, policy=split_policy),
        panel_sessions=sessions,
        target_policy=target_policy,
        target_method=target_method,
        outcome_method=outcome_method,
        target_standardization_id=target_standardization_id,
        sector_by_listing_id=sector_by_listing_id,
        ordered_base_feature_ids=ordered_base_feature_ids,
    )


def _assert_admissible_base_axis(requested: tuple[str, ...], authorized: tuple[str, ...]) -> None:
    """Admit a declared base axis, exactly as declared or not at all.

    Order is admitted rather than repaired because every array here is
    positional: the requested order decides which column each factor occupies, so
    silently sorting a declared axis would train a model against columns it was
    not given. Membership is checked against the foundation's own axis, so a
    document cannot widen its authority by naming a factor the parent evidence
    never answered for.
    """

    if not requested:
        raise AlphaArrayBoundaryError("ALPHA_BASE_FEATURE_AXIS_EMPTY")
    if requested != tuple(dict.fromkeys(requested)):
        raise AlphaArrayBoundaryError("ALPHA_BASE_FEATURE_AXIS_DUPLICATED")
    if not set(requested).issubset(set(authorized)):
        raise AlphaArrayBoundaryError("ALPHA_BASE_FEATURE_AXIS_NOT_AUTHORIZED")


__all__ = [
    "AlphaArrayBoundaryError",
    "AlphaCurrentRefitArrays",
    "AlphaFoldArrayPlan",
    "AlphaFoldArrays",
    "PreparedAlphaArrays",
    "build_alpha_fold_commitment",
    "build_alpha_split_plan",
    "prepare_alpha_fold_plan",
]
