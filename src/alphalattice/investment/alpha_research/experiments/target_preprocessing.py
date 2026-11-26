"""The paired target-preprocessing study: its Program, its clipping evidence, its pairing.

Three arms run through the ordinary Alpha development path, and three separate
runs are not a study. Nothing in three independent receipts records that they
were meant to be compared, that they shared a Feature axis and an evaluation
surface, that the model was held fixed, or which two of them form the
bounded/unbounded pair. Without that, "paired" is a property of how somebody ran
the commands rather than of the evidence.

``AlphaTargetPreprocessingComparisonProgram`` is that record, sealed *before*
execution so the arms cannot be chosen after their numbers are known.
``AlphaTargetPreprocessingComparisonEvidence`` is sealed after, and binds the
three receipts to the Program that authorized them.

This is deliberately not a generic experiment-matrix framework. It knows about
exactly one study shape -- a canonical reference, an unbounded sensitivity
control over the same clock, and a longer-horizon candidate on its own clock --
because that is the study this milestone runs. A framework able to express any
comparison would have to be configured to express this one, and the
configuration would be the thing nobody sealed.

The clipping evidence lives here rather than beside the target compiler for the
same reason the pairing does: what a bound did is a fact about a *run*, and the
compiler is a pure function that has no run to describe.
"""

from __future__ import annotations

from datetime import date
from typing import Literal, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.capabilities.alpha_modeling.contracts import AlphaModelRecipeEnvelope
from alphalattice.kernel.shared_kernel.identity import canonical_hash

type FloatArray = npt.NDArray[np.float64]

SCIENTIFIC_TARGET_BOUND_DISPOSITION = "SCIENTIFIC_TARGET_BOUND_RAW_ECONOMIC_LANE_RETAINED"
"""The one disposition a bounded observation may carry.

A scientifically bounded value is not a Data defect and has not been verified by
anybody as a genuine tail. Data quarantine and scientific target bounding are
separate authorities with separate evidence, and a single string that could mean
either is how the two get confused.
"""

INFLUENCE_SELECTION_ID = "MAXIMUM_TRAINING_SESSION_LOSS_CONTRIBUTION_LEAVE_ONE_SESSION_OUT"
"""The frozen definition of "high influence", so the implementer does not pick one.

Fit the base model; take the mean squared residual of each training session;
select the largest, breaking ties toward the earliest session; drop that session
and refit exactly once. One refit per fold, never an exhaustive jackknife.
"""


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class AlphaTargetClippingRow(_Contract):
    """One bounded observation, with both sides of the bound it crossed."""

    formation_session: date
    listing_id: str = Field(min_length=1, max_length=64)
    raw_residual: float
    lower_bound: float
    upper_bound: float
    bounded_residual: float
    disposition: Literal["SCIENTIFIC_TARGET_BOUND_RAW_ECONOMIC_LANE_RETAINED"] = (
        SCIENTIFIC_TARGET_BOUND_DISPOSITION  # type: ignore[assignment]
    )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_row(self) -> Self:
        if not (self.lower_bound <= self.upper_bound):
            raise ValueError("alpha_research.target_clipping_row_bounds_invalid")
        expected = min(max(self.raw_residual, self.lower_bound), self.upper_bound)
        if self.bounded_residual != expected:
            # The row states what the bound did, so it must state what the bound
            # would do. A row recording a different value describes some other
            # operation under this method's name.
            raise ValueError("alpha_research.target_clipping_row_value_invalid")
        if self.raw_residual == self.bounded_residual:
            # An unbounded observation in the bounded-detail child would inflate
            # every count reconciled against it.
            raise ValueError("alpha_research.target_clipping_row_not_bounded")
        return self


class AlphaTargetClippingDetail(_Contract):
    """The bounded rows of one arm, content-addressed and ordered.

    Bounded in size by construction: it holds only rows the bound actually moved,
    which is a small fraction of a surface by design. If a method ever bounded
    most of its rows, the right response is to refuse the method rather than to
    publish a detail child the size of the surface.
    """

    kind: Literal["AlphaTargetClippingDetail"] = "AlphaTargetClippingDetail"
    target_recipe_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    rows: tuple[AlphaTargetClippingRow, ...]
    detail_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls, *, target_recipe_binding_hash: str, rows: tuple[AlphaTargetClippingRow, ...]
    ) -> Self:
        ordered = tuple(sorted(rows, key=lambda row: (row.formation_session, row.listing_id)))
        values: dict[str, object] = {
            "kind": "AlphaTargetClippingDetail",
            "target_recipe_binding_hash": target_recipe_binding_hash,
            "rows": [row.model_dump(mode="json") for row in ordered],
        }
        return cls(
            target_recipe_binding_hash=target_recipe_binding_hash,
            rows=ordered,
            detail_hash=str(canonical_hash(values)),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        keys = tuple((row.formation_session, row.listing_id) for row in self.rows)
        if keys != tuple(sorted(keys)):
            # Order is part of the identity, so a reordered child would be a
            # different document claiming the same content.
            raise ValueError("alpha_research.target_clipping_detail_not_ordered")
        if len(set(keys)) != len(keys):
            raise ValueError("alpha_research.target_clipping_detail_duplicated")
        if self.detail_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"detail_hash"})
        ):
            raise ValueError("alpha_research.target_clipping_detail_identity_invalid")
        return self


class AlphaTargetPreprocessingReceipt(_Contract):
    """What one arm's target preprocessing did, reconciled against its own child.

    Every count here is checkable against something else: the detail child's row
    count, the target evidence's lane identities, or the surface the arrays were
    built from. A receipt whose numbers only agree with themselves would be
    exactly the self-consistency this program keeps closing.
    """

    kind: Literal["AlphaTargetPreprocessingReceipt"] = "AlphaTargetPreprocessingReceipt"
    target_recipe_id: str = Field(min_length=1, max_length=96)
    target_recipe_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_row_count: int = Field(ge=0)
    eligible_row_count: int = Field(ge=0)
    missing_row_count: int = Field(ge=0)
    unavailable_rows_by_reason: dict[str, int]
    duplicate_row_count: Literal[0] = 0
    nonfinite_row_count: int = Field(ge=0)
    bounded_row_count: int = Field(ge=0)
    bounded_row_fraction: float = Field(ge=0.0, le=1.0)
    bounded_sessions: int = Field(ge=0)
    zero_dispersion_sessions: int = Field(ge=0)
    nonfinite_dispersion_sessions: int = Field(ge=0)
    future_fit_violation_count: Literal[0] = 0
    """Fixed at zero by type. A causal boundary that could report a nonzero count
    would be a boundary that admitted the violation and then described it."""

    pre_bound_finite_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    post_bound_finite_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_residual_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    bounded_residual_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    redemeaned_residual_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    dispersion_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    fit_target_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_log_return_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    simple_economic_return_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    clipping_detail_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        if self.eligible_row_count > self.input_row_count:
            raise ValueError("alpha_research.target_preprocessing_receipt_counts_invalid")
        if self.bounded_row_count > self.eligible_row_count:
            raise ValueError("alpha_research.target_preprocessing_receipt_counts_invalid")
        if any(value < 0 for value in self.unavailable_rows_by_reason.values()):
            raise ValueError("alpha_research.target_preprocessing_receipt_counts_invalid")
        expected_fraction = (
            0.0
            if self.eligible_row_count == 0
            else self.bounded_row_count / self.eligible_row_count
        )
        if abs(self.bounded_row_fraction - expected_fraction) > 1e-12:
            raise ValueError("alpha_research.target_preprocessing_receipt_fraction_invalid")
        if self.receipt_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"receipt_hash"})
        ):
            raise ValueError("alpha_research.target_preprocessing_receipt_identity_invalid")
        return self


def reconcile_target_preprocessing_receipt(
    *, receipt: AlphaTargetPreprocessingReceipt, detail: AlphaTargetClippingDetail
) -> None:
    """Require the receipt's claims to agree with the child that substantiates them.

    Separate from the receipt's own validator on purpose: the validator proves
    the receipt is internally coherent, which a forged receipt also is. This
    compares it with a document sealed at a different moment, which is the only
    place the two can disagree.
    """

    if detail.target_recipe_binding_hash != receipt.target_recipe_binding_hash:
        raise ValueError("alpha_research.target_preprocessing_detail_not_this_arm")
    if detail.detail_hash != receipt.clipping_detail_hash:
        raise ValueError("alpha_research.target_preprocessing_detail_not_named")
    if len(detail.rows) != receipt.bounded_row_count:
        # The count and the rows are two statements about the same fact, written
        # at different times. A dropped row is only visible here.
        raise ValueError("alpha_research.target_preprocessing_bounded_count_mismatch")
    if len({row.formation_session for row in detail.rows}) != receipt.bounded_sessions:
        raise ValueError("alpha_research.target_preprocessing_bounded_sessions_mismatch")


class AlphaComparisonArm(_Contract):
    """One arm of the study, fixed by Host resolution before anything runs.

    Every hash here is a resolved fact, never a caller assertion: the target
    method from the installed catalog, the outcome snapshot and its method seal
    from the workspace's own readers, the split policy from the maturity clock.
    A caller able to state any of them would be asserting what the study exists
    to establish.
    """

    kind: Literal["AlphaComparisonArm"] = "AlphaComparisonArm"
    arm_id: str = Field(min_length=1, max_length=96)
    target_recipe_id: str = Field(min_length=1, max_length=96)
    target_method_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_outcome_recipe_id: str = Field(min_length=1, max_length=96)
    causal_outcome_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    outcome_method_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    maturity_lag_sessions: int = Field(ge=2)
    split_policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_policy_id: str = Field(min_length=1, max_length=96)
    role: Literal[
        "CANONICAL_BOUNDED_REFERENCE",
        "UNBOUNDED_SENSITIVITY_CONTROL",
        "LONGER_HORIZON_CANDIDATE",
    ]


class AlphaTargetPreprocessingComparisonProgram(_Contract):
    """The study, sealed before execution: which arms, which common surface, which model.

    The common surface is a single set of identities rather than one per arm. Two
    arms that read different Feature axes or different listings are not a paired
    comparison of target preprocessing; they are a comparison of everything at
    once, and the difference would be attributed to the bound.
    """

    kind: Literal["AlphaTargetPreprocessingComparisonProgram"] = (
        "AlphaTargetPreprocessingComparisonProgram"
    )
    ordered_arms: tuple[AlphaComparisonArm, ...] = Field(min_length=3, max_length=3)
    common_authority_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_panel_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    factor_development_receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_feature_ids: tuple[str, ...] = Field(min_length=1)
    ordered_listing_ids_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_capability_handle: str = Field(min_length=1, max_length=64)
    model_recipe: AlphaModelRecipeEnvelope
    """The complete Host-admitted adapter recipe for the one fixed model --
    route, parameters and their one digest -- resolved by admitting the
    documents' proposal through the installed mandate before any arm runs.
    Every arm receipt must carry exactly this envelope's ``recipe_hash``, which
    is what makes "the same model" a checked fact rather than a parameter echo.

    The whole envelope rather than its bare hash, because a verifier must be
    able to re-admit the recipe through the *installed* catalog and compare the
    result -- a free-standing digest is an authority claim nothing can
    re-derive, which is exactly the kind of field a forged Program would keep.
    There is still deliberately no separate parameters hash beside it: the
    envelope hashes its route *and* its parameters as one identity."""

    model_search_domain_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    """The search domain the capability handle resolved to at admission. The
    admitted decision is "(domain, recipe)", not "recipe": re-verification
    compares this against the domain the installed mandate resolves for the
    same handle, so a rotated or substituted domain is named precisely instead
    of surfacing as an opaque binding-hash mismatch."""

    influence_selection_id: Literal[
        "MAXIMUM_TRAINING_SESSION_LOSS_CONTRIBUTION_LEAVE_ONE_SESSION_OUT"
    ] = INFLUENCE_SELECTION_ID  # type: ignore[assignment]
    paired_arm_ids: tuple[str, str]
    """The two arms whose difference is attributable to bounding.

    Named rather than inferred from roles, because "the pair" is a scientific
    claim about which comparison is meaningful. The longer-horizon arm is
    deliberately not in it: comparing raw metric magnitudes across horizons
    measures the horizon.
    """

    program_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        ordered_arms: tuple[AlphaComparisonArm, ...],
        common_authority_hash: str,
        feature_panel_snapshot_hash: str,
        factor_development_receipt_hash: str,
        ordered_feature_ids: tuple[str, ...],
        ordered_listing_ids_hash: str,
        model_capability_handle: str,
        model_recipe: AlphaModelRecipeEnvelope,
        model_search_domain_hash: str,
        paired_arm_ids: tuple[str, str],
    ) -> Self:
        values: dict[str, object] = {
            "kind": "AlphaTargetPreprocessingComparisonProgram",
            "ordered_arms": [arm.model_dump(mode="json") for arm in ordered_arms],
            "common_authority_hash": common_authority_hash,
            "feature_panel_snapshot_hash": feature_panel_snapshot_hash,
            "factor_development_receipt_hash": factor_development_receipt_hash,
            "ordered_feature_ids": list(ordered_feature_ids),
            "ordered_listing_ids_hash": ordered_listing_ids_hash,
            "model_capability_handle": model_capability_handle,
            "model_recipe": model_recipe.model_dump(mode="json"),
            "model_search_domain_hash": model_search_domain_hash,
            "influence_selection_id": INFLUENCE_SELECTION_ID,
            "paired_arm_ids": list(paired_arm_ids),
        }
        return cls(
            ordered_arms=ordered_arms,
            common_authority_hash=common_authority_hash,
            feature_panel_snapshot_hash=feature_panel_snapshot_hash,
            factor_development_receipt_hash=factor_development_receipt_hash,
            ordered_feature_ids=ordered_feature_ids,
            ordered_listing_ids_hash=ordered_listing_ids_hash,
            model_capability_handle=model_capability_handle,
            model_recipe=model_recipe,
            model_search_domain_hash=model_search_domain_hash,
            paired_arm_ids=paired_arm_ids,
            program_hash=str(canonical_hash(values)),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        arm_ids = tuple(arm.arm_id for arm in self.ordered_arms)
        if len(set(arm_ids)) != len(arm_ids):
            raise ValueError("alpha_research.comparison_program_arm_duplicated")
        roles = tuple(arm.role for arm in self.ordered_arms)
        if roles != (
            "CANONICAL_BOUNDED_REFERENCE",
            "UNBOUNDED_SENSITIVITY_CONTROL",
            "LONGER_HORIZON_CANDIDATE",
        ):
            # Order is the study design, not a presentation choice: the reference
            # is what the control is a control *for*, and it must exist first.
            raise ValueError("alpha_research.comparison_program_arm_order_invalid")
        by_id = {arm.arm_id: arm for arm in self.ordered_arms}
        if set(self.paired_arm_ids) - set(by_id):
            raise ValueError("alpha_research.comparison_program_pair_unknown")
        first, second = self.paired_arm_ids
        if first == second:
            raise ValueError("alpha_research.comparison_program_pair_degenerate")
        paired = (by_id[first], by_id[second])
        if len({arm.execution_outcome_recipe_id for arm in paired}) != 1:
            # A pair on two different clocks cannot isolate the bound.
            raise ValueError("alpha_research.comparison_program_pair_clock_mismatch")
        if len({arm.split_policy_hash for arm in paired}) != 1:
            raise ValueError("alpha_research.comparison_program_pair_split_mismatch")
        if self.ordered_feature_ids != tuple(dict.fromkeys(self.ordered_feature_ids)):
            raise ValueError("alpha_research.comparison_program_feature_axis_duplicated")
        if self.program_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"program_hash"})
        ):
            raise ValueError("alpha_research.comparison_program_identity_invalid")
        return self


class AlphaComparisonArmResult(_Contract):
    """One executed arm, joined to the Program entry that authorized it."""

    kind: Literal["AlphaComparisonArmResult"] = "AlphaComparisonArmResult"
    arm_id: str = Field(min_length=1, max_length=96)
    development_receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    preprocessing_receipt_hash: str | None = None
    """Absent for an arm with no bounding step. ``None`` is the honest value: a
    zero-row clipping receipt would assert that a bound ran and moved nothing."""

    numerical_call_count: int = Field(ge=0)


class AlphaTargetPreprocessingComparisonEvidence(_Contract):
    """The executed study, bound to the Program that fixed it beforehand."""

    kind: Literal["AlphaTargetPreprocessingComparisonEvidence"] = (
        "AlphaTargetPreprocessingComparisonEvidence"
    )
    program_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_results: tuple[AlphaComparisonArmResult, ...] = Field(min_length=3, max_length=3)
    common_evaluation_rows: int = Field(ge=0)
    """Rows both paired arms scored. Reported rather than assumed equal to either
    arm's own count, because unavailable formations need not coincide."""

    evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        program_hash: str,
        ordered_results: tuple[AlphaComparisonArmResult, ...],
        common_evaluation_rows: int,
    ) -> Self:
        values: dict[str, object] = {
            "kind": "AlphaTargetPreprocessingComparisonEvidence",
            "program_hash": program_hash,
            "ordered_results": [result.model_dump(mode="json") for result in ordered_results],
            "common_evaluation_rows": common_evaluation_rows,
        }
        return cls(
            program_hash=program_hash,
            ordered_results=ordered_results,
            common_evaluation_rows=common_evaluation_rows,
            evidence_hash=str(canonical_hash(values)),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        arm_ids = tuple(result.arm_id for result in self.ordered_results)
        if len(set(arm_ids)) != len(arm_ids):
            raise ValueError("alpha_research.comparison_evidence_arm_duplicated")
        if self.evidence_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"evidence_hash"})
        ):
            raise ValueError("alpha_research.comparison_evidence_identity_invalid")
        return self


def verify_comparison_evidence(
    *,
    program: AlphaTargetPreprocessingComparisonProgram,
    evidence: AlphaTargetPreprocessingComparisonEvidence,
) -> None:
    """Require the executed arms to be exactly the arms the Program authorized.

    Order included. A study that ran the same three arms in a different sequence
    is a different study only in bookkeeping, but a study whose evidence lists
    arms the Program never named is not the study anybody sealed.
    """

    if evidence.program_hash != program.program_hash:
        raise ValueError("alpha_research.comparison_evidence_program_mismatch")
    declared = tuple(arm.arm_id for arm in program.ordered_arms)
    executed = tuple(result.arm_id for result in evidence.ordered_results)
    if declared != executed:
        raise ValueError("alpha_research.comparison_evidence_arms_not_as_programmed")


__all__ = [
    "INFLUENCE_SELECTION_ID",
    "SCIENTIFIC_TARGET_BOUND_DISPOSITION",
    "AlphaComparisonArm",
    "AlphaComparisonArmResult",
    "AlphaTargetClippingDetail",
    "AlphaTargetClippingRow",
    "AlphaTargetPreprocessingComparisonEvidence",
    "AlphaTargetPreprocessingComparisonProgram",
    "AlphaTargetPreprocessingReceipt",
    "reconcile_target_preprocessing_receipt",
    "verify_comparison_evidence",
]
