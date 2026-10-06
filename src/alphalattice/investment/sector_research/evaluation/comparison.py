"""Compare installed Sector methods on one shared target, under frozen rules.

Two rules live here, both pre-registered before any number exists, both
re-derivable by the verifier from the published children.

The **selection rule** is deliberately biased toward closing the branch. A
candidate replaces ``ZERO`` only when it both earns a positive cross-fitted
scale and strictly lowers the held-out squared error against the raw Sector
economic-return lane. Anything else -- a slope shrunk to zero, a tie, a method
that only beats another candidate -- selects ``ZERO``. A Sector expected-return
contribution has to be paid for out of the money lane, and "better than EWMA"
is not that payment.

The **joint-dynamics trigger** decides whether the conditional VAR family is
allowed to run at all. It fires only when the distributed-lag challenger is the
strictly best evaluated method *and* carries a positive scale: that is the one
evidence pattern where regularized lags have said something the persistence and
strength controls did not, which is what would make a genuinely joint model the
next question rather than the next thing to try. Any other outcome publishes
``NOT_TRIGGERED`` and no VAR numerical call is made.

Both rules read only published evidence, so a reader can check the disposition
without rerunning a method.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash

from ..contracts import SectorResearchError
from ..models.zero import ZERO_SECTOR_FORECAST_METHOD_ID

SECTOR_SELECTION_RULE_ID = "POSITIVE_SCALE_AND_STRICTLY_LOWER_ECONOMIC_MSE_ELSE_ZERO"
SECTOR_JOINT_DYNAMICS_TRIGGER_ID = "DISTRIBUTED_LAG_STRICTLY_BEST_WITH_POSITIVE_SCALE"


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


class SectorMethodComparisonRow(_Contract):
    """One method's comparable line, every field derived from its own children."""

    method_id: str = Field(min_length=1, max_length=96)
    disposition: Literal["EVALUATED", "NOT_TRIGGERED", "PENDING_HUMAN_CONFIRMATION", "REFUSED"]
    refusal_reason: str | None = None
    experiment_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    surface_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    evaluation_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    calibration_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    forecast_cell_count: int = Field(default=0, ge=0)
    available_cell_count: int = Field(default=0, ge=0)
    evaluated_pair_count: int = Field(default=0, ge=0)
    mean_cross_fitted_slope: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    mean_calibrated_economic_squared_error: float | None = Field(
        default=None, ge=0.0, allow_inf_nan=False
    )
    mean_identity_economic_squared_error: float | None = Field(
        default=None, ge=0.0, allow_inf_nan=False
    )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_row(self) -> Self:
        """Require evidence for evaluated methods and reasons for held/refused methods.

        Returns:
            This contract after checking its declared consistency rules.

        Raises:
            SectorResearchError: Disposition disagrees with experiment/child evidence or
                refusal-reason presence.
        """
        evaluated = self.disposition == "EVALUATED"
        has_evidence = self.experiment_hash is not None
        if evaluated != has_evidence:
            # A method that ran has a graph; one that did not must not name one.
            raise SectorResearchError("sector_research.comparison_row_evidence_invalid")
        if evaluated and (self.surface_hash is None or self.calibration_hash is None):
            raise SectorResearchError("sector_research.comparison_row_children_missing")
        if (self.disposition in {"PENDING_HUMAN_CONFIRMATION", "REFUSED"}) != (
            self.refusal_reason is not None
        ):
            raise SectorResearchError("sector_research.comparison_row_reason_invalid")
        return self


class SectorMethodComparisonEvidence(_Contract):
    """The Campaign's comparable table, its frozen rules, and what they decided."""

    kind: Literal["SectorMethodComparisonEvidence"] = "SectorMethodComparisonEvidence"
    target_evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_rule_id: Literal["POSITIVE_SCALE_AND_STRICTLY_LOWER_ECONOMIC_MSE_ELSE_ZERO"] = (
        "POSITIVE_SCALE_AND_STRICTLY_LOWER_ECONOMIC_MSE_ELSE_ZERO"
    )
    joint_dynamics_trigger_id: Literal["DISTRIBUTED_LAG_STRICTLY_BEST_WITH_POSITIVE_SCALE"] = (
        "DISTRIBUTED_LAG_STRICTLY_BEST_WITH_POSITIVE_SCALE"
    )
    rows: tuple[SectorMethodComparisonRow, ...] = Field(min_length=2)
    recommended_method_id: str = Field(min_length=1, max_length=96)
    recommended_experiment_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    joint_dynamics_triggered: bool
    comparison_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_identity(self) -> Self:
        """Require unique methods, the zero control and an evaluated recommendation route.

        Returns:
            This contract after checking its declared consistency rules.

        Raises:
            SectorResearchError: A method repeats, the zero control is absent, the recommendation
                route is invalid or comparison_hash differs.
        """
        ids = tuple(value.method_id for value in self.rows)
        if len(set(ids)) != len(ids):
            raise SectorResearchError("sector_research.comparison_method_duplicated")
        if ZERO_SECTOR_FORECAST_METHOD_ID not in set(ids):
            # The null control must be present or the table cannot answer the
            # only question that matters.
            raise SectorResearchError("sector_research.comparison_zero_control_absent")
        recommended = {value.method_id: value for value in self.rows}.get(
            self.recommended_method_id
        )
        if recommended is None or recommended.disposition != "EVALUATED":
            raise SectorResearchError("sector_research.comparison_recommendation_invalid")
        if recommended.experiment_hash != self.recommended_experiment_hash:
            raise SectorResearchError("sector_research.comparison_recommendation_route_invalid")
        if self.comparison_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"comparison_hash"})
        ):
            raise SectorResearchError("sector_research.comparison_identity_invalid")
        return self


def _score(row: SectorMethodComparisonRow) -> float | None:
    if row.disposition != "EVALUATED" or row.evaluated_pair_count == 0:
        return None
    return row.mean_calibrated_economic_squared_error


def derive_sector_recommendation(rows: Sequence[SectorMethodComparisonRow]) -> tuple[str, bool]:
    """Apply both frozen rules to a comparable table; return the choice and trigger.

    Ordering is by method id inside a tie so the result does not depend on the
    order the Campaign happened to run methods in.
    """
    indexed = {value.method_id: value for value in rows}
    control = indexed.get(ZERO_SECTOR_FORECAST_METHOD_ID)
    control_score = _score(control) if control is not None else None
    if control is None or control_score is None:
        raise SectorResearchError("sector_research.comparison_zero_control_unscored")

    eligible = [
        value
        for value in rows
        if value.method_id != ZERO_SECTOR_FORECAST_METHOD_ID
        and _score(value) is not None
        and (value.mean_cross_fitted_slope or 0.0) > 0.0
        and cast(float, _score(value)) < control_score
    ]
    selected = (
        min(eligible, key=lambda value: (cast(float, _score(value)), value.method_id)).method_id
        if eligible
        else ZERO_SECTOR_FORECAST_METHOD_ID
    )

    scored = [value for value in rows if _score(value) is not None]
    best = min(scored, key=lambda value: (cast(float, _score(value)), value.method_id))
    challenger = indexed.get("DISTRIBUTED_LAG_ELASTIC_NET")
    triggered = bool(
        challenger is not None
        and best.method_id == "DISTRIBUTED_LAG_ELASTIC_NET"
        and (challenger.mean_cross_fitted_slope or 0.0) > 0.0
        and cast(float, _score(challenger)) < control_score
        # Strictly best: no other scored method may match the challenger.
        and all(
            cast(float, _score(value)) > cast(float, _score(challenger))
            for value in scored
            if value.method_id != "DISTRIBUTED_LAG_ELASTIC_NET"
        )
    )
    return selected, triggered


def build_sector_method_comparison(
    *,
    target_evidence_hash: str,
    catalog_hash: str,
    rows: Sequence[SectorMethodComparisonRow],
) -> SectorMethodComparisonEvidence:
    """Seal the comparable table together with what its own rules decided."""
    ordered = tuple(sorted(rows, key=lambda value: value.method_id))
    recommended, triggered = derive_sector_recommendation(ordered)
    indexed: Mapping[str, SectorMethodComparisonRow] = {value.method_id: value for value in ordered}
    values: dict[str, object] = {
        "kind": "SectorMethodComparisonEvidence",
        "target_evidence_hash": target_evidence_hash,
        "catalog_hash": catalog_hash,
        "selection_rule_id": SECTOR_SELECTION_RULE_ID,
        "joint_dynamics_trigger_id": SECTOR_JOINT_DYNAMICS_TRIGGER_ID,
        "rows": [value.model_dump(mode="json") for value in ordered],
        "recommended_method_id": recommended,
        "recommended_experiment_hash": indexed[recommended].experiment_hash,
        "joint_dynamics_triggered": triggered,
    }
    return cast(
        SectorMethodComparisonEvidence,
        SectorMethodComparisonEvidence.model_validate(
            {**values, "comparison_hash": canonical_hash(values)}
        ),
    )


__all__ = [
    "SECTOR_JOINT_DYNAMICS_TRIGGER_ID",
    "SECTOR_SELECTION_RULE_ID",
    "SectorMethodComparisonEvidence",
    "SectorMethodComparisonRow",
    "build_sector_method_comparison",
    "derive_sector_recommendation",
]
