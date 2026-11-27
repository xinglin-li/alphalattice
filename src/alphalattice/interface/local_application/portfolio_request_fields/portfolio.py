"""Portfolio request fields and their public descriptions."""

from typing import Literal
from uuid import UUID

from pydantic import ConfigDict, Field

from alphalattice.kernel.identity import BLAKE3_PATTERN

from .contracts import RequestFieldModel


class RequestFields(RequestFieldModel):
    """Portfolio study and update fields for their operations."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    performance: Literal["latest"] | None = None
    """`latest` reads this exact book's latest retained Forward composite;
    omitted reads its saved report."""
    portfolio_scope: Literal["holdings"] | None = None
    """`holdings` reads only this saved date's positions, without its Forward report."""
    spec: dict[str, object] | None = None
    """The Portfolio book's controls (`strategy-book controls` lists them); those left out keep
    their defaults."""
    strategy_package_id: str | None = Field(default=None, min_length=1)
    """The installed strategy package, by its id."""
    recovery_task_id: UUID | None = None
    """The stopped Task whose owner replan this existing planner/admission continues."""
    recovery_task_hash: str | None = Field(default=None, pattern=BLAKE3_PATTERN)
    """The exact stopped Task version named by `recovery_task_id`."""
    left_result_hash: str | None = Field(default=None, pattern=BLAKE3_PATTERN)
    """The first of two Portfolio results to compare."""
    right_result_hash: str | None = Field(default=None, pattern=BLAKE3_PATTERN)
    """The second of two Portfolio results to compare."""
    candidate_hash: str | None = Field(default=None, pattern=BLAKE3_PATTERN)
    """The frozen candidate to finalize, by its hash."""
    score_plan_hash: str | None = Field(default=None, pattern=BLAKE3_PATTERN)
    """A score plan, by the hash its PLAN answered."""
    score_snapshot_hash: str | None = Field(default=None, pattern=BLAKE3_PATTERN)
    """The scores a calibration reads, by their snapshot hash."""
    calibration_plan_hash: str | None = Field(default=None, pattern=BLAKE3_PATTERN)
    """A calibration plan, by the hash its PLAN answered."""
    prepared_input_hash: str | None = Field(default=None, pattern=BLAKE3_PATTERN)
    """The prepared input a Portfolio update reads, by its hash."""
    observed_through: str | None = None
    """The last session the update observes, as YYYY-MM-DD."""
    positions_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    """The trade date whose positions the update is for, as YYYY-MM-DD; decided at the close
    before it."""
    formation_session: str | None = None
    """The session the scores are formed on, as YYYY-MM-DD."""
    automation_enabled: bool | None = Field(default=None, strict=True)
    """true lets the daily research update run by itself; false stops it."""
    automation_package_ids: tuple[str, ...] | None = None
    """The strategy packages the automatic research update covers."""
