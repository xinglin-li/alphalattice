"""The retired goal loop's Task Board, kept so a published marker's board reads back (GR4)."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from ..experiments.mandate import AlphaModelRecipeProposal
from .contracts import (
    LegacyModelSpec,
    _Contract,
    _validate_hash,
    _validate_hash_compatible,
)


class AlphaGoalTaskKind(StrEnum):
    """Name the ordered goal research tasks from orientation through terminal publication."""

    ORIENT_TO_GOAL = "ORIENT_TO_GOAL"
    ESTABLISH_EVIDENCE_BASE = "ESTABLISH_EVIDENCE_BASE"
    DESIGN_INITIAL_BATCH = "DESIGN_INITIAL_BATCH"
    EXECUTE_INITIAL_BATCH = "EXECUTE_INITIAL_BATCH"
    REVIEW_INITIAL_BATCH = "REVIEW_INITIAL_BATCH"
    NOMINATE_INITIAL_QUALIFICATION = "NOMINATE_INITIAL_QUALIFICATION"
    ASSESS_GOAL = "ASSESS_GOAL"
    DESIGN_REFINEMENT_BATCH = "DESIGN_REFINEMENT_BATCH"
    EXECUTE_REFINEMENT_BATCH = "EXECUTE_REFINEMENT_BATCH"
    REVIEW_REFINEMENT_BATCH = "REVIEW_REFINEMENT_BATCH"
    NOMINATE_FINAL_QUALIFICATION = "NOMINATE_FINAL_QUALIFICATION"
    ASSESS_FINAL_GOAL = "ASSESS_FINAL_GOAL"
    FORM_CURRENT_CANDIDATE_SET = "FORM_CURRENT_CANDIDATE_SET"
    FINALIZE_SCIENTIFIC_STOP = "FINALIZE_SCIENTIFIC_STOP"
    FINALIZE_LIMITATIONS = "FINALIZE_LIMITATIONS"


class AlphaGoalTaskStatus(StrEnum):
    """Name pending, running, waiting, verified or blocked goal-task execution."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING = "WAITING"
    VERIFIED = "VERIFIED"
    BLOCKED = "BLOCKED"


class AlphaGoalRunDisposition(StrEnum):
    """Name running/tool-wait/blocked execution or the two terminal research outputs."""

    RUNNING = "RUNNING"
    WAITING_FOR_TOOL = "WAITING_FOR_TOOL"
    BLOCKED = "BLOCKED"
    CURRENT_ALPHA_CANDIDATE_SET = "CURRENT_ALPHA_CANDIDATE_SET"
    EVIDENCE_COMPLETE_SCIENTIFIC_STOP = "EVIDENCE_COMPLETE_SCIENTIFIC_STOP"


class AlphaGoalTaskRecord(_Contract):
    """Seal one task instance, completion evidence and exclusive legacy/current frozen recipes."""

    kind: Literal["AlphaGoalTaskRecord"] = "AlphaGoalTaskRecord"
    task_instance_id: str = Field(min_length=1)
    task_kind: AlphaGoalTaskKind
    generation: int = Field(ge=0, le=1)
    status: AlphaGoalTaskStatus
    completion_criteria_ids: tuple[str, ...] = Field(min_length=1)
    verified_result_refs: tuple[str, ...] = ()
    frozen_specs: tuple[LegacyModelSpec, ...] = ()
    frozen_model_recipes: tuple[AlphaModelRecipeProposal, ...] = ()
    nominated_candidate_ids: tuple[str, ...] = ()
    feedback_ref: str | None = None
    record_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_record(self) -> Self:
        """Require result evidence for verified tasks and frozen recipes for verified batch design.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Verified results/recipes are absent, legacy and current recipe authority is
                mixed or compatible record_hash is inconsistent.
        """
        if self.status is AlphaGoalTaskStatus.VERIFIED and not self.verified_result_refs:
            raise ValueError("verified Alpha goal task lacks result evidence")
        if (
            self.task_kind
            in {
                AlphaGoalTaskKind.DESIGN_INITIAL_BATCH,
                AlphaGoalTaskKind.DESIGN_REFINEMENT_BATCH,
            }
            and self.status is AlphaGoalTaskStatus.VERIFIED
        ):
            frozen = self.frozen_model_recipes or self.frozen_specs
            if not frozen:
                raise ValueError("verified Alpha design task lacks its frozen recipes")
        if self.frozen_model_recipes and self.frozen_specs:
            raise ValueError("Alpha design task mixes legacy specs and model recipes")
        _validate_hash_compatible(
            self,
            "record_hash",
            optional_fields=("frozen_model_recipes",),
        )
        return self


class AlphaGoalTaskBoard(_Contract):
    """Seal the unique bounded task sequence, verified prefix and run disposition."""

    kind: Literal["AlphaGoalTaskBoard"] = "AlphaGoalTaskBoard"
    board_id: str = Field(min_length=1)
    program_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    research_goal_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    revision: int = Field(ge=0, le=64)
    tasks: tuple[AlphaGoalTaskRecord, ...] = Field(min_length=1, max_length=15)
    current_index: int = Field(ge=0, le=15)
    final_disposition: AlphaGoalRunDisposition
    board_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_board(self) -> Self:
        """Require a verified prefix, pending future tasks and cursor-consistent disposition.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: The cursor exceeds tasks, task identifiers repeat, prefix/future statuses
                disagree, final disposition disagrees with completion or board_hash is inconsistent.
        """
        if self.current_index > len(self.tasks):
            raise ValueError("Alpha goal Task Board cursor is invalid")
        if len({value.task_instance_id for value in self.tasks}) != len(self.tasks):
            raise ValueError("Alpha goal Task Board repeats a task instance")
        for index, task in enumerate(self.tasks):
            if index < self.current_index and task.status is not AlphaGoalTaskStatus.VERIFIED:
                raise ValueError("Alpha goal Task Board verified prefix is discontinuous")
            if index > self.current_index and task.status is not AlphaGoalTaskStatus.PENDING:
                raise ValueError("future Alpha goal tasks must remain pending")
        if self.current_index == len(self.tasks):
            if self.final_disposition not in {
                AlphaGoalRunDisposition.CURRENT_ALPHA_CANDIDATE_SET,
                AlphaGoalRunDisposition.EVIDENCE_COMPLETE_SCIENTIFIC_STOP,
                AlphaGoalRunDisposition.BLOCKED,
            }:
                raise ValueError("terminal Alpha goal board lacks a terminal disposition")
        elif self.final_disposition not in {
            AlphaGoalRunDisposition.RUNNING,
            AlphaGoalRunDisposition.WAITING_FOR_TOOL,
            AlphaGoalRunDisposition.BLOCKED,
        }:
            raise ValueError("unfinished Alpha goal board has a terminal disposition")
        _validate_hash(self, "board_hash")
        return self


__all__ = [
    "AlphaGoalRunDisposition",
    "AlphaGoalTaskBoard",
    "AlphaGoalTaskKind",
    "AlphaGoalTaskRecord",
    "AlphaGoalTaskStatus",
]
