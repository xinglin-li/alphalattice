"""A research experiment's sealed plan: the contract its admission, projection and upgrade read.

The application (`research_experiments`) builds and admits a plan; the read-only projection
(`research_experiment_projection`) and the upgrade overview read one. The plan lives apart
from both so the projection, split out of the application, reads it without importing the
application back.
"""

from __future__ import annotations

from typing import Any, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceExperimentInput,
    ResearchWorkspaceModelTrainingInput,
)
from alphalattice.control.task_control.contracts import (
    ResearchGoal,
    ResearchPlan,
    TaskInputEnvelope,
    TaskRecord,
)
from alphalattice.foundation.factor_research.experiments.authoring import FACTOR_EXPERIMENT_KIND
from alphalattice.investment.alpha_research.experiments.authoring import ALPHA_EXPERIMENT_KIND
from alphalattice.investment.alpha_research.experiments.family_qualification import (
    AlphaQualificationFamily,
)
from alphalattice.investment.portfolio_strategy_lab.application.research_experiment import (
    KIND as PORTFOLIO_EXPERIMENT_KIND,
)
from alphalattice.investment.portfolio_strategy_lab.application.research_experiment import (
    PortfolioExperimentSource,
)
from alphalattice.investment.risk_research.experiments.compiler import RISK_EXPERIMENT_KIND
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.sealing import seal_model
from alphalattice.protocols.actor_execution.contracts import ActorSubmissionBinding
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    ResearchExperimentEnvelope,
    ResolvedResearchAuthority,
    SealedResearchProgram,
)

TASK_KIND = "research_experiment"
STAGES = ("execute_sealed_experiment", "verify_experiment_evidence")


class AlphaExperimentSource(BaseModel):  # type: ignore[misc]
    """Verified parent lineage and both input roots needed by recovery."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    factor_task_id: UUID
    factor_binding: ResearchWorkspaceExperimentInput
    factor_receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    curation_receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    foundation_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    foundation_admission_hash: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$", exclude_if=lambda v: v is None
    )


class ExperimentPlan(BaseModel):  # type: ignore[misc]
    """Bind an authored experiment to sealed program, authority and exact upstream sources."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    workspace_id: str
    binding: ResearchWorkspaceExperimentInput
    document: dict[str, Any]
    program: SealedResearchProgram
    authority: ResolvedResearchAuthority
    execution_preview: dict[str, Any]
    implementation_hash: str
    origin_task_id: UUID | None = Field(default=None, exclude_if=lambda value: value is None)
    alpha_source: AlphaExperimentSource | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    portfolio_source: PortfolioExperimentSource | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    model_training_source: ResearchWorkspaceModelTrainingInput | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    qualification_family: AlphaQualificationFamily | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    """The family of development studies an Alpha qualification concludes (GR3)."""
    plan_hash: str

    @classmethod
    def create(cls, **values: Any) -> Self:
        """Seal a retained experiment plan.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical plan_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        return seal_model(cls, values, field="plan_hash")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def verify(self) -> Self:
        """Require exact experiment plan identity and mutually consistent typed upstream sources.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            AuthoringError: Plan/program/authority/kind or the selected Alpha/Portfolio input
                bindings differ.
        """
        envelope = ResearchExperimentEnvelope.create(**self.document["experiment"])
        if (
            self.plan_hash != canonical_hash(self.model_dump(mode="json", exclude={"plan_hash"}))
            or envelope.envelope_hash != self.program.envelope_hash
            or self.authority.authority_hash != self.program.authority_hash
            or self.program.kind
            not in {
                FACTOR_EXPERIMENT_KIND,
                ALPHA_EXPERIMENT_KIND,
                PORTFOLIO_EXPERIMENT_KIND,
                RISK_EXPERIMENT_KIND,
            }
            or (self.program.kind == ALPHA_EXPERIMENT_KIND)
            != (
                self.alpha_source is not None
                or self.model_training_source is not None
                or self.qualification_family is not None
            )
            or sum(
                value is not None
                for value in (
                    self.alpha_source,
                    self.model_training_source,
                    self.qualification_family,
                )
            )
            > 1
            or (
                self.model_training_source is not None
                and self.model_training_source.input_binding_hash != self.binding.binding_hash
            )
            or (self.program.kind == PORTFOLIO_EXPERIMENT_KIND)
            != (self.portfolio_source is not None)
        ):
            raise AuthoringError("research_experiment.plan_identity_invalid")
        return self


def _task_contract(
    plan: ExperimentPlan, actor: ActorSubmissionBinding
) -> tuple[TaskInputEnvelope, ResearchGoal, ResearchPlan]:
    if actor.submission_hash != plan.program.program_hash:
        raise AuthoringError("research_experiment.actor_program_mismatch")
    envelope = TaskInputEnvelope.create(
        task_kind=TASK_KIND,
        input_schema_id="research-experiment-task-input",
        payload={"plan": plan.model_dump(mode="json"), "actor": actor.model_dump(mode="json")},
    )
    # Each summary is bytes of every stored Task's goal, which `checked_task_plan` rebuilds and
    # compares by equality: rewriting a summary refused every stored Task of its kind and
    # was reverted, so the texts are pinned with their reason in the tests.
    goal = ResearchGoal.create(
        goal_kind="EXECUTE_RESEARCH_EXPERIMENT",
        input_hash=envelope.input_hash,
        deliverable_kind="ResearchExecutionEvidence",
        summary={
            FACTOR_EXPERIMENT_KIND: (
                "Run an admitted Factor experiment, not a strategy or current publication."
            ),
            ALPHA_EXPERIMENT_KIND: (
                "Run an admitted Alpha development experiment, "
                "not a strategy or current publication."
            ),
            PORTFOLIO_EXPERIMENT_KIND: (
                "Replay selected Alpha evidence through an EW research book; no activation."
            ),
            RISK_EXPERIMENT_KIND: (
                "Run installed Risk diagnostics over a sealed research input; "
                "no allocation or activation."
            ),
        }[plan.program.kind],
    )
    workflow = ResearchPlan.from_stages(
        goal_hash=goal.goal_hash, stages=STAGES, verifier_namespace="research_experiment"
    )
    return envelope, goal, workflow


def checked_task_plan(
    task: TaskRecord, *, workspace_id: str
) -> tuple[ExperimentPlan, ActorSubmissionBinding]:
    """Check a retained Task against its exact plan, actor and workspace contract."""
    if task.task_kind != TASK_KIND:
        raise AuthoringError("research_experiment.task_kind_mismatch")
    plan = ExperimentPlan.model_validate(task.input.payload.get("plan"))
    actor = ActorSubmissionBinding.model_validate(task.input.payload.get("actor"))
    envelope, goal, workflow = _task_contract(plan, actor)
    if (
        task.input != envelope
        or task.goal != goal
        or task.plan != workflow
        or plan.workspace_id != workspace_id
    ):
        raise AuthoringError("research_experiment.task_binding_mismatch")
    return plan, actor


__all__ = ["STAGES", "TASK_KIND", "AlphaExperimentSource", "ExperimentPlan", "checked_task_plan"]
