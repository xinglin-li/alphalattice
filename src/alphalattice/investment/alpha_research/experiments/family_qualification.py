"""The Alpha owner's qualification of a family of development experiments: its contracts,
the family's rule and the receipt reader (GR3). The Task that runs it lives beside the
committer whose candidate set it seals, outside the reusable experiment packages.

A qualification is a Task over every development experiment on one question admitted from a
goal's opening on, whether a goal's session ran it or not, so every recipe tried enters the Holm
family and trying more raises the bar. It refits the nominated candidates on the current window,
applies the historical-mean benchmark and the stability rule, and seals a candidate set or an
evidence-complete stop that a goal cites; no stable model is a result. The sealed holdout stays
unread: the question's fold plan never materializes it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    ResearchExecutionEvidence,
    SealedResearchProgram,
)

from ..inputs.folds import AlphaFoldArrayPlan
from .contracts import AlphaDevelopmentProgram, candidate_id_for_spec
from .development_artifacts import AlphaDevelopmentArtifactStore
from .mandate import AlphaResearchModelRecipe

METHOD = "ALPHA_FAMILY_QUALIFICATION"
CATEGORY = "family-qualifications"


def qualification_section(document: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The document's qualification declaration, when it declares this method."""
    section = document.get("alpha")
    return (
        section
        if isinstance(section, Mapping) and section.get("methodology_id") == METHOD
        else None
    )


def alpha_question_fields(program: AlphaDevelopmentProgram) -> dict[str, Any]:
    """The question a development experiment answers: its Program without the model it ran.

    A development Program seals only the model it admits: its mandate and its catalog
    binding both name that model alone, so a ridge and a LightGBM study of one target on one
    foundation differ there alone. They ask one question, and both belong to its Holm family,
    so the question sets both aside.

    Args:
        program: A development study's Program.

    Returns:
        The Program's fields that make its question, by name.
    """
    return cast(
        dict[str, Any],
        program.model_dump(
            mode="json", exclude={"program_hash", "model_mandate_hash", "model_catalog_hash"}
        ),
    )


def alpha_question_hash(program: AlphaDevelopmentProgram) -> str:
    """The identity of the question a development experiment answers (``alpha_question_fields``)."""
    return str(canonical_hash(alpha_question_fields(program)))


class _Sealed(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> Self:
        payload = cls.model_construct(**values, content_hash="0" * 64).model_dump(
            mode="json", exclude={"content_hash"}
        )
        return cls(**payload, content_hash=canonical_hash(payload))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def _verify_identity(self) -> Self:
        if self.content_hash != canonical_hash(
            self.model_dump(mode="json", exclude={"content_hash"})
        ):
            raise ValueError("alpha_research.qualification_identity_invalid")
        return self


class AlphaFamilyMember(_Sealed):
    """One development experiment on the question: its Task, its sealed result, its recipe."""

    task_id: str = Field(min_length=36, max_length=36)
    admitted_at: datetime
    output_workspace: str = Field(min_length=1, max_length=512)
    receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    batch_result_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    recipe: AlphaResearchModelRecipe


class AlphaQualificationFamily(_Sealed):
    """Every development experiment on one question from a goal's opening on.

    A study cancelled before it sealed a result is attempted without evidence: it is named, and
    it adds no hypothesis the correction could count.
    """

    question_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    goal_id: str = Field(min_length=36, max_length=36)
    opened_at: datetime
    members: tuple[AlphaFamilyMember, ...] = Field(min_length=1, max_length=200)
    unfinished_task_ids: tuple[str, ...] = Field(default=(), max_length=200)

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        """Every candidate the family attempted, in admission order."""
        return tuple(candidate_id_for_spec(value.recipe) for value in self.members)


class AlphaQualificationReceipt(_Sealed):
    """What one qualification sealed: the family, the Program it read and its end."""

    program_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    authority_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    method_binding_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    family_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    qualification_program_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    registry_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    qualification_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    marker_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    disposition: Literal["CURRENT_ALPHA_CANDIDATE_SET_READY", "NO_STABLE_CURRENT_ALPHA_MODEL"]
    candidate_set_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    scientific_stop_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    selected_candidate_ids: tuple[str, ...] = ()
    attempted_candidate_ids: tuple[str, ...] = Field(min_length=1)
    current_qualified_ids: tuple[str, ...] = ()
    fit_call_count: int = Field(ge=0)
    predict_call_count: int = Field(ge=0)
    metric_call_count: int = Field(ge=0)


@dataclass(frozen=True, slots=True)
class AlphaStudy:
    """One Alpha development study as the Host found it: when it was admitted, where its
    Task stands, the question it asked and the recipe it ran; a sealed study's result."""

    task_id: str
    admitted_at: datetime
    lifecycle: str
    question_hash: str
    recipe: AlphaResearchModelRecipe
    sealed: Callable[[], tuple[str, str, str]] | None = None
    """For a study that succeeded: its output (relative to the workspace), receipt and
    batch result, read only when the rule counts it."""


def alpha_family(
    *,
    studies: Iterable[AlphaStudy],
    question_hash: str,
    goal_id: str,
    opened_at: datetime,
) -> AlphaQualificationFamily:
    """The family a qualification concludes (GR3): every study on the question admitted
    from the goal's opening on, whether a goal's session ran it or not.

    A study still moving or waiting on someone leaves the family unsettled; one cancelled
    before its result is named without evidence; a recipe studied twice counts once, its
    first study standing, since it asks nothing new.
    """
    members: list[AlphaFamilyMember] = []
    unfinished: list[str] = []
    seen: set[str] = set()
    for study in sorted(studies, key=lambda value: (value.admitted_at, value.task_id)):
        if study.admitted_at < opened_at or study.question_hash != question_hash:
            continue
        if study.lifecycle == "CANCELLED":
            unfinished.append(study.task_id)
            continue
        if study.lifecycle != "SUCCEEDED" or study.sealed is None:
            raise AuthoringError("alpha_research.qualification_family_member_unsettled")
        candidate_id = candidate_id_for_spec(study.recipe)
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        output, receipt_hash, batch_result_hash = study.sealed()
        members.append(
            AlphaFamilyMember.create(
                task_id=study.task_id,
                admitted_at=study.admitted_at,
                output_workspace=output,
                receipt_hash=receipt_hash,
                batch_result_hash=batch_result_hash,
                recipe=study.recipe,
            )
        )
    if not members:
        raise AuthoringError("alpha_research.qualification_family_empty")
    return AlphaQualificationFamily.create(
        question_hash=question_hash,
        goal_id=goal_id,
        opened_at=opened_at,
        members=tuple(members),
        unfinished_task_ids=tuple(unfinished),
    )


@dataclass(frozen=True, slots=True)
class AlphaQuestionPreparation:
    """The question's fold plan and Program, prepared as its development studies prepare them."""

    fold_plan: AlphaFoldArrayPlan
    development_program: AlphaDevelopmentProgram


def planned_calls(family: AlphaQualificationFamily, nominated: tuple[str, ...]) -> int:
    """The calls a qualification makes at most: a fit and a prediction for each nominated
    candidate, and one metric for each attempted candidate and the benchmark."""
    return 2 * len(nominated) + len(family.members) + 1


def read_qualification_receipt(
    *,
    program: SealedResearchProgram,
    evidence: ResearchExecutionEvidence,
    output_workspace: Path,
) -> AlphaQualificationReceipt:
    """The receipt a qualification's evidence names, bound to its Program."""
    if len(evidence.artifact_uris) != 1:
        raise AuthoringError("alpha_research.qualification_receipt_not_unique")
    store = AlphaDevelopmentArtifactStore(output_workspace / "alpha-qualification")
    identity = store._hash_from_uri(evidence.artifact_uris[0], CATEGORY)
    receipt: AlphaQualificationReceipt = AlphaQualificationReceipt.model_validate_json(
        store._path(CATEGORY, identity, "json").read_bytes()
    )
    if (
        receipt.content_hash != identity
        or receipt.program_hash != program.program_hash
        or receipt.authority_hash != program.authority_hash
        or receipt.method_binding_hash != program.method_binding_hash
        or evidence.desk_input_binding_hash != program.authority_hash
    ):
        raise AuthoringError("alpha_research.qualification_receipt_binding_invalid")
    return receipt


QUALIFICATION_CATEGORY = CATEGORY


__all__ = [
    "METHOD",
    "QUALIFICATION_CATEGORY",
    "AlphaFamilyMember",
    "AlphaQualificationFamily",
    "AlphaQualificationReceipt",
    "AlphaQuestionPreparation",
    "AlphaStudy",
    "alpha_family",
    "alpha_question_fields",
    "alpha_question_hash",
    "planned_calls",
    "qualification_section",
    "read_qualification_receipt",
]
