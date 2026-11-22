"""Research request fields and their public descriptions."""

from typing import Any, Literal
from uuid import UUID

from pydantic import ConfigDict, Field

from .contracts import RequestFieldModel


class FactorCurationChoice(RequestFieldModel):
    """A Factor selection and its recorded rationale."""

    # The published nested schema has no description.
    __doc__ = None
    model_config = ConfigDict(extra="forbid", frozen=True)
    factor_id: str = Field(min_length=1)
    role: Literal["CORE", "CONDITIONAL"]
    rationale: str = Field(min_length=1, max_length=500)


class FactorCurationRequest(RequestFieldModel):
    """Actor intent and the report it answers, never caller-supplied authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    expected_receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    choices: tuple[FactorCurationChoice, ...]
    limitations_acknowledged: tuple[str, ...]


class ResearchDeliveryCommentary(RequestFieldModel):
    """Attributed caller text, never a product-verified actor identity or decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    attribution: str = Field(min_length=1, max_length=160)
    text: str = Field(min_length=1, max_length=20_000)


class RequestFields(RequestFieldModel):
    """Research, experiment and delivery fields for their operations."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    history_limit: int | None = Field(default=None, ge=1, le=50, strict=True)
    """How many entries one page reads, 1 to 50."""
    history_entry_id: str | None = Field(default=None, json_schema_extra={"reference": "issued"})
    """One research history entry, by the id the history listed."""
    history_kind: str | None = None
    """Only the history entries of this kind."""
    experience_hypothesis: str | None = Field(default=None, min_length=1, max_length=256)
    """A stated hypothesis ID to test or select in a read."""
    experience_method: str | None = Field(default=None, min_length=1, max_length=256)
    """Only trials using this method."""
    experience_workspace: str | None = Field(default=None, min_length=1, max_length=256)
    """Only trials from this workspace ID."""
    experience_trial: str | None = Field(default=None, min_length=1, max_length=64)
    """The trial, by its ID, whose class to read."""
    experience_width: Literal["recipe", "method_version", "method"] | None = None
    """The class's width: the same recipe; the same method version with other parameters (the
    default); any version of the method."""
    experience_known_at: str | None = Field(default=None, min_length=1, max_length=64)
    """Read the class as the history knew it at this ISO-8601 time with its offset; now when
    absent."""
    experience_budget: int | None = Field(default=None, ge=100, le=32_000, strict=True)
    """A brief's budget in tokens, 100 to 32,000; 2,000 when absent."""
    experience_query: str | None = Field(default=None, min_length=1, max_length=512)
    """The text to recall experiences like, 1 to 512 characters."""
    experience_limit: int | None = Field(default=None, ge=1, le=20, strict=True)
    """How many experiences a recall returns, 1 to 20; 5 when absent."""
    review_subject: str | None = Field(
        default=None, pattern=r"^(trial|task|goal):[0-9A-Za-z-]{1,64}$"
    )
    """What a note reviews: `trial:<id>` or `task:<id>` once it has ended, or `goal:<id>`."""
    review_verdict: Literal["success", "failure", "inconclusive"] | None = None
    """The review's verdict on what it reviews."""
    review_note: dict[str, Any] | None = None
    """The note: its hypothesis, method and conclusion, each a sentence, its lessons, and for a
    review task the `trials` it reviewed."""
    left_task_id: UUID | None = None
    """The first of two studies to compare, by its Task."""
    right_task_id: UUID | None = None
    """The second of two studies to compare, by its Task."""
    left_candidate_id: str | None = Field(default=None, min_length=1)
    """The first of two Alpha candidates to compare."""
    right_candidate_id: str | None = Field(default=None, min_length=1)
    """The second of two Alpha candidates to compare."""
    delivery_question: str | None = Field(default=None, min_length=1, max_length=2400)
    """The research question a delivery answers, in the researcher's words."""
    delivery_commentary: tuple[ResearchDeliveryCommentary, ...] | None = Field(
        default=None, max_length=16
    )
    """The researcher's comments a delivery carries, each on the part it names."""
    research_input_plan_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The research input plan to confirm, by the hash its PLAN answered."""
    origin_task_id: UUID | None = None
    """The study a new plan continues from, by its Task."""
    factor_task_id: UUID | None = None
    """The Factor study an Alpha or Foundation plan builds on, by its Task."""
    input_binding_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The research input's version, by its binding hash; with `research_input_id` it names the
    data a study reads."""
    research_input_id: str | None = None
    """The research input, by its id (RESEARCH_INPUTS lists them)."""
    experiment_document: dict[str, Any] | None = None
    """The study's declaration as a document, as `study controls` or `study draft`
    writes it."""
    feature_preparation_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The prepared Feature input whose controls a study reads, by its hash."""
    experiment_kind: str | None = None
    """The study kind whose controls to show, as `study controls` lists them; Factor when
    omitted."""
    risk_task_id: UUID | None = None
    """The Risk study to link, by its Task."""
    risk_report_hash: str | None = None
    """A Risk report, by its hash, to export or deliver."""
    experiment_yaml: str | None = None
    """The study's declaration as YAML text; send it or `experiment_document`, not both."""
    experiment_plan_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """A saved plan, by the hash its PLAN answered."""
    risk_report_scope: Literal["POST_OBSERVED_PORTFOLIO_WINDOW"] | None = None
    """`POST_OBSERVED_PORTFOLIO_WINDOW` links the Risk report over the Portfolio's observed
    window."""
    experiment_curation: FactorCurationRequest | None = None
    """The Factor curation decision: which factors go on, and why."""
    curation_receipt_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The Factor curation decision, by its receipt hash."""
    foundation_admission_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The Foundation, by its admission hash."""
    candidate_id: str | None = Field(default=None, min_length=1)
    """The Alpha candidate a Portfolio draft is built from."""
    component_id: str | None = Field(default=None, min_length=1)
    """The strategy component, by its id."""
    model_lifecycle: Literal["LIGHT", "FULL"] | None = None
    """A component's model lifecycle: `LIGHT`, the default, or `FULL` by name."""
    model_id: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]*$")
    """An Alpha model, by the id its declaration names (`model list` lists them)."""
    task_id: UUID | None = None
    """The Task, by its id."""
