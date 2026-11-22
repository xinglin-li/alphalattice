"""Data and preparation request fields and their public descriptions."""

from pydantic import ConfigDict, Field

from ..questions import PersonConfirmation
from .contracts import RequestFieldModel


class RequestFields(RequestFieldModel):
    """Data preparation and update fields for their operations."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    history_cursor: str | None = None
    """Where a list reads on: the `next_cursor` its previous page answered."""
    preparation_plan_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The workspace preparation plan to confirm, by the hash its PLAN answered."""
    data_issue_case_token: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The data issue, by the case token the issue list gave."""
    data_issue_evidence_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The evidence the issue was shown with; an issue that moved is refused."""
    data_issue_option_id: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_-]{2,119}$")
    """The option chosen for the data issue, one of those it offered."""
    data_issue_option_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The chosen option's hash as offered; a changed option is refused."""
    data_issue_grant_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """A person's recorded grant for a data decision, by its hash."""
    storage_plan_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The storage clean-up plan to confirm, by the hash its PLAN answered."""
    evidence_index_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The Evidence retrieval index to rebuild, by the id the storage view listed."""
    input_pinned: bool | None = Field(default=None, strict=True)
    """true keeps the named evidence index from clean-up; false lets clean-up release it. A
    research input's version takes no pin (`storage.input_pin_retired`)."""
    update_plan_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """An update plan, by the hash its PLAN answered."""
    storage_cap_bytes: str | None = Field(default=None, min_length=1, max_length=32)
    """`auto`, or a positive whole byte cap for managed writes; changes no result identity."""
    backup_generations_kept: int | None = Field(default=None, ge=1, le=100, strict=True)
    """How many backup generations to keep: seven unless a request keeps another count."""
    person_confirmation: PersonConfirmation | None = None
    """The person's yes to this decision, asked in one line and relayed whole: `--person-said`
    their words and `--asked` your question, which the client binds to this request. Only a
    person's decision takes it, once."""
