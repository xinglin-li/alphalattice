"""Goal and committee request fields and their public descriptions."""

from typing import Any, Literal
from uuid import UUID

from pydantic import ConfigDict, Field

from .contracts import RequestFieldModel


class CommitteeMessage(RequestFieldModel):
    """One message a committee member sends to its floor (`committee submit`)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["STANCE", "CHALLENGE", "REPLY", "PASS", "RULING", "VERDICT", "PERSON_ANSWER"]
    """STANCE (blind first round), CHALLENGE, REPLY, PASS (a yielded turn); the PM's RULING,
    VERDICT, the person's answer to what the PM handed them (PERSON_ANSWER)."""
    text: str = Field(min_length=1, max_length=4000)
    """Its words; a holding or point goes by its alias (H3, T2), never a number."""
    targets: tuple[str, ...] = Field(default=(), max_length=16)
    """The tension points (T1), holdings (H3) or messages (M4) it is about."""
    reply_to: str | None = Field(default=None, pattern=r"^M[0-9]{1,4}$")
    """The message it answers, which a REPLY (any message), a RULING (a challenge not yet
    ruled) and a PERSON_ANSWER (an item the PM handed the person) name; no other kind."""
    positions: dict[str, Literal["SUPPORT", "OBJECT", "RESERVE"]] = Field(
        default_factory=dict, max_length=32
    )
    """A stance's position on each agenda point."""
    outcome: (
        Literal["ADOPT", "REJECT", "FOR_THE_PERSON", "PROCEED", "PROCEED_WITH_NOTES", "DECLINE"]
        | None
    ) = None
    """Required of a RULING (ADOPT, REJECT, FOR_THE_PERSON) and the VERDICT (PROCEED,
    PROCEED_WITH_NOTES, DECLINE, FOR_THE_PERSON); other kinds take none. A wrong shape is
    refused with its correction."""


class RequestFields(RequestFieldModel):
    """Goal and committee fields for the operations that admit them."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    goal_id: UUID | None = None
    """The goal, by the id its GOAL_OPEN answered with."""
    goal_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    """The goal's version as last read; a goal that moved since refuses the change."""
    goal_reference_id: str | None = Field(default=None, min_length=1, max_length=64)
    """The reference ID saved in the exact goal revision; only that reference is verified."""
    goal_declaration: dict[str, Any] | None = None
    """The goal as written: its statement, scope and completion rule (`goal schema`)."""
    goal_reference: dict[str, Any] | None = None
    """What to attach to the goal: a Task, a study, a result or a publication, by reference."""
    goal_statement: dict[str, Any] | None = None
    """A note for the goal's record, written by the agent working for it."""
    goal_submission: dict[str, Any] | None = None
    """The completion document claiming the goal is met, naming its evidence."""
    change_reason: str | None = None
    """Why the goal changes, kept in its record beside the change."""
    committee_daily_enabled: bool | None = Field(default=None, strict=True)
    """true: each daily update's review convenes the committee; false: the PM's view."""
    agent_session: str | None = Field(default=None, min_length=1, max_length=200)
    """An agent session: a Task listing shows the Tasks it submitted, 50 to a page, and a goal
    listing the goals whose record names it."""
    committee_role: Literal["PM", "ALPHA", "RISK", "CRO"] | None = None
    """The committee member this request speaks or reads for: the PM (the lead), ALPHA, RISK
    or CRO."""
    committee_key: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    """That member's key: a specialist's is in its bundle, the PM's in its own open's answer."""
    committee_message: CommitteeMessage | None = None
    """One committee message, as its YAML or JSON file."""
    committee_seen: int | None = Field(default=None, ge=0, le=9999, strict=True)
    """The last floor message the reader has seen, by its number (M4 is 4)."""
