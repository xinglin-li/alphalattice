"""The actor-neutral answer boundary: an agent writes judgment, the Host screens it.

An answer is one JSON object holding a list of items and, at most, a few
optional texts. The Host checks each item on its own against what the agent
was given and names every problem in plain words -- which item, what is wrong,
what would be right -- never as a code. What an agent leaves out is never a
problem: a subset is an answer and an empty list is an answer. Trivial spelling
of a closed value (`adverse`, `legal regulatory`) or a single citation written
without its list is normalized, not refused.

How many corrections one bundle receives is the Host's bound: two after the
first answer. An answer with problems is returned for correction while one is
left (`CORRECT`); after the last the valid items are accepted and the rest
recorded as dropped (`DONE`), so an answer always finishes and a report always
comes out. A clean answer -- an empty one included -- is accepted at once.
"""

from __future__ import annotations

import types
import typing
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from alphalattice.kernel.shared_kernel.identity import canonical_hash

ANSWER_CORRECTION_BOUND: int = 2
"""Corrections allowed after the first answer (the first trial needed one)."""

MAXIMUM_ANSWER_COUNT: int = 1 + ANSWER_CORRECTION_BOUND
"""Answers a built-in agent gives in one stage: the first and its corrections."""
SPECIALIST_TEXT_CHARACTERS = 4000
SPECIALIST_REFERENCE_COUNT = 64
SPECIALIST_REFERENCE_CHARACTERS = 200


class AnswerVerdict(StrEnum):
    """What the Host did with one answer."""

    ACCEPTED = "ACCEPTED"
    """Every item written is valid; the answer is admitted as written."""
    CORRECT = "CORRECT"
    """Problems are named and a correction is still allowed; nothing admitted."""
    DONE = "DONE"
    """The bound is reached: the valid items are admitted, the rest dropped."""


def answer_verdict(*, clean: bool, corrections_used: int) -> AnswerVerdict:
    """Choose a verdict from answer validity and prior corrections."""
    if clean:
        return AnswerVerdict.ACCEPTED
    if corrections_used < ANSWER_CORRECTION_BOUND:
        return AnswerVerdict.CORRECT
    return AnswerVerdict.DONE


class AnswerProblem(BaseModel):  # type: ignore[misc]
    """One problem with one answer, in plain words."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    item: int | None = Field(default=None, ge=1)
    """The item's number as the agent wrote it, from one; None for the
    answer as a whole."""
    text: str = Field(min_length=1, max_length=400)


class AgentRun(BaseModel):  # type: ignore[misc]
    """Which host, session, agent, model and effort made one answer, as found when it arrived.

    Provenance beside the judgment, never in its identity (LAWS.md ID7, V300). A request names
    its session, never its subagent, so the agent is the specialist the session's lead assigned
    the answered bundle to, started in the session under the bundle's role; with no such link
    the author is unknown, the host and session alone (AU3, V555). ``basis`` says where the
    model came from: ``HOOK``, the agent's start hook; ``ROLE_CARD``, its card's pin;
    ``SESSION_FILE``, the lead's newest reading of its own file, on a record written before
    V555; ``NOT_OBSERVED``, nothing but the session.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    host: Literal["claude-code", "codex"]
    session_id: str = Field(min_length=1, max_length=128)
    agent_id: str | None = Field(default=None, min_length=1, max_length=128)
    role: str | None = Field(default=None, min_length=1, max_length=64)
    model: str | None = Field(default=None, min_length=1, max_length=64)
    efforts: tuple[str, ...] = Field(default=(), max_length=8)
    basis: Literal["HOOK", "ROLE_CARD", "SESSION_FILE", "NOT_OBSERVED"]


class AgentAnswerRecord(BaseModel):  # type: ignore[misc]
    """Record one answer the Host received for a bundle.

    The Host stores it durably so the correction count survives a restart.
    Its bundle key and answer number form a write-once slot.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["AgentAnswerRecord"] = "AgentAnswerRecord"
    bundle_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    """The binding the answer answers: the prepared packet or dossier and the
    policies it was prepared under, as the Host keys them."""
    number: int = Field(ge=1)
    verdict: AnswerVerdict
    corrections_used: int = Field(ge=0)
    """Corrections returned for the bundle, this answer's included."""
    problems: tuple[AnswerProblem, ...] = Field(max_length=256)
    accepted_items: tuple[int, ...] = Field(max_length=256)
    answer_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    """The canonical digest of the answer as written."""
    read_files: tuple[str, ...] | None = Field(default=None, max_length=256)
    """The bundle files the answer named as read whole, as it named them: provenance, never a
    condition of reading it (OP11, V260); absent on a record written before it was kept."""
    agent_run: AgentRun | None = None
    """Who made the answer (AU3): absent when the request named no agent session, and on a
    record written before it was kept."""
    record_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    accepted_text: str | None = Field(
        default=None, min_length=1, max_length=SPECIALIST_TEXT_CHARACTERS
    )
    accepted_references: tuple[str, ...] = Field(default=(), max_length=SPECIALIST_REFERENCE_COUNT)
    accepted_at: datetime | None = None
    """The generic specialist's accepted contribution, retained in this existing record."""

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_record(self) -> Self:
        """Require the record hash to match its bundle and answer number."""
        if self.record_hash != answer_slot(self.bundle_key, self.number):
            raise ValueError("actor_execution.answer_record_slot_invalid")
        return self


class SpecialistAnswer(BaseModel):  # type: ignore[misc]
    """An interpretation's bounded text and exact references, without a science-quality verdict."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1, max_length=SPECIALIST_TEXT_CHARACTERS)
    references: tuple[str, ...] = Field(default=(), max_length=SPECIALIST_REFERENCE_COUNT)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def nonempty_text(self) -> Self:
        """Require a contribution, never a whitespace-only accepted answer."""
        if not self.text.strip():
            raise ValueError("actor_execution.specialist_text_empty")
        if any(
            not reference or len(reference) > SPECIALIST_REFERENCE_CHARACTERS
            for reference in self.references
        ):
            raise ValueError("actor_execution.specialist_reference_invalid")
        return self


def answer_slot(bundle_key: str, number: int) -> str:
    """Where the `number`-th answer to a bundle is kept."""
    return str(canonical_hash({"bundle": bundle_key, "answer": number}))


def answer_digest(raw: object) -> str:
    """Hash an answer as written, whatever its shape."""
    if isinstance(raw, BaseModel):
        raw = raw.model_dump(mode="json")
    try:
        return str(canonical_hash(raw))
    except (TypeError, ValueError):
        return str(canonical_hash({"unreadable": repr(raw)[:2000]}))


@dataclass(frozen=True, slots=True)
class ScreenedAnswer[ItemT: BaseModel]:
    """Hold one answer after the Host's screen.

    It retains the agent's original item numbers, valid optional texts, and
    every problem.
    """

    items: tuple[tuple[int, ItemT], ...]
    texts: Mapping[str, str] = field(default_factory=dict)
    problems: tuple[AnswerProblem, ...] = ()

    @property
    def clean(self) -> bool:
        """Return whether screening found no problems."""
        return not self.problems

    @property
    def accepted_numbers(self) -> tuple[int, ...]:
        """Return the original item numbers accepted by screening."""
        return tuple(number for number, _ in self.items)


def screen_specialist_answer(
    raw: object, *, allowed_references: Iterable[str]
) -> ScreenedAnswer[SpecialistAnswer]:
    """Check only a generic answer's format, size and exact prepared-reference membership."""
    try:
        parsed = SpecialistAnswer.model_validate(raw)
    except ValidationError as error:
        return ScreenedAnswer(
            items=(), problems=tuple(AnswerProblem(text=text) for text in plain_words(error))
        )
    allowed = set(allowed_references)
    if any(reference not in allowed for reference in parsed.references):
        return ScreenedAnswer(
            items=(),
            problems=(
                AnswerProblem(text="Use only exact references listed in this assigned bundle."),
            ),
        )
    return ScreenedAnswer(items=((1, parsed),))


def screen_answer[ItemT: BaseModel](
    raw: object,
    *,
    items_field: str,
    item_model: type[ItemT],
    maximum_items: int,
    check_item: Callable[[ItemT], Iterable[str]],
    text_fields: Mapping[str, int] | None = None,
) -> ScreenedAnswer[ItemT]:
    """Check what is written, item by item, and nothing that is not.

    `check_item` names the problems a schema cannot see -- an alias the bundle
    does not hold, an issuer it does not cover -- in plain words. An item with
    no problem is accepted whatever the others say; a missing items list is an
    empty answer.
    """
    texts_allowed = dict(text_fields or {})
    if isinstance(raw, BaseModel):
        # An answer a transport already parsed is screened as written.
        raw = raw.model_dump(mode="json")
    if not isinstance(raw, dict):
        return ScreenedAnswer(
            items=(),
            problems=(
                AnswerProblem(
                    text=f'The answer must be one JSON object, such as {{"{items_field}": []}}.'
                ),
            ),
        )
    problems: list[AnswerProblem] = []
    names = ", ".join(f"'{value}'" for value in (items_field, *texts_allowed))
    for key in raw:
        if key != items_field and key not in texts_allowed:
            problems.append(
                AnswerProblem(text=_bounded(f"'{key}' is not a field of this answer; use {names}."))
            )
    written = raw.get(items_field)
    if written is None:
        written = []
    if not isinstance(written, list):
        problems.append(AnswerProblem(text=f"'{items_field}' must be a list."))
        written = []
    if len(written) > maximum_items:
        problems.append(
            AnswerProblem(
                text=(
                    f"The answer has {len(written)} {items_field}; only the first {maximum_items} "
                    "are read."
                )
            )
        )
        written = written[:maximum_items]
    accepted: list[tuple[int, ItemT]] = []
    for number, value in enumerate(written, start=1):
        if not isinstance(value, dict):
            problems.append(AnswerProblem(item=number, text="The item must be a JSON object."))
            continue
        try:
            parsed = item_model.model_validate(_normalized(value, item_model))
        except ValidationError as error:
            problems.extend(
                AnswerProblem(item=number, text=text) for text in plain_words(error, items_field)
            )
            continue
        found = tuple(check_item(parsed))
        if found:
            problems.extend(AnswerProblem(item=number, text=_bounded(text)) for text in found)
            continue
        accepted.append((number, parsed))
    texts: dict[str, str] = {}
    for name, maximum in texts_allowed.items():
        value = raw.get(name)
        if value is None or value == "":
            continue
        if not isinstance(value, str):
            problems.append(AnswerProblem(text=f"'{name}' must be text."))
        elif len(value) > maximum:
            problems.append(
                AnswerProblem(text=f"'{name}' is longer than {maximum} characters; shorten it.")
            )
        elif value.strip():
            texts[name] = value.strip()
    return ScreenedAnswer(items=tuple(accepted), texts=texts, problems=tuple(problems))


def correction_text(
    problems: tuple[AnswerProblem, ...],
    *,
    accepted: tuple[int, ...],
    rounds_left: int,
) -> str:
    """Describe the corrections needed before the answer bound is reached.

    The text names problems by item, identifies acceptable items, and states
    how many more answers the Host will read.
    """
    lines = ["The Host could not accept every part of your answer."]
    for value in problems:
        where = "The answer" if value.item is None else f"Item {value.item}"
        lines.append(f"- {where}: {value.text}")
    if accepted:
        lines.append(
            "Already acceptable as written: items " + ", ".join(str(n) for n in accepted) + "."
        )
    lines.append(
        "Send the whole answer again with the named items fixed or removed. "
        f"{rounds_left} more answer{'s' if rounds_left != 1 else ''} will be read; after the "
        "last, the Host keeps the acceptable items and drops the rest. Leaving an item out is "
        "never an error."
    )
    return "\n".join(lines)


def plain_words(error: ValidationError, items_field: str = "") -> tuple[str, ...]:
    """Describe schema problems in plain words, one per failed field."""
    texts: list[str] = []
    for value in error.errors(include_url=False):
        name = _field_name(tuple(value.get("loc", ())))
        kind = value.get("type", "")
        context = value.get("ctx") or {}
        if kind == "missing":
            text = f"'{name}' is missing."
        elif kind == "extra_forbidden":
            text = f"'{name}' is not a field of an item of '{items_field}'."
        elif kind == "enum":
            text = f"'{name}' must be one of {context.get('expected', 'the listed values')}."
        elif kind == "literal_error":
            text = f"'{name}' must be {context.get('expected', 'one of the listed values')}."
        elif kind == "string_too_short":
            text = f"'{name}' is empty."
        elif kind == "string_too_long":
            text = f"'{name}' is longer than {context.get('max_length')} characters; shorten it."
        elif kind == "too_short":
            minimum = context.get("min_length")
            text = f"'{name}' needs at least {minimum} entr{'y' if minimum == 1 else 'ies'}."
        elif kind == "too_long":
            text = f"'{name}' has more than {context.get('max_length')} entries."
        elif kind == "string_type":
            text = f"'{name}' must be text."
        elif kind in {"tuple_type", "list_type"}:
            text = f"'{name}' must be a list."
        else:
            text = f"'{name}': {value.get('msg', 'invalid')}."
        texts.append(_bounded(text))
    return tuple(dict.fromkeys(texts))


def _field_name(loc: tuple[Any, ...]) -> str:
    names = [str(part) for part in loc if isinstance(part, str)]
    indexes = [part for part in loc if isinstance(part, int)]
    name = ".".join(names) or "item"
    if indexes:
        return f"{name} (entry {indexes[0] + 1})"
    return name


def _normalized(value: Mapping[str, object], model: type[BaseModel]) -> dict[str, object]:
    """Apply deterministic spelling fixes before the schema check.

    Closed values are uppercased with spaces and hyphens made underscores.
    A string where a sequence is expected becomes a one-item list; no other
    guesses are made.
    """
    result = dict(value)
    for name, info in model.model_fields.items():
        if name not in result:
            continue
        written = result[name]
        annotation = info.annotation
        if isinstance(written, str) and _enumeration(annotation) is not None:
            result[name] = written.strip().upper().replace(" ", "_").replace("-", "_")
        elif isinstance(written, str) and _is_sequence(annotation):
            result[name] = [written]
    return result


def _enumeration(annotation: object) -> type[StrEnum] | None:
    for candidate in (annotation, *typing.get_args(annotation)):
        if isinstance(candidate, type) and issubclass(candidate, StrEnum):
            return candidate
    return None


def _is_sequence(annotation: object) -> bool:
    origin = typing.get_origin(annotation)
    if origin in {tuple, list}:
        return True
    if origin in {typing.Union, types.UnionType}:
        return any(_is_sequence(value) for value in typing.get_args(annotation))
    return False


def _bounded(text: str) -> str:
    return text if len(text) <= 400 else text[:397] + "..."


__all__ = [
    "ANSWER_CORRECTION_BOUND",
    "MAXIMUM_ANSWER_COUNT",
    "SPECIALIST_REFERENCE_CHARACTERS",
    "SPECIALIST_REFERENCE_COUNT",
    "SPECIALIST_TEXT_CHARACTERS",
    "AgentAnswerRecord",
    "AgentRun",
    "AnswerProblem",
    "AnswerVerdict",
    "ScreenedAnswer",
    "SpecialistAnswer",
    "answer_digest",
    "answer_slot",
    "answer_verdict",
    "correction_text",
    "plain_words",
    "screen_answer",
    "screen_specialist_answer",
]
