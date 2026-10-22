"""Registered person-question words and reply metadata for the CLI and Host boundary."""

from __future__ import annotations

from functools import cache
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.project_layout import runtime_json


@cache
def _catalog() -> dict[str, Any]:
    return {
        code: {**entry, "catalog_hash": canonical_hash(entry)}
        for code, entry in runtime_json("config/registries/messages.json")["entries"].items()
        if entry["kind"] == "PERSON_QUESTION"
    }


def person_question(code: str, *, operation: str | None = None) -> dict[str, Any]:
    """Read an explicit catalog code; a reply code cannot cover another operation."""
    entry = _catalog().get(code)
    if entry is None:
        raise ValueError("person_confirmation.question_unknown")
    if operation is not None and operation not in entry["operations"]:
        raise ValueError("person_confirmation.question_mismatch")
    return {
        "code": code,
        "catalog_hash": entry["catalog_hash"],
        "words": entry["words"]["en"],
        "words_zh": entry["words"]["zh-CN"],
        "scope": dict(entry["scope"]),
        "way_on": dict(entry["way_on"]),
    }


def asked_questions(operation: str, proposed: object) -> list[dict[str, Any]]:
    """Resolve catalog-declared Host producers; foreign return fields remain ordinary data."""
    codes = (
        {
            item.get("code")
            for item in proposed
            if isinstance(item, dict) and isinstance(item.get("code"), str)
        }
        if isinstance(proposed, list)
        else set()
    )
    return [
        person_question(code)
        for code, entry in _catalog().items()
        if code in codes and operation in entry["ask_operations"]
    ]


class PersonConfirmation(BaseModel):  # type: ignore[misc]
    """The person's yes to one of their decisions, asked in one line and relayed whole.

    The Host cannot prove who typed the words, so it keeps them as they were said for audit and
    revocation. `decision_hash` binds them to the one request they answer, and each is used
    once; the Host's clock dates it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    question: str = Field(min_length=1, max_length=600)
    """The one-line question the agent asked, as it asked it."""
    words: str = Field(min_length=1, max_length=600)
    """The person's answer, verbatim, never paraphrased."""
    decision_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    """The request it answers, without this confirmation."""
    nonce: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    """For the person's next yes to a decision already made: the nonce its refusal named."""
    question_code: str | None = None
    """The offered catalog code; None explicitly retains an uncoded legacy reply."""
