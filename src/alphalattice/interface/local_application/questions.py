"""Registered person-question words and reply metadata for the CLI and Host boundary."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
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


QUESTION_SCOPE_FIELDS = ("goal_id", "agent_session")


def _question_scope(refs: Mapping[str, Any], subject: Mapping[str, Any]) -> tuple[str, str] | None:
    scopes = {field: refs.get(field, subject.get(field)) for field in QUESTION_SCOPE_FIELDS}
    for field, value in scopes.items():
        if field in refs and field in subject and refs[field] != subject[field]:
            return None
        if value is None:
            continue
        if not isinstance(value, str) or not value:
            return None
    return next(((field, value) for field, value in scopes.items() if value is not None), None)


def _observed_question(
    record: Any, subject: Mapping[str, Any], payload: Mapping[str, Any], row: Mapping[str, Any]
) -> dict[str, Any] | None:
    if not isinstance(record, dict) or not isinstance(record.get("references"), dict):
        return None
    refs = record["references"]
    scope = _question_scope(refs, subject)
    code, digest = record.get("code"), record.get("catalog_hash")
    if (
        not isinstance(code, str)
        or code not in _catalog()
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or scope is None
        or record.get("state") not in {"ASKED", "ANSWERED"}
        or not isinstance(row.get("correlation_ids"), list | tuple)
        or not isinstance(refs.get("operation_ref"), str)
        or not refs["operation_ref"]
        or refs["operation_ref"] != payload.get("operation_ref")
        or refs["operation_ref"] not in row["correlation_ids"]
    ):
        return None
    return dict(
        code=code,
        catalog_hash=digest,
        state=record["state"],
        references=dict(refs),
        scope_field=scope[0],
        scope_id=scope[1],
    )


def current_person_questions(
    observations: Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Latest state per (code, Goal), else session; a later ASKED reopens that scope.

    Recorded catalog hashes are provenance, not today's definition; this grants no authority.
    """
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    invalid: list[dict[str, Any]] = []
    for ordinal, row in enumerate(observations):
        payload = row.get("payload") or {}
        if (
            row.get("schema_kind"),
            row.get("source_kind"),
            row.get("authority"),
            payload.get("phase"),
        ) != ("ProductOperationObserved", "PRODUCT_OPERATION", "OPERATIONAL_ASSERTION", "RETURNED"):
            continue
        subject = payload.get("subject") or {}
        records = subject.get("person_questions", ())
        if not isinstance(records, list | tuple):
            invalid.append({"ordinal": ordinal})
            continue
        for record in records:
            projected = _observed_question(record, subject, payload, row)
            if projected is None:
                invalid.append({"ordinal": ordinal})
                continue
            projected["ordinal"] = ordinal
            latest[(projected["code"], projected["scope_field"], projected["scope_id"])] = projected
    questions = list(latest.values())
    return {
        "questions": questions,
        "pending": [q for q in questions if q["state"] == "ASKED"],
        "invalid": invalid,
    }


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
