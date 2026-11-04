"""Filed review-thread outcomes and their exact keyed replay."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from datetime import datetime
from typing import Any, Protocol

from alphalattice.kernel.shared_kernel.identity import canonical_hash


class AnswerThread(Protocol):
    """The stored thread messages used to identify a filed answer."""

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Return the filed messages on this thread."""
        ...


def answer_replay(
    floor: AnswerThread, member: tuple[str, str, Any]
) -> tuple[str, dict[str, Any]] | None:
    """Read a keyed answer's first filed response, never infer an unreferenced legacy correction.

    Queue-full and busy refusals file no outcome, so the same request is evaluated again.
    """
    role, _reference, answer = member
    digest = str(canonical_hash(answer))
    said = {message["id"]: message for message in floor.messages}
    for verdict in floor.messages:
        if verdict.get("kind") not in {"CORRECTION", "ACCEPTANCE"}:
            continue
        original = said.get(verdict.get("reply_to"))
        reference = verdict.get("correction_reference") or verdict.get("answer_reference")
        response = verdict.get("answer_response")
        if (
            isinstance(original, Mapping)
            and original.get("role") == role
            and original.get("answer_digest") == digest
            and isinstance(reference, str)
            and isinstance(response, dict)
        ):
            return reference, dict(response)
    return None


def legacy_acceptance(floor: AnswerThread, member: tuple[str, str, Any]) -> tuple[str, str] | None:
    """Find an old accepted receipt by its exact answer and Task; never invent its lifecycle."""
    role, _reference, answer = member
    digest = str(canonical_hash(answer))
    said = {message["id"]: message for message in floor.messages}
    for verdict in floor.messages:
        original = said.get(verdict.get("reply_to"))
        references = verdict.get("references")
        if (
            verdict.get("kind") == "ACCEPTANCE"
            and "answer_response" not in verdict
            and isinstance(original, Mapping)
            and original.get("role") == role
            and original.get("answer_digest") == digest
            and isinstance(verdict.get("answer_reference"), str)
            and isinstance(references, list)
            and len(references) >= 2
            and references[0] == verdict["answer_reference"]
            and isinstance(references[-1], str)
        ):
            return verdict["answer_reference"], references[-1]
    return None


def file_answer(
    store: Any,
    floor: Any,
    member: Any,
    body: Any,
    now: datetime,
    file: Any,
    *,
    correction_reference: str | None,
    floor_type: Any,
    speaker: Any,
    accepted: Any,
    append: Any,
) -> None:
    """The member's answer and the Host's verdict on it, on the thread `admit` read."""
    (role, _reference, answer), status = member, body.get("status")
    if not isinstance(floor, floor_type) or status not in {"CORRECT", "ACCEPTED", "DONE"}:
        return
    # Filed on the bundle's thread in the lead's session, the member its author, whoever sent it.
    session, thread = floor.opened["session"], floor.opened["update_task_id"]
    own = {"author_session": floor.author_session} if floor.author_session else {}
    body["conversation"] = {"status": "RECORDED_ON_THREAD", "thread": thread, "host": session[0]}
    body["conversation"] |= {
        "session_id": session[1],
        "author": speaker(floor, own | {"role": role}),
    }
    with store.lock:
        if accepted(floor) is not None:
            return  # the accepted answer read again: its receipt, said once
        last = floor.messages[-1]["id"] if floor.messages else None  # its issue, or correction
        said = {"role": role, "kind": "ANSWER", "text": "Answer submitted.", "reply_to": last}
        said["answer_digest"] = str(canonical_hash(answer))
        said |= own
        verdict: dict[str, Any] = {"role": "HOST", "reply_to": append(store, floor, now, **said)}
        if status == "CORRECT":
            problems, left = len(body.get("problems") or ()), body.get("rounds_left")
            text = f"Correct the {problems} named items; {left} rounds left."
            if correction_reference is not None:
                verdict |= {
                    "correction_reference": correction_reference,
                    "answer_response": dict(body),
                }
            append(store, floor, now, kind="CORRECTION", text=text, **verdict)
        else:
            refs = [str(body[k]) for k in ("answer_reference", "task_id") if body.get(k)]
            text = "Accepted." if status == "ACCEPTED" else "Rounds spent; the valid part stands."
            verdict |= {"outcome": status, "references": refs}
            if isinstance(body.get("answer_reference"), str):
                verdict["answer_response"] = dict(body)
            answer_reference = body.get("answer_reference")
            append(
                store,
                floor,
                now,
                kind="ACCEPTANCE",
                text=text,
                answer_reference=answer_reference,
                **verdict,
            )
    with suppress(ValueError, OSError):
        file(floor.opened["update_task_id"], now)
