"""Bounded representations of local-application answers, emitted by the final serializer."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from alphalattice.interface.local_application import cli_contract as contract

if TYPE_CHECKING:
    from alphalattice.interface.local_application import client


def emit_owner(
    args: argparse.Namespace,
    result: dict[str, Any],
    *,
    output_error: client.LocalResearchClientError | None = None,
    replace: bool | None = None,
) -> int:
    """Save exact owner JSON and render its selected view with named local file failures."""
    from alphalattice.interface.local_application import client

    output: Path | None = getattr(args, "output", None)
    line: tuple[str, ...] = args.command_line
    indent: int | None = getattr(args, "legacy_indent", None)
    view: str | None = getattr(args, "view", None)
    if output is not None and replace is None and output_error is None:
        try:
            replace = client.output_replacement(output, line=line, prefix=())
        except client.LocalResearchClientError as error:
            output_error, result = error, {}
    if output is not None and output_error is None:
        try:
            raw = json.dumps(result, indent=indent, default=str).encode("utf-8")
            client._write_new(output, raw, replace=bool(replace))
            result.update(client.output_fields(output, replaced=bool(replace)))
        except client.LocalResearchClientError as error:
            output_error = error
    if output_error is not None:
        code = str(output_error)
        refusal = contract.client_refusal(code)
        failure = {
            "failure_code": code,
            "detail": contract.worded(refusal.detail),
            "next_action": refusal.next_action,
            "path": str(output),
            "next_commands": output_error.next_commands,
        }
        result = (
            {**result, "local_failure": failure} if result else {"status": "REFUSED", **failure}
        )
    reading = contract.READ_VIEW.set((view or "compact", line, summary_display))
    try:
        print(  # noqa: T201  # legacy/full serializer
            json.dumps(result, indent=indent, default=str)
            if view is None
            else contract.answer_json(result, default=str)
        )
        return 2 if result.get("status") == "REFUSED" else 0
    finally:
        contract.READ_VIEW.reset(reading)


def summary_display(
    answer: dict[str, Any], *, default: Callable[[object], Any] | None = None
) -> dict[str, Any]:
    """Bound the whole encoded read; keep references and offers as indivisible units."""
    from alphalattice.interface.local_application import client

    def size(value: object) -> int:
        return len(json.dumps(value, default=default).encode("utf-8"))

    body = answer.get("data", answer)
    data = body if isinstance(body, dict) else {}

    def pointer_to_full() -> dict[str, Any]:
        snapshot = data.get("snapshot")
        saved = answer.get("output_file") or (
            snapshot.get("file") if isinstance(snapshot, dict) else None
        )
        line = list(contract.READ_VIEW.get()[1])
        end = line.index("--") if "--" in line else len(line)
        for index, part in enumerate(line[:end]):
            option, equal, _value = part.partition("=")
            if len(option) > 2 and option.startswith("--") and "--view".startswith(option):
                if equal:
                    line[index] = option + "=full"
                else:
                    line[index + 1] = "full"
        reads = contract.command_table()["reads"]
        read = data.get("read_request")
        read = read if isinstance(read, dict) and read.get("operation") in reads else None
        pointer = {"file": saved} if saved else {"request": read} if read else {"command": line}
        if not saved and not read and not line:
            pointer = {"available": False, "local_failure": answer.get("local_failure")}
        if size(pointer) > client.COMPACT_ANSWER_BYTES // 8:
            pointer = {"argument": "--output" if saved else "command_line", "view": "full"}
        return pointer

    pointer = pointer_to_full()
    shown: dict[str, Any] = {
        "representation": "alphalattice.read-summary.v1",
        "message_id": "read.summary",
        "summary_detail": str(contract.worded("read.summary")).format(pointer="full_answer"),
        "full_answer": pointer,
    }
    views = ("next_requests", "next_commands", "next_templates")
    skip = (*views, "data", *shown)
    for key, value in answer.items():
        if key not in skip and size(shown | {key: value}) < client.COMPACT_ANSWER_BYTES // 2:
            shown[key] = value
    shown.update({key: answer.get(key) for key in views})
    client._hold_next(shown, client.COMPACT_NEXT_BYTES, strict=True)
    if left := shown.get("next_left"):
        shown["next_page"] = {"full_answer": pointer, **left}
    shown["data"] = {}
    for key, value in sorted(
        data.items(),
        key=lambda item: (
            item[0] != "person_questions",
            item[0] not in ("status", "failure_code", "detail", "next_action"),
            item[0] not in client._IMPORTANT,
        ),
    ):
        candidate = {**shown["data"], key: value}
        if key not in views and size({**shown, "data": candidate}) < client.COMPACT_ANSWER_BYTES:
            shown["data"] = candidate
    return shown
