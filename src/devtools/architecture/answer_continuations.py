"""Generated continuation checks for the operation and answer registries (V523).

Source offers are parsed, not executed. Recorded answers are replayed through the client
without a Host: only edge names, fields, classifications and counts leave the replay.
"""

from __future__ import annotations

import ast
import json
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


def reference_checks(root: Path) -> list[dict[str, Any]]:
    """Check the compact display's round trip by the request contract (V523, V561).

    An issued reference, a field the request contract types or a hash, is shown short and read
    back whole under its own name. An answer's own id that the contract does not take is shown
    short and read back whole in the typed request field a reader sends it through: a UUID
    through `task_id`, a hash through `result_hash`, a prefixed entry through
    `history_entry_id`. An authored request field the contract leaves untyped is shown and sent
    as written, and a value of a short reference's shape in it is never rewritten.
    """
    from alphalattice.interface.local_application.client import (
        REFERENCE_LEDGER_NAME,
        short_references,
        whole_references,
    )

    base = root / "src/alphalattice/interface/local_application"
    operations = json.loads((base / "operations.json").read_text(encoding="utf-8"))
    answers = json.loads((base / "answers.json").read_text(encoding="utf-8"))
    typed = set(operations["references"])
    requested = set(operations["types"])
    fields = set(typed)
    fields.update(
        field["name"]
        for row in answers.values()
        for field in row["fields"]
        if field["name"].endswith(("_id", "_hash", "_ids", "_hashes"))
    )
    fields.update(name for name in requested if name.endswith(("_id", "_hash", "_ids", "_hashes")))
    values = ("12345678-1234-4321-8123-123456789abc", "abcdef0123456789" * 4)
    authored_value = "fedcba987654"
    rows = []
    with tempfile.TemporaryDirectory(prefix="s1-reference-") as scratch:
        workspace = Path(scratch)
        (workspace / "runtime").mkdir()
        ledger = (*values, authored_value + "0" * 52)
        (workspace / "runtime" / REFERENCE_LEDGER_NAME).write_text(
            "".join(f"{value}\n" for value in ledger), encoding="utf-8", newline="\n"
        )
        for field in sorted(fields):
            issued = field in typed or field.endswith(("_hash", "_hashes"))
            role = "issued" if issued else "authored" if field in requested else "answer"
            listed = field.endswith(("_ids", "_hashes"))
            for prefix in ("", "experiment:"):
                for value in values:
                    original: Any = [prefix + value] if listed else prefix + value
                    shown = short_references({field: original})
                    if role == "issued":
                        whole_references(shown, workspace)
                        bound = shown[field] == original
                    elif role == "authored":
                        sent = dict(shown)
                        whole_references(sent, workspace)
                        kept: Any = [authored_value] if listed else authored_value
                        written = {field: kept}
                        whole_references(written, workspace)
                        bound = shown[field] == original == sent[field] and written[field] == kept
                    else:
                        carrier = (
                            "history_entry_id"
                            if prefix
                            else "task_id"
                            if "-" in value
                            else "result_hash"
                        )
                        items = shown[field] if listed else [shown[field]]
                        carried = []
                        for item in items:
                            request = {carrier: item}
                            whole_references(request, workspace)
                            carried.append(request[carrier])
                        bound = carried == (original if listed else [original])
                    rows.append(
                        {
                            "field": field,
                            "role": role,
                            "form": "prefixed" if prefix else "whole",
                            "kind": "uuid" if "-" in value else "hash",
                            "class": "BOUND" if bound else "FAIL",
                        }
                    )
    return rows


def continuation_checks(root: Path) -> list[dict[str, Any]]:
    """Every required reference binds from its name/alias or refuses an absent answer."""
    from alphalattice.interface.local_application.cli_contract import client_refusal
    from alphalattice.interface.local_application.client import (
        LOCATORS,
        LocalResearchClientError,
        continued,
        reference_field,
    )

    table = json.loads(
        (root / "src/alphalattice/interface/local_application/operations.json").read_text(
            encoding="utf-8"
        )
    )
    rows = []
    for operation, contract in table["fields"].items():
        references = {name for name in contract["allowed"] if reference_field(name)}
        required = references & set(contract["required"])
        for name in sorted(references):
            if operation == "GOAL_OPEN" and name == "goal_id":
                # GOAL_OPEN creates its target; the field is context, not an inherited
                # target. This is the client's existing new-goal rule (OP17).
                continue
            for alias in LOCATORS.get(name, (name,)):
                bundle = {field: "abcdef0123456789" * 4 for field in required}
                bundle.update(
                    {
                        field: "12345678-1234-4321-8123-123456789abc"
                        for field in ("task_id", "goal_id")
                        if field in contract["allowed"]
                    }
                )
                bundle.pop(name, None)
                bundle[alias] = "abcdef0123456789" * 4
                try:
                    request = continued(operation, bundle, {}, frozenset(contract["allowed"]))
                    result = "BOUND" if request.get(name) == bundle[alias] else "FAIL"
                    reason = None
                except LocalResearchClientError as error:
                    reason = str(error).partition(":")[0]
                    result = "REFUSED" if client_refusal(str(error)).detail else "FAIL"
                rows.append(
                    {
                        "to": operation,
                        "field": name,
                        "alias": alias,
                        "class": result,
                        "reason": reason,
                    }
                )
        if required:
            try:
                continued(operation, {}, {}, frozenset(contract["allowed"]))
                result, reason = "FAIL", "missing reference accepted"
            except LocalResearchClientError as error:
                result = "REFUSED" if client_refusal(str(error)).detail else "FAIL"
                reason = str(error).partition(":")[0]
            rows.append(
                {
                    "to": operation,
                    "field": ",".join(sorted(required)),
                    "alias": None,
                    "class": result,
                    "reason": reason,
                }
            )
    return rows


def source_offers(root: Path) -> list[dict[str, Any]]:
    """All literal operation requests, with their owner and offered reference fields.

    The operation-registry check uses the same literal requests. Dynamic expressions are
    retained as evidence, and observed offers are additionally held at the answer boundary.
    """
    table = json.loads(
        (root / "src/alphalattice/interface/local_application/operations.json").read_text(
            encoding="utf-8"
        )
    )
    rows = []
    sources = []
    for path in sorted((root / "src/alphalattice").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        sources.append((path, text, ast.parse(text)))
    for path, _text, tree in sources:
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            fields = {
                key.value: value
                for key, value in zip(node.keys, node.values, strict=True)
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            op = fields.get("operation")
            if op is None:
                continue
            owner = parents.get(node)
            while owner is not None and not isinstance(
                owner, ast.FunctionDef | ast.AsyncFunctionDef
            ):
                owner = parents.get(owner)
            destinations: list[str] = (
                [str(op.value)]
                if isinstance(op, ast.Constant)
                else sorted(
                    {
                        item.value
                        for item in ast.walk(owner or tree)
                        if isinstance(item, ast.Constant)
                        and isinstance(item.value, str)
                        and item.value in table["fields"]
                    }
                )
            )
            # A helper's operation parameter is bound by its callers (e.g. bound() in
            # Evidence, reused_read(), or a Literal annotation in remediation).
            if not isinstance(op, ast.Constant) and owner is not None:
                for _caller, text, caller_tree in sources:
                    if owner.name not in text:
                        continue
                    for call in ast.walk(caller_tree):
                        if isinstance(call, ast.Call) and (
                            (isinstance(call.func, ast.Name) and call.func.id == owner.name)
                            or (
                                isinstance(call.func, ast.Attribute)
                                and call.func.attr == owner.name
                            )
                        ):
                            destinations += [
                                item.value
                                for item in ast.walk(call)
                                if isinstance(item, ast.Constant)
                                and isinstance(item.value, str)
                                and item.value in table["fields"]
                            ]
            destinations = sorted(set(destinations) & set(table["fields"]))
            if not destinations:
                # Generic request/answer transformers take the operation from the registry
                # or an observed answer, whose boundary check holds every concrete edge.
                rows.append(
                    {
                        "from": f"{path.relative_to(root).as_posix()}:{node.lineno}",
                        "owner": owner.name if owner else "module",
                        "to": "REGISTRY_DISPATCH",
                        "fields": sorted(set(fields) - {"operation"}),
                        "required": [],
                        "choices": [],
                        "spread": True,
                        "operation_expression": ast.unparse(op),
                    }
                )
            for destination in destinations:
                contract = table["fields"][destination]
                rows.append(
                    {
                        "from": f"{path.relative_to(root).as_posix()}:{node.lineno}",
                        "owner": owner.name if owner else "module",
                        "to": destination,
                        "fields": sorted(set(fields) - {"operation"}),
                        "required": contract["required"],
                        "choices": sorted(
                            name
                            for name in contract["required"]
                            if name not in fields
                            or (
                                isinstance(value := fields[name], ast.Constant)
                                and value.value is None
                            )
                        ),
                        "spread": any(key is None for key in node.keys),
                        "operation_expression": None
                        if isinstance(op, ast.Constant)
                        else ast.unparse(op),
                    }
                )
    return rows


def documented_edges(root: Path) -> list[dict[str, Any]]:
    """Saved-output to --from chains in the Skill commands and its references."""
    from scripts.materialize_claude_host import SKILL_COMMANDS

    from alphalattice.interface.local_application.cli_contract import client_refusal
    from alphalattice.interface.local_application.client import (
        LocalResearchClientError,
        continued,
        reference_field,
    )

    table = json.loads(
        (root / "src/alphalattice/interface/local_application/operations.json").read_text(
            encoding="utf-8"
        )
    )
    answers = json.loads(
        (root / "src/alphalattice/interface/local_application/answers.json").read_text(
            encoding="utf-8"
        )
    )
    commands = [
        ("scripts/materialize_claude_host.py:SKILL_COMMANDS", c.form) for c in SKILL_COMMANDS
    ]
    for path in sorted((root / ".agents/skills/alphalattice-research/references").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        commands += [
            (path.relative_to(root).as_posix(), " ".join(m.group(1).split()))
            for m in re.finditer(r"`([^`]+)`", text)
            if "--from" in m.group(1) or "--output" in m.group(1)
        ]

    def operations(command: str) -> list[str]:
        # A reference may print the full Python launcher and workspace prefix. The
        # registered command still owns its operation; the first two words need not.
        found = {
            operation
            for name, declared in table["commands"].items()
            if re.search(r"(?<![\w-])" + re.escape(name) + r"(?![\w-])", command)
            for operation in declared
        }
        return sorted(found)

    producers: dict[str, set[str]] = {}
    for _, command in commands:
        output = re.search(r"--output\s+[\"\']?([^\s\"\']+)", command)
        if output:
            producers.setdefault(output[1].rsplit("/", 1)[-1], set()).update(operations(command))
    rows = []
    for source, command in commands:
        saved = re.search(r"--from\s+[\"\']?([^\s\"\']+)", command)
        if not saved:
            continue
        destinations = operations(command)
        for destination in destinations or ["request --action"]:
            origins = sorted(producers.get(saved[1].rsplit("/", 1)[-1], set()))
            contract = table["fields"].get(destination, {})
            required = [name for name in contract.get("required", []) if reference_field(name)]
            probes = []
            for origin in origins:
                if destination not in table["fields"]:
                    continue
                # This tests declared top-level locators, not fabricated nested offers.
                # Runtime offers and real saved answers are checked separately.
                bundle = {
                    field["name"]: "abcdef0123456789" * 4
                    for field in answers.get(origin, {}).get("fields", [])
                    if reference_field(field["name"])
                }
                try:
                    request = continued(destination, bundle, {}, frozenset(contract["allowed"]))
                    missing = [name for name in required if not request.get(name)]
                    probes.append(
                        {"from": origin, "class": "FAIL" if missing else "BOUND", "reason": None}
                    )
                except LocalResearchClientError as error:
                    probes.append(
                        {
                            "from": origin,
                            "class": "REFUSED" if client_refusal(str(error)).detail else "FAIL",
                            "reason": str(error).partition(":")[0],
                        }
                    )
            rows.append(
                {
                    "source": source,
                    "from": origins,
                    "to": destination,
                    "command": command,
                    "required": contract.get("required", []),
                    "reference_probes": probes,
                    "selection": "named offered action"
                    if destination == "request --action"
                    else ("declared answer" if origins else "reader-selected saved answer"),
                }
            )
    return rows


def recorded_edges(root: Path, evidence: Path) -> dict[str, Any]:
    """Replay real saved answers without sending requests or retaining private values."""
    from alphalattice.interface.local_application.cli_contract import (
        choices,
        client_refusal,
        named_read,
        offered_requests,
        request_problem,
    )
    from alphalattice.interface.local_application.client import LocalResearchClientError, continued

    table = json.loads(
        (root / "src/alphalattice/interface/local_application/operations.json").read_text(
            encoding="utf-8"
        )
    )
    counts: Counter[tuple[str, str, str, str]] = Counter()
    files, answers = 0, 0
    for path in sorted(evidence.rglob("native-answers.json")):
        files += 1
        recorded = json.loads(path.read_text(encoding="utf-8"))
        for row in recorded if isinstance(recorded, list) else recorded.get("answers", []):
            answer = row.get("answer", {})
            body = answer.get("data")
            operation = str(answer.get("operation"))
            answers += 1
            if not isinstance(body, dict):
                continue
            offers = offered_requests(body)
            for request in offers.values():
                destination = str(request.get("operation"))
                if destination not in table["fields"]:
                    counts[(operation, destination, "HISTORICAL", "operation retired")] += 1
                    continue
                selected = {**body, "next_requests": {"selected": request}}
                # Do not retain other nested offers when selecting one edge for replay.
                selected = {
                    key: value
                    for key, value in selected.items()
                    if not isinstance(value, dict | list) or key == "next_requests"
                }
                try:
                    rebuilt = continued(
                        destination,
                        selected,
                        {},
                        frozenset(table["fields"][destination]["allowed"]),
                    )
                    problem = request_problem(rebuilt)
                    changed = [
                        name
                        for name, value in request.items()
                        if value is not None and rebuilt.get(name) != value
                    ]
                    if changed:
                        result, reason = "FAIL", "bound fields lost"
                    elif problem:
                        # An unchanged old offer can fail today's request contract. Keep
                        # that historical fact without copying its payload or pretending
                        # the continuation algorithm lost a field. The current answer
                        # boundary refuses the same malformed offer by name and words.
                        original_problem = request_problem(request)
                        result = "HISTORICAL_INVALID" if original_problem else "FAIL"
                        reason = "recorded offer fails the current request contract"
                    else:
                        result, reason = "BOUND", ""
                except LocalResearchClientError as error:
                    result = "CHOICE" if choices(request) else "REFUSED"
                    reason = str(error).partition(":")[0]
                    if not client_refusal(str(error)).detail:
                        result = "FAIL"
                counts[(operation, destination, result, str(reason))] += 1
            if operation in table["reads"]:
                # Older receipts may predate named_read. Explicitly keep the recorded
                # request when present; never infer a selection from a different answer.
                request = row.get("request")
                named = named_read(request, body) if isinstance(request, dict) else body
                try:
                    continued(
                        operation, named, {}, frozenset(table["fields"][operation]["allowed"])
                    )
                    result, reason = "BOUND", ""
                except LocalResearchClientError as error:
                    result, reason = "REFUSED", str(error).partition(":")[0]
                counts[(operation, operation, result, reason)] += 1
    return {
        "files": files,
        "answers": answers,
        "edges": [
            {"from": a, "to": b, "class": c, "reason": reason, "count": count}
            for (a, b, c, reason), count in sorted(counts.items())
        ],
    }
