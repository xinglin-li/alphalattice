"""Shared refusal inputs for the structural, CLI and Workbench consumers."""

from __future__ import annotations

import ast
import itertools
import json
import re
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/alphalattice"


@cache
def door_words():
    return json.loads(
        (SOURCE / "interface/local_application/refusal_words.json").read_text("utf-8")
    )


def fill_subject(text, subject="SUBJECT-7f3a", *, numbered=False):
    index = iter(range(text.count("{subject}")))
    return re.sub(
        r"\{subject\}",
        lambda _match: subject + "-" + str(next(index)) if numbered else subject,
        text,
    )


@cache
def refusal_codes():
    from alphalattice.control.product_host.composition.plain_refusals import SOURCE_SHORT_CODES

    def spellings(node: ast.expr) -> set[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return {node.value}
        if isinstance(node, ast.JoinedStr):
            head = itertools.takewhile(lambda part: isinstance(part, ast.Constant), node.values)
            return {"".join(str(part.value) for part in head) + "subject"}
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return {a + b for a in spellings(node.left) for b in spellings(node.right)}
        if isinstance(node, ast.IfExp):
            return spellings(node.body) | spellings(node.orelse)
        return {"subject"}

    registered = set(
        json.loads((ROOT / "config/registries/refusals.json").read_text("utf-8"))["entries"]
    )
    client = {
        code
        for path in SOURCE.rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", None)) == "LocalResearchClientError"
        and node.args
        for code in spellings(node.args[0])
    }
    writer = {
        item.value.value
        for path in (SOURCE / "control/task_control").glob("*.py")
        for item in ast.walk(ast.parse(path.read_text("utf-8")))
        if isinstance(item, ast.keyword)
        and item.arg == "failure_code"
        and isinstance(item.value, ast.Constant)
        and isinstance(item.value.value, str)
    }
    tree = ast.parse((SOURCE / "control/guanyin/data/workspace_maintenance.py").read_text("utf-8"))
    owner = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "maintenance_failure_detail"
    )
    maintenance = next(
        set(ast.literal_eval(n.value))
        for n in owner.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "details" for t in n.targets)
    )
    maintenance_dict_keys = [
        key.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    ]
    evidence = {
        code
        for path in (SOURCE / "evidence/alternative_evidence").rglob("*.py")
        for code in re.findall(r'"(alternative_evidence\.[a-z_]+)', path.read_text("utf-8"))
    }
    units = {
        *evidence,
        "alternative_evidence.task_failed:knowledge.semantic_pack_not_pinned",
        *(code + ":0 of 78 issuers hold a source, 47 needed" for code in SOURCE_SHORT_CODES),
        "an_owner." + "x" * (120 - len("an_owner.")),
    }
    read_whole = [
        *("alternative_evidence.unit_not_prepared:" + code for code in sorted(units)),
        *(
            fill_subject(f"portfolio_research.benchmark_support_absent:{cause}:{{subject}}")
            for cause in ("returns_unavailable", "no_eligible_name")
        ),
    ]
    codes = (
        registered | set(door_words()) | client | writer | maintenance | evidence | set(read_whole)
    )
    open_family = {
        "TASK_STAGE_RAISED:MemoryError",
        "TASK_RECOVERY_EVIDENCE_INVALID:recovery_prefix_invalid:stage-1",
        "AUTHORITY_FAILURE:an_owner.unknown_case",
        "OPERATIONAL_FAILURE:an_owner.unknown_case",
        "an_owner." + "x" * (120 - len("an_owner.")),
    }
    codes.update(open_family)
    return {
        "all": codes,
        "registered": registered,
        "client": client,
        "writer": writer,
        "maintenance": maintenance,
        "maintenance_dict_keys": maintenance_dict_keys,
        "evidence": evidence,
        "read_whole": read_whole,
        "open_family": open_family,
    }
