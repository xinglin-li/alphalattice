"""One admitted-snapshot decision across research planning, execution and readback."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_every_research_snapshot_reuse_reader_calls_the_cover_owner():
    """V599/TE12: pin the consumer set, including receipt verification."""
    callers = set()
    for path in (ROOT / "src/alphalattice").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for owner in tree.body:
            if not isinstance(owner, ast.ClassDef):
                continue
            for method in owner.body:
                if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if any(
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "historical_inputs_cover"
                    for node in ast.walk(method)
                ):
                    callers.add((path.relative_to(ROOT).as_posix(), owner.name, method.name))
    filename = "src/alphalattice/control/product_host/composition/decision_advancement.py"
    assert callers == {
        (filename, "DecisionAdvancementApplication", name)
        for name in ("_plan_body", "_execute", "_verify_products")
    }
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    # The three consumers must not independently compare the held bounds.
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            assert not any(
                isinstance(value, ast.Attribute)
                and value.attr in {"data_through", "adjusted_through", "panel_through"}
                for value in ast.walk(node)
            ), ast.unparse(node)


def test_planning_and_both_network_gates_read_one_provider_work_owner():
    """V599/TE12: no unconditional gate can return beside the common decision."""
    callers = set()
    for path in (ROOT / "src/alphalattice/control/product_host").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for owner in tree.body:
            if isinstance(owner, ast.ClassDef):
                for method in owner.body:
                    if isinstance(method, ast.FunctionDef) and any(
                        isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Attribute)
                        and n.func.attr == "network_work"
                        for n in ast.walk(method)
                    ):
                        callers.add((owner.name, method.name))
    assert callers == {
        ("WorkspaceDataUpdateApplication", "historical_inputs_cover"),
        ("WorkspaceDataUpdateApplication", "source_access"),
        # The plan's answer, sealed fresh or the data update's that has not ended (V604).
        ("WorkspaceDataUpdateApplication", "_plan_answer"),
        ("WorkspaceDataUpdateApplication", "admit"),
        ("WorkspaceDataUpdateApplication", "_step"),
        ("DecisionAdvancementApplication", "_plan_body"),
        ("DecisionAdvancementApplication", "_execute"),
    }
    path = ROOT / "src/alphalattice/control/product_host/maintenance/data_update.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    gates = {
        method.name: method
        for owner in tree.body
        if isinstance(owner, ast.ClassDef)
        for method in owner.body
        if isinstance(method, ast.FunctionDef)
        and method.name != "resume_stopped"  # V600 checks a kept Task's already named stop.
        and any(
            isinstance(n, ast.Constant)
            and n.value == "workspace_data_update.source_access_not_admitted"
            for n in ast.walk(method)
        )
    }
    assert set(gates) == {"admit", "_step"}
    for method in gates.values():
        assert any(
            isinstance(n, ast.If)
            and isinstance(n.test, ast.BoolOp)
            and isinstance(n.test.op, ast.And)
            and any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "network_work"
                for call in ast.walk(n.test)
            )
            for n in ast.walk(method)
        ), method.name


def test_both_plan_labels_delegate_to_the_gate_owners_provider_boundary():
    root = ROOT / "src/alphalattice/control/product_host"
    owners = {
        "maintenance/data_update.py": ("WorkspaceDataUpdateApplication", "_plan_answer"),
        "composition/decision_advancement.py": ("DecisionAdvancementApplication", "_plan_body"),
    }
    for filename, (owner, method) in owners.items():
        tree = ast.parse((root / filename).read_text(encoding="utf-8"))
        producer = next(
            n
            for c in tree.body
            if isinstance(c, ast.ClassDef) and c.name == owner
            for n in c.body
            if isinstance(n, ast.FunctionDef) and n.name == method
        )
        assert any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "source_access"
            for n in ast.walk(producer)
        )
    tree = ast.parse((root / "maintenance/data_update.py").read_text(encoding="utf-8"))
    readers = {
        n.name
        for c in tree.body
        if isinstance(c, ast.ClassDef)
        for n in c.body
        if isinstance(n, ast.FunctionDef)
        and any(
            isinstance(v, ast.Attribute) and v.attr == "_provider_requires_network"
            for v in ast.walk(n)
        )
    }
    assert readers == {"source_access", "_source_access_admitted"}


def test_materialization_and_readiness_use_the_same_inclusive_bound_meaning():
    """V599: complementary missing/ready checks never turn equal into missing."""
    root = ROOT / "src/alphalattice/control/data_platform"
    paths = {
        "maintenance/coordinator.py": {
            "candidate_raw_through < request.target_market_session",
            "candidate_adjusted_through < request.target_market_session",
            "active_panel['as_of_session'] >= request.target_market_session",
            "receipt.covered_through_session < request.target_market_session",
        },
        "maintenance/reconciliation.py": {"self.panel_as_of_session < self.target_market_session"},
        "readiness.py": {"_utc(last_checked_at).date() < reference_session"},
    }
    for filename, wanted in paths.items():
        tree = ast.parse((root / filename).read_text(encoding="utf-8"))
        found = {ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Compare)}
        assert wanted <= found, (filename, wanted - found)


def test_held_child_and_candidate_parent_read_the_same_inclusive_bound_rule():
    tree = ast.parse(
        (ROOT / "src/alphalattice/control/product_host/maintenance/data_update.py").read_text(
            encoding="utf-8"
        )
    )
    readers = {
        method.name
        for owner in tree.body
        if isinstance(owner, ast.ClassDef)
        for method in owner.body
        if isinstance(method, ast.FunctionDef)
        and any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "_bound_covers"
            for n in ast.walk(method)
        )
    }
    assert readers == {"_held_lanes", "_candidate_source_covers"}
