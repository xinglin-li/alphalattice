"""CONTRACT: every composed Task admission reaches the recovery re-plan view.

Discover kinds from input construction, independently of re-plan declarations.
Unknown dynamic kind expressions require an explicit discovery update, never a
silent omission. Imported envelope factories count alongside in-module ones.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from typing import get_args, get_type_hints
from uuid import UUID

from alphalattice.control.product_host.composition.local_web_session import (
    HANDLED_OPERATION_ROUTES,
    OPERATION_ROUTES,
)
from alphalattice.control.product_host.composition.portfolio_research_operations import (
    PortfolioResearchOperations,
)
from alphalattice.control.task_control.contracts import TaskLifecycle, TaskReplan
from alphalattice.interface.local_application.cli_contract import command_table
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchOperationRequest,
    PortfolioResearchRequestDocument,
)


def _kinds(source: str, namespace: dict[str, object]) -> set[str]:
    kinds = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or ast.unparse(node.func) != "TaskInputEnvelope.create":
            continue
        expression = next(item.value for item in node.keywords if item.arg == "task_kind")
        if isinstance(expression, ast.Name):
            value = namespace[expression.id]
        elif isinstance(expression, ast.Constant):
            value = expression.value
        else:
            raise AssertionError(f"Review dynamic admitted kind: {ast.unparse(expression)}")
        assert isinstance(value, str) and value
        kinds.add(value)
    return kinds


def _admitted_kinds(owner: type) -> set[str]:
    module = inspect.getmodule(owner)
    assert module is not None
    kinds = _kinds(inspect.getsource(module), vars(module))
    # Portfolio and review owners import their input constructors from their
    # adapter modules. Discover those constructors by their bodies, not names.
    for value in vars(module).values():
        if inspect.isfunction(value) and value.__module__.startswith("alphalattice."):
            factory_module = inspect.getmodule(value)
            assert factory_module is not None
            kinds.update(_kinds(inspect.getsource(value), vars(factory_module)))
    return kinds


def test_every_composed_task_input_kind_reaches_recovery(plan_admission_check) -> None:
    operations = object.__new__(PortfolioResearchOperations)
    hints = get_type_hints(PortfolioResearchOperations)
    declarations = {}
    admissions = set()
    for item in fields(PortfolioResearchOperations):
        setattr(operations, item.name, None)
        hint = hints[item.name]
        for owner in get_args(hint) or (hint,):
            if not inspect.isclass(owner) or not owner.__module__.startswith("alphalattice."):
                continue
            kinds = _admitted_kinds(owner)
            if not kinds:
                continue
            declared = {replan.task_kind: replan for replan in getattr(owner, "replans", ())}
            assert kinds == declared.keys(), (item.name, kinds, declared.keys())
            for replan in declared.values():
                if replan.preview is not None:
                    _required, allowed = PortfolioResearchOperationRequest.field_contract(
                        replan.preview
                    )
                    if allowed - {"recovery_task_id", "recovery_task_hash"}:
                        assert callable(getattr(owner, "replan_request", None)), (
                            item.name,
                            replan.preview,
                            "A preview with choices must bind the verified durable declaration "
                            "or name the missing selector before offering REPLAN",
                        )
            assert not admissions.intersection(kinds), (item.name, kinds)
            admissions.update(kinds)
            declarations.update(declared)
            setattr(operations, item.name, SimpleNamespace(replans=tuple(declared.values())))
    assert admissions
    # A new admission factory anywhere under the Host cannot disappear merely
    # because its composer forgot to declare or wire its owner.
    root = Path(__file__).resolve().parents[2] / "src"
    for path in (root / "alphalattice/control/product_host").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if "TaskInputEnvelope.create" not in source:
            continue
        module = importlib.import_module(".".join(path.relative_to(root).with_suffix("").parts))
        assert _kinds(source, vars(module)) <= admissions, path
    assert operations.replans() == declarations
    # Every operation the recovery view can offer is a CLI operation and a
    # session route, including an owner that declares no separate preview.
    replan_operations = {item.preview or item.admitting for item in declarations.values()}
    registered = command_table()["fields"]
    route_operations = {
        operation for _method, _path, operation in (*OPERATION_ROUTES, *HANDLED_OPERATION_ROUTES)
    }
    assert replan_operations <= registered.keys(), replan_operations - registered.keys()
    assert replan_operations <= route_operations, replan_operations - route_operations
    routes = {
        operation: {"method": method, "path": path}
        for method, path, operation in (*OPERATION_ROUTES, *HANDLED_OPERATION_ROUTES)
    }
    for declared in declarations.values():
        # Every named route remains a request the Host actually accepts.
        _required, allowed = PortfolioResearchOperationRequest.field_contract(
            declared.preview or declared.admitting
        )
        PortfolioResearchOperationRequest.field_contract(declared.admitting)
        assert routes[declared.admitting]["method"] == "POST", declared
        if declared.preview is None or declared.preview == declared.admitting:
            continue
        if not allowed - {"recovery_task_id", "recovery_task_hash"}:
            source = SimpleNamespace(
                task_kind=declared.task_kind,
                lifecycle=TaskLifecycle.BLOCKED,
                task_id=UUID(int=83),
                record_hash="a" * 64,
            )
            offered = operations._task_replan(source)
            assert offered == {
                "operation": declared.preview,
                "recovery_task_id": str(source.task_id),
                "recovery_task_hash": source.record_hash,
            }, declared
            PortfolioResearchRequestDocument.model_validate(offered)
        admission = {
            "operation": declared.admitting,
            **{
                name: {} if name == "spec" else "a" * 64
                for name in registered[declared.admitting]["required"]
            },
        }
        body = {"status": "PLANNED", "next_requests": {"run": admission}}

        def check(answer, table=routes, preview=declared.preview):
            return plan_admission_check(preview, answer, declarations, table)

        assert check(body) is None, declared
        assert check({"status": "PLANNED"}), declared
        assert check({"status": "PLANNED", "next_requests": {}}), declared
        for status in ("PLANNED", "UNKNOWN", None):
            assert check({"status": status, "task_id": "11111111-1111-1111-1111-111111111111"})
        assert check({"status": "DEFINITION_PLANNED", "failure_code": "synthetic.failure"})
        if declared.preview != "RESEARCH_INPUT_PLAN":
            assert check({"status": "REUSED_EXACT"})
        if declared.preview != "WORKSPACE_PREPARE_PLAN":
            assert check({"status": "ALREADY_PREPARED"})
        if declared.preview not in {"RESEARCH_UPDATE_PLAN", "PORTFOLIO_UPDATE_PLAN"}:
            assert check({"status": "OBSERVATIONS_PENDING"})
        if declared.preview == "WORKSPACE_PREPARE_PLAN":
            assert check({"status": "PLANNED", "confirmation_available": False})
        if declared.preview == "DATA_UPDATE_PLAN":
            consent = {**body, "status": "CONFIRMATION_REQUIRED"}
            assert check(consent), "A run cannot replace the required human confirmation"
            confirm = {**admission, "operation": "DATA_CHANGE_CONFIRM"}
            assert check({**consent, "next_requests": {"confirm": confirm}}) is None
        assert check({**body, "next_requests": {"run": {"operation": "RESULTS"}}}), declared
        assert check(
            body, {key: value for key, value in routes.items() if key != declared.admitting}
        )
        assert check(body, {**routes, declared.admitting: {"method": "GET", "path": "/bad"}})
        for field in registered[declared.admitting]["required"]:
            for value in (None, "<unbound>"):
                assert check({**body, "next_requests": {"run": {**admission, field: value}}})
                assert check(
                    {
                        **body,
                        "next_requests": {
                            "run": admission,
                            "another": {**admission, field: value},
                        },
                    }
                )
    method = next(
        node
        for node in ast.walk(ast.parse(inspect.getsource(inspect.getmodule(operations))))
        if isinstance(node, ast.FunctionDef) and node.name == "recovery_view"
    )
    calls = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "build_task_recovery_view"
    ]
    assert len(calls) == 1
    assert any(
        keyword.arg == "replans" and ast.unparse(keyword.value) == "self.replans()"
        for keyword in calls[0].keywords
    ), "The recovery view must receive the complete composed declarations"


def test_a_new_composed_owner_reaches_recovery_without_a_second_owner_list() -> None:
    @dataclass
    class ExtendedOperations(PortfolioResearchOperations):
        future_owner: object = None

    operations = object.__new__(ExtendedOperations)
    for item in fields(ExtendedOperations):
        setattr(operations, item.name, None)
    declared = TaskReplan(task_kind="synthetic.future", preview="PLAN", admitting="RUN")
    request = {"operation": "PLAN", "spec": {"strategy_package_id": "synthetic-selected"}}
    sentinel = SimpleNamespace(task_kind=declared.task_kind, lifecycle=TaskLifecycle.SUCCEEDED)
    seen = []

    def bind(task):
        seen.append(task)
        return request

    operations.future_owner = SimpleNamespace(replans=(declared,), replan_request=bind)
    assert operations.replans() == {declared.task_kind: declared}
    filled = operations._task_replan(sentinel)
    assert filled == request and seen == [sentinel]
    PortfolioResearchRequestDocument.model_validate(filled)
