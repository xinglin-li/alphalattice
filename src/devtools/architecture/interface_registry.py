"""Generate the interface inventory from the operation, Door and package owners."""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from griffe import Alias, find_breaking_changes, visit
from pydantic import BaseModel, TypeAdapter

from alphalattice.interface.local_application.answers import ANSWERS
from alphalattice.interface.local_application.cli_invocation import ForwardedCli
from alphalattice.interface.local_application.operations import GRAMMAR, OPERATIONS, fields
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchRequestDocument,
)
from alphalattice.kernel import storage
from alphalattice.kernel.door import CLI_OPERATION, OPERATIONS_PATH, SCHEMA_VERSION
from alphalattice.protocols import plugin_contract
from devtools.architecture import structural
from devtools.architecture.git_inputs import read_blobs
from devtools.architecture.neutral_changes import source_syntax_tree
from devtools.architecture.operation_registry import _compatible, _typed
from devtools.architecture.registries import tracked

SNAPSHOT = "config/registries/interfaces.json"
_COMPOSITION = "alphalattice.control.product_host.composition"
_ROUTES = "src/alphalattice/control/product_host/composition/operation_route_families.py"
_FORWARDED = "src/alphalattice/interface/local_application/cli_invocation.py"
_STORAGE_DRIVER = "alphalattice.control.workspace_runtime.reader_threads"
_PLUGIN = "src/alphalattice/protocols/plugin_contract"
_PLUGIN_ROW = "extension-point:plugin_action"
_PLUGIN_EXPORTS = f"{_PLUGIN}/__init__.py"
_PLUGIN_CONTRACTS = f"{_PLUGIN}/contracts.py"


def _implementations(root: Path) -> dict[str, str]:
    """Read the Door's registered families without importing their handlers."""
    implementations: dict[str, str] = {}
    for node in ast.walk(source_syntax_tree(root / _ROUTES)):
        match node:
            case ast.Call(
                func=ast.Name(id="FamilyRoute"),
                args=[
                    ast.Tuple(elts=names),
                    ast.Constant(value=str() as module),
                    ast.Constant(value=str() as handler),
                ],
            ):
                module = (
                    module if module.startswith("alphalattice.") else f"{_COMPOSITION}.{module}"
                )
                for name in names:
                    operation = (
                        CLI_OPERATION
                        if isinstance(name, ast.Name) and name.id == "CLI_OPERATION"
                        else ast.literal_eval(name)
                    )
                    if operation in implementations:
                        raise ValueError(f"interface_registry.duplicate_operation: {operation}")
                    implementations[operation] = f"{module}:{handler}"
    if missing := {*OPERATIONS, CLI_OPERATION} ^ implementations.keys():
        missing_names = ", ".join(sorted(missing))
        raise ValueError(f"interface_registry.operation_routes_disagree: {missing_names}")
    return implementations


def _forwarded_schema(root: Path) -> dict[str, Any]:
    tree = source_syntax_tree(root / _FORWARDED)
    invoke = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "invoke"
    )
    required: list[str] | None = None
    replies: dict[str, list[str]] = {}
    for node in ast.walk(invoke):
        match node:
            case ast.Compare(
                left=ast.Call(func=ast.Attribute(value=ast.Name(id="payload"), attr="keys")),
                ops=[ast.NotEq()],
                comparators=[ast.Set(elts=fields)],
            ):
                required = sorted(map(ast.literal_eval, fields))
            case ast.Return(value=ast.Dict(keys=keys, values=values)):
                assert all(isinstance(key, ast.Constant) for key in keys)
                reply_fields = list(map(ast.literal_eval, cast(list[ast.Constant], keys)))
                replies[ast.literal_eval(values[reply_fields.index("cli")])] = sorted(reply_fields)
    if required is None or not replies:
        raise ValueError(f"{_FORWARDED}: expected the forwarded request fields and reply forms")
    return {
        "path": OPERATIONS_PATH,
        "version": SCHEMA_VERSION,
        "request": {
            "model": f"{ForwardedCli.__module__}.{ForwardedCli.__name__}",
            "required": required,
        },
        "answer": replies,
    }


def _storage_driver(root: Path) -> str:
    path = root.joinpath("src", *_STORAGE_DRIVER.split(".")).with_suffix(".py")
    nodes = source_syntax_tree(path).body
    installed = next(
        stmt.value.args[0]
        for stmt in nodes
        if isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and ast.unparse(stmt.value.func) == "storage.install_driver"
    )
    if isinstance(installed, ast.Name):
        installed = next(
            stmt.value
            for stmt in nodes
            if isinstance(stmt, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == installed.id
                for target in stmt.targets
            )
        )
    if not isinstance(installed, ast.Call):
        raise ValueError(f"{_STORAGE_DRIVER}: expected a constructor for the installed driver")
    return f"{_STORAGE_DRIVER}:{ast.unparse(installed.func)}"


def _row(
    kind: str,
    owner: str,
    description: str,
    schema: dict[str, Any],
    stability: str,
    implementations: list[str],
    consumers: list[str],
) -> dict[str, Any]:
    return {
        "kind": kind,
        "owner": owner,
        "description": description,
        "schema": schema,
        "stability": stability,
        "implementations": implementations,
        "consumers": consumers,
        "conformance_suite": [],
    }


def _port_schema(source: Path) -> dict[str, list[str]]:
    return {
        node.name: [
            f"{method.name}({ast.unparse(method.args)})"
            + (f" -> {ast.unparse(method.returns)}" if method.returns else "")
            for method in node.body
            if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        for node in source_syntax_tree(source).body
        if isinstance(node, ast.ClassDef)
        and any(isinstance(base, ast.Name) and base.id == "Protocol" for base in node.bases)
    }


def interface_inventory(root: Path) -> dict[str, Any]:
    """Return deterministic rows without starting a Host or loading a plugin."""
    routes = _implementations(root)
    request = PortfolioResearchRequestDocument
    request_model = f"{request.__module__}.{request.__name__}"
    entries: dict[str, Any] = {}
    for operation in OPERATIONS:
        required, allowed = fields(operation)
        answer = ANSWERS[operation]
        implementation = routes[operation]
        command = GRAMMAR[operation]
        entries[f"operation:{operation}"] = _row(
            "operation",
            implementation.split(":", 1)[0],
            command.purpose,
            {
                "request": {
                    "model": request_model,
                    "required": sorted(required),
                    "allowed": sorted(allowed),
                },
                "answer": f"{answer.__module__}.{answer.__name__}",
            },
            "userspace",
            [implementation],
            [f"cli:{command.noun} {command.verb}"],
        )

    forwarding = routes[CLI_OPERATION]
    entries[f"operation:{CLI_OPERATION}"] = _row(
        "operation",
        forwarding.split(":", 1)[0],
        "Runs a forwarded thin CLI command with the caller's captured streams.",
        _forwarded_schema(root),
        "userspace",
        [forwarding],
        ["alphalattice.interface.local_application.thin_cli"],
    )

    source_root = root / "src"
    paths = tuple(root / path for path in tracked(root) if path.startswith("src/alphalattice/"))
    consumers: dict[str, list[str]] = defaultdict(list)
    for caller, target in structural.discover_package_dependencies(paths, source_root=source_root):
        consumers[target].append(caller)
    architecture = structural.load_package_architecture_registry(
        root / "config/package-architecture.json"
    )
    owners_map = root / "config/registries/owners-map.json"
    nouns = json.loads(owners_map.read_bytes())["nouns"]
    for package in architecture.packages:
        if package.authority_kind != "KERNEL_PORT":
            continue
        owner = package.physical_namespace_target
        source = source_root.joinpath(*owner.split("."), "port.py")
        schema: dict[str, Any] = _port_schema(source)
        implementations = [owner]
        if owner == storage.__name__:
            schema["callables"] = {
                name: str(inspect.signature(export))
                for name in storage.__all__
                if inspect.isfunction(export := getattr(storage, name))
            }
            implementations.append(_storage_driver(root))
        entries[f"port:{package.package_id}"] = _row(
            "port",
            owner,
            nouns[package.capability_owner.lower()]["description"],
            schema,
            "internal",
            implementations,
            consumers[package.package_id],
        )

    installed = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    capabilities = installed.get("entry-points", {}).get("alphalattice.capabilities", {})
    plugin_schemas = {}
    for name in sorted(plugin_contract.__all__):
        contract = getattr(plugin_contract, name)
        plugin_schemas[name] = (
            contract.model_json_schema()
            if issubclass(contract, BaseModel)
            else TypeAdapter(contract).json_schema()
        )
    entries[_PLUGIN_ROW] = _row(
        "extension point",
        plugin_contract.__name__,
        nouns["the plugin contract"]["description"],
        plugin_schemas,
        "published",
        [f"{name}={target}" for name, target in sorted(capabilities.items())],
        consumers["plugin_contract"],
    )
    return {"schema": "alphalattice.playpen.registry.interfaces", "entries": entries}


def inventory_text(root: Path) -> str:
    """Serialize the code-derived inventory in its committed form."""
    return json.dumps(interface_inventory(root), indent=2, sort_keys=True) + "\n"


def snapshot_problems(path: Path, generated: str) -> list[str]:
    """Refuse an inventory that differs from a fresh generation."""
    if path.is_file() and path.read_bytes() == generated.encode("utf-8"):
        return []
    return [
        f"interface_registry.stale_generation: {SNAPSHOT}; "
        "put src on PYTHONPATH and run "
        "python -m devtools.architecture.interface_registry --write"
    ]


def published_problems(
    old_exports: str, old_contracts: str, new_exports: str, new_contracts: str
) -> list[str]:
    """Compare the published plugin symbols with Griffe's static API model."""
    exports_path = Path(_PLUGIN_EXPORTS)
    contracts_path = Path(_PLUGIN_CONTRACTS)
    old_package = visit("plugin_contract", exports_path, old_exports)
    new_package = visit("plugin_contract", exports_path, new_exports)
    old = visit("contracts", contracts_path, old_contracts)
    new = visit("contracts", contracts_path, new_contracts)
    problems: list[str] = []
    for name in sorted(old_package.exports or ()):
        before = old_package.members.get(name)
        after = new_package.members.get(name)
        if name not in (new_package.exports or ()) or not isinstance(after, Alias):
            problems.append(f"interface_registry.published_breaking: {name}: export removed")
            continue
        if not isinstance(before, Alias) or before.target_path != after.target_path:
            problems.append(
                f"interface_registry.published_breaking: {name}: export binding changed"
            )
            continue
        target = after.target_path.removeprefix("plugin_contract.contracts.")
        if target not in old.members or target not in new.members:
            problems.append(
                f"interface_registry.published_breaking: {name}: export target unavailable"
            )
            continue
        for change in find_breaking_changes(old.members[target], new.members[target]):
            problems.append(f"interface_registry.published_breaking: {change.explain()}")
    return problems


def problems(root: Path, base_ref: str) -> list[str]:
    """Check the generated inventory and the published API against the selected base."""
    current = interface_inventory(root)
    generated = json.dumps(current, indent=2, sort_keys=True) + "\n"
    out = snapshot_problems(root / SNAPSHOT, generated)
    export, contracts, snapshot = (
        f"{base_ref}:{path}" for path in (_PLUGIN_EXPORTS, _PLUGIN_CONTRACTS, SNAPSHOT)
    )
    prior = read_blobs([export, contracts, snapshot], root)
    if export not in prior or contracts not in prior:
        return out
    current_exports = (root / _PLUGIN_EXPORTS).read_text(encoding="utf-8")
    current_contracts = (root / _PLUGIN_CONTRACTS).read_text(encoding="utf-8")
    out.extend(
        published_problems(prior[export], prior[contracts], current_exports, current_contracts)
    )
    if snapshot in prior:
        before = json.loads(prior[snapshot])["entries"][_PLUGIN_ROW]["schema"]
        after = current["entries"][_PLUGIN_ROW]["schema"]
        for name, contract in before.items():
            if name not in after or not _compatible(_typed(contract), _typed(after[name])):
                out.append(f"interface_registry.published_breaking: {name}: schema changed")
    elif (prior[export], prior[contracts]) != (current_exports, current_contracts):
        out.append(f"interface_registry.published_schema_baseline_missing: {base_ref}")
    return out


def main() -> int:
    """Regenerate or check the current checkout's inventory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Regenerate the inventory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    if args.write:
        (root / SNAPSHOT).write_text(inventory_text(root), encoding="utf-8", newline="\n")
        return 0
    found = problems(root, "HEAD")
    for line in found:
        print(line)
    return int(bool(found))


if __name__ == "__main__":
    raise SystemExit(main())
