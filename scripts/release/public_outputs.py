"""Prove the exact local outputs that PUBLIC producers create, by parsing their source."""

from __future__ import annotations

import ast
import posixpath
from collections.abc import Collection, Mapping
from pathlib import PurePosixPath

_BUILDER = "scripts/build_local_web_ui.py"
_SETUP = "src/alphalattice/interface/local_application/native_setup.py"
_BRIDGE = "src/alphalattice/interface/local_application/native_bridge.py"
_PERSISTENCE = "src/alphalattice/kernel/shared_kernel/persistence.py"
_ASSETS = "src/alphalattice/interface/local_application/assets"


def _tree(path: str, blobs: Mapping[str, bytes], public: Collection[str]) -> ast.Module | None:
    if path not in public or path not in blobs:
        return None
    try:
        return ast.parse(blobs[path].decode("utf-8-sig"), filename=path)
    except (SyntaxError, UnicodeError):
        return None


def _bindings(tree: ast.Module) -> dict[str, ast.expr]:
    values: dict[str, ast.expr] = {}
    repeated: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign | ast.AnnAssign) or node.value is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                if target.id in values:
                    repeated.add(target.id)
                values[target.id] = node.value
    return {name: value for name, value in values.items() if name not in repeated}


def _value(node: ast.AST, names: Mapping[str, ast.expr], path: str, seen=()) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        if node.id == "__file__":
            return "/__checkout__/" + path
        if node.id in names and node.id not in seen:
            return _value(names[node.id], names, path, (*seen, node.id))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left, right = _value(node.left, names, path, seen), _value(node.right, names, path, seen)
        if left is not None and right is not None:
            return posixpath.normpath(posixpath.join(left, right))
    if isinstance(node, ast.Attribute) and node.attr == "parent":
        value = _value(node.value, names, path, seen)
        return posixpath.dirname(value) if value is not None else None
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "parents"
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, int)
    ):
        value = _value(node.value.value, names, path, seen)
        if value is not None and 0 <= node.slice.value < len(PurePosixPath(value).parents):
            return str(PurePosixPath(value).parents[node.slice.value])
    if isinstance(node, ast.Call):
        if ast.unparse(node.func) in {"Path", "pathlib.Path"} and len(node.args) == 1:
            return _value(node.args[0], names, path, seen)
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"resolve", "absolute"}:
            return _value(node.func.value, names, path, seen)
    return None


def _function(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    return found[0] if len(found) == 1 else None


def _parts(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _parts(node.left) + _parts(node.right)
    if isinstance(node, ast.Name):
        return ("$" + node.id,)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return (node.value,)
    return ("<unresolved>",)


def _imported(tree: ast.Module, module: str, member: str) -> bool:
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == module
        and node.level == 0
        and any(alias.name == member and alias.asname in {None, member} for alias in node.names)
        for node in tree.body
    )


def _asset_writer(build: ast.FunctionDef, tree: ast.Module, persistence: ast.Module | None):
    replace = _function(persistence, "replace_with_retry") if persistence is not None else None
    replaces = replace is not None and any(
        isinstance(node, ast.Call)
        and ast.unparse(node.func) == "os.replace"
        and [_parts(arg) for arg in node.args[:2]] == [("$source",), ("$target",)]
        for node in ast.walk(replace)
    )
    imported = _imported(
        tree, "alphalattice.kernel.shared_kernel.persistence", "replace_with_retry"
    )
    for loop in ast.walk(build):
        if not isinstance(loop, ast.For) or ast.unparse(loop.iter) != "outputs.items()":
            continue
        if not isinstance(loop.target, ast.Tuple) or len(loop.target.elts) != 2:
            continue
        if not all(isinstance(n, ast.Name) for n in loop.target.elts):
            continue
        name, data = (n.id for n in loop.target.elts)
        calls = [node for node in ast.walk(loop) if isinstance(node, ast.Call)]
        for call in calls:
            if (
                isinstance(call.func, ast.Attribute)
                and call.func.attr in {"write_bytes", "write_text"}
                and _parts(call.func.value) == ("$ASSETS", "$" + name)
                and any(isinstance(n, ast.Name) and n.id == data for n in ast.walk(call))
            ):
                return call.lineno
        if not (replaces and imported):
            continue
        staged = {
            node.targets[0].id
            for node in ast.walk(loop)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.BinOp)
            and isinstance(node.value.op, ast.Div)
            and isinstance(node.value.left, ast.Name)
            and node.value.left.id == "ASSETS"
            and isinstance(node.value.right, ast.JoinedStr)
            and any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(node.value.right))
        }
        written = {
            call.func.value.id
            for call in calls
            if isinstance(call.func, ast.Attribute)
            and call.func.attr in {"write_bytes", "write_text"}
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id in staged
            and any(isinstance(n, ast.Name) and n.id == data for n in ast.walk(call))
        }
        for call in calls:
            if (
                ast.unparse(call.func) == "replace_with_retry"
                and len(call.args) >= 2
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id in written
                and _parts(call.args[1]) == ("$ASSETS", "$" + name)
            ):
                return call.lineno
    return None


def _output_declarations(
    scope: ast.FunctionDef, tree: ast.Module, limit: int, seen: tuple[str, ...] = ()
) -> list[ast.Assign]:
    """Follow the writer's mapping through an explicit, key-preserving local render.

    An unknown call, recursive helper, conditional return, changed key or filtered mapping
    supplies no output permission. The write still belongs to the original PUBLIC builder.
    """
    resets = [
        node
        for node in ast.walk(scope)
        if isinstance(node, ast.Assign | ast.AnnAssign)
        and node.lineno < limit
        and any(
            _parts(target) == ("$outputs",)
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
    ]
    reset = max(resets, key=lambda node: node.lineno, default=None)
    if reset is not None and isinstance(reset.value, ast.Call):
        call = reset.value
        if not isinstance(call.func, ast.Name) or call.func.id in seen:
            return []
        helper = _function(tree, call.func.id)
        if helper is None:
            return []
        returns = [node for node in ast.walk(helper) if isinstance(node, ast.Return)]
        if len(returns) != 1:
            return []
        returned = returns[0]
        value = returned.value
        preserves_keys = _parts(value) == ("$outputs",)
        if isinstance(value, ast.DictComp) and len(value.generators) == 1:
            generator = value.generators[0]
            preserves_keys = (
                isinstance(generator.target, ast.Tuple)
                and len(generator.target.elts) == 2
                and all(isinstance(item, ast.Name) for item in generator.target.elts)
                and isinstance(value.key, ast.Name)
                and value.key.id == generator.target.elts[0].id
                and ast.unparse(generator.iter) == "outputs.items()"
                and not generator.ifs
                and not generator.is_async
            )
        # An opaque function handed this mutable mapping could remove its declared keys.
        opaque_mutation = any(
            isinstance(node, ast.Call) and any(_parts(arg) == ("$outputs",) for arg in node.args)
            for node in ast.walk(helper)
        )
        if not preserves_keys or opaque_mutation:
            return []
        return _output_declarations(helper, tree, returned.lineno, (*seen, call.func.id))
    first = reset.lineno if reset is not None else 0
    return [
        node
        for node in ast.walk(scope)
        if isinstance(node, ast.Assign) and first < node.lineno < limit
    ]


def generated_outputs(blobs: Mapping[str, bytes], public: Collection[str]) -> dict[str, str]:
    """Missing Workbench assets and native binding, each proved by its PUBLIC writer.

    These are exact outputs, never directory prefixes. A missing/private producer or
    constant dependency, a declaration without a write, or an existing tracked path
    supplies no permission to omit another source dependency.
    """
    result: dict[str, str] = {}
    builder = _tree(_BUILDER, blobs, public)
    if builder is not None:
        names = _bindings(builder)
        build = _function(builder, "build")
        assets = _value(names.get("ASSETS", ast.Constant(None)), names, _BUILDER)
        manifest = _value(names.get("MANIFEST", ast.Constant(None)), names, _BUILDER)
        if build is not None and assets == "/__checkout__/" + _ASSETS:
            writer = _asset_writer(build, builder, _tree(_PERSISTENCE, blobs, public))
            if writer is not None:
                for node in _output_declarations(build, builder, writer):
                    for target in node.targets:
                        if not isinstance(target, ast.Subscript) or _parts(target.value) != (
                            "$outputs",
                        ):
                            continue
                        name = _value(target.slice, names, _BUILDER)
                        if (
                            name == "workbench.html"
                            or name == manifest == "workbench-manifest.json"
                        ):
                            output = _ASSETS + "/" + name
                            if output not in blobs:
                                result[output] = (
                                    f"{_BUILDER}:{node.lineno}; write {_BUILDER}:{writer}"
                                )
    setup, bridge = _tree(_SETUP, blobs, public), _tree(_BRIDGE, blobs, public)
    if setup is not None and bridge is not None:
        binding = _value(
            _bindings(bridge).get("BINDING_NAME", ast.Constant(None)), _bindings(bridge), _BRIDGE
        )
        writer, bind = _function(setup, "_create_or_match"), _function(setup, "bind_session")
        if (
            binding == "native-research.local.json"
            and writer is not None
            and bind is not None
            and _imported(
                setup, "alphalattice.interface.local_application.native_bridge", "BINDING_NAME"
            )
            and any(
                isinstance(n, ast.Call)
                and ast.unparse(n.func) == "_create_or_match"
                and n.args
                and _parts(n.args[0]) == ("$BINDING_NAME",)
                for n in ast.walk(bind)
            )
            and any(
                isinstance(n, ast.Assign)
                and any(_parts(t) == ("$path",) for t in n.targets)
                and _parts(n.value) == ("$root", ".codex", "$name")
                for n in ast.walk(writer)
            )
        ):
            for node in ast.walk(writer):
                if not isinstance(node, ast.With):
                    continue
                for item in node.items:
                    call = item.context_expr
                    if (
                        isinstance(call, ast.Call)
                        and ast.unparse(call.func) == "path.open"
                        and call.args
                        and isinstance(call.args[0], ast.Constant)
                        and call.args[0].value == "xb"
                        and isinstance(item.optional_vars, ast.Name)
                        and any(
                            isinstance(n, ast.Call)
                            and ast.unparse(n.func) == item.optional_vars.id + ".write"
                            for statement in node.body
                            for n in ast.walk(statement)
                        )
                    ):
                        output = ".codex/" + binding
                        if output not in blobs:
                            result[output] = (
                                f"{_SETUP}:{call.lineno}; constant {_BRIDGE}:BINDING_NAME"
                            )
    return dict(sorted(result.items()))
