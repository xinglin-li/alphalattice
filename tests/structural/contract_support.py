"""SC2's contract reading: every pydantic model src defines, read from source for the commit stage's
holder, and the same rows read by importing src for the Large check that proves the two agree."""

from __future__ import annotations

import ast
import re
from functools import cache
from pathlib import Path

_MODEL_ROOTS = {"pydantic.BaseModel", "pydantic.RootModel"}
_SUBCLASS = re.compile(
    rb"^[ \t]*class[ \t]+[^\s(:]+[ \t]*\((?![ \t]*\))", re.MULTILINE
)  # a model has a base


@cache
def contracts(src: Path) -> frozenset[str]:
    """Every pydantic model a src module defines at its top level, as
    `module:Name:frozen|mutable:extra`, read from source with no import and no process. A config
    merges as pydantic merges it: each base's in order, then the class's own `model_config`. A
    config that is not a literal refuses; what only an import shows is the Large check's."""

    files = {}
    for path in sorted(src.rglob("*.py")):
        parts = path.relative_to(src).with_suffix("").parts
        files[".".join(parts[:-1] if parts[-1] == "__init__" else parts)] = path
    symbols: dict[str, dict[str, str]] = {}  # module -> top-level name -> dotted origin
    classes: dict[str, tuple[str, ast.ClassDef]] = {}

    def names(module: str) -> dict[str, str]:  # parsed on first use
        if module not in symbols and module in files:
            path, table = files[module], symbols.setdefault(module, {})
            package = path.name == "__init__.py"
            for node in ast.parse(path.read_bytes(), str(path)).body:
                if isinstance(node, ast.ClassDef):
                    table[node.name] = f"{module}.{node.name}"
                    classes[table[node.name]] = (module, node)
                elif isinstance(node, ast.ImportFrom):
                    base = module if package else module.rpartition(".")[0]
                    for _ in range(node.level - 1):
                        base = base.rpartition(".")[0]
                    origin = ".".join(filter(None, (base if node.level else "", node.module)))
                    table.update({a.asname or a.name: f"{origin}.{a.name}" for a in node.names})
        return symbols.get(module, {})

    def origin(module: str, base: ast.expr) -> str:  # through src's re-exports
        while isinstance(base, ast.Subscript):
            base = base.value
        dotted = names(module).get(getattr(base, "id", ""), ast.unparse(base))
        while dotted not in classes:
            owner, _, name = dotted.rpartition(".")
            if (onward := names(owner).get(name, dotted)) == dotted:
                break
            dotted = onward
        return dotted

    configs: dict[str, dict[str, object] | None] = {}

    def config(key: str) -> dict[str, object] | None:  # None: not a pydantic model
        if key in _MODEL_ROOTS or key not in classes:
            return {} if key in _MODEL_ROOTS else None
        if key not in configs:
            module, node = classes[key]
            inherited = [
                found for base in node.bases if (found := config(origin(module, base))) is not None
            ]
            merged = {name: value for found in inherited for name, value in found.items()}
            for statement in node.body if inherited else ():
                targets = (
                    statement.targets
                    if isinstance(statement, (ast.Assign, ast.Delete))
                    else (
                        [statement.target]
                        if isinstance(statement, (ast.AnnAssign, ast.AugAssign)) and statement.value
                        else []
                    )
                )
                for target in targets:
                    if any(
                        isinstance(part, ast.Name) and part.id == "model_config"
                        for part in ast.walk(target)
                    ):
                        assert isinstance(target, ast.Name) and not isinstance(
                            statement, (ast.AugAssign, ast.Delete)
                        ), key
                if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                    func = statement.value.func
                    assert not (
                        isinstance(func, ast.Attribute)
                        and isinstance(func.value, ast.Name)
                        and func.value.id == "model_config"
                    ), key
                if "model_config" in map(ast.unparse, targets):
                    value = statement.value  # a literal ConfigDict, or only an import can tell
                    assert isinstance(value, ast.Call) and not value.args, key
                    assert ast.unparse(value.func) == "ConfigDict", key
                    assert None not in [k.arg for k in value.keywords], key  # no **mapping
                    try:
                        merged.update(
                            {str(k.arg): ast.literal_eval(k.value) for k in value.keywords}
                        )
                    except (ValueError, TypeError, SyntaxError) as error:
                        raise AssertionError(key) from error
            configs[key] = merged if inherited else None
        return configs[key]

    for module, path in files.items():
        if _SUBCLASS.search(path.read_bytes()):
            names(module)
    rows = set()
    for key, (module, node) in list(classes.items()):
        if (found := config(key)) is not None:
            frozen = "frozen" if found.get("frozen") else "mutable"
            rows.add(f"{module}:{node.name}:{frozen}:{found.get('extra')}")
    return frozenset(rows)


RUNTIME_CONTRACTS = r"""
import importlib, json, sys
from pathlib import Path
import pydantic
root = Path(sys.argv[1])
rows = []
for path in sorted(root.rglob("*.py")):
    parts = path.relative_to(root).with_suffix("").parts
    module = importlib.import_module(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
    for value in vars(module).values():
        if (
            isinstance(value, type)
            and value.__module__ == module.__name__
            and issubclass(value, pydantic.BaseModel)
        ):
            config = value.model_config
            frozen = "frozen" if config.get("frozen") else "mutable"
            rows.append(f"{module.__name__}:{value.__qualname__}:{frozen}:{config.get('extra')}")
print(json.dumps(sorted(set(rows))))
"""
