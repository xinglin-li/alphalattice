"""Parse the local dependencies of a manifest's PUBLIC files without executing them."""

from __future__ import annotations

import ast
import fnmatch
import json
import posixpath
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit

from release.public_outputs import generated_outputs


@dataclass(frozen=True, order=True)
class Reference:
    """A source dependency, with the read or link that requires it."""

    reader: str
    line: int
    target: str
    kind: str


def _path(value: str) -> str:
    return posixpath.normpath(value.replace("\\", "/"))


def _scope_nodes(tree: ast.AST) -> Iterator[ast.AST]:
    yield tree
    for child in ast.iter_child_nodes(tree):
        if not isinstance(
            child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda
        ):
            yield from _scope_nodes(child)


class _Paths:
    """Only a proven __file__/checkout anchor denotes a repository input."""

    def __init__(
        self,
        path: str,
        tree: ast.AST,
        files: set[str],
        inherited: Mapping[str, set[str]] | None = None,
        arguments: Mapping[str, set[str]] | None = None,
    ) -> None:
        self.file = "/__checkout__/" + path
        self.files = files
        self.names: dict[str, set[str]] = {
            key: set(values) for key, values in (inherited or {}).items()
        }
        self.names["__file__"] = {self.file}
        if isinstance(tree, ast.FunctionDef | ast.AsyncFunctionDef):
            for arg in [*tree.args.posonlyargs, *tree.args.args, *tree.args.kwonlyargs]:
                self.names.pop(arg.arg, None)  # A fixture/parameter shadows the module binding.
        self.names.update(arguments or {})
        assignments = [n for n in _scope_nodes(tree) if isinstance(n, ast.Assign | ast.AnnAssign)]
        for _ in range(8):
            changed = False
            for node in assignments:
                value = self.values(node.value)
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and value:
                        old = self.names.setdefault(target.id, set())
                        changed |= not value <= old
                        old.update(value)
            if not changed:
                break

    def values(self, node: ast.AST | None) -> set[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return {node.value}
        if isinstance(node, ast.Name):
            return self.names.get(node.id, set())
        if isinstance(node, ast.List | ast.Tuple | ast.Set):
            return set().union(*(self.values(item) for item in node.elts))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div | ast.Add):
            left, right = self.values(node.left), self.values(node.right)
            return {
                _path(a + "/" + b) if isinstance(node.op, ast.Div) else a + b
                for a in left
                for b in right
            }
        if isinstance(node, ast.Attribute) and node.attr == "parent":
            return {posixpath.dirname(p) for p in self.values(node.value)}
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "parents"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, int)
        ):
            return {
                str(PurePosixPath(p).parents[node.slice.value])
                for p in self.values(node.value.value)
                if len(PurePosixPath(p).parents) > node.slice.value >= 0
            }
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if name in {"Path", "pathlib.Path", "PurePath", "str", "os.fspath"} and node.args:
                return self.values(node.args[0])
            if name in {"os.path.join", "posixpath.join"}:
                parts = {""}
                for arg in node.args:
                    parts = {
                        _path(a + "/" + b) if a else b for a in parts for b in self.values(arg)
                    }
                return parts
            if isinstance(node.func, ast.Attribute):
                base = self.values(node.func.value)
                if node.func.attr in {"resolve", "absolute"}:
                    return base
                if node.func.attr in {"with_name", "joinpath", "with_suffix"} and node.args:
                    result = base
                    for arg in node.args:
                        tail = self.values(arg)
                        if node.func.attr == "with_name":
                            result = {
                                _path(posixpath.dirname(a) + "/" + b) for a in result for b in tail
                            }
                        elif node.func.attr == "with_suffix":
                            result = {
                                str(PurePosixPath(a).with_suffix(b)) for a in result for b in tail
                            }
                        else:
                            result = {_path(a + "/" + b) for a in result for b in tail}
                    return result
        return set()

    def repository(self, node: ast.AST | None) -> set[str]:
        result = set()
        roots = {p.split("/", 1)[0] for p in self.files}
        for value in self.values(node):
            value = _path(value)
            if value.startswith("/__checkout__/"):
                result.add(value.removeprefix("/__checkout__/"))
            elif value in self.files or value.split("/", 1)[0] in roots:
                result.add(value)
        return result


def _python(path: str, text: str, files: set[str]) -> set[Reference]:
    tree = ast.parse(text, filename=path)
    global_paths = _Paths(path, tree, files)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    scopes = {tree: global_paths}

    def context(node: ast.AST) -> _Paths:
        scope = node
        while scope in parents and not isinstance(
            scope, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
        ):
            scope = parents[scope]
        if scope not in scopes:
            scopes[scope] = _Paths(path, scope, files, global_paths.names)
        return scopes[scope]

    result = set()
    module_paths: dict[str, set[str]] = {}
    for target in files:
        if not target.endswith(".py"):
            continue
        for spelling in {target, target.removeprefix("src/"), target.removeprefix("scripts/")}:
            module = spelling[:-3].replace("/", ".").removesuffix(".__init__")
            module_paths.setdefault(module, set()).add(target)
    own = path.removeprefix("src/")[:-3].replace("/", ".").removesuffix(".__init__")
    package = own if path.endswith("/__init__.py") else own.rpartition(".")[0]
    roots = {name.split(".", 1)[0] for name in module_paths}
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update({a.asname: a.name for a in node.names if a.asname})
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            aliases.update({a.asname or a.name: node.module + "." + a.name for a in node.names})

    def imports(name: str, line: int, *, required: bool = True) -> None:
        candidates = set(module_paths.get(name, ()))
        stem = posixpath.dirname(path) + "/" + name.replace(".", "/")
        candidates.update(p for p in (stem + ".py", stem + "/__init__.py") if p in files)
        namespace = any(
            p.startswith((name.replace(".", "/") + "/", "src/" + name.replace(".", "/") + "/"))
            for p in files
        )
        if not candidates and not namespace and required and name.split(".", 1)[0] in roots:
            prefix = (
                "src/"
                if any(p.startswith("src/" + name.split(".")[0] + "/") for p in files)
                else ""
            )
            candidates.add(prefix + name.replace(".", "/") + ".py")
        for part in range(1, len(name.split("."))):
            candidates.update(module_paths.get(".".join(name.split(".")[:part]), ()))
        for target in candidates:
            result.add(Reference(path, line, target, "PYTHON_IMPORT"))

    functions = {
        n.name: n for n in tree.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
    }

    optional_cache: dict[ast.AST, bool] = {}

    def optional(node: ast.AST) -> bool:
        if node in optional_cache:
            return optional_cache[node]
        current = node
        while current in parents:
            child = current
            current = parents[current]
            if isinstance(current, ast.Try) and child in current.body:
                for handler in current.handlers:
                    caught = (
                        {n.id for n in ast.walk(handler.type) if isinstance(n, ast.Name)}
                        if handler.type
                        else set()
                    )
                    defaults = [
                        n.value
                        for n in handler.body
                        if isinstance(n, ast.Return | ast.Assign | ast.AnnAssign)
                    ]
                    if (
                        caught & {"OSError", "FileNotFoundError"}
                        and defaults
                        and len(defaults) == len(handler.body)
                        and not any(isinstance(n, ast.Raise) for n in ast.walk(handler))
                        and all(
                            isinstance(
                                value, ast.Constant | ast.Dict | ast.List | ast.Tuple | ast.Set
                            )
                            for value in defaults
                        )
                    ):
                        optional_cache[node] = True
                        return True
        optional_cache[node] = False
        return False

    visited: set[tuple[str, tuple[tuple[str, tuple[str, ...]], ...]]] = set()

    def calls(node: ast.Call, paths: _Paths, depth: int = 0) -> None:
        name = ast.unparse(node.func)
        first, dot, rest = name.partition(".")
        name = aliases.get(first, first) + (dot + rest if dot else "")
        args: list[ast.AST] = []
        kind = "PYTHON_OPTIONAL_READ" if optional(node) else "PYTHON_READ"
        if isinstance(node.func, ast.Attribute) and node.func.attr in {
            "read_text",
            "read_bytes",
            "open",
        }:
            mode = [*node.args, *(k.value for k in node.keywords if k.arg == "mode")]
            if node.func.attr != "open" or not any(
                isinstance(a, ast.Constant)
                and isinstance(a.value, str)
                and any(c in a.value for c in "wax")
                for a in mode
            ):
                args.append(node.func.value)
        if name in {"open", "io.open"} and node.args:
            mode = node.args[1:2] + [k.value for k in node.keywords if k.arg == "mode"]
            if not any(
                isinstance(a, ast.Constant)
                and isinstance(a.value, str)
                and any(c in a.value for c in "wax")
                for a in mode
            ):
                args.append(node.args[0])
        if (
            name
            in {
                "shutil.copy",
                "shutil.copy2",
                "shutil.copyfile",
                "shutil.copytree",
                "runpy.run_path",
            }
            and node.args
        ):
            args.append(node.args[0])
            kind = "PYTHON_COPY_TREE" if name == "shutil.copytree" else kind
        if (
            name in {"importlib.util.spec_from_file_location", "SourceFileLoader"}
            and len(node.args) > 1
        ):
            args.append(node.args[1])
        if (
            name
            in {
                "subprocess.run",
                "subprocess.Popen",
                "subprocess.call",
                "subprocess.check_call",
                "subprocess.check_output",
            }
            and node.args
        ):
            command = node.args[0]
            if (
                isinstance(command, ast.List | ast.Tuple)
                and len(command.elts) >= 2
                and paths.values(command.elts[0]) == {"git"}
                and paths.values(command.elts[1]) == {"log"}
            ):
                end = next(
                    (i for i, item in enumerate(command.elts) if paths.values(item) == {"--"}),
                    len(command.elts),
                )
                args.extend(command.elts[:end])  # History path filters do not read the checkout.
            else:
                args.append(command)
            kind = "PYTHON_EXEC"
        if name in {"importlib.import_module", "__import__"} and node.args:
            for value in paths.values(node.args[0]):
                imports(value, node.lineno)
        for arg in args:
            for target in paths.repository(arg):
                if kind == "PYTHON_EXEC" and not PurePosixPath(target).suffix:
                    continue  # cwd and workspace arguments are not executable files.
                result.add(Reference(path, node.lineno, target, kind))
        if name in functions and depth < 4:
            function = functions[name]
            parameters = [*function.args.posonlyargs, *function.args.args]
            bound = {
                parameter.arg: paths.values(arg)
                for parameter, arg in zip(parameters, node.args, strict=False)
            }
            bound.update({k.arg: paths.values(k.value) for k in node.keywords if k.arg})
            key = (
                name,
                tuple(sorted((key, tuple(sorted(values))) for key, values in bound.items())),
            )
            if not any(bound.values()) or key in visited:
                return
            visited.add(key)
            local = _Paths(path, function, files, global_paths.names, bound)
            for child in _scope_nodes(function):
                if isinstance(child, ast.Call):
                    calls(child, local, depth + 1)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports(alias.name, node.lineno)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".")
                base = ".".join(parts[: len(parts) - node.level + 1] + ([base] if base else []))
            imports(base, node.lineno)
            for alias in node.names:
                imports(base + "." + alias.name, node.lineno, required=False)
        elif isinstance(node, ast.Call):
            calls(node, context(node))
    return result


_JS_TOKEN = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`|[\w$]+|[^\s]",
    re.DOTALL,
)
_LINK = re.compile(r"\[[^\]]*\]\(<?([^\s)>]+)>?(?:\s+[^)]*)?\)")
_CODE = re.compile(r"`([^`\n]+)`")
_FENCE = re.compile(r"(?m)^\s*(```|~~~)[^\n]*\n(.*?)^\s*\1\s*$", re.DOTALL)
_CODE_PATH = re.compile(r"(?<![\w/])(?:[\w.-]+/)+[\w.*-]+(?:\.[\w.-]+)?")


def references(path: str, raw: bytes, files: set[str]) -> set[Reference]:
    """Source reads only: policy/synthetic string literals are not file dependencies."""
    text = raw.decode("utf-8-sig")
    if path.endswith(".py"):
        return _python(path, text, files)
    result = set()
    if path.endswith((".js", ".cjs", ".mjs")):
        tokens = [m for m in _JS_TOKEN.finditer(text) if not m.group().startswith(("//", "/*"))]
        for index, match in enumerate(tokens[:-3]):
            values = [m.group() for m in tokens[index : index + 4]]
            if (
                values[0:2] != ["require", "("]
                or values[3] != ")"
                or not values[2].startswith(("'", '"'))
            ):
                continue
            name = values[2][1:-1]
            if not name.startswith("."):
                continue  # A package import is provided by its environment, not the source tree.
            target = _path(posixpath.join(posixpath.dirname(path), name))
            variants = [
                target,
                *(target + ext for ext in (".js", ".cjs", ".json")),
                target + "/index.js",
            ]
            target = next((p for p in variants if p in files), target)
            result.add(
                Reference(path, text.count("\n", 0, match.start()) + 1, target, "CJS_REQUIRE")
            )
    if path.endswith(".md"):
        for match in _LINK.finditer(text):
            link = urlsplit(match.group(1))
            if link.scheme or link.netloc or not link.path:
                continue
            target = _path(posixpath.join(posixpath.dirname(path), unquote(link.path))).lstrip("/")
            result.add(
                Reference(path, text.count("\n", 0, match.start()) + 1, target, "MARKDOWN_LINK")
            )
        for match in [*_CODE.finditer(text), *_FENCE.finditer(text)]:
            content = match.group(2) if match.re is _FENCE else match.group(1)
            for value in _CODE_PATH.findall(content):
                target = _path(value)
                if target in files or target.split("/", 1)[0] in {
                    p.split("/", 1)[0] for p in files
                }:
                    result.add(
                        Reference(
                            path,
                            text.count("\n", 0, match.start()) + 1,
                            target,
                            "MARKDOWN_CODE_PATH",
                        )
                    )
    return result


def _installed_node(target: str, blobs: dict[str, bytes], public: set[str]) -> bool:
    if "/node_modules/" not in target:
        return False
    base, package = target.split("/node_modules/", 1)
    parts = package.split("/")
    name = "/".join(parts[:2]) if package.startswith("@") else parts[0]
    if ".." in parts or not name:
        return False
    declaration, lock = base + "/package.json", base + "/package-lock.json"
    if declaration not in public or lock not in public:
        return False
    manifest = json.loads(blobs[declaration])
    locked = json.loads(blobs[lock])
    declared = {**manifest.get("dependencies", {}), **manifest.get("devDependencies", {})}
    entry = locked.get("packages", {}).get("node_modules/" + name, {})
    return bool(name in declared and entry.get("version") == declared[name])


def violations(manifest: dict, blobs: dict[str, bytes]) -> list[Reference]:
    """Refuse a PUBLIC source's dependency absent from the manifest's PUBLIC set.

    The caller supplies its manifest. No default private inventory or workspace is read.
    Directory links require a public directory; recursive copies require every file.
    Node modules require the specifically published, exact declaration and lock.
    """
    public = {row["path"] for row in manifest["public"]}
    files = set(blobs)
    produced = generated_outputs(blobs, public)
    failures = set()
    for path in sorted(public):
        if path not in blobs:
            failures.add(Reference(path, 1, path, "PUBLIC_BLOB_MISSING"))
            continue
        if not path.endswith((".py", ".cjs", ".mjs", ".js", ".md")):
            continue
        for ref in references(path, blobs[path], files):
            target = ref.target
            children = {p for p in files if p.startswith(target.rstrip("/") + "/")}
            if target in public or _installed_node(target, blobs, public):
                continue
            if target not in files and target in produced:
                continue
            if ref.kind == "PYTHON_OPTIONAL_READ" and target not in files:
                private = any(
                    rule["kind"] == "PRIVATE"
                    and any(fnmatch.fnmatchcase(target, pattern) for pattern in rule["patterns"])
                    for rule in manifest.get("rules", [])
                )
                if not private:
                    continue
            if (
                children
                and ref.kind in {"MARKDOWN_LINK", "MARKDOWN_CODE_PATH"}
                and children & public
            ):
                continue
            if children and ref.kind in {"PYTHON_COPY_TREE", "PYTHON_READ"} and children <= public:
                continue
            if any(c in target for c in "*?["):
                matches = {p for p in files if fnmatch.fnmatchcase(p, target)}
                if matches and matches <= public:
                    continue
            failures.add(ref)
    return sorted(failures)
