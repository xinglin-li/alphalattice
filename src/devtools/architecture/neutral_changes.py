"""Which changed Python files are behaviour-neutral, and which files never are (GN).

A module equal to its previous version in syntax -- the parsed tree without its docstrings
and other bare strings, which is what every identity closure by rule hashes (R1) -- runs as
it did and moves no such closure. It is neutral, and the gate's impact step selects no
tests for it (V80), unless something reads its text (V95):

- a byte closure tracks it (a literal path of `source_file_closure_hash`,
  `source_component_hash` or `source_content_hash`); for a byte closure whose files are
  chosen at runtime, every module its file names by path or dotted name, and its whole
  package when it names none, since no scan can then name its files;
- `inspect.getsource` (or `getsourcelines`, `getsourcefile`) reads it, or its `__file__`
  is read, or a `/`-joined path ending in its name is read -- save a path handed to a rule
  primitive (`switched_source_identity`, `source_rule_closure_hash`, or `source_syntax_tree`
  for a reader that walks the syntax rather than hashing it), directly or by the name it is
  assigned to, or joined in a module that hashes by rule and reads no text itself, since
  the rule reads a Python file as syntax;
- its `__doc__` is read (a command's help, a tool's description);
- a function in it is a model's tool (`@tool`, `from_function`), whose docstring is the
  tool's description.

A model's docstring is prose its JSON schema shows but no hash binds: every schema hash
takes the schema's structure alone (SH, SC3), so a docstring edit is neutral too.

A move is proved the same way, definition by definition (V215): a changed file whose only
changes are definitions moved syntax-equal to or from another changed file, and the imports
and `__all__` entries that follow them, behaves as before; its own tests and the tests that
name its module (patching its objects) are the ones a move can break, not every importer's.

    python -m devtools.architecture.neutral_changes [--base REV | --staged]
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath

from alphalattice.kernel.shared_kernel.source_identity import source_bytes_syntax_sha256
from devtools.architecture.identity_closures import closure_sites

_BYTE_PRIMITIVES = frozenset(
    {"source_file_closure_hash", "source_component_hash", "source_content_hash"}
)
_RULE_PRIMITIVES = frozenset(
    {
        "switched_source_identity",
        "source_rule_closure_hash",
        # The Desk helpers that hand their listed paths to `switched_source_identity`.
        "portfolio_adapter_implementation_hash",
        "implementation_content_hash",
        # A reader that walks a module's syntax tree, as the operation table does (V356).
        "source_syntax_tree",
    }
)
"""Calls that read a Python file as its syntax, hashing or walking it: a path handed to them
is not a text read."""
_SOURCE_READERS = frozenset({"getsource", "getsourcelines", "getsourcefile"})
_BYTE_READERS = frozenset({"read_bytes", "read_text"})
_SCANNED = ("src", "scripts")
_PREFILTER = ("getsource", "__file__", "__doc__", "tool", ".py")
_PRIMITIVES_MODULE = "src/alphalattice/kernel/shared_kernel/source_identity.py"
"""The primitives' own module reads every file a rule closure hashes, as syntax: not a text read."""


def source_syntax_tree(path: Path) -> ast.Module:
    """A module's syntax as a tree, for a reader that walks it rather than hashing it.

    The parsed tree without its docstrings and other bare string statements, what a rule
    closure hashes (R1): no prose edit reaches what such a reader finds, so a path handed here
    is not a text read (V356).

    Args:
        path: The module's file.

    Returns:
        Its tree, each body without its bare strings (a body left empty holds `pass`).
    """

    tree = ast.parse(path.read_bytes(), filename=path.name)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.stmt):
            kept = [
                statement
                for statement in body
                if not (
                    isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)
                )
            ]
            node.body = kept or [ast.Pass()]  # type: ignore[attr-defined]
    return tree


@dataclass(frozen=True, slots=True)
class NeutralVerdict:
    """Whether one changed file is behaviour-neutral, and why."""

    path: str
    neutral: bool
    reason: str


def text_sensitive_files(root: Path) -> dict[str, str]:
    """Every source file whose text some code reads, each with the first reason found.

    Args:
        root: The checkout.

    Returns:
        Repository-relative path to the reason its text is read.
    """

    found: dict[str, str] = {}
    sources = {
        path.relative_to(root).as_posix(): path
        for base in _SCANNED
        for path in sorted((root / base).rglob("*.py"))
        if "__pycache__" not in path.parts
    }
    runtime: dict[str, str] = {}
    for site in closure_sites(root):
        if site.primitive not in _BYTE_PRIMITIVES:
            continue
        if site.tracked_paths is None:
            runtime.setdefault(site.path, site.name)
            continue
        for tracked in site.tracked_paths:
            if tracked.endswith(".py"):
                found.setdefault(tracked, f"a byte closure tracks it: {site.name}")
    for relative, name in sorted(runtime.items()):
        tree = ast.parse(sources[relative].read_text(encoding="utf-8"))
        named = _named_modules(tree, sources) | _built_paths(tree, relative, sources) | {relative}
        if named == {relative}:  # it names no module: its package is what it may read
            package = PurePosixPath(relative).parent.as_posix() + "/"
            named = {candidate for candidate in sources if candidate.startswith(package)}
        for candidate in sorted(named):
            found.setdefault(candidate, f"a byte closure chooses it at runtime: {name}")
    for relative, source in sources.items():
        text = source.read_text(encoding="utf-8")
        if relative == _PRIMITIVES_MODULE or not any(token in text for token in _PREFILTER):
            continue
        for target, reason in _text_readers(ast.parse(text), relative, sources):
            found.setdefault(target, f"{reason} in {relative}")
    return found


def _text_readers(
    tree: ast.Module, relative: str, sources: Mapping[str, Path]
) -> Iterable[tuple[str, str]]:
    """The files one module reads the text of, with how."""

    imported = _imported_modules(tree, relative, sources)
    bindings = _path_bindings(tree)
    rule_entries = _rule_entry_nodes(tree)
    call_names = {_name_of(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    # A module that hashes by rule and reads no text itself joins its paths for the rule.
    rule_module = bool(call_names & _RULE_PRIMITIVES) and not (
        call_names & (_BYTE_PRIMITIVES | _BYTE_READERS | {"open"})
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            called = _name_of(node.func)
            if called in _SOURCE_READERS:
                yield relative, "inspect.getsource reads it"
                for argument in node.args:
                    root_name = _root_name(argument)
                    if root_name in imported:
                        yield imported[root_name], "inspect.getsource reads it"
            elif called == "from_function":
                yield relative, "a tool takes its description from a docstring"
            elif called in _BYTE_READERS and isinstance(node.func, ast.Attribute):
                if any(
                    isinstance(inner, ast.Name) and inner.id == "__file__"
                    for inner in ast.walk(node.func.value)
                ):
                    yield relative, "it reads its own bytes"
            elif (
                _hands_its_own_path(node)
                and not _locates_the_root(called)
                and called not in _RULE_PRIMITIVES
            ):
                yield relative, f"its path is handed to {called}"
        elif isinstance(node, ast.Attribute) and node.attr in {"__file__", "__doc__"}:
            if node.attr == "__file__" and (rule_module or id(node) in rule_entries):
                continue
            root_name = _root_name(node.value)
            if root_name in imported:
                yield imported[root_name], f"its {node.attr} is read"
            elif node.attr == "__doc__":
                yield relative, "its __doc__ is read"
        elif isinstance(node, ast.Name) and node.id == "__doc__":
            yield relative, "its __doc__ is read"
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if any(_name_of(_called_or_self(value)) == "tool" for value in node.decorator_list):
                yield relative, "a tool's description is its docstring"
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            suffix = _joined_suffix(node)
            if suffix is None or (
                (rule_module or id(node) in rule_entries) and suffix.endswith(".py")
            ):
                continue
            exact = _checkout_path(_path_parts(node, bindings, relative))
            if exact is not None and exact in sources:
                yield exact, "a path to it is read"
                continue
            # Unresolved, or resolved to no source file: every file the path could end at.
            for candidate in sources:
                if candidate.endswith("/" + suffix):
                    yield candidate, "a path to it is read"


def _rule_entry_nodes(tree: ast.Module) -> set[int]:
    """The nodes whose paths flow into a rule primitive, which reads a Python file as syntax.

    A path expression is a rule entry when it sits in an argument of `switched_source_identity`
    or `source_rule_closure_hash`, or in the value of a name one of them is handed in the same
    function (the listed sources a byte closure had, W9b).
    """

    found: set[int] = set()
    scopes = [
        tree,
        *(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)),
    ]
    for scope in scopes:
        handed: set[str] = set()
        for node in ast.walk(scope):
            if isinstance(node, ast.Call) and _name_of(node.func) in _RULE_PRIMITIVES:
                for argument in (*node.args, *(keyword.value for keyword in node.keywords)):
                    for inner in ast.walk(argument):
                        found.add(id(inner))
                        if isinstance(inner, ast.Name):
                            handed.add(inner.id)
        for node in ast.walk(scope):
            if isinstance(node, ast.Assign | ast.AnnAssign) and node.value is not None:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(target, ast.Name) and target.id in handed for target in targets):
                    found.update(id(inner) for inner in ast.walk(node.value))
    return found


def _hands_its_own_path(call: ast.Call) -> bool:
    """Whether a call receives the module's own path, as ``__file__`` or ``Path(__file__)``.

    The Risk estimators hand ``Path(__file__)`` to ``implementation_content_hash``, which
    hashes the bytes (IS2); the path, not a read, is what the module itself writes.
    """

    def own(node: ast.expr) -> bool:
        if isinstance(node, ast.Name):
            return node.id == "__file__"
        return (
            isinstance(node, ast.Call)
            and _name_of(node.func) == "Path"
            and len(node.args) == 1
            and own(node.args[0])
        )

    return any(own(value) for value in (*call.args, *(k.value for k in call.keywords)))


def _locates_the_root(called: str | None) -> bool:
    """Whether the call only locates a path, as ``Path`` and the root finders do.

    A root finder (`resolve_playpen_root`) takes a module's path to find the checkout; it
    reads no text.
    """

    return called is None or called == "Path" or called.endswith("root")


def _imported_modules(
    tree: ast.Module, relative: str, sources: Mapping[str, Path]
) -> dict[str, str]:
    """Local names bound to a module, or to an object of one, mapped to that module's file."""

    package = PurePosixPath(relative).with_suffix("").parts[1:-1]
    bound: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                file = _module_file(alias.name, sources)
                if file is not None and alias.asname is not None:
                    bound[alias.asname] = file
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parent = package[: len(package) - node.level + 1]
                base = ".".join((*parent, *filter(None, base.split("."))))
            for alias in node.names:
                file = _module_file(f"{base}.{alias.name}", sources) or _module_file(base, sources)
                if file is not None:
                    bound[alias.asname or alias.name] = file
    return bound


def _named_modules(tree: ast.Module, sources: Mapping[str, Path]) -> set[str]:
    """The source files a module names in its strings, by path or by dotted name."""

    named: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        value = node.value.strip()
        if value.endswith(".py") and "/" in value:
            tail = value.lstrip("/")
            # A repository path names its file whole; a shorter one, any file it ends.
            named.update(path for path in sources if path == tail or path.endswith("/" + tail))
        elif "." in value and all(part.isidentifier() for part in value.split(".")):
            tail = value.replace(".", "/")
            named.update(
                path
                for path in sources
                if path.endswith("/" + tail + ".py") or path.endswith("/" + tail + "/__init__.py")
            )
    return named


def _module_file(dotted: str, sources: Mapping[str, Path]) -> str | None:
    base = "src/" + dotted.replace(".", "/")
    for candidate in (base + ".py", base + "/__init__.py"):
        if candidate in sources:
            return candidate
    return None


_ROOT = "<root>"
"""A path's first part when it is the checkout's root: a parameter, an attribute or a call."""


def _path_bindings(tree: ast.Module) -> dict[str, list[ast.expr]]:
    """The values each name is assigned in the module, for resolving the paths built from it."""

    bindings: dict[str, list[ast.expr]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                bindings.setdefault(target.id, []).append(node.value)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            if isinstance(node.target, ast.Name):
                bindings.setdefault(node.target.id, []).append(node.value)
    return bindings


def _path_parts(
    node: ast.expr,
    bindings: Mapping[str, list[ast.expr]],
    module: str,
    depth: int = 0,
    seen: frozenset[str] = frozenset(),
) -> tuple[str, ...] | None:
    """A path built with `/` from string literals, as its parts from the checkout's root.

    A name is followed through its one assigned value; `Path(__file__)`, `.parent`,
    `.parents[n]` and `.resolve()` are read from the module's own path; a name the module
    never assigns, another attribute or another call stands for the root (`playpen_root`,
    `self.root`). None when a part cannot be read: a name assigned different paths, or a
    variable part.
    """

    if depth > 32:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return tuple(part for part in node.value.split("/") if part)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _path_parts(node.left, bindings, module, depth + 1, seen)
        right = _path_parts(node.right, bindings, module, depth + 1, seen)
        if left is None or right is None or _ROOT in right:
            return None
        return left + right
    if isinstance(node, ast.Name):
        if node.id == "__file__":
            return (_ROOT, *module.split("/"))
        values = bindings.get(node.id)
        if values is None or node.id in seen:  # a parameter, or rebound from itself
            return (_ROOT,)
        inner_seen = seen | {node.id}
        resolved = {_path_parts(value, bindings, module, depth + 1, inner_seen) for value in values}
        return next(iter(resolved)) if len(resolved) == 1 else None
    if isinstance(node, ast.Attribute) and node.attr == "parent":
        inner = _path_parts(node.value, bindings, module, depth + 1, seen)
        return None if inner is None or len(inner) < 2 else inner[:-1]
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "parents"
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, int)
    ):
        inner = _path_parts(node.value.value, bindings, module, depth + 1, seen)
        cut = node.slice.value + 1
        return None if inner is None or len(inner) <= cut else inner[:-cut]
    if isinstance(node, ast.Call):
        called = _name_of(node.func)
        if called in {"resolve", "absolute"} and isinstance(node.func, ast.Attribute):
            return _path_parts(node.func.value, bindings, module, depth + 1, seen)
        if called in {"Path", "PurePath", "PurePosixPath"} and len(node.args) == 1:
            inner = _path_parts(node.args[0], bindings, module, depth + 1, seen)
            if inner is None:
                return None
            return inner if inner[:1] == (_ROOT,) else (_ROOT, *inner)
        if any(isinstance(inner, ast.Name) and inner.id == "__file__" for inner in ast.walk(node)):
            return None
        return (_ROOT,)
    if isinstance(node, ast.Attribute):
        return (_ROOT,)
    return None


def _checkout_path(parts: tuple[str, ...] | None) -> str | None:
    """The checkout-relative path of resolved parts, or None when they start anywhere else."""

    if not parts or parts[0] != _ROOT or len(parts) == 1:
        return None
    return "/".join(parts[1:])


def _built_paths(tree: ast.Module, relative: str, sources: Mapping[str, Path]) -> set[str]:
    """The source files a byte closure reaches through paths it builds: joins and globs (V254)."""

    bindings = _path_bindings(tree)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            exact = _checkout_path(_path_parts(node, bindings, relative))
            if exact is not None and exact in sources:
                found.add(exact)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"glob", "rglob"}
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            directory = _checkout_path(_path_parts(node.func.value, bindings, relative))
            if directory is None:
                continue
            pattern = node.args[0].value
            for candidate in sources:
                place = PurePosixPath(candidate)
                parent = place.parent.as_posix()
                inside = parent == directory or (
                    node.func.attr == "rglob" and parent.startswith(directory + "/")
                )
                if inside and fnmatch(place.name, pattern):
                    found.add(candidate)
    return found


def _joined_suffix(node: ast.BinOp) -> str | None:
    """`root / "a" / "b.py"` read as `a/b.py`, when its last part names a module."""

    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.BinOp) and isinstance(current.op, ast.Div):
        if not isinstance(current.right, ast.Constant) or not isinstance(current.right.value, str):
            break
        parts.append(current.right.value)
        current = current.left
    if not parts or not parts[0].endswith(".py") or parts == ["__init__.py"]:
        return None  # a bare `__init__.py` names every package, so it names none
    return "/".join(reversed(parts))


def _name_of(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _called_or_self(node: ast.expr) -> ast.expr:
    return node.func if isinstance(node, ast.Call) else node


def _root_name(node: ast.expr) -> str | None:
    while isinstance(node, ast.Attribute | ast.Call | ast.Subscript):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None


def neutral_verdicts(
    root: Path,
    changed: Iterable[str],
    previous: Callable[[str], bytes | None],
    *,
    sensitive: Mapping[str, str] | None = None,
) -> tuple[NeutralVerdict, ...]:
    """Which of the changed Python files are behaviour-neutral.

    Args:
        root: The checkout, holding each file's current version.
        changed: The changed paths, repository-relative; other than `.py` files are skipped.
        previous: Each path's previous bytes, or None when it did not exist.
        sensitive: `text_sensitive_files(root)`, when the caller has it.

    Returns:
        One verdict per changed Python file.
    """

    sensitive = text_sensitive_files(root) if sensitive is None else sensitive
    verdicts = []
    for path in sorted(set(changed)):
        if not path.endswith(".py"):
            continue
        before = previous(path)
        current = root / path
        if before is None:
            verdicts.append(NeutralVerdict(path, False, "added"))
        elif not current.is_file():
            verdicts.append(NeutralVerdict(path, False, "deleted"))
        elif path in sensitive:
            verdicts.append(NeutralVerdict(path, False, sensitive[path]))
        else:
            verdicts.append(_compare(path, before, current.read_bytes()))
    return tuple(verdicts)


def _compare(path: str, before: bytes, after: bytes) -> NeutralVerdict:
    try:
        equal = source_bytes_syntax_sha256(before) == source_bytes_syntax_sha256(after)
    except ValueError:
        return NeutralVerdict(path, False, "unparsable")
    if not equal:
        return NeutralVerdict(path, False, "syntax differs")
    return NeutralVerdict(path, True, "syntax equal")


# -- moves ------------------------------------------------------------------------------------

_Definitions = dict[str, str]


def moved_files(
    root: Path, changed: Iterable[str], previous: Callable[[str], bytes | None]
) -> dict[str, frozenset[str]]:
    """The changed Python files whose changes are only a move, with the names each moved.

    A definition is moved when a top-level function, class or assignment leaves one changed
    file and arrives syntax-equal in another. A file is only part of a move when, beyond the
    definitions it gave or took, the only lines that changed are its imports and `__all__`,
    and every import it changed names a moved definition.

    Args:
        root: The checkout, holding each file's current version.
        changed: The changed paths.
        previous: Each path's previous bytes, or None when it did not exist.

    Returns:
        Each moved file's path to the names it gave or took.
    """

    trees: dict[str, tuple[ast.Module | None, ast.Module | None]] = {}
    for path in sorted(set(changed)):
        if not path.endswith(".py"):
            continue
        before, after = previous(path), root / path
        try:
            trees[path] = (
                None if before is None else ast.parse(before),
                ast.parse(after.read_bytes()) if after.is_file() else None,
            )
        except SyntaxError:
            return {}
    gone: dict[tuple[str, str], str] = {}
    came: dict[tuple[str, str], str] = {}
    for path, (before_tree, after_tree) in trees.items():
        old = _definitions(before_tree)
        new = _definitions(after_tree)
        for name, digest in old.items():
            if new.get(name) != digest:
                gone[(name, digest)] = path
        for name, digest in new.items():
            if old.get(name) != digest:
                came[(name, digest)] = path
    moved = {key for key in gone.keys() & came.keys() if gone[key] != came[key]}
    names = {name for name, _digest in moved}
    result: dict[str, frozenset[str]] = {}
    for path, (before_tree, after_tree) in trees.items():
        out = {name for (name, digest) in moved if gone[(name, digest)] == path}
        into = {name for (name, digest) in moved if came[(name, digest)] == path}
        if _residue(before_tree, drop=out) != _residue(after_tree, drop=into):
            continue
        if not _imports_changed_only_for(before_tree, after_tree, names):
            continue
        if out or into or _imports(before_tree) != _imports(after_tree):
            result[path] = frozenset(out | into)
    return result


def _definitions(tree: ast.Module | None) -> _Definitions:
    found: _Definitions = {}
    for node in tree.body if tree is not None else ():
        name = _defined_name(node)
        if name is not None and name != "__all__":
            found[name] = _digest(node)
    return found


def _defined_name(node: ast.stmt) -> str | None:
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return node.name
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        return target.id if isinstance(target, ast.Name) else None
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def _residue(tree: ast.Module | None, *, drop: set[str]) -> str:
    """The module without its imports, its `__all__` and the named definitions."""

    kept = [
        _digest(node)
        for node in (tree.body if tree is not None else ())
        if not isinstance(node, ast.Import | ast.ImportFrom)
        and _defined_name(node) not in {*drop, "__all__"}
        and not _bare_string(node)
    ]
    return hashlib.sha256("\n".join(kept).encode("utf-8")).hexdigest()


def _imports(tree: ast.Module | None) -> frozenset[tuple[str, str]]:
    """Each name a module imports, with the module it comes from."""

    found: set[tuple[str, str]] = set()
    for node in tree.body if tree is not None else ():
        if isinstance(node, ast.Import):
            found.update((alias.asname or alias.name, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            source = "." * node.level + (node.module or "")
            found.update((alias.asname or alias.name, source) for alias in node.names)
    return frozenset(found)


def _imports_changed_only_for(
    before: ast.Module | None, after: ast.Module | None, moved: set[str]
) -> bool:
    changed = _imports(before) ^ _imports(after)
    return all(name in moved for name, _source in changed)


def _digest(node: ast.AST) -> str:
    """A statement's syntax digest, docstrings aside, wherever it stands."""

    return source_bytes_syntax_sha256(ast.unparse(node).encode("utf-8"))


def _bare_string(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


# -- the command ------------------------------------------------------------------------------


def git_previous(root: Path, revision: str) -> Callable[[str], bytes | None]:
    """Read each path's bytes at `revision`, None where it did not exist.

    Args:
        root: The checkout.
        revision: The commit the change is measured against.

    Returns:
        The reader.
    """

    def read(path: str) -> bytes | None:
        completed = subprocess.run(
            ("git", "-C", str(root), "show", f"{revision}:{path}"),
            capture_output=True,
            check=False,
        )
        return completed.stdout if completed.returncode == 0 else None

    return read


def main(argv: Sequence[str] | None = None) -> int:
    """Print each changed Python file's verdict and the moves.

    Args:
        argv: The command's arguments.

    Returns:
        The exit code.
    """

    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--base")
    choice.add_argument("--staged", action="store_true")
    args = parser.parse_args(argv)
    root = Path.cwd()
    diff = ["git", "-C", str(root), "diff", "--name-only", "--diff-filter=ACDMR"]
    if args.staged:
        diff.append("--cached")
    elif args.base:
        diff.append(args.base)
    changed = subprocess.run(diff, capture_output=True, text=True, check=True).stdout.split()
    previous = git_previous(root, args.base or "HEAD")
    for verdict in neutral_verdicts(root, changed, previous):
        print(f"{'NEUTRAL' if verdict.neutral else 'CHANGED'} {verdict.path}: {verdict.reason}")
    for path, names in sorted(moved_files(root, changed, previous).items()):
        print(f"MOVED {path}: {', '.join(sorted(names)) or 'imports only'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
