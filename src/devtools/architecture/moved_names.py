"""Find usages before a safe rename: no reference or test patch left at a name's old place."""

from __future__ import annotations

import ast
import subprocess
from collections.abc import Mapping
from pathlib import Path

SCANNED = ("src", "scripts", "tests", "tools/tests")
PATCHES = {"setattr", "delattr", "object", "patch"}
FIRST_PARTY = ("alphalattice.", "devtools.", "scripts.", "tests.")
DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
Change = tuple[str, str | None, str | None, str | None]  # path, before, after, new path


def module_of(path: str) -> str:  # its dotted module; `src/` is the import root
    parts = path.removesuffix(".py").removeprefix("src/").split("/")
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _package(path: str) -> str:  # where its relative imports resolve
    return module_of(path) if path.endswith("__init__.py") else module_of(path).rpartition(".")[0]


def _from(node: ast.ImportFrom, package: str) -> str:
    parts = package.split(".")[: len(package.split(".")) - node.level + 1] if node.level else []
    return ".".join([*parts, *filter(None, [node.module])])


def _bound(text: str | None, package: str) -> tuple[set[str], dict[str, str]]:
    """A module's own top-level names, and those it binds from another module at import."""
    own, imported = set[str](), dict[str, str]()
    top = ast.parse(text).body if text else []  # and one level into a top-level if or try
    nested = [n for s in top if isinstance(s, ast.If | ast.Try) for n in ast.iter_child_nodes(s)]
    for node in [*top, *nested]:
        if isinstance(node, DEFS):
            own.add(node.name)
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            own |= {t.id for t in targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.ImportFrom):
            imported |= {a.asname or a.name: f"{_from(node, package)}.{a.name}" for a in node.names}
    return own, imported


def _dotted(node: ast.AST, names: Mapping[str, set[str]]) -> list[str]:  # every path it may name
    if isinstance(node, ast.Attribute):
        return [f"{head}.{node.attr}" for head in _dotted(node.value, names)]
    return sorted(names.get(node.id, ())) if isinstance(node, ast.Name) else []


def _named(node: ast.AST, names: Mapping[str, set[str]], package: str) -> list[tuple[object, bool]]:
    """What a node names, and if it patches it: an import, attribute, (m, "n") call or string."""
    if isinstance(node, ast.Import | ast.ImportFrom):
        source = _from(node, package) if isinstance(node, ast.ImportFrom) else ""
        return [(source, False), *((f"{source}.{a.name}".lstrip("."), False) for a in node.names)]
    if isinstance(node, ast.Call) and node.args:
        patch = getattr(node.func, "attr", getattr(node.func, "id", "")) in PATCHES
        name = getattr(node.args[1], "value", None) if node.args[1:] else None
        objects = [(f"{d}.{name}", patch) for d in _dotted(node.args[0], names)]
        return [*objects, (getattr(node.args[0], "value", None), patch)]
    if isinstance(node, ast.Attribute):
        return [(d, False) for d in _dotted(node, names)]
    return [(node.value, False)] if isinstance(node, ast.Constant) else []


def references(text: str, targets: set[str], package: str) -> dict[tuple[int, str], bool]:
    """Each line and target `text` names (at or below it), and whether it patches it itself."""
    tree, names, found = ast.parse(text), dict[str, set[str]](), dict[tuple[int, str], bool]()
    for node in ast.walk(tree):
        for a in node.names if isinstance(node, ast.Import | ast.ImportFrom) else ():
            source = f"{_from(node, package)}." if isinstance(node, ast.ImportFrom) else ""
            alias = a.asname or (a.name if source else a.name.split(".")[0])
            names.setdefault(alias, set()).add(source + a.name if source or a.asname else alias)
    for node in ast.walk(tree):
        for d, patch in _named(node, names, package):
            for target in [t for t in targets if f"{d}.".startswith(f"{t}.")]:
                key = (getattr(node, "lineno", 0), target)
                found[key] = found.get(key, False) or (patch and d == target)
    return found


def _moved(changes: list[Change]) -> tuple[dict[str, str | None], dict[str, str], dict[str, str]]:
    """The names gone (with a renamed one's home), those newly bound at import, and new names."""
    gone, hoisted, homes = dict[str, str | None](), dict[str, str](), dict[str, str]()
    for path, before, after, renamed in changes:
        module, package = module_of(path), _package(path)
        (own, imports), (now, now_imports) = _bound(before, package), _bound(after, package)
        homes |= (added := {n: f"{module}.{n}" for n in now - own})
        only = next(iter(added.values())) if len(added) == 1 else None
        removed = {*own, *imports} - now - set(now_imports) if after is not None else set()
        gone |= {f"{module}.{n}": only for n in removed}
        gone |= {module: renamed and module_of(renamed)} if after is None else {}
        bound = {t: f"{module}.{n}" for n, t in now_imports.items() if imports.get(n) != t}
        hoisted |= {t: u for t, u in bound.items() if before and t.startswith(FIRST_PARTY)}
    return gone, hoisted, homes


def _uses(texts: Mapping[str, str], target: str) -> bool:  # its home's code reads it when called
    home, _, name = target.rpartition(".")
    tops = ast.parse(texts[home]).body if name in texts.get(home, "") else []
    bodies = [s for d in tops if isinstance(d, DEFS) for s in d.body]
    return any(isinstance(n, ast.Name) and n.id == name for s in bodies for n in ast.walk(s))


def moved_name_problems(changes: list[Change], sources: Mapping[str, str]) -> list[str]:
    """Each reference left at a moved name's old place, and each test patch a hoist bypasses."""
    (gone, hoisted, homes), found = _moved(changes), []
    texts = {module_of(p): t for p, t in sources.items()} if hoisted else {}
    hoisted = {t: u for t, u in hoisted.items() if not _uses(texts, t)}  # else a patch there acts
    for path, text in sources.items():
        scope = {*gone, *hoisted} if path.startswith("tests/") else set(gone)
        wanted = {t for t in scope if all(w in text for w in t.rsplit(".", 2)[-2:])}
        hits = references(text, wanted, _package(path)) if wanted else {}
        for (line, target), patch in hits.items():
            name, user = target.rsplit(".", 1)[-1], hoisted.get(target, "")
            if target in gone:
                home = gone[target] or homes.get(name)
                where = f"it is now {home}" if home else "it is gone"
                found.append(f"{path}:{line}: {target} moved or was removed; {where}")
            elif patch and user.rsplit(".", 2)[-2] in text:
                why = f"bound at import as {user}; patch {user} where it is used"
                found.append(f"{path}:{line}: patches {target}, {why}")
    return sorted(found)


def problems(root: Path, *, staged: bool) -> list[str]:
    """This commit's (or the working tree's) changed modules, checked across `SCANNED`."""

    def git(*args: str) -> str:
        done = subprocess.run(("git", *args), cwd=root, capture_output=True, encoding="utf-8")
        return done.stdout if done.returncode == 0 else ""

    def read(p: str) -> str:  # a staged check reads the index where the working tree differs
        return git("show", f":{p}") if p in dirty else (root / p).read_text("utf-8", "replace")

    changes: list[Change] = []
    dirty = set(git("diff", "--name-only").splitlines()) if staged else set()
    diff = ("diff", "--cached" if staged else "HEAD", "--name-status", "-M", "--", "*.py")
    for status, *paths in (row.split("\t") for row in git(*diff).splitlines()):
        old, new = paths[0], paths[-1]
        after = None if status[0] in "DR" else read(new)
        changes.append((old, git("show", f"HEAD:{old}") or None, after, (old != new) * new or None))
    listed = git("ls-files", "--", *(f"{r}/*.py" for r in SCANNED)).splitlines() if changes else []
    files = {p: read(p) for p in listed if p in dirty or (root / p).is_file()}
    return moved_name_problems(changes, files)
