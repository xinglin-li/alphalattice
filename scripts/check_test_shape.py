"""Refuse growing source shape and test-writing violations in staged Git blobs.

The weekly JSON report records existing debt and static review candidates.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import subprocess
import sys
import tokenize
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAP_BYTES = 100 * 1024
SLEEP = re.compile(r"\bsleep\(\s*(\d+(?:\.\d+)?)\s*\)")
# a literal of five or more words that an assert or an includes() checks: a sentence pin
PIN = re.compile(r"""(?:\bincludes\(\s*|\bassert\s+)(['"`])([^'"`\n]{0,400})\1""")


ASSETS = "src/alphalattice/interface/local_application/assets/"
GENERATED = frozenset(
    ASSETS + "workbench" + suffix
    for suffix in (".html", ".css", ".js", ".zh.js", "-prelude.js", "-manifest.json")
)
CONTROL_OWNER = "src/alphalattice/kernel/shared_kernel/environment.py"
WRITE = re.compile(
    r'\b(?:INSERT(?:\s+OR\s+\w+)?\s+INTO|UPDATE|DELETE\s+FROM|MERGE\s+INTO)\s+([\w."`]+)',
    re.I,
)
CODE = frozenset(
    {".py", ".js", ".cjs", ".mjs", ".ts", ".tsx", ".jsx", ".css", ".html", ".sql", ".sh", ".ps1"}
)
DEFINITION_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
FUNCTION_TYPES = DEFINITION_TYPES[:-1]
TOKENS = re.compile(
    r""""(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`"""
    r"|/\*.*?\*/|<!--.*?-->|//[^\n]*|===|!==|=>|\+\+|--|&&|\|\||\?\?"
    r"""|\w+|[^\w\s"'`/]|/(?![/*])""",
    re.S,
)


def _tree(text: str) -> ast.Module:
    try:
        return ast.parse(text)
    except SyntaxError:
        return ast.Module(body=[], type_ignores=[])


def _body(node: ast.AST):
    for child in ast.iter_child_nodes(node):
        if not isinstance(child, DEFINITION_TYPES):
            yield child
            yield from _body(child)


def _definitions(tree: ast.AST, prefix: str = ""):
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, DEFINITION_TYPES):
            name = prefix + node.name
            yield name, node
            yield from _definitions(node, name + ".")
        else:
            yield from _definitions(node, prefix)


def _metrics(node: ast.AST) -> tuple[int, int]:
    decisions = sum(
        isinstance(
            item, ast.If | ast.For | ast.AsyncFor | ast.While | ast.IfExp | ast.ExceptHandler
        )
        + (len(item.values) - 1 if isinstance(item, ast.BoolOp) else 0)
        + (len(item.cases) if isinstance(item, ast.Match) else 0)
        + (1 + len(item.ifs) if isinstance(item, ast.comprehension) else 0)
        for item in _body(node)
    )
    return (node.end_lineno or node.lineno) - node.lineno + 1, decisions


def _sql(node: ast.AST, values: dict[str, str]) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return values.get(node.id, "")
    if isinstance(node, ast.JoinedStr):
        return "".join(_sql(part, values) or "?" for part in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _sql(node.left, values) + _sql(node.right, values)
    return ""


def _lint(path: str, tree: ast.Module) -> set[str]:
    imports = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update((item.asname or item.name, item.name) for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.update(
                (item.asname or item.name, (node.module or "") + "." + item.name)
                for item in node.names
            )

    def called(node):
        if isinstance(node, ast.Name):
            return imports.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return called(node.value) + "." + node.attr
        return ""

    found = set()
    definitions = list(_definitions(tree))
    factories = {
        node.name
        for _name, node in definitions
        if called(getattr(node, "returns", None)) == "duckdb.DuckDBPyConnection"
    }
    for _name, unit in [("", tree), *definitions]:
        nodes = list(_body(unit))
        values = {}
        connections = {
            node.arg
            for node in nodes
            if isinstance(node, ast.arg)
            and node.annotation
            and called(node.annotation) == "duckdb.DuckDBPyConnection"
        }
        for node in nodes:
            if isinstance(node, ast.Assign):
                values.update(
                    (target.id, _sql(node.value, values))
                    for target in node.targets
                    if isinstance(target, ast.Name)
                )
            pairs = (
                [(target, node.value) for target in node.targets]
                if isinstance(node, ast.Assign)
                else [(item.optional_vars, item.context_expr) for item in node.items]
                if isinstance(node, ast.With | ast.AsyncWith)
                else []
            )
            connections.update(
                target.id
                for target, value in pairs
                if isinstance(target, ast.Name)
                and (
                    (isinstance(value, ast.Name) and value.id in connections)
                    or (
                        isinstance(value, ast.Call)
                        and (
                            called(value.func) == "duckdb.connect"
                            or called(value.func).rsplit(".", 1)[-1] in factories
                            or (
                                isinstance(value.func, ast.Attribute)
                                and value.func.attr == "cursor"
                                and called(value.func.value) in connections
                            )
                        )
                    )
                )
            )
        loop_calls = {
            id(child)
            for node in nodes
            if isinstance(node, ast.For | ast.AsyncFor | ast.While)
            for child in _body(node)
            if isinstance(child, ast.Call)
        }
        for node in nodes:
            if not isinstance(node, ast.Call):
                continue
            if path != CONTROL_OWNER and called(node.func) in {
                "threadpoolctl.threadpool_limits",
                "threadpoolctl.ThreadpoolController",
            }:
                found.add(
                    f"{path}:{node.lineno}: threadpoolctl control; use numerical_thread_limit"
                )
            if (
                not node.args
                or not isinstance(node.func, ast.Attribute)
                or node.func.attr not in {"execute", "executemany", "sql"}
            ):
                continue
            writes = list(WRITE.finditer(_sql(node.args[0], values)))
            if any(
                match.group(1).lower().rsplit(".", 1)[-1].strip('"`') == "feature_ineligibility"
                for match in writes
            ):
                found.add(f"{path}:{node.lineno}: view mutation; write feature_ineligibility_run")
            if (
                writes
                and id(node) in loop_calls
                and node.func.attr in {"execute", "executemany"}
                and called(node.func.value) in connections
            ):
                found.add(
                    f"{path}:{node.lineno}: DuckDB row-loop write; batch rows at storage owner"
                )
    return found


@lru_cache(maxsize=64)
def _code_lines(path: str, text: str) -> list[tuple[int, str]]:
    if Path(path).suffix not in CODE or path in GENERATED:
        return []
    rows = defaultdict(list)
    if path.endswith(".py"):
        docs = {
            (node.body[0].value.lineno, node.body[0].value.col_offset)
            for node in ast.walk(_tree(text))
            if isinstance(node, (ast.Module, *DEFINITION_TYPES))
            and ast.get_docstring(node) is not None
        }
        try:
            for token in tokenize.generate_tokens(io.StringIO(text).readline):
                if (
                    token.type
                    in {
                        tokenize.NAME,
                        tokenize.NUMBER,
                        tokenize.STRING,
                        tokenize.OP,
                        tokenize.ERRORTOKEN,
                    }
                    and token.start not in docs
                ):
                    rows[token.start[0]].append(token.string)
        except (tokenize.TokenError, IndentationError):
            return []
    else:
        line, position = 1, 0
        for token in TOKENS.finditer(text):
            line += text.count("\n", position, token.start())
            position = token.start()
            if not token.group().startswith(("/*", "//", "<!--")):
                rows[line].append(token.group())
    return [(line, " ".join(tokens)) for line, tokens in sorted(rows.items())]


def _windows(rows: list[tuple[int, str]]):
    return [
        (rows[index][0], tuple(text for _line, text in rows[index : index + 8]))
        for index in range(len(rows) - 7)
    ]


def _added_windows(path: str, old: str | None, new: str):
    before, after = _code_lines(path, old or ""), _code_lines(path, new)
    added = {
        index
        for tag, _a, _b, start, end in SequenceMatcher(
            a=[row[1] for row in before], b=[row[1] for row in after]
        ).get_opcodes()
        if tag in {"insert", "replace"}
        for index in range(start, end)
    }
    kept = Counter(window for _line, window in _windows(before))
    counts = Counter(window for _line, window in _windows(after))
    wanted = {
        window
        for index, (_line, window) in enumerate(_windows(after))
        if counts[window] > kept[window] and all(item in added for item in range(index, index + 8))
    }
    return after, wanted


def clone_problems(path: str, old: str | None, new: str, corpus: dict[str, str]) -> list[str]:
    """New exact eight-line code copies, with their original source locations."""
    after, wanted = _added_windows(path, old, new)
    if not wanted:
        return []
    anchors = re.compile(
        "|".join(re.escape("".join(max(window, key=len).split())) for window in wanted)
    )
    origins = defaultdict(list)
    for source, text in sorted(corpus.items()):
        if source not in GENERATED and anchors.search("".join(text.split())):
            for line, window in _windows(_code_lines(source, text)):
                if window in wanted:
                    origins[window].append((source, line))
    found, last = [], -8
    for index, (line, window) in enumerate(_windows(after)):
        others = [place for place in origins[window] if place != (path, line)]
        if window in wanted and others and index >= last + 8:
            original, start = others[0]
            found.append(f"{path}:{line}: copied 8 lines from {original}:{start}; call its owner")
            last = index
    return found


def _docstring_lines(text: str) -> dict[str, int]:
    return {
        node.name: len((ast.get_docstring(node) or "").splitlines())
        for node in ast.walk(_tree(text))
        if isinstance(node, FUNCTION_TYPES) and node.name.startswith("test_")
    }


def problems(path: str, old: str | None, new: str) -> list[str]:
    """Growth a staged source or test change refuses; existing debt only shrinks."""
    if path in GENERATED:
        return []
    if path.startswith("src/"):
        found = []
        size = len(new.encode("utf-8", errors="surrogateescape"))
        if size > CAP_BYTES and size > len((old or "").encode("utf-8", errors="surrogateescape")):
            found.append(f"{path}:1: source size {size} > {CAP_BYTES}; split at its owner")
        if path.endswith(".py"):
            before, after = _tree(old or ""), _tree(new)
            was = {
                name: _metrics(node)
                for name, node in _definitions(before)
                if isinstance(node, FUNCTION_TYPES)
            }
            for name, node in _definitions(after):
                if isinstance(node, FUNCTION_TYPES):
                    for metric, count, previous, cap in zip(
                        ("physical lines", "decisions"),
                        _metrics(node),
                        was.get(name, (0, 0)),
                        (120, 20),
                        strict=True,
                    ):
                        if count > cap and count > previous:
                            found.append(
                                f"{path}:{node.lineno}: function {name} "
                                f"{count} {metric} > {cap}; split its rule"
                            )
            debt = Counter(line.split(": ", 1)[1] for line in _lint(path, before))
            current = sorted(_lint(path, after))
            counts = Counter(line.split(": ", 1)[1] for line in current)
            found += [
                line
                for line in current
                if counts[line.split(": ", 1)[1]] > debt[line.split(": ", 1)[1]]
            ]
        return found
    if not path.startswith("tests/"):
        return []
    found = []
    size = len(new.encode("utf-8", errors="surrogateescape"))
    if size > CAP_BYTES and size > len((old or "").encode("utf-8", errors="surrogateescape")):
        found.append(f"{path}: grows to {size // 1024} KB; a test file stays under 100 KB (rule 9)")
    # a line only re-indented is kept, not added
    before = {line.strip() for line in (old or "").splitlines()}
    added = [line for line in new.splitlines() if line.strip() not in before]
    if not path.startswith("tests/structural/"):
        for line in added:
            pin = PIN.search(line)
            if pin and len(pin.group(2).split()) >= 5:
                found.append(
                    f"{path}: pins '{pin.group(2)[:40]}'; check its key and facts (rule 4)"
                )
    if not path.endswith(".py"):
        return found
    if not path.startswith("tests/structural/") and any("inspect.getsource(" in x for x in added):
        found.append(f"{path}: reads source text; test the behaviour at the owner's seam (rule 3)")
    for line in added:
        match = SLEEP.search(line)
        if match and float(match.group(1)) >= 1:
            found.append(f"{path}: sleeps {match.group(1)} s; drive the clock or the step (rule 7)")
    was = _docstring_lines(old or "")
    for name, count in _docstring_lines(new).items():
        if count > 3 and count > was.get(name, 0):
            found.append(
                f"{path}::{name}: a {count}-line docstring; one requirement sentence (rule 1)"
            )
    return found


def _git(*args: str, root: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        input="",
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _blobs(specs: list[str], root: Path = ROOT) -> dict[str, str]:
    result = subprocess.run(
        ["git", "cat-file", "--batch"],
        input=("\n".join(specs) + "\n").encode(),
        cwd=root,
        capture_output=True,
        check=True,
    )
    stream, found = io.BytesIO(result.stdout), {}
    for spec in specs:
        header = stream.readline().split()
        if header[-1:] == [b"missing"]:
            continue
        data = stream.read(int(header[-1]))
        stream.read(1)
        found[spec] = data.decode("utf-8", errors="surrogateescape")
    return found


def weekly_report(root: Path = ROOT) -> dict:
    """Static review candidates, exact copies, seven-day growth and current shape debt."""
    paths = _git(
        "ls-files", "-z", "--", "src", "scripts", "tests", "config", root=root
    ).stdout.split("\0")
    texts = _blobs(["HEAD:" + path for path in paths if path], root)
    sources = {spec[5:]: text for spec, text in texts.items()}
    trees = {path: _tree(text) for path, text in sources.items() if path.endswith(".py")}

    def readers(tree):
        found = Counter()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                found.update([node.id])
            elif isinstance(node, ast.Attribute):
                found.update([node.attr])
            elif isinstance(node, ast.alias):
                found.update(node.name.split("."))
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                found.update(re.findall(r"\b[A-Za-z_]\w*\b", node.value))
        return found

    mentions = sum((readers(tree) for tree in trees.values()), Counter())
    for path, text in sources.items():
        if path.startswith("config/"):
            mentions.update(re.findall(r"\b[A-Za-z_]\w*\b", text))
    candidates = [
        {"path": path, "line": node.lineno, "name": name, "kind": type(node).__name__}
        for path, tree in trees.items()
        if path.startswith("src/")
        for name, node in _definitions(tree)
        if mentions[node.name] <= readers(node)[node.name]
    ]
    index, copies = {}, []
    for path, text in sorted(sources.items()):
        if not path.startswith("src/"):
            continue
        last = -8
        for offset, (line, window) in enumerate(_windows(_code_lines(path, text))):
            if window in index and offset >= last + 8:
                original, start = index[window]
                copies.append(
                    {"original": original, "original_line": start, "copy": path, "line": line}
                )
                last = offset
            index.setdefault(window, (path, line))
    base = _git("rev-list", "-1", "--before=7 days ago", "HEAD", root=root).stdout.strip()
    growth = _git(
        "diff",
        "--numstat",
        base or _git("hash-object", "-t", "tree", "--stdin", root=root).stdout.strip(),
        "HEAD",
        "--",
        "src",
        root=root,
    )
    rows = []
    for row in growth.stdout.splitlines():
        added, removed, path = row.split("\t", 2)
        if added.isdecimal() and removed.isdecimal():
            rows.append(
                {
                    "path": path,
                    "added": int(added),
                    "removed": int(removed),
                    "net": int(added) - int(removed),
                }
            )
    rows.sort(key=lambda row: (-row["net"], row["path"]))
    frequency = Counter((copy["original"], copy["original_line"]) for copy in copies)
    copies.sort(key=lambda copy: -frequency[copy["original"], copy["original_line"]])
    return {
        "candidates": candidates,
        "clones": copies[:20],
        "growth": rows[:20],
        "debt": [
            problem
            for path, text in sources.items()
            if path.startswith("src/")
            for problem in problems(path, None, text)
        ],
        "limit": "Static reader candidates require review; dynamic readers are not proven absent.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weekly", action="store_true", help="Print a read-only source review as JSON"
    )
    if parser.parse_args(argv).weekly:
        print(json.dumps(weekly_report(), ensure_ascii=False))
        return 0
    staged = _git("diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR").stdout.split("\0")
    # A merge adds only what neither parent holds: what the merged branch brings was checked there.
    merging = _git("rev-parse", "-q", "--verify", "MERGE_HEAD").returncode == 0
    parents = ("HEAD", "MERGE_HEAD") if merging else ("HEAD",)
    selected = [
        path
        for path in staged
        if path.startswith("src/") or (path.startswith("tests/") and path.endswith((".py", ".cjs")))
    ]
    texts = _blobs([revision + ":" + path for revision in ("", *parents) for path in selected])
    found, corpus = [], None
    for path in selected:
        if ":" + path in texts:
            new = texts[":" + path]
            each = []
            for parent in parents:
                old = texts.get(parent + ":" + path)
                current = problems(path, old, new)
                if path.startswith("src/") and _added_windows(path, old, new)[1]:
                    if corpus is None:
                        paths = _git("ls-files", "-z", "--", "src").stdout.split("\0")
                        corpus = {
                            spec[1:]: text
                            for spec, text in _blobs([":" + item for item in paths if item]).items()
                        }
                    current += clone_problems(path, old, new, corpus)
                each.append(set(current))
            found += sorted(set.intersection(*each))
    for line in found:
        print(line, file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
