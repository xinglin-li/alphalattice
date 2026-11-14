"""Refuse growing source shape and test-writing violations in staged Git blobs.

The weekly JSON report records existing debt and static review candidates.
"""

from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tokenize
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from devtools.architecture.git_inputs import read_blobs as _blobs
from devtools.architecture.git_inputs import staged_changes, staged_moves, unique_moves
from devtools.testing.contracts import import_violations
from devtools.testing.layout import REGISTRIES, new_data_file_problems, new_test_file_problems
from devtools.testing.requirements import PROVES_LINE, file_holders
from devtools.testing.requirements import parse as parse_requirements
from devtools.testing.sizes import validate_policy

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
# word catalogs are data: every new word needs its Chinese key, so a size ratchet would block every
# feature; the dead-key censuses hold their growth instead
LOCAL = "src/alphalattice/interface/local_application/"
WORD_CATALOGS = frozenset(
    {
        LOCAL + "refusal_words.json",
        LOCAL + "answers.json",
        LOCAL + "assets/workbench-source/js/data/zh.js",
    }
)
# third-party data the package ships whole, never edited: tiktoken's cl100k encoding (NOTICE), kept
# so a token count needs no download; its size is the upstream file's, checked by its SHA-256
VENDORED_DATA = frozenset({"src/alphalattice/protocols/actor_execution/cl100k_base.tiktoken.gz"})
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


def _metrics(node: ast.AST, ruff: dict[int, int] | None = None) -> tuple[int, int]:
    # Ruff's McCabe score and this guard's decision count agree for these
    # functions. Other syntax keeps the original rule, which counts boolean
    # operators and comprehensions and excludes nested definitions differently.
    excluded = (
        ast.BoolOp,
        ast.IfExp,
        ast.comprehension,
        ast.Match,
        ast.Try,
        ast.TryStar,
        ast.AsyncFor,
    )
    children = tuple(_body(node))
    simple = not any(isinstance(item, excluded) for item in children) and not any(
        isinstance(item, (ast.For, ast.AsyncFor, ast.While)) and item.orelse for item in children
    )
    simple = simple and not any(
        isinstance(item, DEFINITION_TYPES) and item is not node for item in ast.walk(node)
    )
    if simple and ruff is not None:
        if node.lineno not in ruff:
            raise ValueError(f"ruff omitted function at line {node.lineno}")
        return (node.end_lineno or node.lineno) - node.lineno + 1, ruff[node.lineno]
    decisions = sum(
        isinstance(
            item, ast.If | ast.For | ast.AsyncFor | ast.While | ast.IfExp | ast.ExceptHandler
        )
        + (len(item.values) - 1 if isinstance(item, ast.BoolOp) else 0)
        + (len(item.cases) if isinstance(item, ast.Match) else 0)
        + (1 + len(item.ifs) if isinstance(item, ast.comprehension) else 0)
        for item in children
    )
    return (node.end_lineno or node.lineno) - node.lineno + 1, decisions


# Ruff's pytest-style rules a test change may not grow, with the fix each names
PT_FIXES = {
    "PT011": "give pytest.raises a match= for the expected message",
    "PT012": "keep one simple statement inside pytest.raises",
    "PT014": "drop the duplicate parametrize case",
    "PT015": "use pytest.fail for an assertion that always fails",
    "PT017": "assert the exception through pytest.raises",
}


def _ruff(snapshots: dict[str, str], *options: str) -> dict[str, list[dict]]:
    """Ruff's JSON rows for each snapshot, from one locked invocation."""
    if not snapshots:
        return {}
    with tempfile.TemporaryDirectory(prefix="test-shape-") as folder:
        root, names = Path(folder), {}
        for index, (key, text) in enumerate(snapshots.items()):
            path = root / f"{index:08}.py"
            path.write_text(text, encoding="utf-8", errors="surrogateescape", newline="\n")
            names[path.name] = key
        done = subprocess.run(
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "--isolated",
                "--ignore-noqa",
                "--target-version",
                "py312",
                *options,
                "--output-format",
                "json",
                str(root),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if done.returncode not in (0, 1):
            raise ValueError("ruff could not read the staged snapshots")
        result = {key: [] for key in snapshots}
        for row in json.loads(done.stdout):
            result[names[Path(row["filename"]).name]].append(row)
        return result


def ruff_decisions(snapshots: dict[str, str]) -> dict[str, dict[int, int]]:
    """Measure each staged Python snapshot with one locked Ruff invocation."""
    rows = _ruff(snapshots, "--select", "C901", "--config", "lint.mccabe.max-complexity=0")
    result = {key: {} for key in rows}
    for key, found in rows.items():
        for row in found:
            match = re.search(r"\((\d+) > 0\)$", row["message"])
            if row["code"] != "C901" or not match:
                raise ValueError("ruff could not measure a staged snapshot")
            line = row["location"]["row"]
            if line in result[key]:
                raise ValueError("ruff returned an ambiguous function location")
            result[key][line] = int(match.group(1)) - 1
    return result


def ruff_test_findings(snapshots: dict[str, str]) -> dict[str, tuple[tuple[str, int], ...]]:
    """Each staged test snapshot's PT findings as (rule, line), from one Ruff invocation."""
    rows = _ruff(snapshots, "--select", ",".join(PT_FIXES))
    if any(row["code"] not in PT_FIXES for found in rows.values() for row in found):
        raise ValueError("ruff returned a rule outside its PT selection")
    return {
        key: tuple(sorted((row["code"], row["location"]["row"]) for row in found))
        for key, found in rows.items()
    }


ANTIPATTERN_BASELINE = "config/registries/antipattern-baseline.json"
ANTIPATTERN_PRIVATE = (
    "config/" + Path(ANTIPATTERN_BASELINE).name.removesuffix(".json") + "-private.json"
)
ANTIPATTERN_SIDES = (ANTIPATTERN_BASELINE, ANTIPATTERN_PRIVATE)
ANTIPATTERN_SEMGREP = "scripts/antipattern_rules.yml"
ANTIPATTERN_CASES = "scripts/antipattern_examples.py.txt"
ANTIPATTERN_RULES = "PGH,G,LOG,C901,PLR0912,PLR0915,S112,S113,DTZ005,ASYNC"
WAIT_RULES = ("S112", "S113", "DTZ005", "ASYNC")


def _mock_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    names = {"patch", "Mock", "MagicMock", "AsyncMock"}
    if isinstance(func, ast.Name):
        return func.id in names
    if not isinstance(func, ast.Attribute):
        return False
    owner = func.value.id if isinstance(func.value, ast.Name) else ""
    return (
        func.attr in names
        or (func.attr == "object" and owner == "patch")
        or (func.attr in {"setattr", "setenv", "setitem"} and owner in {"monkeypatch", "binding"})
    )


def _mock_scores(path: str, definitions: list) -> dict[str, int]:
    scores = {}
    for name, node in definitions:
        if isinstance(node, FUNCTION_TYPES) and node.name.startswith("test_"):
            count = sum(_mock_call(item) for item in ast.walk(node))
            if count > 4:
                scores[f"{path}::test-too-many-mocks::{name}"] = count
    return scores


def ruff_antipatterns(snapshots: dict[str, str]) -> dict[str, int]:
    """Count rules by file and complex functions by their measured score."""
    rows = _ruff(snapshots, "--select", ANTIPATTERN_RULES)
    counts: Counter[str] = Counter()
    for path, findings in rows.items():
        definitions = list(_definitions(_tree(snapshots[path])))
        if path.startswith("tests/"):
            counts.update(_mock_scores(path, definitions))
        for row in findings:
            code = row["code"]
            if path.startswith("tests/") and code.startswith(WAIT_RULES):
                continue
            key, amount = f"{path}::{code}", 1
            if code in {"C901", "PLR0912", "PLR0915"}:
                score = re.search(r"\((\d+) > \d+\)", row["message"])
                line = row["location"]["row"]
                owners = [
                    (node.end_lineno - node.lineno, name)
                    for name, node in definitions
                    if isinstance(node, FUNCTION_TYPES) and node.lineno <= line <= node.end_lineno
                ]
                if not score or not owners:
                    raise ValueError(f"Ruff omitted the complexity owner for {path}:{line}")
                key += "::" + min(owners)[1]
                amount = int(score.group(1))
            counts[key] += amount
    return dict(counts)


def antipattern_growth(held: dict[str, int], found: dict[str, int]) -> list[str]:
    """A finding or allowance may fall, but neither may grow above the held count."""
    return [
        f"{key}: anti-pattern findings grow {held.get(key, 0)} -> {count}"
        for key, count in sorted(found.items())
        if count > held.get(key, 0)
    ]


def antipattern_baseline(text: str) -> dict[str, int]:
    data = json.loads(text)
    baseline = data["counts"]
    if not isinstance(baseline, dict) or any(
        not isinstance(k, str) or "::" not in k or type(v) is not int or v < 1
        for k, v in baseline.items()
    ):
        raise ValueError("invalid anti-pattern allowance")
    return baseline


def antipattern_sides(texts: dict[str, str], prefix: str) -> dict[str, int]:
    counts = Counter(antipattern_baseline(texts[prefix + ANTIPATTERN_BASELINE]))
    if (private := texts.get(prefix + ANTIPATTERN_PRIVATE)) is not None:
        counts.update(antipattern_baseline(private))
    return dict(counts)


def load_antipattern_baseline(root: Path) -> dict[str, int]:
    texts = {ANTIPATTERN_BASELINE: (root / ANTIPATTERN_BASELINE).read_text(encoding="utf-8")}
    with contextlib.suppress(FileNotFoundError):  # the public checkout has no private side
        texts[ANTIPATTERN_PRIVATE] = (root / ANTIPATTERN_PRIVATE).read_text(encoding="utf-8")
    return antipattern_sides(texts, "")


def relocated_antipatterns(held: dict[str, int], moves: dict[str, str]) -> dict[str, int]:
    """Move a file's allowance only with Git's byte-identical move."""
    back = {old: new for new, old in moves.items()}
    return {
        back.get(path, path) + "::" + rule: count
        for key, count in held.items()
        for path, _, rule in [key.partition("::")]
    }


def test_data_antipatterns(paths: list[str]) -> dict[str, int]:
    return {
        f"{path}::test-data-file": 1
        for path in paths
        if path.startswith("tests/")
        and Path(path).suffix.lower() not in CODE | {".md"}
        and not path.endswith((".test.yml", ".test.yaml", "/pytest.ini"))
    }


def _semgrep_environment(root: Path) -> dict[str, str]:
    return {
        **os.environ,
        "SEMGREP_SETTINGS_FILE": str(root / "settings.yml"),
        "SEMGREP_LOG_FILE": str(root / "semgrep.log"),
        "SEMGREP_VERSION_CACHE_PATH": str(root / "version"),
    }


def _semgrep_selftest(rules: str, cases: str) -> None:
    with tempfile.TemporaryDirectory(prefix="antipattern-cases-") as folder:
        root = Path(folder)
        (root / "rules.yml").write_text(rules, encoding="utf-8")
        (root / "rules.py").write_text(cases, encoding="utf-8")
        exe = Path(sys.executable).parent / ("semgrep.exe" if os.name == "nt" else "semgrep")
        result = subprocess.run(
            [
                str(exe),
                "--test",
                "--config",
                "rules.yml",
                "rules.py",
                "--json",
                "--metrics=off",
                "--disable-version-check",
            ],
            cwd=root,
            env=_semgrep_environment(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=180,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode:
            raise ValueError(f"semgrep rule cases failed: {result.stderr[-400:]}")
        report = json.loads(result.stdout)
        checks = {
            name: row
            for config in report["results"].values()
            for name, row in config["checks"].items()
        }
        expected = set(re.findall(r"(?m)^  - id: (\S+)$", rules))
        if (
            report.get("config_missing_tests")
            or set(checks) != expected
            or any(not row["passed"] for row in checks.values())
        ):
            raise ValueError("semgrep rule cases are incomplete or failed")


def semgrep_antipatterns(snapshots: dict[str, str], rules: str) -> dict[str, int]:
    """Run the slower rules on the same committed or staged source snapshots."""
    if not snapshots:
        return {}
    with tempfile.TemporaryDirectory(prefix="antipattern-") as folder:
        root = Path(folder)
        config = root / "rules.yml"
        config.write_text(rules, encoding="utf-8")
        for path, text in snapshots.items():
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8", errors="surrogateescape", newline="\n")
        done = subprocess.run(
            [
                str(
                    Path(sys.executable).parent / ("semgrep.exe" if os.name == "nt" else "semgrep")
                ),
                "scan",
                "--config",
                str(config),
                "--json",
                "--disable-nosem",
                "--disable-version-check",
                "--metrics=off",
                "--no-git-ignore",
                ".",
            ],
            cwd=root,
            env=_semgrep_environment(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=180,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if done.returncode:
            raise ValueError(f"semgrep exited {done.returncode}: {done.stderr[-400:]}")
        report = json.loads(done.stdout)
        if report.get("errors"):
            raise ValueError(f"semgrep could not measure its inputs: {report['errors']}")
        scanned = {Path(path).as_posix() for path in report["paths"]["scanned"]}
        if missing := set(snapshots) - scanned:
            raise ValueError(f"semgrep skipped source: {sorted(missing)[0]}")
        counts = Counter()
        for row in report["results"]:
            path = Path(row["path"]).as_posix()
            counts[f"{path}::{row['check_id'].split('.')[-1]}"] += 1
        return dict(counts)


def auxiliary_antipatterns(snapshots: dict[str, str]) -> dict[str, int]:
    """Check staged workflows and shell scripts with their locked executables."""
    if not snapshots:
        return {}
    counts = Counter()
    with tempfile.TemporaryDirectory(prefix="antipattern-tools-") as folder:
        root = Path(folder)
        for path, text in snapshots.items():
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8", newline="\n")
        windows = ".exe" if os.name == "nt" else ""
        for tool, files, options in (
            (
                "actionlint",
                [p for p in snapshots if p.startswith(".github/workflows/")],
                ["-no-color", "-shellcheck=", "-pyflakes=", "-format", "{{json .}}"],
            ),
            (
                "shellcheck",
                [p for p in snapshots if p.endswith(".sh")],
                ["--norc", "--format=json1"],
            ),
        ):
            if not files:
                continue
            executable = Path(sys.executable).parent / (tool + windows)
            if not executable.is_file():
                raise ValueError(
                    f"{tool} is not installed at {executable}; sync the locked environment "
                    "(uv sync --locked --all-extras)"
                )
            done = subprocess.run(
                [str(executable), *options, *files],
                cwd=root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=30,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if done.returncode not in (0, 1):
                raise ValueError(f"{tool} exited {done.returncode}: {done.stderr[-400:]}")
            report = json.loads(done.stdout)
            rows = report if tool == "actionlint" else report["comments"]
            for row in rows:
                path = Path(row["filepath"] if tool == "actionlint" else row["file"]).as_posix()
                rule = row["kind"] if tool == "actionlint" else f"SC{row['code']}"
                counts[f"{path}::{tool}-{rule}"] += 1
        for path, text in snapshots.items():
            if path.endswith(".sh"):
                counts[f"{path}::shellcheck-inline-disable"] += sum(
                    bool(re.match(r"\s*#\s*shellcheck\b.*\bdisable\s*=", line, re.I))
                    for line in text.splitlines()
                )
    return {key: count for key, count in counts.items() if count}


def _antipattern_problems(
    staged: list[str],
    sources: list[str],
    texts: dict[str, str],
    parents: tuple[str, ...],
    head: str,
    *,
    moves: dict[str, str],
    semgrep: bool = False,
) -> list[str]:
    baseline = antipattern_sides(texts, head)
    found = []
    carried = len(parents) > 1 and any(
        parent + ":" + ANTIPATTERN_BASELINE in texts for parent in parents
    )
    for parent in parents:
        previous = texts.get(parent + ":" + ANTIPATTERN_BASELINE)
        if previous is None:
            if not carried and (
                ANTIPATTERN_BASELINE not in staged or "scripts/check_test_shape.py" not in staged
            ):
                raise ValueError("anti-pattern baseline missing from parent")
        else:
            found += antipattern_growth(
                relocated_antipatterns(antipattern_sides(texts, parent + ":"), moves), baseline
            )
    missing = [path for path in sources if head + path not in texts]
    if missing:
        raise ValueError(f"changed source unreadable: {missing[0]}")
    measured = ruff_antipatterns(
        {path: texts[head + path] for path in sources if path.endswith(".py")}
    )
    measured.update(test_data_antipatterns(staged))
    measured.update(
        auxiliary_antipatterns(
            {path: texts[head + path] for path in sources if not path.endswith(".py")}
        )
    )
    if semgrep:
        if ANTIPATTERN_SEMGREP in staged or ANTIPATTERN_CASES in staged:
            _semgrep_selftest(texts[head + ANTIPATTERN_SEMGREP], texts[head + ANTIPATTERN_CASES])
        measured.update(
            semgrep_antipatterns(
                {
                    path: texts[head + path]
                    for path in sources
                    if path.endswith(".py") and not path.startswith("tests/")
                },
                texts[head + ANTIPATTERN_SEMGREP],
            )
        )
    return found + antipattern_growth(baseline, measured)


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
    if path.startswith("tests/unit/"):
        found.update(
            import_violations(
                tuple(imports.values())
                + tuple(called(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)),
                "unit",
            )
        )
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


def _node_environment() -> dict[str, str]:
    """Do not let inherited Node injection replace the verified scanner."""
    environment = dict(os.environ)
    for name in ("NODE_OPTIONS", "NODE_PATH"):
        environment.pop(name, None)
    return environment


def _npm_package(installed: Path, entry: dict, cache: Path) -> None:
    """Check installed bytes against the locked archive, not an installed stamp."""
    integrity = entry.get("integrity", "")
    if not isinstance(integrity, str) or not integrity.startswith("sha512-"):
        raise ValueError("jscpd.lock_unreadable: package integrity is missing")
    try:
        digest = base64.b64decode(integrity[7:], validate=True)
        if len(digest) != 64:
            raise ValueError("invalid SHA-512 digest")
        key = digest.hex()
        data = (cache / "_cacache/content-v2/sha512" / key[:2] / key[2:4] / key[4:]).read_bytes()
        if hashlib.sha512(data).digest() != digest:
            raise ValueError("cached archive hash differs from the lock")
        expected = {}
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            for item in archive.getmembers():
                if item.isdir():
                    continue
                parts = item.name.split("/")
                if (
                    not item.isfile()
                    or len(parts) < 2
                    or parts[0] != "package"
                    or any(part in {"", ".", ".."} for part in parts[1:])
                ):
                    raise ValueError("unsafe archive member")
                stream = archive.extractfile(item)
                name = "/".join(parts[1:])
                if stream is None or name in expected:
                    raise ValueError("duplicate or unreadable archive member")
                expected[name] = hashlib.sha512(stream.read()).digest()
        if installed.is_symlink() or installed.is_junction():
            raise ValueError("installed package is a link")
        actual = {}

        def unreadable(error: OSError) -> None:
            raise error

        for folder, directories, files in os.walk(installed, followlinks=False, onerror=unreadable):
            for name in (*directories, *files):
                path = Path(folder) / name
                if path.is_symlink() or path.is_junction():
                    raise ValueError("installed package contains a link")
            for name in files:
                path = Path(folder) / name
                actual[path.relative_to(installed).as_posix()] = hashlib.sha512(
                    path.read_bytes()
                ).digest()
        if not expected or actual != expected:
            raise ValueError("installed package differs from its locked archive")
    except (OSError, ValueError, tarfile.TarError) as error:
        raise ValueError(
            f"jscpd.integrity_unavailable: {installed}: {error}; run npm ci and retry"
        ) from error


def jscpd_runner(
    root: Path, interpreter: Path, manifests: tuple[bytes, bytes] | None = None
) -> Path:
    """Bind every executable npm package to the staged lock's archive bytes."""
    package = Path("scripts/test-shape-tools")
    try:
        manifest, lock = manifests or tuple(
            (root / package / name).read_bytes() for name in ("package.json", "package-lock.json")
        )
        manifest, lock = json.loads(manifest), json.loads(lock)
        packages = lock["packages"]
        if manifest["devDependencies"] != packages[""]["devDependencies"]:
            raise ValueError("package manifest differs from its lock")
        entry = packages["node_modules/jscpd"]
        if entry["version"] != manifest["devDependencies"]["jscpd"]:
            raise ValueError("jscpd version differs from its lock")
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"jscpd.lock_unreadable: restore the staged package manifests: {error}"
        ) from error
    modules = root / package / "node_modules"
    if not (modules / "jscpd/run-jscpd.js").is_file():
        if (
            interpreter.parent.name.lower() not in {"scripts", "bin"}
            or interpreter.parent.parent.name != ".venv"
        ):
            raise ValueError(
                "jscpd.unavailable: install the locked tool in scripts/test-shape-tools"
            )
        modules = interpreter.parent.parent.parent / package / "node_modules"
    if not (modules / "jscpd/run-jscpd.js").is_file():
        raise ValueError("jscpd.unavailable: install the locked tool in scripts/test-shape-tools")
    cache = os.environ.get("npm_config_cache") or os.environ.get("NPM_CONFIG_CACHE")  # noqa: SIM112
    npm, node = shutil.which("npm"), shutil.which("node")
    if not node or (not cache and not npm):
        raise ValueError("jscpd.unavailable: install Node.js and the locked tool")
    try:
        if not cache:
            cache = subprocess.run(
                [npm, "config", "get", "cache", "--offline"],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
                env=_node_environment(),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ).stdout.strip()
        _npm_package(modules / "jscpd", entry, Path(cache))
        choose = (
            "const m=require(process.argv[1]);const k=m.getPlatformKey();"
            "if(!k)process.exit(2);process.stdout.write(m.PLATFORM_MAP[k].packageName)"
        )
        name = subprocess.run(
            [node, "-e", choose, str(modules / "jscpd/platform-map.js")],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
            env=_node_environment(),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout.strip()
        if not re.fullmatch(r"jscpd-[a-z0-9-]+", name):
            raise ValueError("platform package name is invalid")
        _npm_package(modules / name, packages["node_modules/" + name], Path(cache))
    except (OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        raise ValueError(f"jscpd.integrity_unavailable: {error}; run npm ci and retry") from error
    return modules / "jscpd/run-jscpd.js"


def clone_problems_batch(
    changes: list[tuple[str, str | None, str]],
    corpus: dict[str, str],
    *,
    manifests: tuple[bytes, bytes] | None = None,
) -> list[list[str]]:
    """Find exact added eight-line copies in staged blobs, using one jscpd scan."""
    prepared = [(path, *_added_windows(path, old, new)) for path, old, new in changes]
    wanted = set().union(*(windows for _path, _rows, windows in prepared))
    if not wanted:
        return [[] for _change in changes]
    anchors = re.compile(
        "|".join(re.escape("".join(max(window, key=len).split())) for window in wanted)
    )
    texts = dict(corpus)
    texts.update((path, new) for path, _old, new in changes)
    candidates = {
        path: _code_lines(path, text)
        for path, text in sorted(texts.items())
        if path not in GENERATED and anchors.search("".join(text.split()))
    }
    candidates = {path: rows for path, rows in candidates.items() if len(rows) >= 8}
    origins = defaultdict(list)
    if candidates:
        node = shutil.which("node")
        if not node:
            raise ValueError(
                "jscpd unavailable; run npm ci --ignore-scripts in scripts/test-shape-tools"
            )
        runner = jscpd_runner(ROOT, Path(sys.executable), manifests)
        with tempfile.TemporaryDirectory(prefix="test-clones-") as folder:
            root = Path(folder)
            inputs, report = root / "input", root / "report"
            inputs.mkdir()
            names = {}
            for index, (path, rows) in enumerate(candidates.items()):
                name = f"{index:08}.py"
                names[name] = (path, rows)
                # Each executable line becomes one opaque token, preserving literal
                # values and cross-language equality while jscpd finds line sequences.
                (inputs / name).write_text(
                    "".join(
                        "x"
                        + hashlib.sha256(line.encode("utf-8", "surrogateescape")).hexdigest()
                        + "()\n"
                        for _number, line in rows
                    ),
                    encoding="utf-8",
                    newline="\n",
                )
            # jscpd counts the distance between line endpoints, so seven admits
            # an exact eight-line block; the existing window check filters shorter hits.
            subprocess.run(
                [
                    node,
                    str(runner),
                    "--min-lines",
                    "7",
                    "--min-tokens",
                    "1",
                    "--skip-comments",
                    "--no-gitignore",
                    "--fail-on-empty",
                    "--workers",
                    "1",
                    "--max-size",
                    str(1 + max(len(rows) for rows in candidates.values()) * 68),
                    "--reporters",
                    "json",
                    "--output",
                    str(report),
                    "--no-tips",
                    str(inputs),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=120,
                env=_node_environment(),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            result = json.loads((report / "jscpd-report.json").read_text(encoding="utf-8"))
            if result["statistics"]["total"]["sources"] != len(candidates):
                raise ValueError("jscpd omitted a staged source")
            matches = result["duplicates"]
            for match in matches:
                locations = []
                for key in ("firstFile", "secondFile"):
                    entry = match[key]
                    path, rows = names[Path(entry["name"]).name]
                    start, end = entry["start"], entry["end"]
                    if not (
                        type(start) is int and type(end) is int and 1 <= start <= end <= len(rows)
                    ):
                        raise ValueError("jscpd returned an invalid source range")
                    locations.append((path, dict(_windows(rows[start - 1 : end]))))
                (first, a), (second, b) = locations
                shared = set(a.values()) & set(b.values()) & wanted
                for path, windows in ((first, a), (second, b)):
                    for line, window in windows.items():
                        if window in shared:
                            origins[window].append((path, line))
    results = []
    for path, after, selected in prepared:
        found, last = [], -8
        for index, (line, window) in enumerate(_windows(after)):
            others = sorted(set(origins[window]) - {(path, line)})
            if window in selected and others and index >= last + 8:
                original, start = others[0]
                found.append(
                    f"{path}:{line}: copied 8 lines from {original}:{start}; call its owner"
                )
                last = index
        results.append(found)
    return results


def clone_problems(path: str, old: str | None, new: str, corpus: dict[str, str]) -> list[str]:
    return clone_problems_batch([(path, old, new)], corpus)[0]


def _docstring_lines(text: str) -> dict[str, int]:
    """A test marked `proves`, on itself, its class or its module, has its sentence in its
    registered requirement: it counts as one."""
    tree = _tree(text)
    held = file_holders("", ast.unparse(tree)) if tree.body else []
    marked = {test.node.split("::", 1)[1].replace("::", ".") for test in held if test.ids}
    return {
        name: len((ast.get_docstring(node) or "").splitlines()) or int(name in marked)
        for name, node in _definitions(tree)
        if isinstance(node, FUNCTION_TYPES) and node.name.startswith("test_")
    }


def _given(value: ast.AST | None) -> bool:
    return value is not None and not (isinstance(value, ast.Constant) and value.value is None)


def _names_every_case(ids: ast.AST | None) -> bool:
    """ids= names each case; a literal list with a None entry leaves pytest's value id there."""
    if isinstance(ids, ast.List | ast.Tuple):
        return bool(ids.elts) and all(_given(entry) for entry in ids.elts)
    return _given(ids)


def _named(case: ast.AST) -> bool:
    return (
        isinstance(case, ast.Call)
        and getattr(case.func, "attr", getattr(case.func, "id", "")) == "param"
        and any(word.arg == "id" and _given(word.value) for word in case.keywords)
    )


def _unnamed(node: ast.AST) -> list[str]:
    """A test's parametrize sites whose cases are not each named by ids= or pytest.param id=."""
    sites = []
    for mark in getattr(node, "decorator_list", ()):
        if not (isinstance(mark, ast.Call) and getattr(mark.func, "attr", "") == "parametrize"):
            continue
        given = {word.arg: word.value for word in mark.keywords}
        cases = mark.args[1] if len(mark.args) > 1 else given.get("argvalues")
        names = mark.args[0] if mark.args else given.get("argnames")
        listed = isinstance(cases, ast.List | ast.Tuple) and cases.elts
        if not _names_every_case(given.get("ids")) and not (
            listed and all(_named(case) for case in cases.elts)
        ):
            sites.append(f"line {mark.lineno} parametrize({ast.unparse(names) if names else ''})")
    return sites


def _tests(tree: ast.AST) -> dict[str, ast.AST]:
    return {
        name: node
        for name, node in _definitions(tree)
        if isinstance(node, FUNCTION_TYPES) and node.name.startswith("test_")
    }


def _owners(tree: ast.AST, found: tuple[tuple[str, int], ...]) -> Counter:
    """Each PT finding counted under the innermost function holding its line, decorators
    included, else the module."""
    spans = [
        (min([node.lineno] + [mark.lineno for mark in node.decorator_list]), node.end_lineno, name)
        for name, node in _definitions(tree)
        if isinstance(node, FUNCTION_TYPES)
    ]
    counts = Counter()
    for rule, line in found:
        holding = [span for span in spans if span[0] <= line <= span[1]]
        counts[(max(holding)[2] if holding else "<module>", rule)] += 1
    return counts


def _test_problems(
    path: str, old: str | None, new: str, ruff: tuple[tuple, tuple] | None
) -> list[str]:
    """An added or changed test names each parametrize case; no function grows a PT finding."""
    before, after = _tree(old or ""), _tree(new)
    was, found = _tests(before), []
    for name, node in _tests(after).items():
        if name in was and ast.dump(was[name]) == ast.dump(node):
            continue
        found += [
            f"{path}::{name}: {site} leaves its cases unnamed; "
            "give each case pytest.param(..., id=...) or ids= (rule 5)"
            for site in _unnamed(node)
        ]
    if ruff is not None:
        held, now = _owners(before, ruff[0]), _owners(after, ruff[1])
        found += [
            f"{path}::{name}: Ruff {rule} findings grow {held[(name, rule)]} -> {count}; "
            f"{PT_FIXES[rule]}"
            for (name, rule), count in sorted(now.items())
            if count > held[(name, rule)]
        ]
    return found


def problems(
    path: str,
    old: str | None,
    new: str,
    ruff: tuple[dict[int, int], dict[int, int]] | tuple[tuple, tuple] | None = None,
) -> list[str]:
    """Growth a staged source or test change refuses; existing debt only shrinks."""
    if path in GENERATED:
        return []
    if path.startswith("src/"):
        found = []
        size = len(new.encode("utf-8", errors="surrogateescape"))
        grew = size > len((old or "").encode("utf-8", errors="surrogateescape"))
        if size > CAP_BYTES and grew and path not in WORD_CATALOGS | VENDORED_DATA:
            found.append(f"{path}:1: source size {size} > {CAP_BYTES}; split at its owner")
        if path.endswith(".py"):
            before, after = _tree(old or ""), _tree(new)
            was = {
                name: _metrics(node, ruff[0] if ruff else None)
                for name, node in _definitions(before)
                if isinstance(node, FUNCTION_TYPES)
            }
            for name, node in _definitions(after):
                if isinstance(node, FUNCTION_TYPES):
                    for metric, count, previous, cap in zip(
                        ("physical lines", "decisions"),
                        _metrics(node, ruff[1] if ruff else None),
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

    def counted(text: str) -> int:  # tracing a test to its requirement never grows its file
        return len(PROVES_LINE.sub("", text).encode("utf-8", errors="surrogateescape"))

    size = counted(new)
    if size > CAP_BYTES and size > counted(old or ""):
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
    found += _test_problems(path, old, new, ruff)
    if path.startswith("tests/unit/"):
        found.extend(
            row for row in _lint(path, _tree(new)) if row.startswith("testing.layer_contract:")
        )
    if not path.startswith("tests/structural/") and any("inspect.getsource(" in x for x in added):
        found.append(f"{path}: reads source text; test the behaviour at the owner's seam (rule 3)")
    for line in added:
        match = SLEEP.search(line)
        if match and float(match.group(1)) >= 1:
            found.append(f"{path}: sleeps {match.group(1)} s; drive the clock or the step (rule 7)")
    was = _docstring_lines(old or "")
    for name, count in _docstring_lines(new).items():
        if name not in was and count == 0:
            found.append(f"{path}::{name}: missing requirement docstring (rule 1)")
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


def _data_file_problems(
    staged: list[str], moved: dict, parents: tuple[str, ...], head: str = ":"
) -> list[str]:
    """The rule for test data: a path no parent holds, under tests/, that is neither code nor
    prose. The tree is its baseline, and a byte-identical move keeps its standing."""
    data = [
        path
        for path in staged
        if path.startswith("tests/")
        and path not in moved
        and Path(path).suffix.lower() not in CODE | {".md"}
    ]
    if not data:
        return []
    try:
        held = set()
        for parent in parents:
            listed = _git("ls-tree", "-r", "-z", "--name-only", parent, "--", *data)
            if listed.returncode:
                raise ValueError(f"{parent} is unreadable")
            held.update(listed.stdout.split("\0"))
        fresh = [path for path in data if path not in held]
        if not fresh:
            return []
        policy = validate_policy(json.loads(_blobs([head + REGISTRIES[0]])[head + REGISTRIES[0]]))
        return new_data_file_problems(fresh, policy)
    except (KeyError, OSError, ValueError, subprocess.SubprocessError) as error:
        return [f"TEST-DATA unmeasured: {error}; restore the staged registry and retry"]


def _shape_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weekly", action="store_true", help="Print a read-only source review as JSON"
    )
    parser.add_argument(
        "--base",
        help="judge the committed head against its merge base with this ref (the commit stage)",
    )
    parser.add_argument("--semgrep", action="store_true", help="include the slower source rules")
    return parser.parse_args(argv)


def _shape_moves(base: str | None) -> tuple[dict[str, str], dict[str, str]]:
    pairs = staged_changes(ROOT, base=base, tree="HEAD") if base else staged_changes(ROOT)
    return staged_moves(ROOT, pairs=pairs), unique_moves(pairs)


def main(argv: list[str] | None = None) -> int:
    args = _shape_args(argv)
    if args.weekly:
        print(json.dumps(weekly_report(), ensure_ascii=False))
        return 0
    if args.base:  # the committed head against its merge base: the stage's snapshot
        found_base = _git("merge-base", args.base, "HEAD")
        if found_base.returncode:
            print(
                f"TEST-LAYOUT base {args.base} unreadable; name a reachable base", file=sys.stderr
            )
            return 1
        parents, head = (found_base.stdout.strip(),), "HEAD:"
        listing = _git("diff", "--name-only", "-z", "--diff-filter=ACMR", parents[0], "HEAD")
    else:  # the staged index against HEAD: the hook's snapshot
        head = ":"
        listing = _git("diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR")
    if listing.returncode:
        print(
            "TEST-LAYOUT changed files unreadable; restore the Git index and retry", file=sys.stderr
        )
        return 1
    staged = listing.stdout.split("\0")
    if not args.base:
        # A merge adds only what neither parent holds: the merged branch was checked there.
        merging = _git("rev-parse", "-q", "--verify", "MERGE_HEAD").returncode == 0
        parents = ("HEAD", "MERGE_HEAD") if merging else ("HEAD",)
    selected, anti_sources = (
        [
            path
            for path in staged
            if path.startswith("src/")
            or (path.startswith("tests/") and path.endswith((".py", ".cjs")))
        ],
        [
            path
            for path in staged
            if path.endswith((".py", ".sh"))
            or (path.startswith(".github/workflows/") and path.endswith((".yml", ".yaml")))
        ],
    )
    try:
        moved, identical = _shape_moves(parents[0] if args.base else None)
        origin = {path: moved.get(path, path) for path in selected}
        candidates = [
            p
            for p in selected
            if p.startswith("tests/")
            and p.endswith(".py")
            and (Path(p).name.startswith("test_") or Path(p).name.endswith("_test.py"))
            and p not in moved
        ]
        texts = _blobs(
            [head + path for path in selected]
            + [head + path for path in anti_sources]
            + [parent + ":" + origin[path] for parent in parents for path in selected]
            + [parent + ":" + path for parent in parents for path in candidates]
            + [head + path for path in ANTIPATTERN_SIDES]
            + ([head + ANTIPATTERN_SEMGREP] if args.semgrep else [])
            + ([head + ANTIPATTERN_CASES] if args.semgrep else [])
            + [parent + ":" + path for parent in parents for path in ANTIPATTERN_SIDES]
            + ([head + "config/registries/requirements.json"] if candidates else [])
        )
        found, corpus = (
            _data_file_problems(staged, moved, parents, head)
            + _antipattern_problems(
                staged, anti_sources, texts, parents, head, moves=identical, semgrep=args.semgrep
            ),
            None,
        )
    except (
        OSError,
        ValueError,
        IndexError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as error:
        print(
            f"TEST-LAYOUT staged snapshot unreadable: {error}; restore the Git index and retry",
            file=sys.stderr,
        )
        return 1
    added = [p for p in candidates if all(parent + ":" + p not in texts for parent in parents)]
    if added:
        try:
            registry = texts.get(head + "config/registries/requirements.json")
            if registry is None:
                raise ValueError("the staged requirements registry is missing")
            rows = parse_requirements(registry)
            sources = _git("ls-files", "-z", "--", "src", "scripts")
            if sources.returncode:
                raise ValueError("staged source listing is unreadable")
            source_paths = set(sources.stdout.split("\0"))
            for path in added:
                if head + path not in texts:
                    raise ValueError(f"staged test {path} is unreadable")
                found += new_test_file_problems(path, texts[head + path], rows, source_paths)
        except (ValueError, TypeError, OSError) as error:
            found.append(
                f"TEST-LAYOUT placement unmeasured: {error}; "
                "restore the staged registry and source index"
            )
    python_sources = [p for p in selected if p.startswith("src/") and p.endswith(".py")]
    specs = [head + path for path in python_sources] + [
        parent + ":" + origin[path] for path in python_sources for parent in parents
    ]
    test_sources = [p for p in selected if p.startswith("tests/") and p.endswith(".py")]
    test_specs = [head + path for path in test_sources] + [
        parent + ":" + origin[path] for path in test_sources for parent in parents
    ]
    try:
        scores = ruff_decisions({spec: texts[spec] for spec in specs if spec in texts})
        scores.update(ruff_test_findings({s: texts[s] for s in test_specs if s in texts}))
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(
            f"TEST-LAYOUT Ruff measurement unavailable: {error}; restore Ruff and retry",
            file=sys.stderr,
        )
        return 1
    grouped, copies, locations = [], [], []
    for path in selected:
        if head + path in texts:
            new = texts[head + path]
            each = []
            for parent in parents:
                old = texts.get(parent + ":" + origin[path])
                measurement = (
                    (scores.get(parent + ":" + origin[path], {}), scores[head + path])
                    if path in python_sources
                    else (scores.get(parent + ":" + origin[path], ()), scores[head + path])
                    if path in test_sources
                    else None
                )
                try:
                    current = problems(path, old, new, measurement)
                except (ValueError, KeyError) as error:
                    print(
                        f"TEST-LAYOUT Ruff measurement unavailable: {error}; retry", file=sys.stderr
                    )
                    return 1
                if path.startswith("src/") and _added_windows(path, old, new)[1]:
                    if corpus is None:
                        listing = _git("ls-files", "-z", "--", "src", "scripts")
                        if listing.returncode:
                            print(
                                "TEST-LAYOUT clone inputs unreadable; restore the Git index",
                                file=sys.stderr,
                            )
                            return 1
                        paths = listing.stdout.split("\0")
                        corpus = {
                            spec[len(head) :]: text
                            for spec, text in _blobs(
                                [head + item for item in paths if item]
                            ).items()
                        }
                    copies.append((path, old, new))
                    locations.append((len(grouped), len(each)))
                each.append(set(current))
            grouped.append(each)
    if copies:
        try:
            names = (
                head + "scripts/test-shape-tools/package.json",
                head + "scripts/test-shape-tools/package-lock.json",
            )
            staged_manifests = _blobs(list(names))
            manifests = tuple(
                staged_manifests[name].encode("utf-8", errors="surrogateescape") for name in names
            )
            for (group, parent), matches in zip(
                locations,
                clone_problems_batch(copies, corpus or {}, manifests=manifests),
                strict=True,
            ):
                grouped[group][parent].update(matches)
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
            print(
                f"TEST-LAYOUT clone detection unmeasured: {error}; "
                "restore the locked tool and retry",
                file=sys.stderr,
            )
            return 1
    for each in grouped:
        found += sorted(set.intersection(*each))
    for line in found:
        print(line, file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
