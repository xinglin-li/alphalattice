"""Registered test requirements, the tests that hold them, and ``tests find`` over both.

A requirement is registered before its test: config/registries/requirements.json holds each one's
id, sentence, owner package, layer and seconds. A test names the ids it proves with
``@pytest.mark.proves("<id>")``. Holders and lifecycle are read from the tree, never stored. A
row passing its checks is admitted at once; a journey, a Large test, an overrun or a near-duplicate
carries the test manager's recorded decision. Tests older than the registry are counted per file
in the private legacy count list, which only shrinks.

    python scripts/tests_registry.py find "detached host survives session"
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from datetime import date
from fnmatch import fnmatchcase
from itertools import product
from pathlib import Path
from typing import Any, NamedTuple

from devtools.testing.sizes import load_policy

REGISTRY = Path("config/registries/requirements.json")
LEGACY = Path("config/registries/tests-unregistered.json")  # private: names private tests
PRIVATE_MARKER = Path("config/release/public-manifest.json")  # tracked here, never exported
LAYERS = ("unit", "integration", "contract", "e2e", "native")
SIZES = dict(unit="small", integration="medium", contract="medium", e2e="large", native="large")
REGISTER = (
    "register it first: add its row to config/registries/requirements.json (id "
    "<owner's last segment>.<words>, description, owner, layer, seconds), then mark its test "
    '@pytest.mark.proves("<id>")'
)
_SHOWN = 8
_ID = re.compile(r"[a-z][a-z0-9_]*\.[a-z0-9_]+")
_ENDINGS = ("ing", "ed", "es", "s")
_STOP = frozenset(  # common words a query does not distinguish by
    re.findall(
        r"\w+",
        "a an and are as at be by each every for from in into is it "
        "its no not of on one only or same than that the then this to when while with",
    )
)


class Holder(NamedTuple):
    """One test function: its node, the ids it proves (none when unregistered), its words and the
    digest of its whole definition, which a move to another path keeps."""

    node: str
    ids: tuple[str, ...]
    words: str
    digest: str = ""


class Index(NamedTuple):
    """The requirements, every test, each requirement's holders and the search entries."""

    rows: dict[str, dict[str, Any]]
    tests: list[Holder]
    held: dict[str, list[Holder]]
    entries: dict[str, frozenset[str]]
    layered: dict[tuple[str, str], set[str]]

    def duplicates(
        self, prior: Mapping[tuple[str, str], set[str]] | None = None
    ) -> set[tuple[str, str]]:
        """Duplicate holder groups, optionally excluding unchanged inherited file debt."""
        prior = {} if prior is None else prior
        return {
            key
            for key, paths in self.layered.items()
            if len(paths) > max(1, len(prior.get(key, set())))
        }


def parse(text: str) -> dict[str, dict[str, Any]]:
    """The requirements by id; a malformed document or row refuses the registry, by name."""
    document = json.loads(text)
    if (
        not isinstance(document, dict)
        or not isinstance(document.get("entries"), dict)
        or (document.get("schema") != "alphalattice.requirements.v1")
    ):
        raise ValueError("requirements.schema")
    for key, row in document["entries"].items():
        seconds = row.get("seconds") if isinstance(row, dict) else None
        if not (
            isinstance(row, dict)
            and isinstance(row.get("description"), str)
            and isinstance(row.get("owner"), str)
            and row.get("layer") in LAYERS
            and isinstance(seconds, int | float)
            and not isinstance(seconds, bool)
            and seconds > 0
        ):
            raise ValueError("requirements.row_invalid:" + key)
    return dict(document["entries"])


def load(root: Path) -> dict[str, dict[str, Any]]:
    """The registered requirements by id."""
    return parse((root / REGISTRY).read_text("utf-8"))


def _proved(nodes: Iterable[ast.expr]) -> tuple[str, ...]:
    """The ids of each ``pytest.mark.proves(...)`` among decorators or a ``pytestmark`` value."""
    found: list[str] = []
    for node in nodes:
        for mark in node.elts if isinstance(node, (ast.List, ast.Tuple)) else (node,):
            if isinstance(mark, ast.Call) and getattr(mark.func, "attr", "") == "proves":
                found += [str(a.value) for a in mark.args if isinstance(a, ast.Constant)]
    return tuple(found)


def selection(value: object) -> tuple[str, ...]:
    """Required cost checks name complete, relative test nodes, never options or outside paths."""
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(n, str)
        or "::" not in n
        or not n.partition("::")[2]
        or not n.partition("::")[0].startswith("tests/")
        or not n.partition("::")[0].endswith(".py")
        or ".." in Path(n.partition("::")[0]).parts
        or any(c in n for c in "\n\r\x00\\")
        for n in value
    ):
        raise ValueError("test_size.required_selection_invalid")
    return tuple(dict.fromkeys(value))


def node_matches(node: str, selector: str) -> bool:
    """A full node or its parameter family; a selected parameter never admits siblings."""
    return node == selector or (
        "[" not in selector.split("::", 1)[-1] and node.startswith(selector + "[")
    )


def file_holders(relative: str, text: str) -> list[Holder]:
    """Each test function in one file, with the ids its function, class and module prove."""
    tree, found = ast.parse(text), []
    lines = text.splitlines(keepends=True)
    module = _proved(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(target, "id", "") == "pytestmark" for target in node.targets)
    )
    scopes: list[tuple[str, tuple[str, ...], ast.stmt]]
    scopes = [(f"{relative}::", (), node) for node in tree.body]
    scopes += [
        (f"{relative}::{c.name}::", _proved(c.decorator_list), node)
        for c in tree.body
        if isinstance(c, ast.ClassDef)
        for node in c.body
    ]
    for prefix, scope, node in scopes:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if node.name.startswith("test_"):
            text = " ".join((ast.get_docstring(node) or "").split("\n\n")[0].split())
            ids = module + scope + _proved(node.decorator_list)
            start = min([node.lineno, *(d.lineno for d in node.decorator_list)]) - 1
            image = json.dumps([ids, "".join(lines[start : node.end_lineno])], ensure_ascii=False)
            digest = hashlib.sha256(image.encode("utf-8")).hexdigest()
            found.append(Holder(prefix + node.name, ids, f"{node.name} {text}", digest))
    return found


def fresh(before: Iterable[Holder], after: Iterable[Holder]) -> list[Holder]:
    """New functions or newly proved ids on an existing function.

    A qualified unchanged definition relocating carries its standing; each removal matches one copy.
    """
    old, now = {test.node: test for test in before}, {test.node: test for test in after}
    gone = Counter((n.split("::", 1)[1], t.digest) for n, t in old.items() if n not in now)
    found = []
    for node, test in sorted(now.items()):
        identity = (node.split("::", 1)[1], test.digest)
        if node in old:
            added_ids = tuple(key for key in test.ids if key not in old[node].ids)
            if added_ids:
                found.append(test._replace(ids=added_ids))
            continue
        if gone[identity]:
            gone[identity] -= 1  # this copy is the move; a second copy is new
        else:
            found.append(test)
    return found


def holders(root: Path) -> list[Holder]:
    """Every test function under tests/."""
    return [
        holder
        for path in sorted((root / "tests").rglob("test_*.py"))
        for holder in file_holders(path.relative_to(root).as_posix(), path.read_text("utf-8"))
    ]


def unregistered(tests: Iterable[Holder]) -> Counter[str]:
    """Each file's count of test functions that prove nothing."""
    return Counter(test.node.split("::", 1)[0] for test in tests if not test.ids)


def legacy(root: Path) -> dict[str, int] | None:
    """Each file's count of tests older than the registry, which only shrinks. The list is private:
    the public export (which never holds the private release manifest) has none, and there the
    backstop stands aside; a private checkout without it is refused by name."""
    if (root / LEGACY).is_file():
        return dict(json.loads((root / LEGACY).read_text("utf-8"))["files"])
    if not (root / PRIVATE_MARKER).is_file():
        return None
    raise ValueError(
        f"requirements.legacy_missing:{LEGACY.as_posix()} is absent in a private checkout; restore "
        "it from git, or count the tree once: python scripts/tests_registry.py shrink --seed"
    )


def legacy_owner(root: Path) -> tuple[str, dict[str, int]]:
    """Read the registered legacy ledger owner and its shrink-only per-file counts."""
    path = root / LEGACY
    try:
        document = json.loads(path.read_text("utf-8"))
        if (
            not isinstance(document, dict)
            or document.get("schema") != "alphalattice.tests-unregistered.v1"
            or document.get("owner") != "devtools.testing"
            or not isinstance(document.get("files"), dict)
            or any(
                not isinstance(name, str) or type(count) is not int or count < 1
                for name, count in document["files"].items()
            )
        ):
            raise ValueError("invalid schema, registered owner, or counts")
        return document["owner"], document["files"]
    except (OSError, UnicodeError, ValueError, TypeError) as error:
        raise ValueError(f"requirements.legacy_unavailable:{LEGACY.as_posix()}: {error}") from None


def shrink(root: Path, *, seed: bool = False) -> dict[str, int]:
    """Lower each file's legacy count to what the tree holds, or with ``seed`` count it first; a
    count never rises, so a new test registers instead; seeding is only the first count."""
    if seed and (root / LEGACY).is_file():
        raise ValueError(
            f"requirements.legacy_seeded:{LEGACY.as_posix()} exists and only shrinks; run "
            "python scripts/tests_registry.py shrink without --seed"
        )
    now = unregistered(holders(root))
    counted = {} if seed else legacy(root) or {}
    files = dict(now) if seed else {p: min(n, now[p]) for p, n in counted.items() if now[p]}
    document = {
        "schema": "alphalattice.tests-unregistered.v1",
        "owner": "devtools.testing",
        "note": "Each file's test functions that predate the requirements registry and prove no "
        "requirement; a count only falls (python scripts/tests_registry.py shrink) and a new test "
        "registers its requirement instead.",
        "files": dict(sorted(files.items())),
    }
    (root / LEGACY).write_bytes((json.dumps(document, indent=2) + "\n").encode("utf-8"))
    return files


def words(text: str) -> frozenset[str]:
    """Lowercased words, snake_case split, common words dropped and plural or tense endings cut."""
    found = set()
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        cut = next((e for e in _ENDINGS if word.endswith(e) and len(word) > len(e) + 3), "")
        if word not in _STOP and len(word) > 1 and (word[: -len(cut)] if cut else word) != "test":
            found.add(word[: -len(cut)] if cut else word)
    return frozenset(found)


def _idf(entries: Mapping[str, frozenset[str]]) -> Callable[[str], float]:
    """Each word's inverse document frequency over the entries."""
    counts = Counter(word for entry in entries.values() for word in entry)
    return lambda word: math.log(len(entries) / (1 + counts[word])) + 1


def share(asked: frozenset[str], entry: frozenset[str], idf: Callable[[str], float]) -> float:
    """The IDF-weighted share of the asked words that an entry holds."""
    total = sum(map(idf, asked))
    return sum(map(idf, asked & entry)) / total if total else 0.0


def matches(query: str, entries: dict[str, frozenset[str]]) -> list[tuple[float, str]]:
    """Entries holding a third or more of the query's IDF-weighted words, best first."""
    asked, idf = words(query), _idf(entries)
    if not entries:
        return []
    scored = [(share(asked, entry, idf), key) for key, entry in entries.items()]
    return sorted(((s, k) for s, k in scored if s >= 1 / 3), key=lambda pair: (-pair[0], pair[1]))


def _places(
    rows: Mapping[str, dict[str, Any]], tests: Iterable[Holder]
) -> dict[tuple[str, str], set[str]]:
    """One physical holder per id/layer, independent of its focused functions and cases."""
    found: dict[tuple[str, str], set[str]] = {}
    for test in tests:
        declared = {key: rows[key]["layer"] for key in test.ids if key in rows}
        level = primary_layer(test.node, declared)
        for key in declared:
            found.setdefault((key, level), set()).add(test.node.partition("::")[0])
    return found


def index(
    root: Path,
    rows: dict[str, dict[str, Any]] | None = None,
    *,
    tests: Iterable[Holder] | None = None,
) -> Index:
    """The requirements and every test, read once for any number of queries."""
    rows = load(root) if rows is None else rows
    tests = list(holders(root) if tests is None else tests)
    held: dict[str, list[Holder]] = {key: [] for key in rows}
    for test in tests:
        for key in test.ids:
            held.setdefault(key, []).append(test)
    entries = {
        f"R {key}": words(" ".join([key, row["description"], *(t.words for t in held[key])]))
        for key, row in rows.items()
    } | {f"T {test.node}": words(test.words) for test in tests if not test.ids}
    return Index(rows, tests, held, entries, _places(rows, tests))


def _requirement(
    key: str, row: dict[str, Any], holding: list[str], quarantine: list[str]
) -> list[str]:
    """A requirement's lines: its derived lifecycle, layer, budget, owner, sentence and holders."""
    quarantined = any(fnmatchcase(node, q) for node in holding for q in quarantine)
    state = "proposed" if not holding else "quarantined" if quarantined else "active"
    places: dict[str, list[str]] = {}
    for node in holding:
        places.setdefault(node.partition("::")[0], []).append(node)
    return [
        f"{key}  {'retired' if row.get('retired') else state}  {row['layer']}  "
        f"{row['seconds']:g} s  {row['owner']}",
        f"  {row['description']}",
        *(
            line
            for nodes in places.values()
            for line in [f"  held by {nodes[0]}", *(f"    case {node}" for node in nodes[1:])]
        ),
    ]


def find(root: Path, query: str, at: Index | None = None) -> list[str]:
    """``tests find``: the requirements and unregistered tests a query matches, with each
    requirement's lifecycle and holders, and the next lawful step."""
    at = index(root) if at is None else at
    ranked = sorted(matches(query, at.entries), key=lambda pair: pair[1][0] != "R")  # rows first
    found = [key.split(" ", 1) for _, key in ranked[:_SHOWN]]
    if not found:
        return [f'Nothing matches "{query}".', f"Next: {REGISTER}."]
    today = date.today().isoformat()  # a quarantine holds until its date, as the gate reads it
    live = [q for q in load_policy(root).get("quarantine", []) if q["until"] >= today]
    quarantine = [q["node"].partition("[")[0] for q in live]
    docstrings = {test.node: test.words.partition(" ")[2] for test in at.tests}
    kinds = Counter(kind for kind, _ in found)
    lines = [f'{kinds["R"]} requirements and {kinds["T"]} unregistered tests match "{query}":']
    for kind, name in found:
        lines += (
            [f"unregistered  {name}", f"  {docstrings[name] or '(no docstring)'}"]
            if kind == "T"
            else _requirement(name, at.rows[name], [t.node for t in at.held[name]], quarantine)
        )
    return [*lines, f"Next: add your case to a holder above; only if none holds it, {REGISTER}."]


def layer(node: str, fallback: str = "") -> str:
    """An explicit layer folder, otherwise the registered requirement's default."""
    parts = node.partition("::")[0].replace("\\", "/").split("/")
    return parts[1] if len(parts) > 1 and parts[0] == "tests" and parts[1] in LAYERS else fallback


def primary_layer(node: str, declared: Mapping[str, str]) -> str:
    """Registered layers are authoritative; an explicit folder must agree before grouping."""
    explicit, levels = layer(node), set(declared.values())
    named = ", ".join(declared)
    if len(levels) > 1:
        raise ValueError(
            f"testing.layer_contract: {node} proves mixed layers for {named}: {sorted(levels)}; "
            "split them into focused tests at each registered layer"
        )
    primary = next(iter(levels), explicit)
    if explicit and primary != explicit:
        raise ValueError(
            f"testing.layer_contract: {node} folder {explicit} disagrees with "
            f"registered layer {primary} for {named}; move the file there "
            "or register its true layer"
        )
    return primary


def registration(
    root: Path,
    new: Iterable[Holder],
    rows: dict[str, dict[str, Any]],
    *,
    at: Index | None = None,
    prior: Mapping[tuple[str, str], set[str]] | None = None,
) -> list[str]:
    """Each new test proves registered, unretired ids, or is refused with what ``tests find``
    returns for its name and docstring, and the next lawful step."""
    lines: list[str] = []
    new = list(new)
    if not new:
        return lines
    at = at or index(root, rows)
    added = {test.node for test in new}
    prior = (
        _places(rows, (test for test in at.tests if test.node not in added))
        if prior is None
        else prior
    )
    anchors = {
        key: sorted(paths, key=lambda path: (path not in prior.get(key, set()), path))[:2]
        for key, paths in at.layered.items()
    }
    for test in new:
        bad = [key for key in test.ids if key not in rows or rows[key].get("retired")]
        if test.ids and not bad:
            for key in dict.fromkeys(test.ids):
                level = rows[key]["layer"]
                place = test.node.partition("::")[0]
                peer = min(set(anchors.get((key, level), ())) - {place}, default=None)
                if peer and place not in prior.get((key, level), set()):
                    lines.append(
                        f"TEST-REGISTRY {test.node}: {key} already has its {level} holder {peer}; "
                        "extend that file, move the remaining holders too, or merge into one file."
                    )
            continue
        query = " ".join(test.words[5:].replace("_", " ").split())
        others = at._replace(entries={k: v for k, v in at.entries.items() if k != f"T {test.node}"})
        what = f"proves {', '.join(bad)}, unregistered or retired" if bad else "proves nothing"
        lines += [
            f"TEST-REGISTRY {test.node} {what}, and an unregistered test does not run in a gate.",
            f'  tests find "{query}" found:',
            *(f"    {line}" for line in find(root, query, others)[:-1]),
            "  Next: if one of these holds the requirement, add your case to that holder (a "
            "parameter or one assertion) and delete this function; an unregistered holder takes "
            f'@pytest.mark.proves("<id>") once its row is registered. Otherwise {REGISTER}, '
            "and run the guard again.",
        ]
    return lines


def _checks(key: str, row: dict[str, Any], owners: set[str]) -> list[str]:
    """The automatic checks a row fails: its id's form, its owner and its one sentence."""
    segment, text = row["owner"].rsplit(".", 1)[-1], row["description"]
    sentence = text.endswith(".") and not re.search(r"[.!?]\s+\S", text) and "\n" not in text
    return [
        why
        for passed, why in (
            (
                _ID.fullmatch(key) and key.partition(".")[0] == segment,
                f"its id is {segment}.<words>",
            ),
            (row["owner"] in owners, "its owner owns a noun in config/registries/owners-map.json"),
            (
                sentence and 5 <= len(text.split()) <= 40,
                "its description is one 5-40 word sentence",
            ),
        )
        if not passed
    ]


def _decided(exception: Any, kinds: list[str]) -> bool:
    """A recorded decision names every kind it covers, a reason, and a dated decision."""
    named = exception.get("kinds") if isinstance(exception, dict) else None
    if not isinstance(named, list) or not all(isinstance(k, str) for k in named):
        return False
    if not set(kinds) <= set(named):
        return False
    why, decided = exception.get("why"), exception.get("decided")
    dated = isinstance(decided, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}: \S.*", decided)
    return isinstance(why, str) and bool(why.strip()) and bool(dated)


def admission(
    root: Path,
    rows: dict[str, dict[str, Any]],
    before: dict[str, dict[str, Any]],
    *,
    at: Index | None = None,
) -> list[str]:
    """What the change's added or edited rows fail: a check, or an exception the test manager has
    not decided (a journey, a Large test, an overrun, a near-duplicate); a row is only retired."""
    limits = load_policy(root)["limits"]  # a layer's limit is its size's: one source
    found = [
        f'TEST-REGISTRY {key} was removed; retire it instead ("retired": "<date>: <why>")'
        for key in sorted(before.keys() - rows.keys())
    ]
    changed = sorted(key for key, row in rows.items() if row != before.get(key))
    if not changed:
        return found
    nouns = json.loads((root / "config/registries/owners-map.json").read_text("utf-8"))["nouns"]
    owners, at = {noun["owner"] for noun in nouns.values()}, at or index(root, rows)
    idf = _idf(at.entries)
    for key in changed:
        row, mine = rows[key], at.entries[f"R {key}"]
        found += [f"TEST-REGISTRY {key} is refused: {why}." for why in _checks(key, row, owners)]
        twins = [
            other.partition(" ")[2]
            for other, entry in at.entries.items()
            if other != f"R {key}"
            and (other[0] == "T" or rows[other[2:]]["owner"] == row["owner"])
            and min(share(mine, entry, idf), share(entry, mine, idf)) >= 0.6
        ]
        needed = {
            "journey": row["layer"] in ("e2e", "native"),
            "large": row["seconds"] > 60,
            "over_budget": row["seconds"] > limits[SIZES[row["layer"]]],
            "near_duplicate": bool(twins),
        }
        kinds = [kind for kind, need in needed.items() if need]
        if kinds and not _decided(row.get("exception"), kinds):
            near = f"; extend {', '.join(twins[:3])} instead, or" if twins else "; or"
            found.append(
                f"TEST-REGISTRY {key} needs the test manager's decision ({', '.join(kinds)}){near} "
                'ask the test manager on the board and record "exception": {"kinds": '
                f'{json.dumps(kinds)}, "why": "<why>", "decided": "<date>: <the decision>"}}.'
            )
    return found


def guard(
    root: Path,
    new: Iterable[Holder],
    text: str | None,
    base: str | None,
    *,
    tests: Iterable[Holder] | None = None,
    before_tests: Iterable[Holder] | None = None,
) -> list[str]:
    """The fast guard's step over a change: its new tests and its added or edited rows, parsed
    once; a tree with no registry at either end has nothing to register."""
    if text is None and base is None:
        return []
    try:
        rows, before = parse(text) if text else {}, parse(base) if base else {}
        new = list(new)
        if not new and rows == before and before_tests is None:
            return []
        at = index(root, rows, tests=tests)
        added_nodes = {test.node for test in new}
        prior = _places(
            before,
            (test for test in at.tests if test.node not in added_nodes)
            if before_tests is None
            else before_tests,
        )
    except ValueError as error:
        return [f"TEST-REGISTRY {REGISTRY.as_posix()} is refused: {error}."]
    changed_rows = {key for key, row in rows.items() if row != before.get(key)}
    affected = at.duplicates(prior) | set(product(changed_rows, LAYERS))
    candidates = {test.node: test for test in new}
    for test in at.tests:
        ids = tuple(
            key for key in test.ids if (key, rows.get(key, {}).get("layer", "")) in affected
        )
        if ids:
            previous = candidates.get(test.node)
            candidates[test.node] = test._replace(
                ids=tuple(dict.fromkeys((*(previous.ids if previous else ()), *ids)))
            )
    return registration(root, candidates.values(), rows, at=at, prior=prior) + admission(
        root, rows, before, at=at
    )
