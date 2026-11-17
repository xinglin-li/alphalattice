"""The fast guard holds tests/README.md's writing rules on what a change adds."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

import check_test_shape as shape
from check_playpen import shrink_only_growth
from check_test_shape import problems
from devtools.testing import os_marks
from devtools.testing.requirements import file_holders

PATH = "tests/owner/test_example.py"
_REAL_RUN = subprocess.run


def _antipattern_cases(monkeypatch) -> None:
    """The real Ruff rules ignore inline escapes and identify each file and rule."""
    source = (
        "import logging\n"
        "def example(value):\n"
        "    logging.warning(f'{value}')  # noqa: G004, LOG015\n"
        "    value = unknown  # type: ignore\n"
        "    try:\n        value()\n    except Exception:\n        pass\n"
        + "    if value:\n        value = 1\n" * 13
        + "    value = 1\n" * 52
    )
    waits = "\n".join(
        [
            "import requests, time",
            "from datetime import datetime",
            "def waits():",
            "    datetime.now()",
            "    requests.get('https://example.test')",
            "    for x in []:",
            "        try:",
            "            x()",
            "        except Exception:",
            "            continue",
            "async def late():",
            "    time.sleep" + "(1)",
            "",
        ]
    )
    mocked = "def test_mocks(monkeypatch):\n" + "    monkeypatch.setattr(a, b, c)\n" * 5
    found = shape.ruff_antipatterns({"src/example.py": source, "src/waits.py": waits, PATH: mocked})
    for rule in ("G004", "LOG015", "PGH003"):
        assert found[f"src/example.py::{rule}"] > 0
    for rule in ("C901", "PLR0912", "PLR0915"):
        assert found[f"src/example.py::{rule}::example"] > 0
    for rule in ("DTZ005", "S112", "S113", "ASYNC251"):
        assert found[f"src/waits.py::{rule}"]
    assert found[f"{PATH}::test-too-many-mocks::test_mocks"] == 5
    workflow = "\n".join(
        [
            "name: test",
            "on: push",
            "jobs:",
            "  a:",
            "    runs-on: ubuntu-latest",
            "    steps:",
            "      - runz: x",
            "",
        ]
    )
    shell = "#!/bin/sh\necho $value\n# shellcheck disable=SC2086\n"
    auxiliary = shape.auxiliary_antipatterns(
        {".github/workflows/example.yml": workflow, "test.sh": shell}
    )
    assert any(key.startswith(".github/workflows/example.yml::actionlint-") for key in auxiliary)
    assert auxiliary["test.sh::shellcheck-inline-disable"] == 1
    assert any(key.startswith("test.sh::shellcheck-SC") for key in auxiliary)
    data = shape.test_data_antipatterns(["tests/a/data.csv", "tests/a/test_example.py"])
    assert data == {"tests/a/data.csv::test-data-file": 1}
    assert shape.antipattern_growth({}, data)
    assert shape.antipattern_growth(found, found) == []
    assert shape.antipattern_growth({}, found)
    loops = {
        "src/endless.py": "while True:\n    tick()\n",
        "src/break.py": "while True:\n    if done():\n        break\n",
        "src/nested.py": "while True:\n    def nested():\n        return 1\n    nested()\n",
        "src/inner-for.py": "while True:\n    for item in items:\n        break\n",
        "src/inner-while.py": "while True:\n    while condition:\n        break\n",
        "src/loop-else.py": (
            "while True:\n    for item in items:\n        pass\n    else:\n        break\n"
        ),
        "scripts/deadline.py": (
            "while True:\n    if time.monotonic() >= deadline:\n        raise TimeoutError()\n"
        ),
    }
    assert set(shape.unbounded_loops(loops)) == {
        "src/endless.py",
        "src/nested.py",
        "src/inner-for.py",
        "src/inner-while.py",
    }
    old = {"src/old.py::PGH003": 1}
    new = {"src/new.py::PGH003": 1}
    assert not shape.antipattern_growth(
        shape.relocated_antipatterns(old, {"src/new.py": "src/old.py"}), new
    )
    assert shape.antipattern_growth(shape.relocated_antipatterns(old, {}), new)
    for name in ("ruff_antipatterns", "test_data_antipatterns", "auxiliary_antipatterns"):
        monkeypatch.setattr(shape, name, lambda _: {})
    baseline = json.dumps({"counts": {"src/a.py::PGH003": 1}})
    snapshots = {
        ":" + shape.ANTIPATTERN_BASELINE: baseline,
        "HEAD:" + shape.ANTIPATTERN_BASELINE: baseline,
    }
    assert not shape._antipattern_problems([], [], snapshots, ("HEAD", "MERGE_HEAD"), ":", moves={})
    with pytest.raises(ValueError, match="anti-pattern baseline missing from parent"):
        shape._antipattern_problems([], [], snapshots, ("MERGE_HEAD",), ":", moves={})
    snapshots[":src/endless.py"] = loops["src/endless.py"]
    snapshots["HEAD:src/endless.py"] = "pass\n"
    assert any(
        "unbounded-loop" in issue
        for issue in shape._antipattern_problems(
            [], ["src/endless.py"], snapshots, ("HEAD",), ":", moves={}
        )
    )
    snapshots["HEAD:src/endless.py"] = loops["src/endless.py"]
    assert not shape._antipattern_problems(
        [], ["src/endless.py"], snapshots, ("HEAD",), ":", moves={}
    )
    snapshots[":" + shape.ANTIPATTERN_BASELINE] = json.dumps({"counts": {"src/a.py::PGH003": 2}})
    assert shape._antipattern_problems([], [], snapshots, ("HEAD", "MERGE_HEAD"), ":", moves={})
    seeded = {
        "src/a.py::PGH003": 1,
        "tests/conftest.py::new-rule": 1,
    }  # a card branched before a rule landed
    snapshots |= {
        "HEAD:" + shape.ANTIPATTERN_BASELINE: json.dumps({"counts": seeded}),
        "MERGE_HEAD:" + shape.ANTIPATTERN_BASELINE: baseline,
        ":" + shape.ANTIPATTERN_BASELINE: json.dumps({"counts": seeded}),
    }
    assert not shape._antipattern_problems([], [], snapshots, ("HEAD", "MERGE_HEAD"), ":", moves={})
    snapshots[":" + shape.ANTIPATTERN_BASELINE] = json.dumps(
        {"counts": {**seeded, "tests/conftest.py::new-rule": 2}}
    )
    assert shape._antipattern_problems([], [], snapshots, ("HEAD", "MERGE_HEAD"), ":", moves={})
    seed = {"src/a.py::PGH003": 1, "tests/a/test_b.py::new-rule": 4}  # a new rule's first counts
    held = {"src/a.py::PGH003": 1}
    assert shape.seeded_rules(held, seed, [shape.TEST_ANTIPATTERN_RULES]) == {"new-rule"}
    assert not shape.seeded_rules(held, seed, ["tests/a/test_b.py"])


def _split_antipattern_cases(monkeypatch, tmp_path) -> None:
    for name in ("ruff_antipatterns", "test_data_antipatterns", "auxiliary_antipatterns"):
        monkeypatch.setattr(shape, name, lambda _: {})
    private = (
        "config/" + Path(shape.ANTIPATTERN_BASELINE).name.removesuffix(".json") + "-private.json"
    )
    key = "probe/private.py::PGH003"
    snapshots = {
        ":" + shape.ANTIPATTERN_BASELINE: json.dumps({"counts": {}}),
        ":" + private: json.dumps({"counts": {key: 2}}),
        "HEAD:" + shape.ANTIPATTERN_BASELINE: json.dumps({"counts": {key: 2}}),
        ":probe/private.py": "x = 1\n",
    }
    monkeypatch.setattr(shape, "ruff_antipatterns", lambda _: {key: 2})
    assert not shape._antipattern_problems(
        [], ["probe/private.py"], snapshots, ("HEAD",), ":", moves={}
    )
    snapshots[":" + private] = json.dumps({"counts": {key: 3}})
    assert shape._antipattern_problems(
        [], ["probe/private.py"], snapshots, ("HEAD",), ":", moves={}
    )
    snapshots[":" + shape.ANTIPATTERN_BASELINE] = json.dumps({"counts": {key: 1}})
    assert shape.antipattern_sides(snapshots, ":") == {key: 4}
    public_file = tmp_path / shape.ANTIPATTERN_BASELINE
    public_file.parent.mkdir(parents=True)
    public_file.write_text(json.dumps({"counts": {"src/a.py::PGH003": 1}}))
    assert shape.load_antipattern_baseline(tmp_path) == {"src/a.py::PGH003": 1}
    (tmp_path / private).write_text(json.dumps({"counts": {key: 2}}))
    assert shape.load_antipattern_baseline(tmp_path) == {"src/a.py::PGH003": 1, key: 2}


LONG = "def test_long():\n" + "    x = 1\n" * 50 + "    assert x\n"
STEPS = "def main():\n" + "    if a:\n        b()\n" * 12 + "    c()\n" * 59
MOCKED = (
    "def test_mocks(monkeypatch):\n" + "    monkeypatch.setattr(a, b, c)\n" * 5 + "    assert a\n"
)
UNBOUNDED = (  # a child with no timeout: one unbounded-child finding in test_a
    "import subprocess\ndef test_a():\n    assert subprocess.run(['git'], input='',"
    " creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))\n"
)
UNBOUNDED_KEY = PATH + "::test-child-unbounded::test_a"
GIT_CASES = (  # a parametrized test that starts Git for each of its two cases
    "import pytest, subprocess\n"
    "@pytest.mark.parametrize('ref', ['a', 'b'])\ndef test_a(ref):\n"
    "    assert subprocess.run(['git', ref], timeout=9, input='',"
    " creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))\n"
)
GIT_KEY = PATH + "::test-parametrized-process::test_a"
HELPER = "tests/owner/case_tables.py"  # a tests/ helper holding a table the test imports
HELPER_CASES = "REFS = ['a', 'b', 'c']\n"
IMPORTED_CASES = GIT_CASES.replace(
    "import pytest, subprocess\n",
    "import pytest, subprocess\nfrom tests.owner.case_tables import REFS\n",
).replace("['a', 'b']", "REFS")
SIZE_CASES = {  # changed file, parent text (None: new), its text; the refused rule, the signal
    "a-51-statement-test-is-accepted": (PATH, None, LONG, None, "PLR0915 51 in test_long"),
    "an-83-statement-script-is-accepted": ("scripts/step.py", None, STEPS, None, "C901 13 in main"),
    "five-mocks-are-a-signal": (PATH, None, MOCKED, None, "test-too-many-mocks 5 in test_mocks"),
    "a-giant-the-change-left-alone-says-nothing": (PATH, LONG, LONG + "X = 1\n", None, None),
    "a-blanket-type-ignore-is-still-refused": (
        "src/a.py",
        None,
        "x = y  # type: ignore\n",
        "src/a.py::PGH003",
        None,
    ),
    "a-test-child-that-opens-a-window-is-refused": (
        PATH,
        None,
        "import subprocess\ndef test_a():\n    assert subprocess.run(['git'], timeout=9, input='')"
        "\n",
        PATH + "::test-child-visible-window",
        None,
    ),
    "a-test-child-with-no-window-is-accepted": (
        PATH,
        None,
        "import subprocess\ndef test_a():\n    assert subprocess.run(['git'], timeout=9, input='',"
        " creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))\n",
        None,
        None,
    ),
    "a-test-child-with-no-timeout-is-refused": (PATH, None, UNBOUNDED, UNBOUNDED_KEY, None),
    "an-allowed-unbounded-child-is-accepted": (PATH, None, UNBOUNDED, None, None),
    "an-allowed-unbounded-child-grown-is-refused": (
        PATH,
        None,
        UNBOUNDED + "    assert subprocess.run(['git'], input='',"
        " creationflags=subprocess.CREATE_NO_WINDOW)\n",
        UNBOUNDED_KEY,
        None,
    ),
    "a-parametrized-git-case-is-refused": (PATH, None, GIT_CASES, GIT_KEY, None),
    "a-parametrized-in-memory-case-is-accepted": (
        PATH,
        None,
        "import pytest\n"
        "@pytest.mark.parametrize('ref', ['a', 'b'])\ndef test_a(ref):\n"
        "    assert ref in 'ab'\n",
        None,
        None,
    ),
    "an-allowed-parametrized-git-case-is-accepted": (PATH, None, GIT_CASES, None, None),
    "an-allowed-parametrized-git-table-grown-is-refused": (
        PATH,
        None,
        GIT_CASES.replace("['a', 'b']", "['a', 'b', 'c']"),
        GIT_KEY,
        None,
    ),
    "an-imported-parametrized-table-counts-its-cases": (PATH, None, IMPORTED_CASES, GIT_KEY, None),
    "a-waited-popen-is-accepted": (
        PATH,
        None,
        "import subprocess\ndef test_a():\n    child = subprocess.Popen(['git'],"
        " stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))\n"
        "    assert child.wait(timeout=9) == 0\n",
        None,
        None,
    ),
    "an-unclassed-model-turn-is-refused": (
        PATH,
        None,
        "import subprocess\ndef test_a():\n    assert subprocess.run(['claude', '-p', 'hi'],"
        " timeout=9, input='', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))\n",
        PATH + "::test-paid-agent::test_a",
        None,
    ),
    "a-classed-model-turn-is-accepted": (
        PATH,
        None,
        "import subprocess, pytest\n"
        "@pytest.mark.paid_agent\ndef test_a():\n"
        "    assert subprocess.run(['claude', '-p', 'hi'], timeout=9, input='',"
        " creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))\n",
        None,
        None,
    ),
    "an-asserted-model-argv-is-no-turn": (
        PATH,
        None,
        "def test_a():\n    assert ['claude', '-p'] == ['claude', '-p']\n",
        None,
        None,
    ),
}
ALLOWED = {  # the cases held in the allowance: file::rule::function and its count
    "an-allowed-unbounded-child-is-accepted": (UNBOUNDED_KEY, 1),
    "an-allowed-unbounded-child-grown-is-refused": (UNBOUNDED_KEY, 1),
    "an-allowed-parametrized-git-case-is-accepted": (GIT_KEY, 2),
    "an-allowed-parametrized-git-table-grown-is-refused": (GIT_KEY, 2),
    "an-imported-parametrized-table-counts-its-cases": (GIT_KEY, 2),
}


@pytest.mark.proves("testing.size_counts_are_review_signals")
@pytest.mark.parametrize(
    ("path", "old", "new", "refused", "signal"), SIZE_CASES.values(), ids=list(SIZE_CASES)
)
def test_a_size_count_is_a_signal_and_a_harm_still_refuses(
    capsys, request, path, old, new, refused, signal
) -> None:
    """A size or form count over its bound never refuses a commit, and is named as a signal only
    in a function the change wrote; a blanket type ignore is still refused (the 11-14 verdict).
    A test's unbounded child or unclassed model turn is refused past its allowance, by function."""
    held = ALLOWED.get(request.node.callspec.id)
    empty = json.dumps({"counts": dict([held]) if held else {}})
    texts = {":" + path: new, ":" + shape.ANTIPATTERN_BASELINE: empty, ":" + HELPER: HELPER_CASES}
    texts["HEAD:" + shape.ANTIPATTERN_BASELINE] = empty
    rules = Path(shape.__file__).parents[1] / shape.TEST_ANTIPATTERN_RULES  # ast-grep, run for real
    texts[":" + shape.TEST_ANTIPATTERN_RULES] = rules.read_text(encoding="utf-8")
    if old is not None:
        texts["HEAD:" + path] = old
    found = shape._antipattern_problems([path], [path], texts, ("HEAD",), ":", moves={})
    assert [line.split(": ")[0] for line in found] == ([refused] if refused else [])
    said = capsys.readouterr().out
    assert (signal in said) if signal else "test_shape.signal" not in said


@pytest.mark.proves("testing.size_counts_are_review_signals")
def test_a_changed_case_table_is_judged_in_each_test_that_imports_it() -> None:
    """A test the change left alone is measured again when a tests/ helper it takes a parametrize
    table from changes, so its allowance by case falls with the table and refuses its growth."""
    listed = []

    def grep(*args):  # git grep -l names the helper's importers (and a conftest, never a reader)
        listed.append(args)
        return subprocess.CompletedProcess(args, 0, f"HEAD:{PATH}\nHEAD:tests/owner/conftest.py\n")

    def read(specs):
        return {s: IMPORTED_CASES for s in specs if PATH in s}

    texts = {"HEAD:" + HELPER: HELPER_CASES}
    assert shape._table_readers([HELPER], texts, "HEAD:", grep=grep, read=read) == [PATH]
    assert texts["HEAD:" + PATH] == IMPORTED_CASES and "-ecase_tables" in listed[0]
    allowance = json.dumps({"counts": {GIT_KEY: 2}})
    for table, refused in ((HELPER_CASES, [GIT_KEY]), ("REFS = ['a']\n", [])):
        texts = {"HEAD:" + HELPER: table, "HEAD:" + PATH: IMPORTED_CASES}
        texts |= {f"{side}:{shape.ANTIPATTERN_BASELINE}": allowance for side in ("HEAD", "base")}
        found = shape._antipattern_problems(
            [HELPER], [HELPER], texts, ("base",), "HEAD:", moves={}, readers=[PATH]
        )
        assert [line.split(": ")[0] for line in found] == refused


@pytest.mark.large
@pytest.mark.proves("testing.antipattern_rules_hold_the_product_tree")
def test_nightly_antipattern_rules_hold_the_product_tree() -> None:
    """The slow rules and their own examples run over the whole tracked source each night."""
    listing = shape._git("ls-files", "-z", "--", "*.py")
    assert listing.returncode == 0
    paths = [path for path in listing.stdout.split("\0") if path and not path.startswith("tests/")]
    specs = [":" + path for path in paths]
    specs += [
        ":" + name
        for name in (shape.ANTIPATTERN_BASELINE, shape.ANTIPATTERN_SEMGREP, shape.ANTIPATTERN_CASES)
    ]
    blobs = shape._blobs([*specs, ":" + shape.ANTIPATTERN_PRIVATE])
    assert all(spec in blobs for spec in specs)
    rules = blobs[":" + shape.ANTIPATTERN_SEMGREP]
    shape._semgrep_selftest(rules, blobs[":" + shape.ANTIPATTERN_CASES])
    found = shape.semgrep_antipatterns({path: blobs[":" + path] for path in paths}, rules)
    assert not shape.antipattern_growth(shape.antipattern_sides(blobs, ":"), found)
    tests = [path for path in listing.stdout.split("\0") if path.startswith("tests/")]
    named = (*tests, shape.TEST_ANTIPATTERN_RULES, shape.TEST_ANTIPATTERN_CASES)
    blobs |= shape._blobs([":" + path for path in named])
    test_rules = blobs[":" + shape.TEST_ANTIPATTERN_RULES]
    shape._astgrep_selftest(test_rules, blobs[":" + shape.TEST_ANTIPATTERN_CASES])
    found = shape.astgrep_antipatterns({path: blobs[":" + path] for path in tests}, test_rules)
    assert not shape.antipattern_growth(shape.antipattern_sides(blobs, ":"), found)


OS_CASES = {  # each way a test reaches Windows: True when it is refused for naming no OS
    "WinDLL": ("def test_a():\n    ctypes.WinDLL('kernel32')\n", True),
    "marked": (
        '@pytest.mark.skipif(sys.platform != "win32", reason="w")\n'
        'def test_a():\n    ctypes.WinDLL("k")\n',
        False,
    ),
    "alias": (
        'WINDOWS = pytest.mark.skipif(os.name != "nt", reason="w")\n'
        "@WINDOWS\ndef test_a():\n    winreg.HKEY_CURRENT_USER\n",
        False,
    ),
    "module": (
        'pytestmark = pytest.mark.skipif(os.name != "nt", reason="w")\n'
        "def test_a():\n    msvcrt.getch()\n",
        False,
    ),
    "fork": ('def test_a():\n    if os.name == "nt":\n        ctypes.windll.kernel32\n', False),
    "and-fork": (
        'def test_a(x):\n    if x and sys.platform == "win32":\n        ctypes.windll.kernel32\n',
        False,
    ),
    "returns": (
        'def test_a():\n    if os.name != "nt":\n        return\n    ctypes.WinDLL("k")\n',
        False,
    ),
    "winerror": (
        "def test_a(tmp_path):\n    with pytest.raises(OSError) as caught:\n"
        "        open(tmp_path)\n    assert caught.value.winerror == 5\n",
        True,
    ),
    "winerror-set": (
        "def test_a():\n    error = OSError()\n    error.winerror = 5\n"
        "    assert error.winerror == 5\n",
        False,
    ),
    "powershell": (
        'def test_a():\n    subprocess.run(["powershell", "-Command", "x"], check=True)\n',
        True,
    ),
    "data": (
        'def test_a():\n    assert count({"command": ["pwsh", "-Command", "x"]}) == 1\n',
        False,
    ),
}


@pytest.mark.proves("testing.an_os_only_test_names_its_os")
@pytest.mark.parametrize("case", sorted(OS_CASES), ids=sorted(OS_CASES))
def test_a_test_that_reaches_an_os_only_api_names_its_os(case: str) -> None:
    """The product runs on macOS, Ubuntu and Windows: a test a change writes that reaches a
    Windows-only API is refused unless it names its OS, with the repair."""
    source, refused = OS_CASES[case]
    found = os_marks.refusals("tests/owner/test_x.py", source, "")
    assert bool(found) is refused
    assert all("os-unmarked" in line and "skipif" in line for line in found)
    assert (
        os_marks.refusals("tests/owner/test_x.py", source, source) == []
    )  # a kept test is not judged


SHORT, LONG = "user/temp", "run/" + "r" * 100  # a user's temp, and a stage's run-allocated one
SCANNER_TEMPS = {  # TEMP, LOCALAPPDATA, on Windows, the folder semgrep is given (None: TEMP)
    "a-short-windows-temp-is-kept": (SHORT, "user/local", True, None),
    "a-stage-run-temp-is-shortened": (LONG, "user/local", True, "user/local/Temp"),
    "no-user-temp-keeps-the-default": (LONG, "", True, None),
    "elsewhere-the-default-holds": (LONG, "", False, None),
}


@pytest.mark.proves("testing.the_semgrep_scan_gets_a_short_temp_on_windows")
@pytest.mark.parametrize(
    ("temp", "local", "windows", "given"), SCANNER_TEMPS.values(), ids=SCANNER_TEMPS.keys()
)
def test_the_semgrep_scan_gets_a_short_temp_on_windows(
    temp: str, local: str, windows: bool, given: str | None
) -> None:
    """semgrep-core fails once its TEMP passes about 80 characters on Windows, as a stage's run
    TEMP does: there the scan goes under the user's own temp folder; elsewhere nothing changes."""
    found = shape.scanner_temp(temp, {"LOCALAPPDATA": local}, windows=windows)
    assert (Path(found) if found else None) == (Path(given) if given else None)


def test_a_declared_finding_set_only_loses_keys_and_keeps_its_owners() -> None:
    """New finding keys or swapped owners refuse, while retained and fixed findings pass."""
    row = dict(state="floor", mutation="shape", refusal_code="shape", owner="committee.py")
    before = dict(shrink_only_set=["state", "mutation", "refusal_code"], entries=[row])
    assert not shrink_only_growth("config/findings.json", before, before)
    assert not shrink_only_growth("config/findings.json", before, {**before, "entries": []})
    assert shrink_only_growth("config/findings.json", before, {"entries": []})
    for field in ("state", "owner"):
        added = {**row, field: "different"}
        assert shrink_only_growth("config/findings.json", before, {**before, "entries": [added]})
    sizes = dict(before, shrink_only_list="large", large=[row], entries=[])
    assert not shrink_only_growth("config/sizes.json", sizes, sizes)
    assert not shrink_only_growth("config/sizes.json", sizes, {**sizes, "large": []})
    assert shrink_only_growth(
        "config/sizes.json", sizes, {**sizes, "large": [dict(row, state="different")]}
    )
    assert shrink_only_growth("config/sizes.json", sizes, {**sizes, "shrink_only_list": "entries"})
    sizes = dict(shrink_only_set=["pattern"], shrink_only_list="large", large=[])
    path = "config/registries/test-sizes.json"
    added = dict(pattern="test_example", owner="tests")
    after = dict(sizes, large=[added])
    assert shrink_only_growth(path, sizes, after)
    for value in (None, "", "accepted", "2026-02-30: accepted", "2026-10-22:"):
        assert shrink_only_growth(path, sizes, dict(sizes, large=[dict(added, admitted=value)]))
    admitted = dict(added, admitted="2026-10-22: accepted")
    after = dict(sizes, large=[admitted])
    assert not shrink_only_growth(path, sizes, after)
    assert not shrink_only_growth(path, after, sizes)
    assert shrink_only_growth("config/sizes.json", sizes, after)
    from devtools.architecture.git_inputs import carried

    moves = {"tests/b/test_x.py": "tests/a/test_x.py"}  # a byte-identical move, new to old
    keys = ["defaults:tests/a/test_x.py*", "x/tests/a/test_x.py", "tests/a/test_x.pyc"]
    assert carried(keys, moves) == {"defaults:tests/b/test_x.py*"}  # whole paths only
    wildcards = ["defaults:tests/a/*", "defaults:tests/c/*", "d::large:tests/a/test_[xy].py::t*"]
    projected = {"defaults:tests/b/test_x.py*", "d::large:tests/b/test_x.py::t*"}  # node kept
    assert carried(wildcards, moves) == projected
    held = dict(sizes, large=[dict(pattern="tests/a/test_x.py::t*", owner="tests")])
    moved = dict(sizes, large=[dict(pattern="tests/b/test_x.py::t*", owner="tests")])
    assert shrink_only_growth(path, held, moved)  # a rename without its move is growth
    assert not shrink_only_growth(path, held, moved, lambda v: next(iter(carried([v], moves)), v))


def _git_backend(texts, paths, state=None):
    def run(command, **kwargs):
        if Path(command[0]).stem.lower() == "ast-grep":
            return _REAL_RUN(command, **kwargs)
        action = command[1]
        output = "base" if action == "rev-list" else "1\t0\tsrc/example.py\n"
        if "-z" in command:
            output = (state or {}).get("raw", "") if "--raw" in command else "\0".join(paths) + "\0"
        if action == "cat-file":
            rows = [
                texts.get(
                    spec,
                    '{"counts":{}}' if spec.endswith(":" + shape.ANTIPATTERN_BASELINE) else None,
                )
                for spec in kwargs["input"].decode().splitlines()
            ]
            output = b"".join(
                f"{'0' * 40} blob {len(row.encode())}\n".encode() + row.encode() + b"\n"
                if row is not None
                else b"x missing\n"
                for row in rows
            )
        code = int(action == "rev-parse" and not (state or {}).get("merge"))
        return shape.subprocess.CompletedProcess(command, code, output)

    return run


def _mock_ruff(monkeypatch):
    monkeypatch.setattr(shape, "ruff_test_findings", lambda snapshots: dict.fromkeys(snapshots, ()))
    monkeypatch.setattr(shape, "ruff_antipatterns", lambda snapshots: {})
    monkeypatch.setattr(shape, "test_data_antipatterns", lambda paths: {})


def test_the_shape_check_refuses_what_a_change_adds_and_spares_what_it_keeps(
    monkeypatch, tmp_path
) -> None:
    """Each rule fires on added content; kept lines, shrinking files and tests/structural pass."""
    _antipattern_cases(monkeypatch)
    _split_antipattern_cases(monkeypatch, tmp_path)
    for path in ("scripts/example.py", "tests/structural/test_example.py"):
        for flag in ("DETACHED_" + "PROCESS", "CREATE_NEW_" + "CONSOLE"):
            for source in (
                f"import subprocess\nx = subprocess.{flag}\n",
                f"import subprocess\nx = getattr(subprocess, {flag!r}, 0)\n",
                f"from subprocess import {flag} as flag\n",
            ):
                assert problems(path, None, source), (path, source)
        assert not problems(
            path, None, "import subprocess\nx = getattr(subprocess, 'CREATE_NO_WINDOW', 0)\n"
        )
    assert not problems(
        "src/example.py", None, "import subprocess\nx = subprocess." + "DETACHED_" + "PROCESS\n"
    )
    big = "x = 1\n" * 20_000
    assert not problems(PATH, big, big + "\u754c")  # a grown test file is a size signal only
    assert not problems(PATH, big + "\u754c", big)
    source = "import inspect\ninspect.getsource(owner)\n"
    assert problems(PATH, "", source) and not problems(PATH, source, source)
    assert not problems("tests/structural/test_example.py", "", source)
    pin = "assert.ok(html.includes('The update waits for the provider'));\n"
    cjs = "tests/owner/workbench_example.cjs"
    assert problems(cjs, "", pin) and not problems(cjs, pin, pin)
    assert not problems(cjs, "", "assert.ok(html.includes('data-key=\"wait\"'));\n")
    assert problems(PATH, "", 'assert "the update waits for the provider" in text\n')
    assert problems(PATH, "", "time.sleep" + "(2)\n") and not problems(
        PATH, "", "time.sleep(0.05)\n"
    )
    long = 'def test_a():\n    """One.\n\n    Two.\n    Three.\n    """\n'
    assert problems(PATH, None, long) and not problems(PATH, long, long)
    missing = "def test_reason():\n    pass\n"
    assert problems(PATH, None, missing) and not problems(PATH, missing, missing)
    reason = 'def test_reason():\n    """Missing authority must refuse publication."""\n    pass\n'
    assert not problems(PATH, None, reason)
    marked = "import pytest\npytestmark = pytest.mark.proves('x.y')\n" + missing
    held = "@pytest.mark.proves('x.y')\nclass TestA:\n    " + missing.replace("\n", "\n    ")
    assert not problems(PATH, None, marked) and not problems(PATH, None, "import pytest\n" + held)
    debt = "class A:\n    " + missing.replace("\n", "\n    ")
    assert problems(PATH, debt, debt.replace("class A", "class B"))
    # an added or changed test names each case pytest would name by its position (rule 5); a
    # case of literal values names itself, as in the harness's lint
    site = "@pytest.mark.parametrize({}, {})\ndef test_a(flag):\n    'One.'\n"
    for names, cases, refused in (
        ("'flag'", "['on', 'off']", False),  # a literal str is its own id
        ("'flag'", "[1, -2, 0.5, b'x', True, None]", False),
        ("'flag'", "[object(), 'off']", True),  # an object is named flag0
        ("'flag'", "[[1], {'a': 1}]", True),  # so are a list and a dict
        ("'flag'", "CASES", True),  # a computed list cannot be read
        ("'a, b'", "[('x', 1), ('y', None)]", False),  # several literal values
        ("('a', 'b')", "[('x', [1])]", True),
        ("'flag'", "[object(), None], ids=['on', 'off']", False),
        ("'flag'", "[object(), None], ids=['on', None]", True),  # None asks for the value id
        ("'flag'", "[pytest.param(object(), id='on'), pytest.param(1)]", False),
        ("'flag'", "[pytest.param(object(), id=None)]", True),
    ):
        assert bool(problems(PATH, None, site.format(names, cases))) is refused, cases
    unnamed = site.format("'flag'", "[object(), 'off']")
    assert problems(PATH, None, unnamed) == [
        f"{PATH}::test_a: line 1 parametrize('flag') has a case pytest names by its position; "
        "give each such case pytest.param(..., id=...), or ids= (rule 5)"
    ]
    assert not problems(PATH, unnamed, unnamed)  # a kept test keeps its standing
    assert problems(PATH, unnamed, unnamed.replace("One.", "Two."))  # a changed one is judged
    raises = 'def test_a():\n    """One."""\n    with pytest.raises(ValueError):\n        f()\n'
    old = "import pytest\n" + raises
    twice = "@pytest.mark.parametrize('a', [1, 1], ids=['x', 'y'])\n" + raises
    snapshots = {
        "old": old,
        "new": old + raises.replace("test_a", "test_b") + twice.replace("test_a", "test_c"),
        "moved": old.replace("import pytest\n", "import pytest\n\n\n"),
    }
    found = shape.ruff_test_findings(snapshots)
    assert found["old"] == (("PT011", 4),)
    assert problems(PATH, old, snapshots["new"], (found["old"], found["new"])) == [
        f"{PATH}::test_b: Ruff PT011 findings grow 0 -> 1; "
        "give pytest.raises a match= for the expected message",
        f"{PATH}::test_c: Ruff PT011 findings grow 0 -> 1; "
        "give pytest.raises a match= for the expected message",
        f"{PATH}::test_c: Ruff PT014 findings grow 0 -> 1; drop the duplicate parametrize case",
    ]
    assert not problems(PATH, old, snapshots["moved"], (found["old"], found["moved"]))


@pytest.mark.proves("testing.only_a_requirement_marker_may_join_an_oversized_test_file")
def test_an_oversized_test_file_takes_a_marker_line_and_nothing_else(capsys) -> None:
    """A whole proves line is free; a comment on it, or any other line, still counts as growth,
    named as a review signal and never refused (a size cap is met by splitting files)."""
    path = "tests/example/test_big.py"
    big = "# " + "x" * shape.CAP_BYTES + "\ndef test_a():\n    pass\n"

    def refused(new: str) -> bool:
        assert not problems(path, big, new)
        return "KB (a review signal, never refused)" in capsys.readouterr().out

    assert not refused(big.replace("def test_a", '@pytest.mark.proves("a.b")\ndef test_a'))
    assert not refused(big.replace("def test_a", '@pytest.mark.proves("a.b")\r\ndef test_a'))
    assert refused(big.replace("def test_a", '@pytest.mark.proves("a.b")  # why\ndef test_a'))
    assert refused(big.replace("def test_a", "@pytest.mark.slow\ndef test_a"))
    assert refused(big + "# one more line\n")
    plain = "def test_a():\n    pass\n"
    mark = '@pytest.mark.proves("a.b")\n'
    marked = mark + plain
    digest = [file_holders("t.py", text, size_scope=True)[0].digest for text in (plain, marked)]
    assert digest[0] == digest[1]  # a Large test keeps its lanes when it gains a marker
    edited = marked.replace("pass", "assert True")
    assert file_holders("t.py", edited, size_scope=True)[0].digest != digest[0]
    fixture = 'def test_a():\n    source = """\n' + mark + '"""\n'
    changed_fixture = fixture.replace('"a.b"', '"a.c"')
    assert (
        file_holders("t.py", fixture, size_scope=True)[0].digest
        != file_holders("t.py", changed_fixture, size_scope=True)[0].digest
    )


@pytest.mark.proves("testing.size_counts_are_review_signals")
def test_a_grown_source_is_a_size_signal_and_a_long_function_is_left_to_ruff(capsys) -> None:
    """A size cap is met by splitting, which leaves the code no better: a grown source names a
    review signal and is never refused, and a long or branching function is Ruff's C901 and
    PLR0915 signal, never this check's refusal."""
    path = "src/example.py"
    big = "x" * shape.CAP_BYTES
    wide = big + "\u754c"
    assert not problems("src/example.txt", big, wide)
    assert "test_shape.signal: src/example.txt" in capsys.readouterr().out
    assert not problems("src/example.txt", wide, big)
    assert "test_shape.signal" not in capsys.readouterr().out
    long = "def run():\n" + "    pass\n" * 130
    branches = "def branch():\n" + "    if ready:\n        pass\n" * 25
    assert not problems(path, None, long) and not problems(path, "def branch(): pass\n", branches)
    for generated in shape.GENERATED | shape.WORD_CATALOGS:
        assert not problems(generated, big, wide)
    assert "test_shape.signal" not in capsys.readouterr().out
    assert not problems(shape.ASSETS + "workbench-source/js/app/new.js", big, wide)
    assert "test_shape.signal" in capsys.readouterr().out


def test_performance_lints_resolve_aliases_and_spare_queries_batches_and_migrations() -> None:
    """Static controls and resolved row writes grow only at their actual owning seams."""
    path = "src/example.py"
    controls = "from threadpoolctl import threadpool_limits as cap\ncap(1)\n"
    assert problems(path, None, controls)
    assert not problems(shape.CONTROL_OWNER, None, controls)
    assert not problems(path, controls, "# moved\n" + controls)
    assert problems(path, controls, controls + "cap(2)\n")
    header = "from duckdb import connect as open_db, DuckDBPyConnection as DB\n"
    header += "def store() -> DB:\n    return open_db(':memory:')\n"
    writer = header + "def write():\n    con = store()\n    cursor = con.cursor()\n"
    write = (
        writer
        + "    for row in rows:\n        cursor.execute('INSERT INTO records VALUES (?)', row)\n"
    )
    assert problems(path, None, write)
    for clean in (
        write.replace("INSERT INTO records VALUES (?)", "SELECT ?"),
        writer + "    cursor.executemany('INSERT INTO records VALUES (?)', rows)\n",
        "for row in rows:\n    logger.execute('INSERT INTO records VALUES (?)', row)\n",
    ):
        assert not problems(path, None, clean)
    for sql in (
        "INSERT INTO feature_ineligibility VALUES (?)",
        'DELETE FROM main."feature_ineligibility"',
    ):
        assert problems(path, None, f"connection.execute({sql!r})\n")
    for sql in (
        "INSERT INTO feature_ineligibility_run VALUES (?)",
        "DELETE FROM feature_ineligibility_legacy",
        "CREATE OR REPLACE VIEW feature_ineligibility AS SELECT 1",
        "ALTER TABLE feature_ineligibility RENAME TO feature_ineligibility_legacy",
    ):
        assert not problems(path, None, f"connection.execute({sql!r})\n")


def test_exact_copy_guard_names_original_and_counts_only_added_executable_runs(
    monkeypatch, tmp_path
) -> None:
    """Added eight-line code copies are refused while debt, comments and literal changes pass."""
    stale = tmp_path / "stale" / "scripts/test-shape-tools"
    (stale / "node_modules/jscpd").mkdir(parents=True)
    source_package = shape.ROOT / "scripts/test-shape-tools"
    for name in ("package.json", "package-lock.json"):
        (stale / name).write_bytes((source_package / name).read_bytes())
    (stale / "node_modules/jscpd/run-jscpd.js").write_text("stale tool")
    with pytest.raises(ValueError, match=r"jscpd\."):
        shape.jscpd_runner(tmp_path / "stale", tmp_path / "owner/.venv/Scripts/python.exe")
    body = "".join(f"    value{index} = {index}\n" for index in range(8))
    original, copy = "src/original.py", "src/copy.py"
    source, duplicate = "def original():\n" + body, "def copy():\n" + body
    corpus = {original: source, copy: duplicate}
    found = shape.clone_problems(copy, None, duplicate, corpus)
    assert found and f"{original}:2" in found[0]
    assert not shape.clone_problems(copy, duplicate, duplicate, corpus)
    assert not shape.clone_problems(
        copy, None, "def copy():\n" + body.rsplit("\n", 2)[0] + "\n", corpus
    )
    assert not shape.clone_problems(copy, None, '"""' + body + '"""\n', corpus)
    js = "".join(f"const value{index} = 'two  words';\n" for index in range(8))
    workbench = shape.ASSETS + "workbench-source/js/app/new.js"
    assert shape.clone_problems(workbench, None, js, {"src/original.js": "// note\n" + js})
    assert not shape.clone_problems(
        workbench, None, js.replace("two  words", "two words"), {"src/original.js": js}
    )
    same_file = source + "def copy():\n" + body
    assert shape.clone_problems(original, source, same_file, {original: same_file})
    assert shape.clone_problems(workbench, None, body, {original: source})
    real_run, commands = shape.subprocess.run, []

    def run(command, **kwargs):
        commands.append(command)
        return real_run(command, **kwargs)

    monkeypatch.setattr(shape.subprocess, "run", run)
    result = shape.clone_problems_batch(
        [(copy, None, duplicate), (copy, duplicate, duplicate)], corpus
    )
    assert result[0] and not result[1]
    assert sum("--min-lines" in command for command in commands) == 1
    runner = getattr(shape, "jscpd_runner", None)
    assert callable(runner), "a fresh gate checkout must bind the matching locked tool"
    staged = tmp_path / "staged"
    package = staged / "scripts/test-shape-tools"
    package.mkdir(parents=True)
    for name in ("package.json", "package-lock.json"):
        (package / name).write_bytes((source_package / name).read_bytes())
    executable = Path(sys.executable)
    installed = source_package / "node_modules/jscpd/run-jscpd.js"
    assert runner(staged, executable) == installed
    with monkeypatch.context() as environment:
        environment.setenv("NODE_OPTIONS", "--require=missing-inherited-module")
        environment.setenv("NODE_PATH", str(tmp_path / "untrusted"))
        assert runner(staged, executable) == installed
    with monkeypatch.context() as environment:
        environment.delenv("npm_config_cache", raising=False)
        environment.setenv("NPM_CONFIG_CACHE", str(tmp_path / "empty-cache"))
        with pytest.raises(ValueError, match=r"jscpd\.integrity_unavailable"):
            runner(staged, executable)
    (package / "package-lock.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match=r"jscpd\.lock_unreadable"):
        runner(staged, executable)
    (package / "package-lock.json").write_bytes((source_package / "package-lock.json").read_bytes())
    with pytest.raises(ValueError, match=r"jscpd\.unavailable"):
        runner(staged, tmp_path / "other-python")
    import shutil

    shutil.copytree(source_package / "node_modules", package / "node_modules")
    assert runner(staged, executable) == package / "node_modules/jscpd/run-jscpd.js"
    with monkeypatch.context() as inaccessible:

        def denied_walk(_path, *, followlinks=False, onerror=None):
            if onerror is not None:
                onerror(PermissionError("fixture unreadable"))
            return iter(())

        inaccessible.setattr(shape.os, "walk", denied_walk)
        with pytest.raises(ValueError, match=r"jscpd\.integrity_unavailable.*fixture unreadable"):
            runner(staged, executable)
    locked = json.loads((package / "package-lock.json").read_bytes())
    locked["packages"]["node_modules/jscpd"]["integrity"] = "sha512-" + "A" * 86 + "=="
    with pytest.raises(ValueError, match=r"jscpd\.integrity_unavailable"):
        runner(
            staged,
            executable,
            ((package / "package.json").read_bytes(), json.dumps(locked).encode()),
        )
    binary = next((package / "node_modules").glob("jscpd-*/bin/jscpd*"))
    original_binary = binary.read_bytes()
    binary.write_bytes(b"tampered native executable")
    with pytest.raises(ValueError, match=r"jscpd\.integrity_unavailable"):
        runner(staged, executable)
    binary.write_bytes(original_binary)
    (package / "node_modules/jscpd/platform-map.js").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match=r"jscpd\.integrity_unavailable"):
        runner(staged, executable)
    monkeypatch.setattr(shape.shutil, "which", lambda _command: None)
    with pytest.raises(ValueError, match="jscpd unavailable"):
        shape.clone_problems(copy, None, duplicate, corpus)


def test_staged_guard_reads_index_blobs_and_intersects_both_merge_parents(monkeypatch) -> None:
    """Only staged content is checked, and inherited parent violations do not block a merge."""
    _mock_ruff(monkeypatch)
    path = "src/example.py"
    bad = "import threadpoolctl as pools\npools.threadpool_limits(1)\n"
    state = {"merge": False}
    texts = {":" + path: bad, "HEAD:" + path: "", "MERGE_HEAD:" + path: ""}
    monkeypatch.setattr(shape.subprocess, "run", _git_backend(texts, [path], state))
    assert shape.main([]) == 1
    for guarded in ("scripts/example.py", "tests/structural/example.py"):
        bad_flag = "import subprocess\nx = subprocess." + "DETACHED_" + "PROCESS\n"
        blobs = {":" + guarded: bad_flag, "HEAD:" + guarded: ""}
        monkeypatch.setattr(shape.subprocess, "run", _git_backend(blobs, [guarded]))
        assert shape.main([]) == 1
        blobs[":" + guarded] = "import subprocess\nx = subprocess.CREATE_NO_WINDOW\n"
        assert shape.main([]) == 0
    monkeypatch.setattr(shape.subprocess, "run", _git_backend(texts, [path], state))
    state["merge"] = True
    texts["HEAD:" + path] = bad
    assert shape.main([]) == 0
    asset = "src/image.png"
    texts = {":" + asset: "\0" * shape.CAP_BYTES, "HEAD:" + asset: ""}
    monkeypatch.setattr(shape.subprocess, "run", _git_backend(texts, [asset]))
    assert shape.main([]) == 0
    texts[":" + asset] += "\0"
    assert shape.main([]) == 0  # a grown asset is a size signal, never refused
    old, new, legacy = (
        "tests/area/test_kept.py",
        "tests/unit/test_kept.py",
        "def test_kept(): pass\n",
    )
    rows = [(f"{'a' * 40} {'0' * 40}", "D", old), (f"{'0' * 40} {'a' * 40}", "A", new)]
    raw = "".join(f":100644 100644 {blobs} {kind}\0{path}\0" for blobs, kind, path in rows)
    texts = {":" + new: legacy, "HEAD:" + old: legacy}  # a byte-identical move keeps its old debt
    for blob, code in (("a", 0), ("b", 1)):  # an edited move is new: its missing docstring refuses
        state = {"raw": raw.replace(f"{'a' * 40} A", f"{blob * 40} A")}
        monkeypatch.setattr(shape.subprocess, "run", _git_backend(texts, [new], state))
        assert shape.main([]) == code

    table = "config/registries/test-layout.json"
    edited = (
        raw.replace(f"{'a' * 40} A", f"{'b' * 40} A")
        + f":100644 100644 {'c' * 40} {'d' * 40} M\0{table}\0"
    )
    for row_old, added, code in (
        (old, "", 0),  # the move's table pairs an edited move: it is an edit of its old file
        (old, "def test_added(): pass\n", 1),  # whose added lines stay judged
        ("tests/area/test_elsewhere.py", "", 1),  # a row names only a path this change deletes
    ):
        texts[":" + table] = json.dumps(
            {"schema": "alphalattice.test-layout.v1", "files": {new: {"old": row_old}}}
        )
        texts[":" + new] = legacy + added
        monkeypatch.setattr(
            shape.subprocess, "run", _git_backend(texts, [new, table], {"raw": edited})
        )
        assert shape.main([]) == code
    texts[":" + new] = legacy

    data, sizes = "tests/integration/testing/panel.csv", "config/registries/test-sizes.json"
    policy = json.loads((shape.ROOT / sizes).read_text("utf-8"))
    trees, moves = {"HEAD": ""}, {}
    texts = {":" + sizes: json.dumps(policy)}

    def holding(command, **kwargs):
        if command[1] == "ls-tree":
            return shape.subprocess.CompletedProcess(command, 0, trees[command[5]] + "\0")
        return _git_backend(texts, [data, "tests/unit/testing/notes.md"], moves)(command, **kwargs)

    monkeypatch.setattr(shape.subprocess, "run", holding)
    assert shape.main([]) == 1  # a new data file names no reason
    trees["HEAD"] = data
    assert shape.main([]) == 0  # the tree is the baseline
    trees["HEAD"] = ""
    row = {"path": data, "reason": "One small panel a test reads in place."}
    texts[":" + sizes] = json.dumps({**policy, "data_files": [row]})
    assert shape.main([]) == 0  # registered with its reason
    for bad in ({**row, "reason": " "}, {**row, "path": "tests/*.csv"}, {**row, "path": "x"}):
        texts[":" + sizes] = json.dumps({**policy, "data_files": [bad]})
        assert shape.main([]) == 1
    texts[":" + sizes] = json.dumps({**policy, "data_files": [row, row]})
    assert shape.main([]) == 1  # a row names one file once
    texts[":" + sizes] = json.dumps(policy)
    moved = raw.replace(old, "tests/integration/testing/old.csv").replace(new, data)
    for blob, code in (("a", 0), ("b", 1)):  # a byte-identical move keeps its standing
        moves["raw"] = moved.replace(f"{'a' * 40} A", f"{blob * 40} A")
        assert shape.main([]) == code

    source = "src/devtools/testing/layout.py"
    wrong, right = "tests/unit/testing/test_wrong.py", "tests/unit/testing/test_layout.py"
    test = (
        f"TEST_SOURCE = {source!r}\nimport pytest\n"
        "@pytest.mark.proves('testing.example')\ndef test_one(): pass\n"
    )
    registry = json.dumps(
        {
            "schema": "alphalattice.requirements.v1",
            "entries": {
                "testing.example": {
                    "description": "One test checks the assigned source.",
                    "owner": "devtools.testing",
                    "layer": "unit",
                    "seconds": 1,
                }
            },
        }
    )
    texts = {":" + wrong: test, ":config/registries/requirements.json": registry}
    paths = [wrong]
    placement_state = {"merge": False}

    def staged(command, **kwargs):
        if command[1] == "ls-files" and "src" in command:
            return shape.subprocess.CompletedProcess(command, 0, source + "\0")
        return _git_backend(texts, paths, placement_state)(command, **kwargs)

    monkeypatch.setattr(shape.subprocess, "run", staged)
    assert shape.main([]) == 1  # the added file must use the declared source's mapped path
    suffix = "tests/unit/testing/layout_test.py"
    paths[:] = [suffix]
    texts[":" + suffix] = test
    assert shape.main([]) == 1  # pytest also collects *_test.py
    paths[:] = [right]
    texts[":" + right] = test
    assert shape.main([]) == 0
    old_source = source
    source = "scripts/release/check_example.py"
    script_test = "tests/unit/testing/release/test_check_example.py"
    paths[:] = [script_test]
    texts[":" + script_test] = test.replace(old_source, source)
    assert shape.main([]) == 0  # the staged source listing includes scripts
    source = old_source
    paths[:] = [wrong]
    texts["HEAD:" + wrong] = test
    assert shape.main([]) == 0  # existing files are not moved by this rule
    del texts["HEAD:" + wrong]
    texts["MERGE_HEAD:" + wrong] = test
    placement_state["merge"] = True
    assert shape.main([]) == 0  # a merge does not classify a parent-owned file as new
    del texts["MERGE_HEAD:" + wrong]
    del texts[":config/registries/requirements.json"]
    assert shape.main([]) == 1  # missing staged ownership must not be treated as empty


@pytest.mark.proves("testing.the_stage_judges_the_committed_change_as_the_hook_judges_the_index")
def test_the_stage_judges_the_committed_head_against_its_merge_base(monkeypatch) -> None:
    """With a base, the committed head is judged against its merge base, never the index."""
    _mock_ruff(monkeypatch)
    path = "src/example.py"
    bad = "import threadpoolctl as pools\npools.threadpool_limits(1)\n"
    texts = {"HEAD:" + path: bad, ":" + path: "", "fork:" + path: ""}
    backend, asked = _git_backend(texts, [path]), []

    def run(command, **kwargs):
        asked.append(command[1:])
        if command[1] == "merge-base":
            known = command[2] == "develop"
            return shape.subprocess.CompletedProcess(command, int(not known), "fork\n" * known)
        return backend(command, **kwargs)

    monkeypatch.setattr(shape.subprocess, "run", run)
    assert shape.main(["--base", "develop"]) == 1  # the committed break, though the index is clean
    assert ["merge-base", "develop", "HEAD"] in asked
    assert shape.main([]) == 0  # the hook judges the clean index
    texts["fork:" + path] = bad
    assert shape.main(["--base", "develop"]) == 0  # inherited from the base: not this change's
    assert shape.main(["--base", "gone"]) == 1  # an unreadable base refuses, never passes


def test_weekly_report_lists_static_candidates_and_respects_config_string_readers(
    monkeypatch,
) -> None:
    """The public weekly seam reports review candidates and retained configuration readers."""
    texts = {
        "HEAD:src/example.py": (
            "def used(): return 1\ndef unused(): return 2\ndef configured(): return 3\n"
        ),
        "HEAD:tests/test_reader.py": "from example import used\nassert used() == 1\n",
        "HEAD:config/example.json": '{"callable": "configured"}\n',
    }
    monkeypatch.setattr(shape.subprocess, "run", _git_backend(texts, [spec[5:] for spec in texts]))
    report = shape.weekly_report()
    assert [row["name"] for row in report["candidates"]] == ["unused"]
    assert report["growth"] == [{"path": "src/example.py", "added": 1, "removed": 0, "net": 1}]
    assert report["debt"] == []
