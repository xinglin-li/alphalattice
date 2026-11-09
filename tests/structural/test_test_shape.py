"""The fast guard holds tests/README.md's writing rules on what a change adds."""

import json
import sys
from pathlib import Path

import pytest

import check_test_shape as shape
from check_playpen import shrink_only_growth
from check_test_shape import problems
from devtools.testing.requirements import file_holders

PATH = "tests/owner/test_example.py"


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
        action = command[1]
        output = "base" if action == "rev-list" else "1\t0\tsrc/example.py\n"
        if "-z" in command:
            output = (state or {}).get("raw", "") if "--raw" in command else "\0".join(paths) + "\0"
        if action == "cat-file":
            rows = [texts.get(spec) for spec in kwargs["input"].decode().splitlines()]
            output = b"".join(
                f"{'0' * 40} blob {len(row.encode())}\n".encode() + row.encode() + b"\n"
                if row is not None
                else b"x missing\n"
                for row in rows
            )
        code = int(action == "rev-parse" and not (state or {}).get("merge"))
        return shape.subprocess.CompletedProcess(command, code, output)

    return run


def test_the_shape_check_refuses_what_a_change_adds_and_spares_what_it_keeps() -> None:
    """Each rule fires on added content; kept lines, shrinking files and tests/structural pass."""
    big = "x = 1\n" * 20_000
    assert problems(PATH, big, big + "\u754c") and not problems(PATH, big + "\u754c", big)
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


@pytest.mark.proves("testing.only_a_requirement_marker_may_join_an_oversized_test_file")
def test_an_oversized_test_file_takes_a_marker_line_and_nothing_else() -> None:
    """A whole proves line is free; a comment on it, or any other line, still counts as growth."""
    path = "tests/example/test_big.py"
    big = "# " + "x" * shape.CAP_BYTES + "\ndef test_a():\n    pass\n"

    def refused(new: str) -> bool:
        return any("a test file stays under 100 KB" in line for line in problems(path, big, new))

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


def test_source_caps_hold_growth_by_qualified_function_and_spare_existing_debt() -> None:
    """Each size and function metric refuses its own growth, including nested definitions."""
    path = "src/example.py"
    big = "x" * shape.CAP_BYTES
    assert not problems("src/example.txt", None, big)
    assert problems("src/example.txt", big, big + "\u754c")
    assert not problems("src/example.txt", big + "\u754c", big)

    def method(name, lines):
        return f"class {name}:\n    def run(self):\n" + "        pass\n" * (lines - 1)

    old = method("A", 120) + method("B", 125)
    new = method("A", 121) + method("B", 124)
    found = problems(path, old, new)
    assert len(found) == 1 and "A.run" in found[0]

    def branch(count):
        return "def branch():\n" + "    if ready:\n        pass\n" * count

    assert problems(path, branch(20), branch(21))
    assert not problems(path, branch(21), branch(20))
    nested = "    def inner():\n" + "        if ready:\n            pass\n" * 21
    assert not problems(path, branch(19) + nested, branch(20) + nested)
    snapshots = {
        "before": branch(20),
        "after": branch(21),
        "nested": branch(20) + nested,
        "mixed": "def shape(a, b):\n    if a and b:\n        return 1 if a else 2\n",
        "loop_else": "def shape(a):\n    for x in a:\n        pass\n    else:\n        pass\n",
        "async_loop": "async def shape(a):\n    async for x in a:\n        pass\n",
    }
    measured = shape.ruff_decisions(snapshots)
    for before, after in (
        ("before", "after"),
        ("after", "before"),
        ("before", "nested"),
        ("before", "mixed"),
        ("before", "loop_else"),
        ("before", "async_loop"),
    ):
        assert problems(
            path, snapshots[before], snapshots[after], (measured[before], measured[after])
        ) == problems(path, snapshots[before], snapshots[after])
    for generated in shape.GENERATED | shape.WORD_CATALOGS:
        assert not problems(generated, big, big + "\u754c")
    assert problems(shape.ASSETS + "workbench-source/js/app/new.js", big, big + "\u754c")


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
    monkeypatch.setattr(shape, "ruff_decisions", lambda snapshots: {key: {} for key in snapshots})
    path = "src/example.py"
    bad = "import threadpoolctl as pools\npools.threadpool_limits(1)\n"
    state = {"merge": False}
    texts = {":" + path: bad, "HEAD:" + path: "", "MERGE_HEAD:" + path: ""}
    monkeypatch.setattr(shape.subprocess, "run", _git_backend(texts, [path], state))
    assert shape.main([]) == 1
    state["merge"] = True
    texts["HEAD:" + path] = bad
    assert shape.main([]) == 0
    asset = "src/image.png"
    texts = {":" + asset: "\0" * shape.CAP_BYTES, "HEAD:" + asset: ""}
    monkeypatch.setattr(shape.subprocess, "run", _git_backend(texts, [asset]))
    assert shape.main([]) == 0
    texts[":" + asset] += "\0"
    assert shape.main([]) == 1
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
