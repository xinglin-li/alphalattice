"""The fast guard holds tests/README.md's writing rules on what a change adds."""

import check_test_shape as shape
from check_test_shape import problems

PATH = "tests/owner/test_example.py"


def _git_backend(texts, paths, state=None):
    def run(command, **kwargs):
        action = command[1]
        output = "base" if action == "rev-list" else "1\t0\tsrc/example.py\n"
        if "-z" in command:
            output = "\0".join(paths) + "\0"
        if action == "cat-file":
            rows = [texts[spec].encode() for spec in kwargs["input"].decode().splitlines()]
            output = b"".join(
                f"{'0' * 40} blob {len(row)}\n".encode() + row + b"\n" for row in rows
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


def test_exact_copy_guard_names_original_and_counts_only_added_executable_runs() -> None:
    """Added eight-line code copies are refused while debt, comments and literal changes pass."""
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


def test_staged_guard_reads_index_blobs_and_intersects_both_merge_parents(monkeypatch) -> None:
    """Only staged content is checked, and inherited parent violations do not block a merge."""
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
