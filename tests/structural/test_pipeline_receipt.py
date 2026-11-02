import contextlib
import json
from types import SimpleNamespace

import pytest
from scripts import pipeline as cli

from devtools.pipeline import receipt

HEAD = "a" * 40
RESULT = {"node": "t::alpha[x::y]", "attempt": 1, "solo": False, "status": "PASSED", "seconds": 1}
CONFIGURATION = {"definition": "d", "definition_sha256": "d" * 64, "rows": ["tests"]}
CONFIGURATION |= {"providers": {"tests": "product"}, "workers": 1, "bound_seconds": 900}
TAIL = '{"id": {"metrics": {}}, "metrics": {}}'
TWIN = TAIL.replace("metrics", "workspace")
END = '{"id": {"finished": {"n": 1}}, "last_message": true, "finished": {"result": "FAIL"}}'
FIRST = {**RESULT, "status": "FAILED", "timeout_seconds": 1}
ALONE = {**FIRST, "attempt": 2, "solo": True, "status": "PASSED"}
RERUNS = {  # a case's node results in file order, its red nodes and the classes it writes
    "a red node its solo rerun passed": ([ALONE, FIRST], [], ["flake"]),  # workers finish unordered
    "a solo rerun past its time": ([FIRST, {**ALONE, "seconds": 2}], [RESULT["node"]], ["slow"]),
    "a node still red alone": ([FIRST, {**ALONE, "status": "FAILED"}], [RESULT["node"]], []),
    "required skipped": ([RESULT | {"status": "SKIPPED"}], [], []),
    "required over": ([RESULT | {"timeout_seconds": 0.5}], [], []),
}
RERUNS |= {"required missing": ([ALONE], [], []), "required empty": ([], [], [])}
SIBLING = ALONE | {"node": RESULT["node"] + "[z]"}
RERUNS |= {
    "exact missing": ([SIBLING], [], []),
    "exact skip": ([ALONE, SIBLING | {"status": "SKIPPED"}], [], []),
}
NEEDED = {"required_tests": ["t::alpha"], "within_limits": True}
CASES = {  # line i's (old, new): ("", new) puts a line before it, (old, None) drops it
    "complete": ({}, {"stage": "commit", "head": HEAD, "passing": True}, None),
    "a red node its solo rerun passed": ({}, NEEDED | {"passing": True}, None),
    "a solo rerun past its time": ({}, {}, None),
    "a node still red alone": ({}, {}, None),
    "a host-skipped step": (
        {3: ('"PASS", "exit": 0', '"SKIPPED_BY_HOST", "exit": null')},
        {},
        None,
    ),
    "a second started": (
        {1: ("", '{"id": {"started": {"n": 1}}, "started": {}}')},
        {},
        "receipt_malformed",
    ),
    "a second metrics": ({5: ("", TAIL), 6: ("", TAIL)}, {}, "receipt_malformed"),
    "a card that is no mapping": (
        {6: ("", TAIL.replace("{}}", '{"card": []}}'))},
        {},
        "receipt_malformed",
    ),
    "infinite seconds": ({3: ('"seconds": 1,', '"seconds": Infinity,')}, {}, "receipt_malformed"),
    "a child missing": ({1: ("-", None)}, {}, "receipt_incomplete"),
    "a tail after finished": ({7: ("", TAIL)}, {}, "receipt_incomplete"),
    "another head": ({}, {"head": "b" * 40}, "receipt_mismatch"),
    "not a receipt": ({0: ("-", None)}, {}, "receipt_unknown"),
    "not JSON": ({0: ("{", "[")}, {}, "receipt_unreadable"),
    "no file": (None, {}, "receipt_missing"),
    "a JSON list": ({0: ("", "[1, 2]")}, {}, "receipt_malformed"),
    "an event with no id": ({1: ("", '{"workspace": {}}')}, {}, "receipt_malformed"),
    "a repeated workspace": ({2: ("", TWIN)}, {}, "receipt_malformed"),
    "a short head": ({1: (HEAD, "a" * 9)}, {}, "receipt_malformed"),
    "no command": ({0: ('["x"]', "[]")}, {}, "receipt_malformed"),
    "PASS over a failed step": ({3: ('"PASS"', '"FAIL"')}, {}, "receipt_inconsistent"),
    "PASS over a red node": ({4: ("PASSED", "FAILED")}, {}, "receipt_inconsistent"),
    "PASS over exit 1": ({6: ('"exit": 0', '"exit": 1')}, {}, "receipt_inconsistent"),
    "a step PASS with exit 1": ({3: ('"exit": 0', '"exit": 1')}, {}, "receipt_inconsistent"),
    "a second finished": ({6: ("", END)}, {}, "receipt_malformed"),
    "no tool version": ({0: ("tool_version", "tool")}, {}, "receipt_malformed"),
    "a short digest": ({2: ("d" * 64, "d")}, {}, "receipt_malformed"),
    "a list payload": ({6: ("", TAIL.replace("{}}", "[1]}"))}, {}, "receipt_malformed"),
    "children that are no list": (
        {6: ("", TAIL.replace("{}}", '{}, "children": 1}'))},
        {},
        "receipt_malformed",
    ),
    "required missing": ({}, NEEDED | {"required_tests": ["t::alpha", "t::a"]}, "receipt_mismatch"),
    "required skipped": ({}, NEEDED, "receipt_mismatch"),
    "required unbounded": ({}, NEEDED, "receipt_mismatch"),
    "required over": ({}, NEEDED, "receipt_mismatch"),
    "required empty": ({}, {"within_limits": True}, "receipt_mismatch"),
}
CASES |= {
    name: ({}, NEEDED | {"required_tests": [RESULT["node"]]}, code)
    for name, code in (("exact missing", "receipt_mismatch"), ("exact skip", None))
}


for _case in (
    "cache hit",
    "cache orphan",
    "cache shadow",
    "cache mismatch",
    "cache mismatch unexplained",
    "cache wrong mode",
):
    CASES[_case] = (
        {},
        {},
        "receipt_inconsistent"
        if _case in {"cache orphan", "cache mismatch unexplained", "cache wrong mode"}
        else None,
    )


@pytest.mark.proves("pipeline.receipt_verifies_only_complete_and_as_asked")
@pytest.mark.parametrize("case", CASES)
def test_a_receipt_verifies_only_complete_and_as_asked(tmp_path, capsys, case):
    """A receipt counts only whole, with its claim standing over its events, and as asked."""
    spoil, asked, code = CASES[case]
    events, path = tmp_path / "events.jsonl", tmp_path / "receipt.jsonl"
    results, red, classes = RERUNS.get(case, ([RESULT], [], []))
    if case.startswith("cache "):
        origin = dict(key="a" * 64, digest="d" * 64, head=HEAD)
        mode = "shadow" if case == "cache shadow" or case.startswith("cache mismatch") else "on"
        results = [RESULT | ({"cached_from": origin} if mode == "on" else {})]
        if case != "cache orphan":
            decision = dict(
                group="f" * 64,
                mode=mode,
                decision="HIT",
                entry=origin,
                reason=None,
                executed=mode != "on",
                expected={
                    RESULT["node"]: "SKIPPED" if case.startswith("cache mismatch") else "PASSED"
                },
            )
            if case == "cache mismatch":
                decision["reason"] = "result_cache.not_passing"
            if case == "cache wrong mode":
                decision["mode"] = "record"
            results = [{"cache": decision}, *results]
    events.write_text("\n".join(map(json.dumps, results)))
    started = {"stage": "commit", "host": "workshop", "command": ["x"]}
    writer = receipt.Writer(path, started, [{"workspace": {}}, {"configuration": {}}])
    writer.write({"workspace": {}}, {"repo": "r", "base": "b" * 40, "head": HEAD, "clean": True})
    writer.write({"configuration": {}}, CONFIGURATION)
    step = {"status": "PASS", "exit": 0, "seconds": 1, "provider": "product"}
    assert writer.tests("tests", step, events) == red  # by its highest attempt, and classed
    writer.finish({"result": "FAIL" if red else "PASS", "exit": int(bool(red)), "red_nodes": red})
    s = path.read_text("utf-8").splitlines()
    assert [c for c in ("flake", "slow") if f'"class": "{c}"' in "".join(s)] == classes
    for i, (old, new) in sorted((spoil or {}).items(), reverse=True):
        s[i : i + bool(old)] = [] if new is None else [s[i].replace(old, new) if old else new]
    _ = path.unlink() if spoil is None else path.write_text("\n".join(s))
    if asked.get("required_tests"):  # CLI and the unchanged default summary
        names = asked["required_tests"]
        argv = ["verify", str(path), "--passing", "--head", HEAD, "--within-limits"]
        argv += ["--required-tests", *names]
        try:
            status = cli.main(argv)
        except SystemExit as error:
            status = error.code
        out = capsys.readouterr()
        assert status == (2 if code else 0) and (not code or code in out.err)
        assert code or json.loads(out.out) == receipt.verify(path)
    with pytest.raises(receipt.ReceiptError) if code else contextlib.nullcontext() as refused:
        try:
            found = receipt.verify(path, **asked)
        except receipt.ReceiptError as error:
            assert code is not None, f"unexpected receipt refusal: {error.code}"
            raise
        assert found["head"] == HEAD
    assert not code or refused.value.code == "pipeline." + code


@pytest.mark.proves("pipeline.receipt_plugin_writes_each_node_once")
def test_the_plugin_writes_each_node_once_with_its_failure_and_disposes_of_itself(tmp_path):
    """A node's result follows its teardown; a failure's signature ignores paths and numbers."""
    config = SimpleNamespace(pluginmanager=pytest.PytestPluginManager())
    receipt.register(config, tmp_path / "events.jsonl")
    plugin = config.pluginmanager.get_plugin("pipeline-events")
    for node, line in (
        ("t::a", r"AssertionError: C:\x\a.py:12 holds 3 rows at abcdef1"),
        ("t::b", "AssertionError: /y/b.py:9 holds 70 rows at 0123fed"),
    ):
        for when, outcome in (("setup", "passed"), ("call", "failed"), ("teardown", "passed")):
            report = pytest.TestReport(node, ("t", 1, node), {}, outcome, line, when, duration=0.5)
            report.user_properties = [("test_size", "small")]
            plugin.pytest_runtest_logreport(report)
    for node, when in (("t::c", "setup"), ("t::c", "???"), ("t::d", "setup")):  # crash; open
        report = pytest.TestReport(node, ("t", 1, node), {}, "passed", "gw1 crashed", when)
        report.outcome = "failed" if when == "???" else "passed"
        plugin.pytest_runtest_logreport(report)
    receipt.pytest_unconfigure(config)
    assert config.pluginmanager.get_plugin("pipeline-events") is None and plugin.stream.closed
    a, b, c, d = map(json.loads, (tmp_path / "events.jsonl").read_text("utf-8").splitlines())
    assert (a["status"], a["phase"], a["seconds"], a["size"]) == ("FAILED", "call", 1.5, "small")
    assert a["failure"]["signature"] == b["failure"]["signature"] != ""
    assert c["status"] == d["status"] == "ERROR"
    assert (c["phase"], d["phase"]) == ("???", "incomplete")


@pytest.mark.parametrize("fault", ["duplicate", "clock", "tree"])
@pytest.mark.proves("pipeline.health_report_counts_the_test_metrics")
def test_the_health_report_counts_the_tree_and_the_stages(tmp_path, synthetic_history, fault):
    """The report counts what the tree and the receipts hold; an unreadable one gives no verdict."""
    from datetime import date
    from pathlib import Path

    from devtools.pipeline.definition import DefinitionError
    from devtools.pipeline.health import report
    from devtools.pipeline.health import stages as stage_health

    _, commit = synthetic_history
    root = Path(__file__).resolve().parents[2]
    policy = json.loads((root / "config/registries/test-sizes.json").read_bytes())
    held = {"node": "tests/a/test_x.py::test_c", "owner": "tests.a", "until": "2026-10-28"}
    sizes = tmp_path / "config/registries/test-sizes.json"
    sizes.parent.mkdir(parents=True)
    sizes.write_text(json.dumps(policy | {"quarantine": [held | {"reason": "r"}]}))
    (tmp_path / "tests/a").mkdir(parents=True)
    mark = '@pytest.mark.proves("a.b")\n'
    body = (
        f"import pytest\n{mark}def test_a(): pass\n{mark}def test_b(): pass\ndef test_c(): pass\n"
    )
    (tmp_path / "tests/a/test_x.py").write_text(body + "#" * 102400 + "\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src/m.py").write_text("x = 1\n" * 14)
    commit("A tree to report on")
    flaky = {"node": "t::f", "attempt": 1, "solo": False, "status": "FAILED", "seconds": 1}
    results = [RESULT, RESULT | {"node": "t::slow", "seconds": 12}, flaky]
    results.append(flaky | {"attempt": 2, "solo": True, "status": "PASSED"})
    paths = []
    for name, result, tested in (("pass", "PASS", results), ("error", "ERROR", [])):
        events, path = tmp_path / f"{name}.jsonl", tmp_path / f"{name}-receipt.jsonl"
        events.write_text("\n".join(map(json.dumps, tested)))
        started = {"stage": "commit", "host": "workshop", "command": ["x"]}
        children = [{"workspace": {}}, {"configuration": {}}]
        writer = receipt.Writer(path, started, children)
        writer.write(
            {"workspace": {}}, {"repo": "r", "base": "b" * 40, "head": HEAD, "clean": True}
        )
        writer.write({"configuration": {}}, CONFIGURATION)
        if tested:
            writer.tests(
                "tests", {"status": "PASS", "exit": 0, "seconds": 1, "provider": "p"}, events
            )
        writer.finish({"result": result, "exit": int(result != "PASS"), "red_nodes": []})
        paths.append(path)
    found = report(tmp_path, [*paths, tmp_path / "missing.jsonl"], date(2026, 10, 25))
    tree, stages = found["tree"], found["stages"]
    assert (tree["test_lines"], tree["source_lines"], tree["test_to_source"]) == (7, 14, 0.5)
    assert (tree["test_functions"], tree["registered_functions"]) == (3, 2)
    assert tree["requirements_with_several_holders"] == ["a.b"]
    assert tree["files_over_100kb"] == ["tests/a/test_x.py"]
    assert tree["quarantine"] == [held | {"days_left": 3}]
    assert stages["by_result"] == {"PASS": 1, "ERROR": 1, "UNREADABLE": 1}
    assert stages["error_rate"] == 0.667 and stages["unreadable"][0]["code"].endswith("missing")
    assert (stages["flaked_nodes"], stages["flake_rate"]) == (["t::f"], 0.3333)
    assert (stages["slow_tests"], stages["slow_share"]) == (1, 0.8)
    assert report(tmp_path, [], date(2026, 10, 25))["stages"]["error_rate"] is None

    if fault == "duplicate":
        copied = tmp_path / "copied.jsonl"
        copied.write_bytes(paths[0].read_bytes())
        assert (
            stage_health([paths[0], paths[0], copied, *paths[1:], tmp_path / "missing.jsonl"])
            == stages
        )
    elif fault == "clock":
        clocks = []
        for name, stamp in (
            ("newer", "2026-10-25T12:00:00+00:00"),
            ("older", "2026-10-25T13:00:00+02:00"),
        ):
            data = [json.loads(line) for line in paths[0].read_text().splitlines()]
            data[0]["started"]["invocation"] = name
            data[-1]["finished"]["finished_at"] = stamp
            path = tmp_path / (name + ".jsonl")
            path.write_text("\n".join(map(json.dumps, data)) + "\n")
            clocks.append(path)
        assert stage_health(clocks)["slow_from"] == str(clocks[0])
        for stamp in ("not-a-time", "2026-10-25T12:00:00"):
            data[-1]["finished"]["finished_at"] = stamp
            clocks[1].write_text("\n".join(map(json.dumps, data)) + "\n")
            invalid = stage_health([clocks[1]])
            assert (
                invalid["by_result"] == {"UNREADABLE": 1}
                and invalid["unreadable"][0]["code"] == "pipeline.receipt_malformed"
            )
    else:
        bad = tmp_path / "tests/a/test_x.py"
        bad.write_bytes(b"\xff")
        with pytest.raises(Exception) as caught:
            report(tmp_path, [], date(2026, 10, 25))
        assert (
            isinstance(caught.value, DefinitionError)
            and caught.value.code == "pipeline.workspace_unreadable"
            and str(bad) in str(caught.value)
        )
